"""End-to-end pipeline: filings -> prices -> abnormal returns -> inference.

Run ``python -m filing_alpha.pipeline --help`` for options.

The pipeline is deliberately resumable. Each stage writes its output to
``data/`` and skips work it has already done, because the ingestion stage takes
tens of minutes and should not restart from zero every time a later stage is
tweaked.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import edgar, prices as prices_mod, stats as stats_mod
from .edgar import (
    SecClient,
    check_eastern_distribution,
    collect_filings,
    diagnose_acceptance_hours,
    load_ticker_map,
)
from .eventstudy import InsufficientData, run_event
from .prices import WindowConfig
from .scoring import build_scorer

DEFAULT_DATA_DIR = Path("data")
DEFAULT_REPORT_DIR = Path("reports")
_TAG = re.compile(r"<[^>]+>")


def build_universe(client: SecClient, max_companies: int, universe_file: Path | None):
    """(cik, ticker) pairs to study."""
    ticker_map = load_ticker_map(client)
    if universe_file:
        wanted = {
            line.strip().upper()
            for line in universe_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        }
        pairs = [(cik, tk) for cik, tk in ticker_map.items() if tk in wanted]
        missing = wanted - {tk for _, tk in pairs}
        if missing:
            print(f"  warning: no CIK for {len(missing)} tickers, e.g. {sorted(missing)[:5]}")
        return pairs
    # SEC orders company_tickers.json roughly by size, so the head of the file
    # is a reasonable large-cap sample. Documented as a limitation in the README.
    return list(ticker_map.items())[:max_companies]


def stage_filings(args, client: SecClient, data_dir: Path) -> pd.DataFrame:
    out = data_dir / "interim" / "filings.parquet"
    if out.exists() and not args.refresh:
        print(f"Using cached filings: {out}")
        return pd.read_parquet(out)

    print("Building universe...")
    universe = build_universe(client, args.max_companies, args.universe_file)
    print(f"  {len(universe)} companies")

    print(f"Collecting 8-K filings {args.start} to {args.end}...")
    filings, report = collect_filings(client, universe, args.start, args.end)
    print(f"  {report.summary()}")

    hours = diagnose_acceptance_hours(filings)
    verdict = check_eastern_distribution(hours["eastern"])
    print(f"  acceptance hours (UTC):     {hours['utc']}")
    print(f"  acceptance hours (Eastern): {hours['eastern']}")
    print(f"  timezone check: {verdict}")

    rows = []
    for f in filings:
        rows.append(
            {
                "cik": f.cik,
                "ticker": f.ticker,
                "company": f.company,
                "accession": f.accession,
                "filed_date": f.filed_date,
                "event_date": f.event_date(),
                "after_hours": f.after_hours,
                "items": ",".join(f.items),
                "primary_document": f.primary_document,
                "document_url": f.document_url,
            }
        )
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out)
    (data_dir / "interim" / "ingestion_report.json").write_text(
        json.dumps({**asdict(report), "acceptance_hours": hours, "timezone_check": verdict}, indent=2), encoding="utf-8"
    )
    return frame


def stage_sentiment(args, client: SecClient, filings: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach a sentiment score per filing. Skipped unless --scorer is set."""
    if not args.scorer:
        filings["sentiment"] = np.nan
        filings["sentiment_backend"] = ""
        return filings

    out = data_dir / "interim" / f"sentiment_{args.scorer}.parquet"
    if out.exists() and not args.refresh:
        print(f"Using cached sentiment: {out}")
        return filings.merge(pd.read_parquet(out), on="accession", how="left")

    print(f"Scoring filing text with '{args.scorer}'...")
    scorer = build_scorer(args.scorer, args.lexicon_csv)
    subset = filings.head(args.max_text) if args.max_text else filings

    records = []
    for n, row in enumerate(subset.itertuples(), start=1):
        try:
            raw = client.get(row.document_url, cache_key=f"docs/{row.accession}.htm")
            text = _TAG.sub(" ", raw)[: args.max_chars]
            result = scorer.score(text)
            records.append(
                {
                    "accession": row.accession,
                    "sentiment": result.score,
                    "sentiment_backend": result.backend,
                }
            )
        except Exception as exc:
            records.append({"accession": row.accession, "sentiment": np.nan, "sentiment_backend": ""})
            if n <= 5:
                print(f"  text fetch failed for {row.accession}: {exc}")
        if n % 100 == 0:
            print(f"  [{n}/{len(subset)}] scored", flush=True)

    scores = pd.DataFrame(records)
    out.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out)
    return filings.merge(scores, on="accession", how="left")


def stage_event_study(args, filings: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    out = data_dir / "interim" / "events.parquet"
    if out.exists() and not args.refresh:
        print(f"Using cached event study: {out}")
        return pd.read_parquet(out)

    config = WindowConfig(
        estimation_start=args.estimation_start,
        estimation_end=args.estimation_end,
        event_windows=tuple(tuple(int(x) for x in w.split(",")) for w in args.windows),
    )

    tickers = sorted(filings["ticker"].dropna().unique().tolist())
    # Pad the download so the earliest event still has a full estimation window.
    pad_days = int(abs(config.earliest_offset) * 1.6) + 10
    price_start = (pd.Timestamp(args.start) - pd.Timedelta(days=pad_days)).strftime("%Y-%m-%d")
    price_end = (pd.Timestamp(args.end) + pd.Timedelta(days=30)).strftime("%Y-%m-%d")

    print(f"Downloading prices for {len(tickers)} tickers ({price_start} to {price_end})...")
    price_frame = prices_mod.download_prices(
        tickers, price_start, price_end, cache_dir=data_dir / "cache"
    )
    returns = prices_mod.to_returns(price_frame)
    if prices_mod.MARKET_PROXY not in returns.columns:
        raise RuntimeError(f"market proxy {prices_mod.MARKET_PROXY} missing from price data")
    market = returns[prices_mod.MARKET_PROXY]
    calendar = returns.index

    print("Running event study...")
    rows = []
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    for n, filing in enumerate(filings.itertuples(), start=1):
        ticker = filing.ticker
        if ticker not in returns.columns:
            skip("no price series")
            continue
        stock = returns[ticker]
        index = prices_mod.locate_event_index(calendar, filing.event_date)
        if index is None:
            skip("event outside price history")
            continue

        for window in config.event_windows:
            extracted = prices_mod.extract_windows(stock, market, index, config, window)
            if extracted is None:
                skip("incomplete window")
                continue
            try:
                result = run_event(*extracted)
            except InsufficientData:
                skip("insufficient estimation data")
                continue
            rows.append(
                {
                    "accession": filing.accession,
                    "ticker": ticker,
                    "event_date": filing.event_date,
                    "after_hours": filing.after_hours,
                    "items": filing.items,
                    "sentiment": getattr(filing, "sentiment", np.nan),
                    "window": config.label(window),
                    "car": result.car,
                    "scar": result.scar,
                    "beta": result.beta,
                    "r_squared": result.r_squared,
                    "estimation_days": result.estimation_days,
                }
            )
        if n % 500 == 0:
            print(f"  [{n}/{len(filings)}] filings processed, {len(rows)} observations", flush=True)

    events = pd.DataFrame(rows)
    print(f"  {len(events)} usable observations; skipped: {skipped}")
    out.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(out)
    return events


def _slices(events: pd.DataFrame, min_events: int):
    """Yield (slice_name, subset) pairs to test."""
    yield "All filings", events
    for item in sorted({i for row in events["items"].fillna("") for i in row.split(",") if i}):
        subset = events[events["items"].fillna("").str.contains(re.escape(item))]
        if len(subset) >= min_events:
            label = edgar.ITEM_LABELS.get(item, "Other")
            yield f"Item {item} ({label})", subset
    if events["sentiment"].notna().any():
        scored = events[events["sentiment"].notna()]
        if len(scored) >= 3 * min_events:
            try:
                tercile = pd.qcut(scored["sentiment"], 3, labels=["negative", "neutral", "positive"])
            except ValueError:
                return  # too few distinct sentiment values to split
            for name in ("negative", "positive"):
                subset = scored[tercile == name]
                if len(subset) >= min_events:
                    yield f"Sentiment {name} tercile", subset


def stage_analysis(args, events: pd.DataFrame, report_dir: Path) -> pd.DataFrame:
    print("Aggregating...")
    rows = []
    for window in events["window"].unique():
        in_window = events[events["window"] == window]
        for name, subset in _slices(in_window, args.min_events):
            if len(subset) < args.min_events:
                continue
            scars = subset["scar"].to_numpy()
            result = stats_mod.analyse(
                subset["car"].to_numpy(),
                scars,
                estimation_days=int(subset["estimation_days"].median()),
            )
            spread = stats_mod.dispersion(scars)
            rows.append(
                {
                    "slice": name,
                    "window": window,
                    "n": result.n,
                    "mean_car": result.mean_car,
                    "median_car": result.median_car,
                    "ci_low": result.ci_low,
                    "ci_high": result.ci_high,
                    "pct_positive": result.positive_fraction,
                    "t_p": result.p_values["t"],
                    "bmp_p": result.p_values["bmp"],
                    "sign_p": result.p_values["sign"],
                    "mde": result.minimum_detectable_effect,
                    "abs_ratio": spread.abs_ratio,
                    "variance_ratio": spread.variance_ratio,
                    "abs_p": spread.abs_test.p_value,
                    "variance_p": spread.variance_test.p_value,
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        print("  no slices met the minimum event count")
        return table

    # FDR within each family of hypotheses. Direction (mean CAR) and magnitude
    # (variance) are different questions, so they are corrected separately
    # rather than pooled -- pooling would penalise each for the other's tests.
    table["significant_fdr"] = stats_mod.benjamini_hochberg(table["bmp_p"].to_numpy(), q=args.fdr_q)
    table["variance_fdr"] = stats_mod.benjamini_hochberg(
        table["variance_p"].to_numpy(), q=args.fdr_q
    )
    table = table.sort_values(["window", "slice"]).reset_index(drop=True)

    report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(report_dir / "results.csv", index=False)
    (report_dir / "results.md").write_text(render_report(table, events, args), encoding="utf-8")
    return table


def render_report(table: pd.DataFrame, events: pd.DataFrame, args) -> str:
    n_tests = len(table)
    n_direction = int(table["significant_fdr"].sum())
    n_variance = int(table["variance_fdr"].sum())
    lines = [
        "# Do 8-K filings move prices?",
        "",
        f"- Events: **{events['accession'].nunique():,}** filings, "
        f"**{events['ticker'].nunique():,}** tickers, {args.start} to {args.end}",
        f"- Observations: {len(events):,} (filing x event window)",
        f"- Market-model estimation window: [{args.estimation_start}, {args.estimation_end}] "
        "trading days; market proxy SPY",
        f"- Sentiment backend: {args.scorer or 'none'}",
        "",
        "Two different questions are asked of the same events:",
        "",
        f"1. **Direction** -- does the filing push the price up or down on average? "
        f"{n_direction} of {n_tests} tests significant after Benjamini-Hochberg at q={args.fdr_q}.",
        f"2. **Magnitude** -- does the filing make the price move *more*, in either direction? "
        f"{n_variance} of {n_tests} tests significant at the same threshold.",
        "",
        "The distinction matters. For earnings releases roughly half the surprises are good and "
        "half are bad, so the signed average can sit at zero even when every announcement moves "
        "the stock several percent. A mean test is structurally blind to that; the variance test "
        "is not.",
        "",
        "## 1. Direction: mean cumulative abnormal return",
        "",
        "`mde` is the smallest true effect this sample could have detected at 80% power. A null "
        "result is only meaningful next to it.",
        "",
        "| Slice | Window | n | Mean CAR | 95% CI | % positive | BMP p | FDR sig | MDE |",
        "|---|---|---:|---:|---|---:|---:|:-:|---:|",
    ]
    for row in table.itertuples():
        lines.append(
            f"| {row.slice} | {row.window} | {row.n:,} | {row.mean_car * 100:+.3f}% | "
            f"[{row.ci_low * 100:+.3f}%, {row.ci_high * 100:+.3f}%] | {row.pct_positive * 100:.1f}% | "
            f"{row.bmp_p:.4f} | {'yes' if row.significant_fdr else 'no'} | {row.mde * 100:.3f}% |"
        )

    lines += [
        "",
        "## 2. Magnitude: was the move bigger than a normal day?",
        "",
        "Both ratios are 1.00 on an ordinary day. `|AR| ratio` is mean absolute standardized "
        "abnormal return against its no-event expectation; `var ratio` is mean squared "
        "standardized abnormal return, so 2.00 means twice the usual return variance. Tests are "
        "one-sided: an unusually *calm* day is not evidence of a reaction.",
        "",
        "| Slice | Window | n | \\|AR\\| ratio | Var ratio | Variance p | FDR sig |",
        "|---|---|---:|---:|---:|---:|:-:|",
    ]
    for row in table.sort_values("variance_p").itertuples():
        lines.append(
            f"| {row.slice} | {row.window} | {row.n:,} | {row.abs_ratio:.2f}x | "
            f"{row.variance_ratio:.2f}x | {row.variance_p:.2e} | "
            f"{'yes' if row.variance_fdr else 'no'} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "A row is evidence only if its p-value survives the FDR column. An uncorrected p below "
        "0.05 means little with this many slices: several will clear that bar by chance alone.",
        "",
        "Where the direction table is null but the variance table is not, the reading is that "
        "filings reliably move prices without being predictable in sign -- consistent with the "
        "market pricing the news in, rather than with a tradeable signal.",
        "",
        "Nothing here is a strategy. Abnormal returns ignore spreads, market impact and borrow "
        "costs, and the variance result in particular is not directional.",
    ]
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="filing-alpha",
        description="Event study of SEC 8-K filings against market-adjusted returns.",
    )
    p.add_argument("--user-agent", required=True, help='SEC requires "Name email@example.com"')
    p.add_argument("--start", default="2024-01-01", help="first filing date (YYYY-MM-DD)")
    p.add_argument("--end", default="2025-12-31", help="last filing date (YYYY-MM-DD)")
    p.add_argument("--max-companies", type=int, default=500, help="universe size when no file given")
    p.add_argument("--universe-file", type=Path, help="file of tickers, one per line")
    p.add_argument("--scorer", choices=["lexicon", "finbert"], help="enable sentiment scoring")
    p.add_argument("--lexicon-csv", type=Path, help="Loughran-McDonald CSV for the lexicon scorer")
    p.add_argument("--max-text", type=int, default=2000, help="cap filings fetched for scoring")
    p.add_argument("--max-chars", type=int, default=20000, help="chars of filing text to score")
    p.add_argument("--estimation-start", type=int, default=-250)
    p.add_argument("--estimation-end", type=int, default=-30)
    p.add_argument("--windows", nargs="+", default=["0,0", "0,1", "-1,1", "0,5"])
    p.add_argument("--min-events", type=int, default=30, help="skip slices smaller than this")
    p.add_argument("--fdr-q", type=float, default=0.05)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--refresh", action="store_true", help="ignore cached stage outputs")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    data_dir = args.data_dir
    client = SecClient(user_agent=args.user_agent, cache_dir=data_dir / "cache" / "sec")

    filings = stage_filings(args, client, data_dir)
    if filings.empty:
        print("No filings found. Widen --start/--end or increase --max-companies.")
        return 1

    filings = stage_sentiment(args, client, filings, data_dir)
    events = stage_event_study(args, filings, data_dir)
    if events.empty:
        print("No usable events. Check price coverage for your universe.")
        return 1

    table = stage_analysis(args, events, args.report_dir)
    if not table.empty:
        print(f"\nWrote {args.report_dir / 'results.md'} and results.csv")
        print(
            f"direction: {int(table['significant_fdr'].sum())} of {len(table)} significant "
            f"after FDR at q={args.fdr_q}"
        )
        print(
            f"magnitude: {int(table['variance_fdr'].sum())} of {len(table)} significant "
            f"after FDR at q={args.fdr_q}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
