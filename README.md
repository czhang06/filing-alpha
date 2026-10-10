# filing-alpha

**Do SEC 8-K filings move stock prices, and can sentiment in the filing text predict which way?**

A reproducible event study. It pulls 8-K filings from EDGAR, scores their text,
measures market-adjusted abnormal returns around each filing, and reports
whether anything survives correction for multiple testing.

The answer this produces may well be "no detectable effect." That is a real
finding, and the pipeline is built to make it a *credible* one — every null
result is reported next to the smallest effect the sample could have detected.

## Why 8-K filings instead of news headlines

Matching a headline to a ticker is unreliable and is where most retail
sentiment studies quietly break. Every 8-K already carries the filer's CIK, the
SEC publishes an official CIK→ticker map, and filings are timestamped to the
second and tagged with Item codes. So the event set is exact, and events can be
sliced by *kind* — a quarterly results release (Item 2.02) is a different animal
from an executive departure (Item 5.02), and pooling them averages real effects
toward zero.

## Method

For every filing:

1. **Date it correctly.** EDGAR's acceptance timestamp is Eastern wall-clock
   time despite its `Z` suffix. Filings accepted after 16:00 cannot move that
   day's close, so they are attributed to the next session. Roughly half of all
   8-Ks are filed after hours, so getting this wrong misdates half the sample.
2. **Fit a market model** on days −250 to −30 relative to the event:
   `R_it = α_i + β_i·R_mt + ε_it`, with SPY as the market proxy. The window ends
   well before the event so pre-announcement drift cannot contaminate β.
3. **Measure abnormal returns** over several event windows — `[0,0]`, `[0,+1]`,
   `[−1,+1]`, `[0,+5]` — and cumulate them into a CAR.
4. **Standardize.** Raw CARs are not comparable across stocks; 2% means
   something different for a utility than a biotech. Patell standardization
   divides each CAR by its own forecast standard error, accounting for the
   stock's residual variance, the error in having estimated α and β on finite
   data, and how unusual the event-window market return was.
5. **Test.** The headline test is BMP (Boehmer-Musumeci-Poulsen), which stays
   calibrated when the event itself inflates return variance — which
   announcements obviously do, and which makes the classic Patell Z over-reject.
   A distribution-free sign test and a bootstrap CI run alongside.
6. **Correct.** Slicing by Item code, window and sentiment tercile produces
   dozens of tests. At α=0.05 one in twenty comes back "significant" on noise
   alone, so Benjamini-Hochberg FDR is applied across *all* tests at once.
7. **Report power.** Every row carries the minimum detectable effect at 80%
   power. Without it, "we found nothing" is indistinguishable from "our sample
   was too small to find anything."

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows PowerShell
# source .venv/bin/activate         # macOS/Linux
pip install -r requirements.txt
```

FinBERT is optional and heavy (torch plus ~500 MB of weights):

```bash
pip install -r requirements-finbert.txt
```

## Running it

SEC fair-access rules require a real contact address in the User-Agent. The
client enforces this and throttles to 8 requests/second.

```bash
# Quick check: 100 companies, no text fetching. A few minutes.
python -m filing_alpha.pipeline \
  --user-agent "Your Name you@example.com" \
  --start 2024-01-01 --end 2025-12-31 \
  --max-companies 100

# Full run with lexicon sentiment. Tens of minutes, mostly EDGAR throttling.
python -m filing_alpha.pipeline \
  --user-agent "Your Name you@example.com" \
  --start 2023-01-01 --end 2025-12-31 \
  --max-companies 500 \
  --scorer lexicon

# Same events through FinBERT, to compare backends.
python -m filing_alpha.pipeline \
  --user-agent "Your Name you@example.com" \
  --start 2023-01-01 --end 2025-12-31 \
  --max-companies 500 \
  --scorer finbert
```

Results land in `reports/results.md` and `reports/results.csv`.

Every stage caches to `data/`, so re-running only redoes what changed. Pass
`--refresh` to force a rebuild.

### Useful options

| Flag | Purpose |
|---|---|
| `--universe-file` | Your own ticker list, one per line, instead of the default universe |
| `--windows` | Event windows as `start,end` offsets, e.g. `--windows 0,0 0,3` |
| `--estimation-start/--estimation-end` | Market-model estimation window |
| `--min-events` | Skip slices with fewer events than this (default 30) |
| `--fdr-q` | False discovery rate (default 0.05) |
| `--lexicon-csv` | Loughran-McDonald dictionary CSV (see below) |

## Tests

```bash
pytest --cov=filing_alpha
```

105 tests. The ones that matter most are the calibration checks, because a
mis-calibrated test turns noise into a discovery:

- Patell-standardized CARs are verified to be **standard normal under the
  null** across 1,500 simulated events — which is what proves the variance
  formula is right.
- An injected +3% shock is verified to come back as a +3% CAR, and a stock that
  merely tracks a 6% market move is verified to produce **zero** abnormal return.
- BMP is shown to hold its 5% rejection rate when event variance triples, in the
  same test where **Patell is shown to over-reject at >20%** — the empirical
  justification for which test is the headline.
- Benjamini-Hochberg is checked against the worked example from the 1995 paper.
- Bootstrap CI coverage is verified at ~95% over 400 simulated samples.

## Limitations

- **Default universe.** Without `--universe-file`, the universe is the head of
  SEC's `company_tickers.json`, which is roughly size-ordered but is not a
  formally defined index. For a clean study, supply your own constituent list.
- **Survivorship.** yfinance serves currently-listed tickers, so companies that
  delisted during the sample are under-represented. This biases toward survivors.
- **Daily data.** Intraday reaction to a 16:05 filing is invisible at daily
  resolution; the `[0,+1]` window is the practical substitute.
- **Sentiment lexicon.** The standard Loughran-McDonald finance dictionary is
  distributed by its authors, not bundled here. Without `--lexicon-csv`, a small
  hand-written fallback list is used, which is meaningfully weaker.
- **No trading costs.** Abnormal returns are not a strategy. Any effect found
  here would still need to clear spreads, impact and borrow costs.

## Layout

```
src/filing_alpha/
  edgar.py        EDGAR client, filing discovery, event dating
  prices.py       price download, trading-day alignment, window slicing
  eventstudy.py   market model, abnormal returns, Patell standardization
  stats.py        BMP, sign test, bootstrap, FDR, power
  scoring.py      lexicon and FinBERT sentiment backends
  pipeline.py     CLI orchestration and reporting
tests/            105 tests, no network required
```
