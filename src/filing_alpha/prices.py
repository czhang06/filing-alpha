"""Price data and trading-day window extraction.

Split deliberately into a thin network layer (``download_prices``) and pure
functions that do the index arithmetic. The arithmetic is where off-by-one
errors live, so it is kept testable without touching the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MARKET_PROXY = "SPY"


@dataclass(frozen=True)
class WindowConfig:
    """Trading-day offsets relative to the event day (day 0).

    The estimation window stops well before the event so that pre-announcement
    drift or leakage does not contaminate the alpha/beta estimate.
    """

    estimation_start: int = -250
    estimation_end: int = -30
    event_windows: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (-1, 1), (0, 5))

    @property
    def estimation_length(self) -> int:
        return self.estimation_end - self.estimation_start + 1

    @property
    def earliest_offset(self) -> int:
        return min(self.estimation_start, min(start for start, _ in self.event_windows))

    @property
    def latest_offset(self) -> int:
        return max(0, max(end for _, end in self.event_windows))

    def label(self, window: tuple[int, int]) -> str:
        start, end = window
        return f"[{start:+d},{end:+d}]"


def select_cached(cached: pd.DataFrame, needed: list[str]) -> pd.DataFrame | None:
    """Pick usable columns out of a cached price frame, or None if unusable.

    The market proxy must survive: every abnormal return is measured against
    it, so a cache missing SPY is worse than no cache at all. Individual
    tickers may legitimately be absent (delisted, bad symbol) and are simply
    dropped -- the event study already skips tickers with no price series.
    """
    if MARKET_PROXY not in cached.columns:
        return None
    available = [ticker for ticker in needed if ticker in cached.columns]
    if len(available) < 2:  # nothing but the proxy is not worth returning
        return None
    return cached[available]


def download_prices(
    tickers: list[str], start: str, end: str, cache_dir: Path | None = None, batch_size: int = 100
) -> pd.DataFrame:
    """Adjusted close prices, one column per ticker, always including the market proxy."""
    needed = sorted(set(tickers) | {MARKET_PROXY})

    cache_path = cache_dir / f"prices_{start}_{end}.parquet" if cache_dir else None
    if cache_path and cache_path.exists():
        usable = select_cached(pd.read_parquet(cache_path), needed)
        if usable is not None:
            return usable
        print("  cached prices unusable (market proxy missing); re-downloading")

    import yfinance as yf

    frames = []
    unique = needed
    for i in range(0, len(unique), batch_size):
        batch = unique[i : i + batch_size]
        print(f"  prices {i + 1}-{min(i + batch_size, len(unique))} of {len(unique)}", flush=True)
        raw = yf.download(
            batch, start=start, end=end, auto_adjust=True, progress=False, threads=True
        )
        if raw is None or raw.empty:
            continue
        closes = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
        if isinstance(closes, pd.Series):
            closes = closes.to_frame(batch[0])
        frames.append(closes)

    if not frames:
        raise RuntimeError("no price data returned; check tickers and date range")

    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.loc[:, ~prices.columns.duplicated()]
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
    return prices


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns, with non-finite values dropped to NaN."""
    returns = prices.astype(float).pct_change()
    return returns.replace([np.inf, -np.inf], np.nan)


def locate_event_index(trading_days: pd.DatetimeIndex, event_date: str) -> int | None:
    """Index of the first trading day on or after ``event_date``.

    An 8-K filed on a Saturday, or after Friday's close, maps forward to the
    next session. Returns None when the date falls outside the price history.
    """
    target = pd.Timestamp(event_date)
    position = trading_days.searchsorted(target, side="left")
    if position >= len(trading_days):
        return None
    return int(position)


def extract_windows(
    stock_returns: pd.Series,
    market_returns: pd.Series,
    event_index: int,
    config: WindowConfig,
    event_window: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Slice estimation and event windows around an event.

    Returns ``(est_stock, est_market, event_stock, event_market)`` or None when
    the event sits too close to either end of the price history for the full
    windows to exist.
    """
    start, end = event_window
    first_needed = event_index + min(config.estimation_start, start)
    last_needed = event_index + max(end, 0)
    if first_needed < 0 or last_needed >= len(stock_returns):
        return None

    est_slice = slice(event_index + config.estimation_start, event_index + config.estimation_end + 1)
    event_slice = slice(event_index + start, event_index + end + 1)

    est_stock = stock_returns.iloc[est_slice].to_numpy(dtype=float)
    est_market = market_returns.iloc[est_slice].to_numpy(dtype=float)
    event_stock = stock_returns.iloc[event_slice].to_numpy(dtype=float)
    event_market = market_returns.iloc[event_slice].to_numpy(dtype=float)

    if event_stock.size == 0 or not np.isfinite(event_stock).all():
        return None
    if not np.isfinite(event_market).all():
        return None
    return est_stock, est_market, event_stock, event_market
