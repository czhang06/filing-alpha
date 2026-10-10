"""Tests for trading-day alignment and window slicing.

Off-by-one errors here would silently measure the wrong days, so the windows
are checked against hand-computed positions on a synthetic calendar.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from filing_alpha.prices import WindowConfig, extract_windows, locate_event_index, to_returns

# Weekdays only, so weekends are genuinely absent from the index.
CALENDAR = pd.bdate_range("2025-01-01", periods=400)


def make_series(n=400, seed=1):
    rng = np.random.default_rng(seed)
    return (
        pd.Series(rng.normal(0, 0.01, n), index=CALENDAR[:n]),
        pd.Series(rng.normal(0, 0.008, n), index=CALENDAR[:n]),
    )


class TestToReturns:
    def test_computes_simple_returns(self):
        prices = pd.DataFrame({"A": [100.0, 110.0, 99.0]})
        returns = to_returns(prices)
        assert np.isnan(returns["A"].iloc[0])
        assert returns["A"].iloc[1] == pytest.approx(0.10)
        assert returns["A"].iloc[2] == pytest.approx(-0.10)

    def test_division_by_zero_becomes_nan(self):
        prices = pd.DataFrame({"A": [0.0, 5.0, 6.0]})
        assert np.isnan(to_returns(prices)["A"].iloc[1])


class TestLocateEventIndex:
    def test_finds_an_exact_trading_day(self):
        idx = locate_event_index(CALENDAR, CALENDAR[10].strftime("%Y-%m-%d"))
        assert idx == 10

    def test_weekend_event_maps_to_the_next_session(self):
        saturday = pd.Timestamp("2025-01-04")
        assert saturday.dayofweek == 5
        idx = locate_event_index(CALENDAR, "2025-01-04")
        assert CALENDAR[idx] == pd.Timestamp("2025-01-06")  # the Monday

    def test_date_after_the_history_returns_none(self):
        assert locate_event_index(CALENDAR, "2030-01-01") is None

    def test_date_before_the_history_maps_to_the_first_day(self):
        assert locate_event_index(CALENDAR, "2020-01-01") == 0


class TestWindowConfig:
    def test_reports_its_own_geometry(self):
        config = WindowConfig(estimation_start=-250, estimation_end=-30)
        assert config.estimation_length == 221
        assert config.earliest_offset == -250
        assert config.latest_offset == 5

    def test_labels_windows_readably(self):
        config = WindowConfig()
        assert config.label((0, 0)) == "[+0,+0]"
        assert config.label((-1, 1)) == "[-1,+1]"


class TestExtractWindows:
    def test_slices_the_expected_lengths(self):
        stock, market = make_series()
        config = WindowConfig()
        result = extract_windows(stock, market, event_index=300, config=config, event_window=(0, 1))
        assert result is not None
        est_stock, est_market, event_stock, event_market = result
        assert est_stock.size == config.estimation_length == 221
        assert est_market.size == 221
        assert event_stock.size == 2
        assert event_market.size == 2

    def test_slices_the_exact_days(self):
        """Day 0 of the event window must be the event day itself."""
        stock, market = make_series()
        result = extract_windows(
            stock, market, event_index=300, config=WindowConfig(), event_window=(0, 0)
        )
        assert result[2][0] == pytest.approx(stock.iloc[300])

    def test_negative_start_includes_the_prior_day(self):
        stock, market = make_series()
        result = extract_windows(
            stock, market, event_index=300, config=WindowConfig(), event_window=(-1, 1)
        )
        event_stock = result[2]
        assert event_stock.size == 3
        assert event_stock[0] == pytest.approx(stock.iloc[299])
        assert event_stock[1] == pytest.approx(stock.iloc[300])
        assert event_stock[2] == pytest.approx(stock.iloc[301])

    def test_estimation_window_ends_before_the_event(self):
        """Nothing from the run-up may leak into the alpha/beta estimate."""
        stock, market = make_series()
        config = WindowConfig(estimation_start=-250, estimation_end=-30)
        est_stock = extract_windows(stock, market, 300, config, (0, 0))[0]
        assert est_stock[0] == pytest.approx(stock.iloc[50])
        assert est_stock[-1] == pytest.approx(stock.iloc[270])

    def test_event_too_early_in_the_history_is_rejected(self):
        stock, market = make_series()
        assert extract_windows(stock, market, 10, WindowConfig(), (0, 1)) is None

    def test_event_too_late_in_the_history_is_rejected(self):
        stock, market = make_series(n=320)
        assert extract_windows(stock, market, 318, WindowConfig(), (0, 5)) is None

    def test_missing_event_day_price_is_rejected(self):
        """A halted or untraded stock must drop out rather than contribute noise."""
        stock, market = make_series()
        stock.iloc[300] = np.nan
        assert extract_windows(stock, market, 300, WindowConfig(), (0, 0)) is None

    def test_missing_estimation_data_is_tolerated(self):
        """Gaps before the event are fine; fit_market_model drops them."""
        stock, market = make_series()
        stock.iloc[100] = np.nan
        result = extract_windows(stock, market, 300, WindowConfig(), (0, 0))
        assert result is not None
        assert np.isnan(result[0]).sum() == 1
