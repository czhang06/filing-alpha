"""Correctness tests for the event-study core.

These use synthetic data where the right answer is known in advance, which is
the only way to be sure the abnormal-return machinery is actually measuring
what it claims to.
"""

from __future__ import annotations

import numpy as np
import pytest

from filing_alpha.eventstudy import (
    InsufficientData,
    abnormal_returns,
    car_forecast_std,
    fit_market_model,
    run_event,
)

RNG_SEED = 20261010
EST_DAYS = 200


def make_returns(rng, n, alpha=0.0, beta=1.0, noise=0.01, market_vol=0.01):
    market = rng.normal(0.0004, market_vol, n)
    stock = alpha + beta * market + rng.normal(0.0, noise, n)
    return stock, market


class TestMarketModelFit:
    def test_recovers_known_alpha_and_beta_without_noise(self):
        rng = np.random.default_rng(RNG_SEED)
        market = rng.normal(0.0005, 0.01, EST_DAYS)
        stock = 0.0003 + 1.4 * market  # exact, no noise

        fit = fit_market_model(stock, market)

        assert fit.beta == pytest.approx(1.4, abs=1e-9)
        assert fit.alpha == pytest.approx(0.0003, abs=1e-9)
        assert fit.residual_std == pytest.approx(0.0, abs=1e-9)
        assert fit.r_squared == pytest.approx(1.0, abs=1e-9)

    def test_recovers_beta_with_noise(self):
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, 2000, alpha=0.0002, beta=0.8, noise=0.005)

        fit = fit_market_model(stock, market)

        assert fit.beta == pytest.approx(0.8, abs=0.05)
        assert fit.residual_std == pytest.approx(0.005, rel=0.1)

    def test_rejects_short_estimation_window(self):
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, 10)
        with pytest.raises(InsufficientData, match="usable days"):
            fit_market_model(stock, market)

    def test_rejects_constant_market(self):
        rng = np.random.default_rng(RNG_SEED)
        market = np.zeros(EST_DAYS)
        stock = rng.normal(0, 0.01, EST_DAYS)
        with pytest.raises(InsufficientData, match="constant"):
            fit_market_model(stock, market)

    def test_drops_non_finite_observations(self):
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, EST_DAYS, beta=1.2, noise=0.0)
        stock[5] = np.nan
        market[9] = np.inf

        fit = fit_market_model(stock, market)

        assert fit.n_obs == EST_DAYS - 2
        assert fit.beta == pytest.approx(1.2, abs=1e-9)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            fit_market_model(np.zeros(10), np.zeros(11))


class TestAbnormalReturns:
    def test_zero_when_stock_follows_the_model_exactly(self):
        rng = np.random.default_rng(RNG_SEED)
        market = rng.normal(0.0005, 0.01, EST_DAYS)
        stock = 0.0002 + 1.1 * market
        fit = fit_market_model(stock, market)

        event_market = np.array([0.01, -0.005, 0.002])
        event_stock = 0.0002 + 1.1 * event_market

        ar = abnormal_returns(fit, event_stock, event_market)

        assert np.allclose(ar, 0.0, atol=1e-12)

    def test_recovers_an_injected_shock(self):
        """A clean +3% event-day jump should show up as a +3% CAR."""
        rng = np.random.default_rng(RNG_SEED)
        market = rng.normal(0.0005, 0.01, EST_DAYS)
        stock = 0.0002 + 1.1 * market
        fit = fit_market_model(stock, market)

        event_market = np.array([0.004, -0.002, 0.001])
        event_stock = 0.0002 + 1.1 * event_market
        event_stock[1] += 0.03  # the shock

        ar = abnormal_returns(fit, event_stock, event_market)

        assert ar.sum() == pytest.approx(0.03, abs=1e-9)
        assert ar[1] == pytest.approx(0.03, abs=1e-9)
        assert ar[0] == pytest.approx(0.0, abs=1e-12)

    def test_market_move_alone_produces_no_abnormal_return(self):
        """A stock that just tracks a big market day is not an event."""
        rng = np.random.default_rng(RNG_SEED)
        market = rng.normal(0.0005, 0.01, EST_DAYS)
        stock = 1.5 * market
        fit = fit_market_model(stock, market)

        event_market = np.array([-0.06])  # market crashes 6%
        event_stock = 1.5 * event_market  # stock falls 9%, exactly as expected

        ar = abnormal_returns(fit, event_stock, event_market)

        assert ar[0] == pytest.approx(0.0, abs=1e-12)


class TestPatellStandardization:
    def test_forecast_std_exceeds_naive_scaling(self):
        """Estimation error must inflate the standard error above s*sqrt(L2)."""
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, EST_DAYS, noise=0.02)
        fit = fit_market_model(stock, market)

        event_market = np.array([0.01, 0.005, -0.003])
        naive = fit.residual_std * np.sqrt(len(event_market))

        assert car_forecast_std(fit, event_market) > naive

    def test_forecast_std_grows_with_unusual_market_moves(self):
        """An event window far from the estimation mean is predicted less precisely."""
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, EST_DAYS, noise=0.01, market_vol=0.008)
        fit = fit_market_model(stock, market)

        typical = car_forecast_std(fit, np.array([fit.market_mean]))
        extreme = car_forecast_std(fit, np.array([fit.market_mean + 0.10]))

        assert extreme > typical

    def test_scar_is_standard_normal_under_the_null(self):
        """The key statistical check.

        Simulate many events with no effect at all. If the standardization is
        right, the SCARs should look like draws from N(0, 1): mean near zero,
        standard deviation near one. A wrong variance formula shows up here
        immediately as an sd meaningfully different from 1.
        """
        rng = np.random.default_rng(RNG_SEED)
        scars = []
        for _ in range(1500):
            stock, market = make_returns(rng, EST_DAYS + 3, beta=1.2, noise=0.015)
            scars.append(
                run_event(stock[:EST_DAYS], market[:EST_DAYS], stock[EST_DAYS:], market[EST_DAYS:]).scar
            )
        scars = np.array(scars)

        assert abs(scars.mean()) < 0.08
        assert scars.std(ddof=1) == pytest.approx(1.0, abs=0.07)
        # A nominal 5% two-sided test should reject near 5% of the time.
        assert 0.025 < float((np.abs(scars) > 1.96).mean()) < 0.085

    def test_scar_detects_a_real_effect(self):
        """With a genuine +4% shock, SCARs should shift strongly positive."""
        rng = np.random.default_rng(RNG_SEED)
        scars = []
        for _ in range(300):
            stock, market = make_returns(rng, EST_DAYS + 1, beta=1.0, noise=0.015)
            stock[EST_DAYS] += 0.04
            scars.append(
                run_event(stock[:EST_DAYS], market[:EST_DAYS], stock[EST_DAYS:], market[EST_DAYS:]).scar
            )
        scars = np.array(scars)

        assert scars.mean() > 2.0
        assert float((scars > 1.96).mean()) > 0.5


class TestRunEvent:
    def test_reports_fit_diagnostics(self):
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, EST_DAYS + 3, beta=0.9, noise=0.01)
        result = run_event(stock[:EST_DAYS], market[:EST_DAYS], stock[EST_DAYS:], market[EST_DAYS:])

        assert result.estimation_days == EST_DAYS
        assert result.event_days == 3
        assert result.beta == pytest.approx(0.9, abs=0.15)
        assert result.forecast_std > 0
        assert result.scar == pytest.approx(result.car / result.forecast_std, rel=1e-12)

    def test_rejects_missing_event_window_data(self):
        rng = np.random.default_rng(RNG_SEED)
        stock, market = make_returns(rng, EST_DAYS + 2)
        event_stock = np.array([0.01, np.nan])
        with pytest.raises(InsufficientData, match="missing returns"):
            run_event(stock[:EST_DAYS], market[:EST_DAYS], event_stock, market[EST_DAYS:])
