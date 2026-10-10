"""Market-model event study.

For each event we estimate a market model on a clean window well before the
event, then measure how far the stock's actual returns during the event window
depart from what that model predicts.

    R_it = alpha_i + beta_i * R_mt + e_it          (estimation window)
    AR_it = R_it - (alpha_i + beta_i * R_mt)       (event window)
    CAR_i = sum of AR_it over the event window

Raw CARs are not comparable across stocks: a 2% move means something different
for a utility than for a biotech. We therefore also report Patell-standardized
CARs, which divide each CAR by its own forecast standard error. That error
accounts for three things, which is the part people usually get wrong:

  1. the residual variance of the stock,
  2. the extra uncertainty from having estimated alpha and beta on a finite
     sample (the ``L2^2 / L1`` term), and
  3. the extra uncertainty from the event window's market return being far from
     the mean market return seen during estimation (the final term).

References: Patell (1976); Boehmer, Musumeci & Poulsen (1991) for the
event-induced-variance-robust cross-sectional test in ``stats.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# An estimation window shorter than this cannot support a stable alpha/beta.
MIN_ESTIMATION_DAYS = 30


class InsufficientData(ValueError):
    """Raised when an event does not have enough clean data to be usable."""


@dataclass(frozen=True)
class MarketModelFit:
    """OLS fit of one stock's returns on the market, over the estimation window."""

    alpha: float
    beta: float
    residual_std: float  # s_i, using (n - 2) degrees of freedom
    n_obs: int  # L1
    market_mean: float  # mean market return over the estimation window
    market_sum_sq_dev: float  # sum (R_mt - mean)^2 over the estimation window
    r_squared: float

    @property
    def degrees_of_freedom(self) -> int:
        return self.n_obs - 2

    def predict(self, market_returns: np.ndarray) -> np.ndarray:
        return self.alpha + self.beta * np.asarray(market_returns, dtype=float)


def fit_market_model(stock_returns: np.ndarray, market_returns: np.ndarray) -> MarketModelFit:
    """Estimate alpha and beta by OLS over the estimation window."""
    y = np.asarray(stock_returns, dtype=float)
    x = np.asarray(market_returns, dtype=float)
    if y.shape != x.shape:
        raise ValueError(f"shape mismatch: stock {y.shape} vs market {x.shape}")

    finite = np.isfinite(y) & np.isfinite(x)
    y, x = y[finite], x[finite]
    n = y.size
    if n < MIN_ESTIMATION_DAYS:
        raise InsufficientData(f"estimation window has {n} usable days, need >= {MIN_ESTIMATION_DAYS}")

    x_mean = x.mean()
    sum_sq_dev = float(((x - x_mean) ** 2).sum())
    if sum_sq_dev <= 0:
        raise InsufficientData("market returns are constant over the estimation window")

    beta = float(((x - x_mean) * (y - y.mean())).sum() / sum_sq_dev)
    alpha = float(y.mean() - beta * x_mean)

    residuals = y - (alpha + beta * x)
    sse = float((residuals**2).sum())
    residual_std = float(np.sqrt(sse / (n - 2)))

    total_ss = float(((y - y.mean()) ** 2).sum())
    r_squared = float(1.0 - sse / total_ss) if total_ss > 0 else 0.0

    return MarketModelFit(
        alpha=alpha,
        beta=beta,
        residual_std=residual_std,
        n_obs=n,
        market_mean=x_mean,
        market_sum_sq_dev=sum_sq_dev,
        r_squared=r_squared,
    )


def abnormal_returns(
    fit: MarketModelFit, stock_returns: np.ndarray, market_returns: np.ndarray
) -> np.ndarray:
    """Event-window abnormal returns: actual minus market-model prediction."""
    y = np.asarray(stock_returns, dtype=float)
    x = np.asarray(market_returns, dtype=float)
    if y.shape != x.shape:
        raise ValueError(f"shape mismatch: stock {y.shape} vs market {x.shape}")
    return y - fit.predict(x)


def car_forecast_std(fit: MarketModelFit, event_market_returns: np.ndarray) -> float:
    """Patell forecast standard error of a CAR over the given event window.

    S_i = s_i * sqrt( L2 + L2^2/L1 + (sum(R_mt - mbar))^2 / sum_est(R_mt - mbar)^2 )
    """
    x = np.asarray(event_market_returns, dtype=float)
    length = x.size
    if length == 0:
        raise ValueError("event window is empty")

    deviation_sum = float((x - fit.market_mean).sum())
    variance_inflation = (
        length
        + (length**2) / fit.n_obs
        + (deviation_sum**2) / fit.market_sum_sq_dev
    )
    return float(fit.residual_std * np.sqrt(variance_inflation))


@dataclass(frozen=True)
class EventResult:
    """One event's outcome over one event window."""

    car: float
    scar: float  # Patell-standardized CAR; ~N(0, 1) under the null
    forecast_std: float
    beta: float
    residual_std: float
    r_squared: float
    estimation_days: int
    event_days: int


def run_event(
    estimation_stock: np.ndarray,
    estimation_market: np.ndarray,
    event_stock: np.ndarray,
    event_market: np.ndarray,
) -> EventResult:
    """Fit the market model on the estimation window and score the event window."""
    fit = fit_market_model(estimation_stock, estimation_market)

    event_stock = np.asarray(event_stock, dtype=float)
    event_market = np.asarray(event_market, dtype=float)
    if not np.isfinite(event_stock).all() or not np.isfinite(event_market).all():
        raise InsufficientData("event window contains missing returns")

    ar = abnormal_returns(fit, event_stock, event_market)
    car = float(ar.sum())
    forecast_std = car_forecast_std(fit, event_market)
    if forecast_std <= 0:
        raise InsufficientData("degenerate forecast standard error")

    return EventResult(
        car=car,
        scar=car / forecast_std,
        forecast_std=forecast_std,
        beta=fit.beta,
        residual_std=fit.residual_std,
        r_squared=fit.r_squared,
        estimation_days=fit.n_obs,
        event_days=int(event_stock.size),
    )
