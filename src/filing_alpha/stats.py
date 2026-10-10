"""Cross-sectional inference for a set of events.

Three things here exist specifically to stop the analysis fooling itself:

* **BMP test** (Boehmer, Musumeci & Poulsen 1991). The naive Patell Z assumes
  the event does not change return variance. Corporate announcements obviously
  do, which makes Patell over-reject. BMP uses the cross-sectional spread of
  the standardized returns instead, so it stays honest when volatility jumps.
* **Benjamini-Hochberg FDR.** Slicing events by Item code and window gives
  dozens of tests. At a 5% threshold, one in twenty comes back "significant"
  on noise alone. Reporting uncorrected p-values across that many slices is
  the single easiest way to manufacture a finding that is not there.
* **Minimum detectable effect.** The honest answer to "we found nothing" is
  only meaningful alongside "and here is the smallest effect we could have
  seen". Without it, a null result is indistinguishable from a sample that was
  simply too small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class TestResult:
    name: str
    statistic: float
    p_value: float
    n: int

    def __str__(self) -> str:
        return f"{self.name}: stat={self.statistic:.3f}, p={self.p_value:.4f}, n={self.n}"


@dataclass(frozen=True)
class CrossSectionResult:
    """Everything reported for one (slice, event window) cell."""

    n: int
    mean_car: float
    median_car: float
    std_car: float
    positive_fraction: float
    ttest: TestResult
    bmp: TestResult
    sign_test: TestResult
    ci_low: float
    ci_high: float
    minimum_detectable_effect: float
    p_values: dict[str, float] = field(default_factory=dict)


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[np.isfinite(arr)]


def ttest_mean_car(cars) -> TestResult:
    """Plain cross-sectional t-test that mean CAR is zero."""
    cars = _clean(cars)
    if cars.size < 2:
        return TestResult("cross-sectional t", float("nan"), float("nan"), cars.size)
    result = scipy_stats.ttest_1samp(cars, 0.0)
    return TestResult("cross-sectional t", float(result.statistic), float(result.pvalue), cars.size)


def bmp_test(scars) -> TestResult:
    """Boehmer-Musumeci-Poulsen standardized cross-sectional test.

        Z = sqrt(N) * mean(SCAR) / sd(SCAR)

    Dividing by the *observed* cross-sectional spread of the SCARs is what makes
    this robust to the event itself inflating variance.
    """
    scars = _clean(scars)
    n = scars.size
    if n < 2:
        return TestResult("BMP", float("nan"), float("nan"), n)
    spread = scars.std(ddof=1)
    if spread <= 0:
        return TestResult("BMP", float("nan"), float("nan"), n)
    z = float(np.sqrt(n) * scars.mean() / spread)
    p = float(2.0 * scipy_stats.norm.sf(abs(z)))
    return TestResult("BMP", z, p, n)


def patell_z(scars, estimation_days: int) -> TestResult:
    """Classic Patell Z. Included for comparison; BMP is the one to trust."""
    scars = _clean(scars)
    n = scars.size
    if n < 2 or estimation_days <= 4:
        return TestResult("Patell Z", float("nan"), float("nan"), n)
    variance = (estimation_days - 2) / (estimation_days - 4)
    z = float(scars.sum() / np.sqrt(n * variance))
    p = float(2.0 * scipy_stats.norm.sf(abs(z)))
    return TestResult("Patell Z", z, p, n)


def sign_test(cars) -> TestResult:
    """Binomial test that positive and negative CARs are equally likely.

    Distribution-free, so it is unmoved by the fat tails that make the t-test
    nervous on return data.
    """
    cars = _clean(cars)
    nonzero = cars[cars != 0]
    n = nonzero.size
    if n < 1:
        return TestResult("sign test", float("nan"), float("nan"), n)
    positives = int((nonzero > 0).sum())
    result = scipy_stats.binomtest(positives, n, 0.5, alternative="two-sided")
    return TestResult("sign test", float(positives / n), float(result.pvalue), n)


# Under the null, SCARs are standard normal (verified in test_eventstudy.py),
# so E[|SCAR|] = sqrt(2/pi) and E[SCAR^2] = 1. Those are the no-event baselines
# the dispersion tests below measure against.
EXPECTED_ABS_STANDARD_NORMAL = float(np.sqrt(2.0 / np.pi))


@dataclass(frozen=True)
class DispersionResult:
    """Did the event make the stock move *more*, regardless of direction?

    This is the question a mean-CAR test cannot answer. For earnings releases
    roughly half the surprises are good and half bad, so the signed average is
    near zero by construction even when every single announcement moves the
    stock several percent. Dispersion is where that effect lives.
    """

    n: int
    abs_ratio: float  # mean|SCAR| / sqrt(2/pi); 1.0 means a normal day
    variance_ratio: float  # mean SCAR^2; 1.0 means a normal day
    abs_test: TestResult
    variance_test: TestResult


def abs_return_test(scars) -> tuple[TestResult, float]:
    """One-sided test that mean |SCAR| exceeds its no-event expectation."""
    values = np.abs(_clean(scars))
    n = values.size
    if n < 2:
        return TestResult("|SCAR| vs no-event", float("nan"), float("nan"), n), float("nan")
    ratio = float(values.mean() / EXPECTED_ABS_STANDARD_NORMAL)
    spread = values.std(ddof=1)
    if spread <= 0:
        return TestResult("|SCAR| vs no-event", float("nan"), float("nan"), n), ratio
    t = float((values.mean() - EXPECTED_ABS_STANDARD_NORMAL) / (spread / np.sqrt(n)))
    p = float(scipy_stats.t.sf(t, df=n - 1))  # one-sided: only an increase is interesting
    return TestResult("|SCAR| vs no-event", t, p, n), ratio


def variance_ratio_test(scars) -> tuple[TestResult, float]:
    """One-sided test that event-window return variance exceeds the normal level.

    Tested as a t-test on SCAR^2 rather than a chi-squared test, because
    returns are fat-tailed and chi-squared assumes normality of the underlying
    far more strongly than a mean test on the squares does.
    """
    values = _clean(scars) ** 2
    n = values.size
    if n < 2:
        return TestResult("variance ratio", float("nan"), float("nan"), n), float("nan")
    ratio = float(values.mean())
    spread = values.std(ddof=1)
    if spread <= 0:
        return TestResult("variance ratio", float("nan"), float("nan"), n), ratio
    t = float((ratio - 1.0) / (spread / np.sqrt(n)))
    p = float(scipy_stats.t.sf(t, df=n - 1))
    return TestResult("variance ratio", t, p, n), ratio


def dispersion(scars) -> DispersionResult:
    """Run both dispersion tests on one slice."""
    abs_test, abs_ratio = abs_return_test(scars)
    var_test, var_ratio = variance_ratio_test(scars)
    return DispersionResult(
        n=_clean(scars).size,
        abs_ratio=abs_ratio,
        variance_ratio=var_ratio,
        abs_test=abs_test,
        variance_test=var_test,
    )


def bootstrap_ci(
    values, confidence: float = 0.95, iterations: int = 10_000, seed: int = 20261010
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean."""
    values = _clean(values)
    if values.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(iterations, values.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    low, high = np.quantile(draws, [tail, 1.0 - tail])
    return (float(low), float(high))


def minimum_detectable_effect(
    n: int, std: float, alpha: float = 0.05, power: float = 0.80
) -> float:
    """Smallest true mean CAR this sample could detect at the given power.

    MDE = (z_{1-alpha/2} + z_{power}) * sd / sqrt(n)
    """
    if n < 2 or not np.isfinite(std) or std <= 0:
        return float("nan")
    z_alpha = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
    z_power = scipy_stats.norm.ppf(power)
    return float((z_alpha + z_power) * std / np.sqrt(n))


def benjamini_hochberg(p_values, q: float = 0.05) -> np.ndarray:
    """Return a boolean mask of hypotheses rejected at FDR level q."""
    p = np.asarray(p_values, dtype=float)
    m = p.size
    rejected = np.zeros(m, dtype=bool)
    if m == 0:
        return rejected

    usable = np.isfinite(p)
    if not usable.any():
        return rejected

    idx = np.where(usable)[0]
    order = idx[np.argsort(p[idx], kind="stable")]
    n_tests = order.size
    thresholds = (np.arange(1, n_tests + 1) / n_tests) * q
    below = p[order] <= thresholds
    if not below.any():
        return rejected
    cutoff = int(np.max(np.where(below)[0]))
    rejected[order[: cutoff + 1]] = True
    return rejected


def analyse(cars, scars, estimation_days: int = 200, alpha: float = 0.05) -> CrossSectionResult:
    """Run the full battery on one slice of events."""
    cars = _clean(cars)
    scars = _clean(scars)
    n = cars.size

    t_result = ttest_mean_car(cars)
    bmp_result = bmp_test(scars)
    sign_result = sign_test(cars)
    low, high = bootstrap_ci(cars)
    std = float(cars.std(ddof=1)) if n > 1 else float("nan")

    return CrossSectionResult(
        n=n,
        mean_car=float(cars.mean()) if n else float("nan"),
        median_car=float(np.median(cars)) if n else float("nan"),
        std_car=std,
        positive_fraction=float((cars > 0).mean()) if n else float("nan"),
        ttest=t_result,
        bmp=bmp_result,
        sign_test=sign_result,
        ci_low=low,
        ci_high=high,
        minimum_detectable_effect=minimum_detectable_effect(n, std, alpha=alpha),
        p_values={
            "t": t_result.p_value,
            "bmp": bmp_result.p_value,
            "sign": sign_result.p_value,
            "patell": patell_z(scars, estimation_days).p_value,
        },
    )
