"""Correctness tests for the cross-sectional statistics.

The tests that matter most are the calibration ones: under the null, each test
should reject at roughly its nominal rate, and the bootstrap interval should
cover the truth about 95% of the time. A test that is silently over-rejecting
is how a null result turns into a fake discovery.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from filing_alpha.stats import (
    analyse,
    benjamini_hochberg,
    bmp_test,
    bootstrap_ci,
    minimum_detectable_effect,
    patell_z,
    sign_test,
    ttest_mean_car,
)

SEED = 20261010


class TestTTest:
    def test_matches_scipy(self):
        rng = np.random.default_rng(SEED)
        cars = rng.normal(0.002, 0.03, 250)
        expected = scipy_stats.ttest_1samp(cars, 0.0)

        result = ttest_mean_car(cars)

        assert result.statistic == pytest.approx(float(expected.statistic), rel=1e-12)
        assert result.p_value == pytest.approx(float(expected.pvalue), rel=1e-12)
        assert result.n == 250

    def test_ignores_non_finite_values(self):
        cars = np.array([0.01, 0.02, np.nan, 0.03, np.inf])
        assert ttest_mean_car(cars).n == 3

    def test_degenerate_input_returns_nan(self):
        assert np.isnan(ttest_mean_car([0.01]).p_value)


class TestBMP:
    def test_rejects_at_nominal_rate_under_the_null(self):
        rng = np.random.default_rng(SEED)
        rejections = 0
        trials = 1200
        for _ in range(trials):
            scars = rng.normal(0.0, 1.0, 120)
            if bmp_test(scars).p_value < 0.05:
                rejections += 1
        assert 0.03 < rejections / trials < 0.075

    def test_stays_calibrated_when_event_inflates_variance(self):
        """The whole point of BMP: a variance jump must not cause over-rejection."""
        rng = np.random.default_rng(SEED)
        rejections = 0
        trials = 1200
        for _ in range(trials):
            scars = rng.normal(0.0, 3.0, 120)  # same null mean, triple the spread
            if bmp_test(scars).p_value < 0.05:
                rejections += 1
        assert 0.03 < rejections / trials < 0.075

    def test_patell_over_rejects_when_variance_is_inflated(self):
        """Contrast case justifying why BMP is the headline test, not Patell."""
        rng = np.random.default_rng(SEED)
        rejections = 0
        trials = 600
        for _ in range(trials):
            scars = rng.normal(0.0, 3.0, 120)
            if patell_z(scars, estimation_days=200).p_value < 0.05:
                rejections += 1
        assert rejections / trials > 0.20  # badly mis-calibrated, as expected

    def test_detects_a_genuine_shift(self):
        rng = np.random.default_rng(SEED)
        scars = rng.normal(0.5, 1.0, 200)
        result = bmp_test(scars)
        assert result.statistic > 4
        assert result.p_value < 1e-4


class TestSignTest:
    def test_balanced_signs_are_not_significant(self):
        cars = np.array([0.01, -0.01] * 60)
        assert sign_test(cars).p_value == pytest.approx(1.0, abs=1e-9)

    def test_lopsided_signs_are_significant(self):
        cars = np.array([0.01] * 45 + [-0.01] * 5)
        result = sign_test(cars)
        assert result.p_value < 1e-6
        assert result.statistic == pytest.approx(0.9, abs=1e-9)

    def test_zero_values_are_excluded(self):
        assert sign_test(np.array([0.0, 0.0, 0.01, -0.01])).n == 2


class TestBootstrap:
    def test_interval_contains_the_sample_mean(self):
        rng = np.random.default_rng(SEED)
        cars = rng.normal(0.004, 0.02, 400)
        low, high = bootstrap_ci(cars)
        assert low < cars.mean() < high

    def test_coverage_is_about_95_percent(self):
        rng = np.random.default_rng(SEED)
        true_mean = 0.003
        covered = 0
        trials = 400
        for i in range(trials):
            sample = rng.normal(true_mean, 0.025, 250)
            low, high = bootstrap_ci(sample, iterations=600, seed=i)
            covered += low <= true_mean <= high
        assert 0.90 < covered / trials < 0.99

    def test_is_deterministic_for_a_fixed_seed(self):
        rng = np.random.default_rng(SEED)
        cars = rng.normal(0.0, 0.02, 100)
        assert bootstrap_ci(cars, seed=7) == bootstrap_ci(cars, seed=7)

    def test_narrows_as_the_sample_grows(self):
        rng = np.random.default_rng(SEED)
        small = bootstrap_ci(rng.normal(0.0, 0.02, 50))
        large = bootstrap_ci(rng.normal(0.0, 0.02, 5000))
        assert (large[1] - large[0]) < (small[1] - small[0])


class TestMinimumDetectableEffect:
    def test_matches_the_closed_form(self):
        mde = minimum_detectable_effect(n=100, std=0.05, alpha=0.05, power=0.80)
        expected = (1.959963985 + 0.841621234) * 0.05 / 10.0
        assert mde == pytest.approx(expected, rel=1e-6)

    def test_shrinks_with_the_square_root_of_n(self):
        small = minimum_detectable_effect(n=100, std=0.05)
        large = minimum_detectable_effect(n=400, std=0.05)
        assert large == pytest.approx(small / 2.0, rel=1e-9)

    def test_tiny_samples_cannot_detect_much(self):
        """Grounds the v1 problem: n=18 could only have found a huge effect."""
        mde = minimum_detectable_effect(n=18, std=0.05)
        assert mde > 0.03  # would have needed a >3% average abnormal return

    def test_degenerate_input_returns_nan(self):
        assert np.isnan(minimum_detectable_effect(n=1, std=0.05))
        assert np.isnan(minimum_detectable_effect(n=100, std=0.0))


class TestBenjaminiHochberg:
    def test_matches_the_benjamini_hochberg_1995_example(self):
        """The 15 p-values from the original paper; it rejects the 4 smallest at q=0.05."""
        p = np.array(
            [
                0.0001, 0.0004, 0.0019, 0.0095, 0.0201,
                0.0278, 0.0298, 0.0344, 0.0459, 0.3240,
                0.4262, 0.5719, 0.6528, 0.7590, 1.0000,
            ]
        )
        rejected = benjamini_hochberg(p, q=0.05)
        assert rejected.sum() == 4
        assert rejected[:4].all()
        assert not rejected[4:].any()

    def test_rejection_set_is_a_prefix_in_p_value_order(self):
        """BH is a step-up procedure: a rejected hypothesis implies all smaller ones are too."""
        p = np.array([0.04, 0.001, 0.2, 0.01, 0.9])
        rejected = benjamini_hochberg(p, q=0.10)
        order = np.argsort(p)
        in_order = rejected[order]
        assert not (np.diff(in_order.astype(int)) > 0).any()

    def test_is_less_strict_than_bonferroni(self):
        p = np.array([0.001, 0.008, 0.02, 0.04, 0.2])
        bh = benjamini_hochberg(p, q=0.05).sum()
        bonferroni = (p <= 0.05 / p.size).sum()
        assert bh >= bonferroni

    def test_rejects_nothing_when_all_p_values_are_large(self):
        assert not benjamini_hochberg(np.array([0.4, 0.6, 0.9])).any()

    def test_controls_false_discoveries_under_the_global_null(self):
        rng = np.random.default_rng(SEED)
        false_positive_runs = 0
        trials = 500
        for _ in range(trials):
            p = rng.uniform(0, 1, 30)  # every null true
            if benjamini_hochberg(p, q=0.05).any():
                false_positive_runs += 1
        assert false_positive_runs / trials < 0.10

    def test_handles_nan_and_empty_input(self):
        assert benjamini_hochberg(np.array([])).size == 0
        mask = benjamini_hochberg(np.array([0.001, np.nan, 0.9]))
        assert mask.tolist() == [True, False, False]


class TestAnalyse:
    def test_reports_a_full_result(self):
        rng = np.random.default_rng(SEED)
        cars = rng.normal(0.0, 0.03, 300)
        scars = cars / 0.03

        result = analyse(cars, scars)

        assert result.n == 300
        assert result.ci_low < result.mean_car < result.ci_high
        assert 0.0 < result.positive_fraction < 1.0
        assert result.minimum_detectable_effect > 0
        assert set(result.p_values) == {"t", "bmp", "sign", "patell"}

    def test_flags_a_real_effect_across_every_test(self):
        rng = np.random.default_rng(SEED)
        cars = rng.normal(0.02, 0.02, 300)
        scars = cars / 0.02

        result = analyse(cars, scars)

        assert result.mean_car > 0.015
        assert result.ci_low > 0
        assert result.p_values["t"] < 1e-6
        assert result.p_values["bmp"] < 1e-6
        assert result.p_values["sign"] < 1e-6
