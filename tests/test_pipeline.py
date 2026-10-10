"""Tests for the aggregation and reporting path.

Network stages are not exercised here. What is tested is the part that turns
per-event numbers into the claims in the report: slicing, the FDR correction
across all tests, and the rendered table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from filing_alpha.pipeline import _slices, parse_args, render_report, stage_analysis


def make_events(n=400, effect=0.0, seed=3, items="2.02", windows=("[+0,+0]",)):
    rng = np.random.default_rng(seed)
    rows = []
    for window in windows:
        for i in range(n):
            car = rng.normal(effect, 0.03)
            rows.append(
                {
                    "accession": f"acc-{window}-{i}",
                    "ticker": f"T{i % 50}",
                    "event_date": "2025-06-02",
                    "after_hours": bool(i % 2),
                    "items": items if isinstance(items, str) else items[i % len(items)],
                    "sentiment": rng.normal(0, 0.4),
                    "window": window,
                    "car": car,
                    "scar": car / 0.03,
                    "beta": 1.0,
                    "r_squared": 0.3,
                    "estimation_days": 221,
                }
            )
    return pd.DataFrame(rows)


def default_args(**overrides):
    args = parse_args(["--user-agent", "Test test@example.com"])
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class TestSlices:
    def test_always_includes_the_full_sample(self):
        events = make_events(n=100)
        names = [name for name, _ in _slices(events, min_events=30)]
        assert names[0] == "All filings"

    def test_splits_by_item_code(self):
        events = make_events(n=200, items=["2.02", "5.02"])
        names = [name for name, _ in _slices(events, min_events=30)]
        assert any("Item 2.02" in n for n in names)
        assert any("Item 5.02" in n for n in names)

    def test_labels_item_codes_with_their_meaning(self):
        events = make_events(n=100, items="2.02")
        names = [name for name, _ in _slices(events, min_events=30)]
        assert any("Results of operations" in n for n in names)

    def test_drops_slices_below_the_minimum(self):
        events = make_events(n=200, items=["2.02"] * 39 + ["9.99"])
        names = [name for name, _ in _slices(events, min_events=30)]
        assert not any("9.99" in n for n in names)

    def test_adds_sentiment_terciles_when_scored(self):
        events = make_events(n=300)
        names = [name for name, _ in _slices(events, min_events=30)]
        assert "Sentiment negative tercile" in names
        assert "Sentiment positive tercile" in names

    def test_omits_sentiment_slices_when_unscored(self):
        events = make_events(n=300)
        events["sentiment"] = np.nan
        names = [name for name, _ in _slices(events, min_events=30)]
        assert not any("Sentiment" in n for n in names)

    def test_handles_filings_carrying_multiple_items(self):
        events = make_events(n=120, items="2.02,7.01")
        names = [name for name, _ in _slices(events, min_events=30)]
        assert any("Item 2.02" in n for n in names)
        assert any("Item 7.01" in n for n in names)


class TestStageAnalysis:
    def test_finds_nothing_when_there_is_nothing(self, tmp_path):
        """Pure noise must not produce a surviving result."""
        events = make_events(n=400, effect=0.0, items=["2.02", "5.02", "7.01", "8.01"])
        table = stage_analysis(default_args(), events, tmp_path)
        assert not table.empty
        assert int(table["significant_fdr"].sum()) == 0

    def test_finds_a_real_effect(self, tmp_path):
        events = make_events(n=400, effect=0.02)
        table = stage_analysis(default_args(), events, tmp_path)
        assert table["significant_fdr"].any()
        assert table.loc[table["slice"] == "All filings", "mean_car"].iloc[0] > 0.015

    def test_fdr_is_applied_across_every_test_at_once(self, tmp_path):
        """Correcting per-window instead of globally would be too permissive."""
        events = make_events(
            n=300, effect=0.0, items=["2.02", "5.02", "7.01"], windows=("[+0,+0]", "[+0,+1]", "[-1,+1]")
        )
        table = stage_analysis(default_args(), events, tmp_path)
        assert len(table) > 6  # several slices x several windows
        assert int(table["significant_fdr"].sum()) == 0

    def test_writes_both_output_files(self, tmp_path):
        stage_analysis(default_args(), make_events(n=200), tmp_path)
        assert (tmp_path / "results.csv").exists()
        assert (tmp_path / "results.md").exists()

    def test_reports_a_minimum_detectable_effect_per_row(self, tmp_path):
        table = stage_analysis(default_args(), make_events(n=200), tmp_path)
        assert (table["mde"] > 0).all()

    def test_confidence_interval_brackets_the_mean(self, tmp_path):
        table = stage_analysis(default_args(), make_events(n=300, effect=0.01), tmp_path)
        assert ((table["ci_low"] < table["mean_car"]) & (table["mean_car"] < table["ci_high"])).all()

    def test_small_sample_yields_no_rows(self, tmp_path):
        table = stage_analysis(default_args(min_events=500), make_events(n=50), tmp_path)
        assert table.empty


class TestRenderReport:
    def test_includes_counts_and_the_fdr_verdict(self):
        events = make_events(n=200)
        args = default_args()
        table = pd.DataFrame(
            {
                "slice": ["All filings"],
                "window": ["[+0,+0]"],
                "n": [200],
                "mean_car": [0.0012],
                "median_car": [0.001],
                "ci_low": [-0.001],
                "ci_high": [0.0034],
                "pct_positive": [0.52],
                "t_p": [0.3],
                "bmp_p": [0.31],
                "sign_p": [0.6],
                "mde": [0.0059],
                "abs_ratio": [1.02],
                "variance_ratio": [1.05],
                "abs_p": [0.4],
                "variance_p": [0.42],
                "significant_fdr": [False],
                "variance_fdr": [False],
            }
        )
        text = render_report(table, events, args)
        assert "significant after Benjamini-Hochberg" in text
        assert "+0.120%" in text  # mean CAR rendered as a percentage
        assert "| no |" in text
        assert "MDE" in text

    def test_states_how_to_read_a_null_result(self):
        events = make_events(n=100)
        table = pd.DataFrame(
            {
                "slice": ["All filings"], "window": ["[+0,+0]"], "n": [100],
                "mean_car": [0.0], "median_car": [0.0], "ci_low": [-0.01], "ci_high": [0.01],
                "pct_positive": [0.5], "t_p": [0.9], "bmp_p": [0.9], "sign_p": [0.9],
                "mde": [0.008], "abs_ratio": [1.0], "variance_ratio": [1.0],
                "abs_p": [0.5], "variance_p": [0.5],
                "significant_fdr": [False], "variance_fdr": [False],
            }
        )
        text = render_report(table, events, default_args())
        # A null result must be reported next to what it could have detected,
        # and must not be dressed up as a tradeable finding.
        assert "smallest true effect this sample could have detected" in text
        assert "Nothing here is a strategy" in text


class TestParseArgs:
    def test_requires_a_user_agent(self):
        with pytest.raises(SystemExit):
            parse_args([])

    def test_defaults_are_sensible(self):
        args = parse_args(["--user-agent", "A a@b.com"])
        assert args.estimation_start == -250
        assert args.estimation_end == -30
        assert args.windows == ["0,0", "0,1", "-1,1", "0,5"]
        assert args.fdr_q == 0.05

    def test_rejects_an_unknown_scorer(self):
        with pytest.raises(SystemExit):
            parse_args(["--user-agent", "A a@b.com", "--scorer", "vader"])


class TestDispersionReporting:
    def test_variance_columns_are_produced(self, tmp_path):
        table = stage_analysis(default_args(), make_events(n=300), tmp_path)
        for column in ("abs_ratio", "variance_ratio", "variance_p", "variance_fdr"):
            assert column in table.columns

    def test_normal_days_give_ratios_near_one(self, tmp_path):
        """make_events draws scar = car/0.03 with car ~ N(0, 0.03), i.e. the null."""
        table = stage_analysis(default_args(), make_events(n=600), tmp_path)
        row = table[table["slice"] == "All filings"].iloc[0]
        assert row["variance_ratio"] == pytest.approx(1.0, abs=0.2)
        assert not bool(row["variance_fdr"])

    def test_detects_inflated_variance_with_no_directional_effect(self, tmp_path):
        rng = np.random.default_rng(11)
        events = make_events(n=600, effect=0.0)
        # Triple the spread, keep the mean at zero.
        events["car"] = rng.normal(0.0, 0.09, len(events))
        events["scar"] = events["car"] / 0.03
        table = stage_analysis(default_args(), events, tmp_path)
        row = table[table["slice"] == "All filings"].iloc[0]

        assert row["variance_ratio"] > 5
        assert bool(row["variance_fdr"])
        assert not bool(row["significant_fdr"])  # mean test stays blind

    def test_report_contains_both_sections(self, tmp_path):
        stage_analysis(default_args(), make_events(n=300), tmp_path)
        text = (tmp_path / "results.md").read_text(encoding="utf-8")
        assert "## 1. Direction" in text
        assert "## 2. Magnitude" in text
        assert "Var ratio" in text
