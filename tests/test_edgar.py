"""Tests for the pure logic in the EDGAR layer.

No network here. The event-date rule is the thing worth testing hardest: about
half of all 8-Ks are filed after the close, so a wrong rule misdates half the
sample by one day and quietly destroys any real effect.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from filing_alpha.edgar import (
    Filing,
    SecClient,
    _parse_accepted,
    check_eastern_distribution,
    diagnose_acceptance_hours,
)


def make_filing(filed="2026-03-10", accepted="2026-03-10T09:15:00.000Z", items=("2.02",)):
    return Filing(
        cik=320193,
        ticker="AAPL",
        company="Apple Inc.",
        accession="0000320193-26-000012",
        filed_date=filed,
        accepted_at=_parse_accepted(accepted),
        items=tuple(items),
        primary_document="a8k.htm",
    )


class TestParseAccepted:
    def test_parses_as_utc(self):
        """EDGAR's acceptanceDateTime is UTC, confirmed from the real hour
        distribution: filings are absent 06:00-09:00 and peak at 20:00-21:00,
        which is only coherent as UTC."""
        parsed = _parse_accepted("2026-03-10T20:31:02.000Z")
        assert parsed == datetime(2026, 3, 10, 20, 31, 2, tzinfo=timezone.utc)
        assert parsed.tzinfo is not None

    def test_handles_missing_and_malformed_values(self):
        assert _parse_accepted(None) is None
        assert _parse_accepted("") is None
        assert _parse_accepted("not a timestamp") is None

    def test_assumes_utc_when_no_offset_is_given(self):
        assert _parse_accepted("2026-03-10T08:00:00") == datetime(
            2026, 3, 10, 8, 0, 0, tzinfo=timezone.utc
        )


class TestEasternConversion:
    def test_converts_to_eastern_daylight_time(self):
        local = make_filing(accepted="2026-07-15T20:30:00.000Z").eastern()
        assert (local.hour, local.minute) == (16, 30)
        assert local.utcoffset().total_seconds() == -4 * 3600

    def test_converts_to_eastern_standard_time(self):
        """Same UTC hour, winter: the offset must change or dates slip in Jan."""
        local = make_filing(accepted="2026-01-15T20:30:00.000Z").eastern()
        assert (local.hour, local.minute) == (15, 30)
        assert local.utcoffset().total_seconds() == -5 * 3600

    def test_utc_after_midnight_is_the_previous_eastern_day(self):
        local = make_filing(filed="2026-03-10", accepted="2026-03-11T01:30:00.000Z").eastern()
        assert local.date().isoformat() == "2026-03-10"
        assert local.hour == 21


class TestEventDate:
    def test_morning_filing_belongs_to_the_same_day(self):
        # 13:15 UTC = 09:15 ET, before the open but same session
        assert make_filing(accepted="2026-03-10T13:15:00.000Z").event_date() == "2026-03-10"

    def test_midsession_filing_belongs_to_the_same_day(self):
        # 18:00 UTC = 14:00 ET
        assert make_filing(accepted="2026-03-10T18:00:00.000Z").event_date() == "2026-03-10"

    def test_filing_just_before_the_close_belongs_to_the_same_day(self):
        # 19:59 UTC = 15:59 ET
        assert make_filing(accepted="2026-03-10T19:59:00.000Z").event_date() == "2026-03-10"

    def test_filing_just_after_the_close_rolls_forward(self):
        # 20:01 UTC = 16:01 ET
        assert make_filing(accepted="2026-03-10T20:01:00.000Z").event_date() == "2026-03-11"

    def test_late_evening_filing_rolls_forward(self):
        # 01:30 UTC next day = 21:30 ET same day
        assert (
            make_filing(filed="2026-03-10", accepted="2026-03-11T01:30:00.000Z").event_date()
            == "2026-03-11"
        )

    def test_winter_close_uses_the_standard_time_offset(self):
        """21:30 UTC is 16:30 EST in January but 17:30 EDT in July; both roll."""
        assert make_filing(filed="2026-01-15", accepted="2026-01-15T21:30:00.000Z").event_date() == "2026-01-16"
        # 20:30 UTC in January is only 15:30 EST, still the same session
        assert make_filing(filed="2026-01-15", accepted="2026-01-15T20:30:00.000Z").event_date() == "2026-01-15"

    def test_rolls_across_a_month_boundary(self):
        filing = make_filing(filed="2026-03-31", accepted="2026-03-31T22:00:00.000Z")
        assert filing.event_date() == "2026-04-01"

    def test_rolls_across_a_year_boundary(self):
        filing = make_filing(filed="2026-12-31", accepted="2026-12-31T22:00:00.000Z")
        assert filing.event_date() == "2027-01-01"

    def test_falls_back_to_the_filing_date_without_a_timestamp(self):
        filing = make_filing(accepted=None)
        assert filing.accepted_at is None
        assert filing.event_date() == "2026-03-10"

    def test_close_hour_is_configurable(self):
        # 18:00 UTC = 14:00 ET; with a 13:00 close it rolls forward
        filing = make_filing(accepted="2026-03-10T18:00:00.000Z")
        assert filing.event_date(market_close_hour=13) == "2026-03-11"

    def test_after_hours_flag_uses_eastern_market_hours(self):
        assert make_filing(accepted="2026-03-10T20:30:00.000Z").after_hours  # 16:30 ET
        assert make_filing(accepted="2026-03-10T12:00:00.000Z").after_hours  # 08:00 ET, pre-market
        assert not make_filing(accepted="2026-03-10T15:00:00.000Z").after_hours  # 11:00 ET


class TestFilingUrls:
    def test_builds_the_archive_url(self):
        filing = make_filing()
        assert filing.accession_nodash == "000032019326000012"
        assert filing.document_url == (
            "https://www.sec.gov/Archives/edgar/data/320193/000032019326000012/a8k.htm"
        )


class TestSecClient:
    def test_requires_a_contact_email_in_the_user_agent(self):
        with pytest.raises(ValueError, match="contact email"):
            SecClient(user_agent="my-scraper/1.0")

    def test_accepts_a_compliant_user_agent(self):
        client = SecClient(user_agent="Charles Zhang charles@example.com")
        assert "@" in client.session.headers["User-Agent"]


class TestDiagnostics:
    def test_reports_both_utc_and_eastern_histograms(self):
        filings = [
            make_filing(accepted="2026-03-10T12:00:00.000Z"),  # 08:00 ET
            make_filing(accepted="2026-03-10T20:30:00.000Z"),  # 16:30 ET
            make_filing(accepted="2026-03-10T20:45:00.000Z"),  # 16:45 ET
            make_filing(accepted=None),
        ]
        result = diagnose_acceptance_hours(filings)
        assert result["utc"] == {12: 1, 20: 2}
        assert result["eastern"] == {8: 1, 16: 2}

    def test_flags_an_implausible_eastern_distribution(self):
        """The check that caught the original UTC-vs-Eastern bug."""
        bad = {0: 295, 1: 58, 2: 173, 20: 295, 21: 174}
        assert "WARNING" in check_eastern_distribution(bad)

    def test_accepts_a_plausible_eastern_distribution(self):
        good = {7: 40, 8: 120, 9: 90, 12: 60, 15: 80, 16: 300, 17: 150, 20: 30}
        assert check_eastern_distribution(good).startswith("OK")
