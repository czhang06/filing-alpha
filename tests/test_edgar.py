"""Tests for the pure logic in the EDGAR layer.

No network here. The event-date rule is the thing worth testing hardest: about
half of all 8-Ks are filed after the close, so a wrong rule misdates half the
sample by one day and quietly destroys any real effect.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from filing_alpha.edgar import (
    Filing,
    SecClient,
    _parse_accepted,
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
    def test_reads_the_wall_clock_hour_not_a_utc_conversion(self):
        parsed = _parse_accepted("2026-03-10T16:31:02.000Z")
        assert parsed == datetime(2026, 3, 10, 16, 31, 2)
        assert parsed.tzinfo is None

    def test_handles_missing_and_malformed_values(self):
        assert _parse_accepted(None) is None
        assert _parse_accepted("") is None
        assert _parse_accepted("not a timestamp") is None

    def test_handles_a_plain_timestamp(self):
        assert _parse_accepted("2026-03-10T08:00:00") == datetime(2026, 3, 10, 8, 0, 0)


class TestEventDate:
    def test_morning_filing_belongs_to_the_same_day(self):
        assert make_filing(accepted="2026-03-10T09:15:00.000Z").event_date() == "2026-03-10"

    def test_filing_just_before_the_close_belongs_to_the_same_day(self):
        assert make_filing(accepted="2026-03-10T15:59:00.000Z").event_date() == "2026-03-10"

    def test_after_hours_filing_rolls_to_the_next_day(self):
        assert make_filing(accepted="2026-03-10T16:31:00.000Z").event_date() == "2026-03-11"

    def test_late_evening_filing_rolls_to_the_next_day(self):
        assert make_filing(accepted="2026-03-10T21:05:00.000Z").event_date() == "2026-03-11"

    def test_rolls_across_a_month_boundary(self):
        filing = make_filing(filed="2026-03-31", accepted="2026-03-31T18:00:00.000Z")
        assert filing.event_date() == "2026-04-01"

    def test_rolls_across_a_year_boundary(self):
        filing = make_filing(filed="2026-12-31", accepted="2026-12-31T17:00:00.000Z")
        assert filing.event_date() == "2027-01-01"

    def test_falls_back_to_the_filing_date_without_a_timestamp(self):
        filing = make_filing(accepted=None)
        assert filing.accepted_at is None
        assert filing.event_date() == "2026-03-10"

    def test_close_hour_is_configurable(self):
        filing = make_filing(accepted="2026-03-10T14:00:00.000Z")
        assert filing.event_date(market_close_hour=13) == "2026-03-11"

    def test_after_hours_flag_matches_the_rule(self):
        assert make_filing(accepted="2026-03-10T16:00:00.000Z").after_hours
        assert not make_filing(accepted="2026-03-10T15:00:00.000Z").after_hours


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
    def test_acceptance_hour_histogram(self):
        filings = [
            make_filing(accepted="2026-03-10T08:00:00.000Z"),
            make_filing(accepted="2026-03-10T16:30:00.000Z"),
            make_filing(accepted="2026-03-10T16:45:00.000Z"),
            make_filing(accepted=None),
        ]
        assert diagnose_acceptance_hours(filings) == {8: 1, 16: 2}
