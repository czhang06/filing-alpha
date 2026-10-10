"""Pull 8-K filings from SEC EDGAR.

Why 8-Ks rather than news headlines: every filing already carries the CIK of
the company that filed it, and the SEC publishes an official CIK-to-ticker map.
That deletes the headline-to-ticker matching problem entirely. Filings are also
timestamped to the second and categorised by Item code, so events can be sliced
by *kind* (earnings release, executive departure, material agreement) instead of
lumped together.

The company submissions endpoint returns every filing a company has ever made
in one response, including acceptance timestamps and Item codes, so building a
universe of tens of thousands of events costs one request per company.

SEC fair-access rules: declare a real contact address in the User-Agent and stay
under 10 requests/second. ``SecClient`` enforces both.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator

import requests

SEC_RATE_LIMIT_PER_SECOND = 8.0  # SEC allows 10; leave headroom
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

# 8-K Item codes worth separating. Markets should react very differently to a
# quarterly results release than to a new credit facility, and pooling them
# would average any real effect away.
ITEM_LABELS = {
    "1.01": "Material agreement entered",
    "1.02": "Material agreement terminated",
    "2.01": "Completion of acquisition",
    "2.02": "Results of operations",
    "2.03": "Direct financial obligation",
    "2.05": "Costs of exit/disposal",
    "2.06": "Material impairment",
    "3.01": "Listing/standard non-compliance",
    "4.01": "Change in accountant",
    "4.02": "Non-reliance on prior statements",
    "5.01": "Change in control",
    "5.02": "Director/officer departure",
    "7.01": "Regulation FD disclosure",
    "8.01": "Other events",
}


class SecClient:
    """Rate-limited HTTP client that complies with SEC fair-access rules."""

    def __init__(self, user_agent: str, cache_dir: Path | None = None, max_retries: int = 3):
        if "@" not in user_agent:
            raise ValueError(
                "SEC requires a contact email in the User-Agent, "
                'e.g. "Charles Zhang charles@example.com"'
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self._min_interval = 1.0 / SEC_RATE_LIMIT_PER_SECOND
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.monotonic()

    def get(self, url: str, cache_key: str | None = None) -> str:
        """Fetch a URL, using the on-disk cache when a key is supplied.

        Caching matters here: re-running the pipeline should not re-download
        thousands of filings, and it keeps repeat runs polite.
        """
        cache_path = self.cache_dir / cache_key if (self.cache_dir and cache_key) else None
        if cache_path and cache_path.exists():
            return cache_path.read_text(encoding="utf-8", errors="replace")

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._throttle()
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 429:  # throttled; back off and retry
                    time.sleep(2.0 * (attempt + 1))
                    continue
                response.raise_for_status()
                text = response.text
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(text, encoding="utf-8")
                return text
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"failed to fetch {url} after {self.max_retries} attempts") from last_error


def load_ticker_map(client: SecClient) -> dict[int, str]:
    """CIK -> ticker, straight from the SEC's own mapping file."""
    raw = json.loads(client.get(TICKER_MAP_URL, cache_key="company_tickers.json"))
    mapping: dict[int, str] = {}
    for entry in raw.values():
        cik = int(entry["cik_str"])
        mapping.setdefault(cik, str(entry["ticker"]).upper())
    return mapping


@dataclass(frozen=True)
class Filing:
    """One 8-K filing."""

    cik: int
    ticker: str
    company: str
    accession: str
    filed_date: str  # YYYY-MM-DD, SEC filing date
    accepted_at: datetime | None  # precise acceptance timestamp, UTC
    items: tuple[str, ...]  # e.g. ("2.02", "7.01")
    primary_document: str

    @property
    def accession_nodash(self) -> str:
        return self.accession.replace("-", "")

    @property
    def document_url(self) -> str:
        return ARCHIVE_URL.format(
            cik=self.cik, accession_nodash=self.accession_nodash, document=self.primary_document
        )

    def event_date(self, market_close_hour: int = 16) -> str:
        """Calendar date the filing should be attributed to.

        A filing accepted after the 16:00 close cannot move that day's closing
        price, so it belongs to the next session. Getting this wrong shifts
        every after-hours filing by a day, which is the most common silent bug
        in event studies -- and roughly half of all 8-Ks are filed after hours.

        Timezone note: EDGAR's ``acceptanceDateTime`` is written with a ``Z``
        suffix but is US/Eastern wall-clock time, not UTC. We therefore read the
        hour as Eastern directly. ``diagnose_acceptance_hours`` prints the
        distribution so this can be confirmed against real data rather than
        taken on faith -- as Eastern it should cluster in business hours with a
        spike just after 16:00; as UTC it would sit implausibly overnight.

        This returns a calendar date. Mapping it onto the next *trading* day
        (weekends, holidays) happens in ``prices.py``, which knows the exchange
        calendar implied by the price series.
        """
        if self.accepted_at is None:
            return self.filed_date
        if self.accepted_at.hour >= market_close_hour:
            return _next_calendar_day(self.filed_date)
        return self.filed_date

    @property
    def after_hours(self) -> bool:
        return self.accepted_at is not None and self.accepted_at.hour >= 16


def _next_calendar_day(date_str: str) -> str:
    from datetime import date, timedelta

    y, m, d = (int(part) for part in date_str.split("-"))
    return (date(y, m, d) + timedelta(days=1)).isoformat()


def _parse_accepted(value: str | None) -> datetime | None:
    """Parse EDGAR's acceptance timestamp as naive US/Eastern wall-clock time.

    EDGAR emits e.g. "2026-02-11T16:31:02.000Z". The trailing Z is misleading --
    the clock reading is Eastern. We strip any offset and keep the wall-clock
    reading rather than converting, because converting would move every
    after-hours filing into the wrong session.
    """
    if not value:
        return None
    cleaned = value.strip().rstrip("Z")
    for suffix_len in (6, 0):  # tolerate a trailing +00:00 style offset
        candidate = cleaned[:-suffix_len] if suffix_len else cleaned
        try:
            return datetime.fromisoformat(candidate).replace(tzinfo=None)
        except ValueError:
            continue
    return None


def diagnose_acceptance_hours(filings: list["Filing"]) -> dict[int, int]:
    """Histogram of acceptance hours, to confirm the Eastern-time reading.

    Expect a business-hours shape with a pronounced spike at 16:00-18:00 (the
    post-close earnings window). A flat overnight distribution would mean the
    timestamps are not what we assume and ``_parse_accepted`` needs revisiting.
    """
    histogram: dict[int, int] = {}
    for filing in filings:
        if filing.accepted_at is not None:
            hour = filing.accepted_at.hour
            histogram[hour] = histogram.get(hour, 0) + 1
    return dict(sorted(histogram.items()))


def iter_company_8ks(
    client: SecClient,
    cik: int,
    ticker: str,
    start_date: str,
    end_date: str,
) -> Iterator[Filing]:
    """Yield every 8-K a company filed in the date range.

    One request per company covers its entire recent filing history.
    """
    payload = json.loads(
        client.get(SUBMISSIONS_URL.format(cik=cik), cache_key=f"submissions/CIK{cik:010d}.json")
    )
    company = payload.get("name", "")
    recent = payload.get("filings", {}).get("recent", {})

    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    documents = recent.get("primaryDocument", [])
    accepted = recent.get("acceptanceDateTime", [])
    items = recent.get("items", [])

    for i, form in enumerate(forms):
        if form != "8-K":
            continue
        filed = dates[i] if i < len(dates) else ""
        if not (start_date <= filed <= end_date):
            continue
        raw_items = items[i] if i < len(items) else ""
        parsed_items = tuple(sorted({part.strip() for part in raw_items.split(",") if part.strip()}))
        yield Filing(
            cik=cik,
            ticker=ticker,
            company=company,
            accession=accessions[i] if i < len(accessions) else "",
            filed_date=filed,
            accepted_at=_parse_accepted(accepted[i] if i < len(accepted) else None),
            items=parsed_items,
            primary_document=documents[i] if i < len(documents) else "",
        )


@dataclass
class IngestionReport:
    """What happened during a run, so the sample can be described honestly."""

    companies_attempted: int = 0
    companies_succeeded: int = 0
    filings_found: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"{self.filings_found} 8-K filings from {self.companies_succeeded}/"
            f"{self.companies_attempted} companies ({len(self.errors)} errors)"
        )


def collect_filings(
    client: SecClient,
    universe: list[tuple[int, str]],
    start_date: str,
    end_date: str,
    progress_every: int = 50,
) -> tuple[list[Filing], IngestionReport]:
    """Walk a universe of (cik, ticker) pairs and gather their 8-Ks."""
    filings: list[Filing] = []
    report = IngestionReport(companies_attempted=len(universe))

    for n, (cik, ticker) in enumerate(universe, start=1):
        try:
            found = list(iter_company_8ks(client, cik, ticker, start_date, end_date))
            filings.extend(found)
            report.companies_succeeded += 1
            report.filings_found += len(found)
        except Exception as exc:  # keep going; one bad company must not kill the run
            report.errors.append(f"{ticker} (CIK {cik}): {exc}")
        if progress_every and n % progress_every == 0:
            print(f"  [{n}/{len(universe)}] {report.filings_found} filings so far", flush=True)

    return filings, report
