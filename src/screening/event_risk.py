"""Friday-only macro event-risk overlay for weekly screening reports.

The overlay is intentionally independent from technical risk.  In ``shadow``
mode it records what the event gate would have decided without changing any
BUY, HOLD, or SELL result.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime, time, timedelta, timezone
from html.parser import HTMLParser
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urlparse
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd
import requests
from exchange_calendars.errors import CalendarError
from icalendar import Calendar


SCHEMA_VERSION = "1.0"
DEFAULT_POLICY_VERSION = "2026-09-shadow-v1"
OFFICIAL_SOURCES = {
    "fed": "https://www.federalreserve.gov/json/calendar.json",
    "bls": "https://www.bls.gov/schedule/news_release/bls.ics",
    "bea": "https://www.bea.gov/news/schedule/ics/online-calendar-subscription.ics",
}
BLS_ANNUAL_SOURCE = "https://www.bls.gov/schedule/{year}/home.htm"
FRED_CALENDAR_SOURCE = "https://fred.stlouisfed.org/releases/calendar"
FRED_BLS_RELEASES = {
    "CPI": {"rid": 10, "name": "Consumer Price Index"},
    "EMPLOYMENT_SITUATION": {"rid": 50, "name": "Employment Situation"},
}
SOURCE_FAMILIES = {
    "fed": {"FOMC_DECISION"},
    "bls": {"CPI", "EMPLOYMENT_SITUATION"},
    "bea": {"PCE"},
}
DEFAULT_POLICY: Dict[str, Any] = {
    "enabled": True,
    "schema_version": SCHEMA_VERSION,
    "policy_version": DEFAULT_POLICY_VERSION,
    "mode": "shadow",
    "timezone": "America/New_York",
    "exchange_calendar": "XNYS",
    "macro_window_sessions": 2,
    "required_macro_families": [
        "FOMC_DECISION",
        "CPI",
        "PCE",
        "EMPLOYMENT_SITUATION",
    ],
    "max_fetch_age_hours": 36,
    "request_timeout_seconds": 15,
    "require_earnings_for_new_buys": False,
    "allow_event_only_sell": False,
    "providers": {
        "fed": "official_json",
        "bls": "official_ics",
        "bea": "official_ics",
        "earnings": "unconfigured",
    },
}


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def _parse_clock(value: str) -> time:
    match = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*([ap])\.?m\.?\s*", value, re.I)
    if not match:
        raise ValueError("unsupported_time_format")
    hour, minute, meridiem = int(match.group(1)), int(match.group(2)), match.group(3).lower()
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        raise ValueError("invalid_time")
    hour = hour % 12 + (12 if meridiem == "p" else 0)
    return time(hour, minute)


def parse_fed_events(payload: Mapping[str, Any], timezone_name: str = "America/New_York") -> List[Dict[str, Any]]:
    """Parse only FOMC decision events, excluding minutes and press conferences."""
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("fed_events_missing")
    local_tz = ZoneInfo(timezone_name)
    parsed: List[Dict[str, Any]] = []
    for item in events:
        if not isinstance(item, Mapping):
            continue
        if _normalize_text(item.get("type")).upper() != "FOMC":
            continue
        if _normalize_text(item.get("title")).casefold() != "fomc meeting":
            continue
        month = _normalize_text(item.get("month"))
        day_numbers = re.findall(r"\d+", _normalize_text(item.get("days")))
        if not re.fullmatch(r"\d{4}-\d{2}", month) or not day_numbers:
            raise ValueError("fed_event_date_invalid")
        event_date = date.fromisoformat(f"{month}-{int(day_numbers[-1]):02d}")
        event_time = _parse_clock(_normalize_text(item.get("time")))
        event_local = datetime.combine(event_date, event_time, tzinfo=local_tz)
        parsed.append(
            {
                "event_id": f"fed:fomc:{event_date.isoformat()}",
                "family": "FOMC_DECISION",
                "name": "FOMC Meeting",
                "symbol": None,
                "event_at_utc": _utc_iso(event_local),
                "event_date_local": event_date.isoformat(),
                "source_timezone": timezone_name,
                "time_precision": "EXACT",
                "timing": "EXACT",
                "confirmation_status": "SCHEDULED",
                "source_id": "fed",
                "source_url": OFFICIAL_SOURCES["fed"],
            }
        )
    return parsed


def _decoded_dtstart(component: Any, timezone_name: str) -> Tuple[Optional[datetime], date, str]:
    decoded = component.decoded("dtstart")
    if isinstance(decoded, datetime):
        if decoded.tzinfo is None:
            decoded = decoded.replace(tzinfo=ZoneInfo(timezone_name))
        local_date = decoded.astimezone(ZoneInfo(timezone_name)).date()
        return decoded, local_date, "EXACT"
    if isinstance(decoded, date):
        return None, decoded, "DATE_ONLY"
    raise ValueError("ics_dtstart_invalid")


def parse_ics_events(content: bytes, provider: str, timezone_name: str = "America/New_York") -> List[Dict[str, Any]]:
    """Parse the relevant BLS or BEA release families from an official ICS feed."""
    if provider not in {"bls", "bea"}:
        raise ValueError("unsupported_ics_provider")
    calendar = Calendar.from_ical(content)
    parsed: List[Dict[str, Any]] = []
    for component in calendar.walk("VEVENT"):
        summary = _normalize_text(component.get("summary"))
        folded = summary.casefold().replace("\\,", ",")
        family: Optional[str] = None
        if provider == "bls":
            if folded.startswith("consumer price index"):
                family = "CPI"
            elif folded.startswith("employment situation"):
                family = "EMPLOYMENT_SITUATION"
        elif re.search(r"\bpersonal income(?: and)? outlays\b", folded):
            family = "PCE"
        if family is None:
            continue
        event_at, local_date, precision = _decoded_dtstart(component, timezone_name)
        uid = _normalize_text(component.get("uid")) or f"{family}:{local_date.isoformat()}"
        parsed.append(
            {
                "event_id": f"{provider}:{uid}",
                "family": family,
                "name": summary,
                "symbol": None,
                "event_at_utc": _utc_iso(event_at) if event_at else None,
                "event_date_local": local_date.isoformat(),
                "source_timezone": timezone_name if event_at else None,
                "time_precision": precision,
                "timing": "EXACT" if event_at else "UNKNOWN",
                "confirmation_status": "SCHEDULED",
                "source_id": provider,
                "source_url": OFFICIAL_SOURCES[provider],
            }
        )
    return parsed


class _BLSScheduleHTMLParser(HTMLParser):
    """Extract table rows from the official annual BLS release calendar."""

    def __init__(self) -> None:
        super().__init__()
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._row: List[str] = []
        self.rows: List[List[str]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        del attrs
        if tag.casefold() == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and tag.casefold() in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        if self._in_cell and folded in {"td", "th"}:
            self._row.append(_normalize_text(" ".join(self._cell_parts)))
            self._in_cell = False
            self._cell_parts = []
        elif self._in_row and folded == "tr":
            if self._row:
                self.rows.append(self._row)
            self._in_row = False
            self._row = []


def parse_bls_html_events(
    content: bytes,
    source_url: str,
    timezone_name: str = "America/New_York",
) -> List[Dict[str, Any]]:
    """Parse CPI and Employment Situation rows from BLS's official annual HTML."""
    parser = _BLSScheduleHTMLParser()
    parser.feed(content.decode("utf-8-sig"))
    local_tz = ZoneInfo(timezone_name)
    parsed: List[Dict[str, Any]] = []
    for row in parser.rows:
        if len(row) < 3:
            continue
        date_text, time_text, release_text = row[0], row[1], row[2]
        folded = release_text.casefold()
        if folded.startswith("consumer price index"):
            family = "CPI"
        elif folded.startswith("employment situation"):
            family = "EMPLOYMENT_SITUATION"
        else:
            continue
        try:
            event_date = datetime.strptime(date_text, "%A, %B %d, %Y").date()
            event_time = _parse_clock(time_text)
        except ValueError as exc:
            raise ValueError("bls_html_event_invalid") from exc
        event_local = datetime.combine(event_date, event_time, tzinfo=local_tz)
        parsed.append(
            {
                "event_id": f"bls-html:{family}:{event_date.isoformat()}",
                "family": family,
                "name": release_text,
                "symbol": None,
                "event_at_utc": _utc_iso(event_local),
                "event_date_local": event_date.isoformat(),
                "source_timezone": timezone_name,
                "time_precision": "EXACT",
                "timing": "EXACT",
                "confirmation_status": "SCHEDULED",
                "source_id": "bls",
                "source_url": source_url,
            }
        )
    return parsed


class _FREDPagerHTMLParser(HTMLParser):
    """Collect table-cell text and links from FRED's release-calendar pager."""

    def __init__(self) -> None:
        super().__init__()
        self._in_row = False
        self._in_cell = False
        self._cell_parts: List[str] = []
        self._cell_hrefs: List[str] = []
        self._row: List[Dict[str, Any]] = []
        self.rows: List[List[Dict[str, Any]]] = []

    def handle_starttag(
        self, tag: str, attrs: List[Tuple[str, Optional[str]]]
    ) -> None:
        folded = tag.casefold()
        if folded == "tr":
            self._in_row = True
            self._row = []
        elif self._in_row and folded in {"td", "th"}:
            self._in_cell = True
            self._cell_parts = []
            self._cell_hrefs = []
        elif self._in_cell and folded == "a":
            href = dict(attrs).get("href")
            if href:
                self._cell_hrefs.append(href)

    def handle_data(self, data: str) -> None:
        if self._in_cell:
            self._cell_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        folded = tag.casefold()
        if self._in_cell and folded in {"td", "th"}:
            self._row.append(
                {
                    "text": _normalize_text(" ".join(self._cell_parts)),
                    "hrefs": list(self._cell_hrefs),
                }
            )
            self._in_cell = False
            self._cell_parts = []
            self._cell_hrefs = []
        elif self._in_row and folded == "tr":
            if self._row:
                self.rows.append(self._row)
            self._in_row = False
            self._row = []


def _fred_release_url(family: str, start: date, end: date) -> str:
    release = FRED_BLS_RELEASES[family]
    query = urlencode(
        {
            "po": 1,
            "ptic": 0,
            "vs": start.isoformat(),
            "ve": end.isoformat(),
            "rid": release["rid"],
        }
    )
    return f"{FRED_CALENDAR_SOURCE}?{query}"


def parse_fred_release_events(
    content: bytes,
    family: str,
    source_url: str,
    timezone_name: str = "America/New_York",
) -> List[Dict[str, Any]]:
    """Parse one strict FRED/BLS release pager response.

    FRED is a secondary institutional calendar, not a BLS-direct source.  Its
    calendar page states that displayed times are US Central Time.
    """
    if family not in FRED_BLS_RELEASES:
        raise ValueError("fred_family_unsupported")
    release = FRED_BLS_RELEASES[family]
    expected_rid = int(release["rid"])
    expected_name = str(release["name"])

    query = parse_qs(urlparse(source_url).query)
    if query.get("rid") != [str(expected_rid)]:
        raise ValueError("fred_source_rid_mismatch")
    try:
        requested_start = date.fromisoformat(query["vs"][0])
        requested_end = date.fromisoformat(query["ve"][0])
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("fred_source_range_invalid") from exc
    if requested_end < requested_start:
        raise ValueError("fred_source_range_invalid")

    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, Mapping):
        raise ValueError("fred_payload_invalid")
    pager = payload.get("pager")
    total = payload.get("ptic")
    if not isinstance(pager, str) or not pager.strip():
        raise ValueError("fred_pager_missing")
    if isinstance(total, bool) or not isinstance(total, int) or total <= 0:
        raise ValueError("fred_total_invalid")

    parser = _FREDPagerHTMLParser()
    parser.feed(pager)
    central_tz = ZoneInfo("America/Chicago")
    output_tz = ZoneInfo(timezone_name)
    current_date: Optional[date] = None
    parsed: List[Dict[str, Any]] = []
    footer: Optional[Tuple[int, int, int]] = None
    seen_dates: Dict[date, datetime] = {}

    date_pattern = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday) "
        r"([A-Za-z]+ \d{1,2}, \d{4})(?: Updated)?$"
    )
    footer_pattern = re.compile(r"^Releases (\d+) - (\d+) of (\d+)$")
    clock_pattern = re.compile(r"^\d{1,2}:\d{2} [ap]m$", re.IGNORECASE)

    for row in parser.rows:
        row_text = _normalize_text(" ".join(str(cell["text"]) for cell in row))
        footer_match = footer_pattern.fullmatch(row_text)
        if footer_match:
            footer = tuple(int(value) for value in footer_match.groups())
            continue

        date_match = date_pattern.fullmatch(row_text)
        if date_match:
            current_date = datetime.strptime(
                f"{date_match.group(1)} {date_match.group(2)}", "%A %B %d, %Y"
            ).date()
            continue

        release_cells = [
            cell
            for cell in row
            if any(urlparse(href).path == "/release" for href in cell["hrefs"])
        ]
        if not release_cells:
            continue
        if current_date is None:
            raise ValueError("fred_release_without_date")

        names = [str(cell["text"]) for cell in release_cells]
        if names != [expected_name]:
            raise ValueError("fred_release_name_mismatch")
        hrefs = [
            href
            for href in release_cells[0]["hrefs"]
            if urlparse(href).path == "/release"
        ]
        if len(hrefs) != 1:
            raise ValueError("fred_release_link_invalid")
        href_rid = parse_qs(urlparse(hrefs[0]).query).get("rid")
        if href_rid != [str(expected_rid)]:
            raise ValueError("fred_release_rid_mismatch")

        clocks = [str(cell["text"]) for cell in row if clock_pattern.fullmatch(str(cell["text"]))]
        if len(clocks) != 1:
            raise ValueError("fred_release_time_invalid")
        if not requested_start <= current_date <= requested_end:
            raise ValueError("fred_release_out_of_range")
        clock = datetime.strptime(clocks[0].upper(), "%I:%M %p").time()
        event_central = datetime.combine(current_date, clock, tzinfo=central_tz)
        if current_date in seen_dates:
            raise ValueError("fred_release_date_conflict")
        seen_dates[current_date] = event_central
        event_output = event_central.astimezone(output_tz)
        parsed.append(
            {
                "event_id": f"fred:{expected_rid}:{current_date.isoformat()}",
                "family": family,
                "name": expected_name,
                "symbol": None,
                "event_at_utc": _utc_iso(event_central),
                "event_date_local": event_output.date().isoformat(),
                "source_timezone": "America/Chicago",
                "time_precision": "EXACT",
                "timing": "EXACT",
                "confirmation_status": "SCHEDULED",
                "source_id": "fred",
                "source_tier": "SECONDARY_INSTITUTIONAL",
                "source_publisher": "Federal Reserve Bank of St. Louis",
                "source_url": source_url,
                "source_release_id": expected_rid,
            }
        )
        current_date = None

    if footer is None:
        raise ValueError("fred_pagination_missing")
    first, last, footer_total = footer
    if first != 1 or last != footer_total or footer_total != total:
        raise ValueError("fred_pagination_incomplete")
    if len(parsed) != total:
        raise ValueError("fred_release_count_mismatch")
    return parsed


def _last_completed_session(calendar: Any, generated_at: datetime, local_tz: ZoneInfo) -> pd.Timestamp:
    local_now = generated_at.astimezone(local_tz)
    local_day = pd.Timestamp(local_now.date())
    session = calendar.date_to_session(local_day, direction="previous")
    if calendar.is_session(local_day):
        close = calendar.session_close(session).to_pydatetime()
        if generated_at.astimezone(timezone.utc) < close.astimezone(timezone.utc):
            session = calendar.previous_session(session)
    return session


def _event_effective_session(event: Mapping[str, Any], calendar: Any) -> pd.Timestamp:
    local_day = pd.Timestamp(date.fromisoformat(str(event["event_date_local"])))
    if calendar.is_session(local_day):
        session = calendar.date_to_session(local_day)
        event_at_raw = event.get("event_at_utc")
        if event_at_raw:
            event_at = datetime.fromisoformat(str(event_at_raw).replace("Z", "+00:00"))
            close = calendar.session_close(session).to_pydatetime()
            if event_at.astimezone(timezone.utc) > close.astimezone(timezone.utc):
                return calendar.next_session(session)
        return session
    return calendar.date_to_session(local_day, direction="next")


def _session_distance(calendar: Any, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if end == start:
        return 0
    if end > start:
        return len(calendar.sessions_in_range(start, end)) - 1
    return -(len(calendar.sessions_in_range(end, start)) - 1)


def _feed_record(
    provider: str,
    content: bytes,
    events: Iterable[Mapping[str, Any]],
    fetched_at: datetime,
    source_url: Optional[str] = None,
    source_format: Optional[str] = None,
    primary_error_code: Optional[str] = None,
) -> Dict[str, Any]:
    event_list = list(events)
    dates = sorted(str(event["event_date_local"]) for event in event_list)
    return {
        "provider": provider,
        "source_url": source_url or OFFICIAL_SOURCES[provider],
        "source_format": source_format,
        "primary_error_code": primary_error_code,
        "fetched_at_utc": _utc_iso(fetched_at),
        "source_updated_at": None,
        "etag": None,
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "coverage_start": dates[0] if dates else None,
        "coverage_end": dates[-1] if dates else None,
        "coverage_verified": bool(event_list),
        "status": "OK" if event_list else "DATA_INCOMPLETE",
        "error_code": None if event_list else "NO_RELEVANT_EVENTS",
    }


def _failed_feed(provider: str, fetched_at: datetime, error_code: str) -> Dict[str, Any]:
    return {
        "provider": provider,
        "source_url": OFFICIAL_SOURCES[provider],
        "source_format": None,
        "primary_error_code": None,
        "fetched_at_utc": _utc_iso(fetched_at),
        "source_updated_at": None,
        "etag": None,
        "content_sha256": None,
        "coverage_start": None,
        "coverage_end": None,
        "coverage_verified": False,
        "status": "DATA_INCOMPLETE",
        "error_code": error_code,
    }


def _request_bytes(
    url: str,
    timeout: float,
    http_get: Callable[..., Any],
    headers: Optional[Mapping[str, str]] = None,
) -> bytes:
    request_headers = (
        {
            "User-Agent": "dx1004-stock-screener/1.0 event-calendar",
            "Accept": "application/json,text/calendar;q=0.9,*/*;q=0.1",
        }
        if headers is None
        else dict(headers)
    )
    request_kwargs: Dict[str, Any] = {"timeout": timeout}
    if request_headers:
        request_kwargs["headers"] = request_headers
    response = http_get(url, **request_kwargs)
    response.raise_for_status()
    content = bytes(response.content)
    if not content:
        raise ValueError("empty_response")
    if len(content) > 5_000_000:
        raise ValueError("payload_too_large")
    return content


def _safe_fetch_error(exc: Exception) -> str:
    if isinstance(exc, requests.Timeout):
        return "FETCH_TIMEOUT"
    if isinstance(exc, requests.HTTPError):
        return "HTTP_ERROR"
    if isinstance(exc, requests.RequestException):
        return "NETWORK_ERROR"
    if isinstance(exc, (ValueError, KeyError, TypeError, UnicodeError, json.JSONDecodeError)):
        return "PARSE_ERROR"
    return "FETCH_ERROR"


def _load_fred_bls_fallback(
    generated_at: datetime,
    timezone_name: str,
    timeout: float,
    http_get: Callable[..., Any],
    primary_error_code: str,
    official_fallback_error_code: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load both BLS families from FRED's secondary institutional calendar."""
    local_date = generated_at.astimezone(ZoneInfo(timezone_name)).date()
    range_start = date(local_date.year, 1, 1)
    range_end = date(local_date.year + 1, 12, 31)
    fetched: List[Tuple[str, bytes]] = []
    parsed: List[Dict[str, Any]] = []

    for family in ("CPI", "EMPLOYMENT_SITUATION"):
        source_url = _fred_release_url(family, range_start, range_end)
        # FRED currently stalls this project's descriptive custom User-Agent,
        # while its normal public response succeeds with the client's default.
        content = _request_bytes(source_url, timeout, http_get, headers={})
        family_events = parse_fred_release_events(
            content, family, source_url, timezone_name
        )
        if not any(
            date.fromisoformat(str(event["event_date_local"])) >= local_date
            for event in family_events
        ):
            raise ValueError("fred_future_coverage_missing")
        fetched.append((source_url, content))
        parsed.extend(family_events)

    combined_content = b"\n--FRED-RESPONSE--\n".join(
        content for _, content in fetched
    )
    feed = _feed_record(
        "bls",
        combined_content,
        parsed,
        generated_at,
        source_url=FRED_CALENDAR_SOURCE,
        source_format="FRED_PAGER_HTML",
        primary_error_code=primary_error_code,
    )
    feed.update(
        {
            "source_id": "fred",
            "source_tier": "SECONDARY_INSTITUTIONAL",
            "source_publisher": "Federal Reserve Bank of St. Louis",
            "source_urls": [url for url, _ in fetched],
            "official_fallback_error_code": official_fallback_error_code,
            "coverage_verified": True,
        }
    )
    return parsed, feed


def _has_future_bls_family_coverage(
    events: Iterable[Mapping[str, Any]],
    generated_at: datetime,
    timezone_name: str,
) -> bool:
    local_date = generated_at.astimezone(ZoneInfo(timezone_name)).date()
    future_families = {
        str(event.get("family"))
        for event in events
        if date.fromisoformat(str(event["event_date_local"])) >= local_date
    }
    return set(FRED_BLS_RELEASES).issubset(future_families)


def _load_official_events(
    policy: Mapping[str, Any],
    generated_at: datetime,
    http_get: Callable[..., Any],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    events: List[Dict[str, Any]] = []
    feeds: List[Dict[str, Any]] = []
    timeout = float(policy.get("request_timeout_seconds", 15))
    timezone_name = str(policy.get("timezone", "America/New_York"))
    for provider in ("fed", "bls", "bea"):
        try:
            content = _request_bytes(OFFICIAL_SOURCES[provider], timeout, http_get)
            if provider == "fed":
                parsed = parse_fed_events(json.loads(content.decode("utf-8-sig")), timezone_name)
            else:
                parsed = parse_ics_events(content, provider, timezone_name)
            if provider == "bls" and not _has_future_bls_family_coverage(
                parsed, generated_at, timezone_name
            ):
                raise ValueError("bls_required_coverage_missing")
            events.extend(parsed)
            feeds.append(
                _feed_record(
                    provider,
                    content,
                    parsed,
                    generated_at,
                    source_format="JSON" if provider == "fed" else "ICS",
                )
            )
        except (requests.RequestException, ValueError, KeyError, TypeError, UnicodeError) as exc:
            primary_error = _safe_fetch_error(exc)
            if provider != "bls":
                feeds.append(_failed_feed(provider, generated_at, primary_error))
                continue
            fallback_url = BLS_ANNUAL_SOURCE.format(
                year=generated_at.astimezone(ZoneInfo(timezone_name)).year
            )
            try:
                fallback_content = _request_bytes(fallback_url, timeout, http_get)
                fallback_events = parse_bls_html_events(
                    fallback_content, fallback_url, timezone_name
                )
                if not _has_future_bls_family_coverage(
                    fallback_events, generated_at, timezone_name
                ):
                    raise ValueError("bls_html_required_coverage_missing")
                events.extend(fallback_events)
                feeds.append(
                    _feed_record(
                        provider,
                        fallback_content,
                        fallback_events,
                        generated_at,
                        source_url=fallback_url,
                        source_format="HTML",
                        primary_error_code=primary_error,
                    )
                )
            except (
                requests.RequestException,
                ValueError,
                KeyError,
                TypeError,
                UnicodeError,
            ) as fallback_exc:
                official_fallback_error = _safe_fetch_error(fallback_exc)
                try:
                    fred_events, fred_feed = _load_fred_bls_fallback(
                        generated_at,
                        timezone_name,
                        timeout,
                        http_get,
                        primary_error,
                        official_fallback_error,
                    )
                    events.extend(fred_events)
                    feeds.append(fred_feed)
                except (
                    requests.RequestException,
                    ValueError,
                    KeyError,
                    TypeError,
                    UnicodeError,
                ) as fred_exc:
                    failed = _failed_feed(
                        provider,
                        generated_at,
                        "PRIMARY_OFFICIAL_AND_FRED_FALLBACK_FAILED",
                    )
                    failed.update(
                        {
                            "primary_error_code": primary_error,
                            "official_fallback_error_code": official_fallback_error,
                            "secondary_error_code": _safe_fetch_error(fred_exc),
                            "secondary_source_url": FRED_CALENDAR_SOURCE,
                        }
                    )
                    feeds.append(failed)
    return events, feeds


def evaluate_macro_events(
    events: Iterable[Mapping[str, Any]],
    feeds: Iterable[Mapping[str, Any]],
    policy: Mapping[str, Any],
    generated_at: datetime,
    calendar: Optional[Any] = None,
) -> Tuple[Dict[str, Any], str, List[Dict[str, Any]]]:
    """Evaluate macro timing with exchange sessions and fail-closed coverage."""
    timezone_name = str(policy.get("timezone", "America/New_York"))
    local_tz = ZoneInfo(timezone_name)
    calendar = calendar or xcals.get_calendar(str(policy.get("exchange_calendar", "XNYS")))
    as_of = _last_completed_session(calendar, generated_at, local_tz)
    window = int(policy.get("macro_window_sessions", 2))
    required = {str(value) for value in policy.get("required_macro_families", [])}
    feed_list = [dict(feed) for feed in feeds]
    event_list: List[Dict[str, Any]] = []
    future_families = set()
    for raw in events:
        event = dict(raw)
        try:
            effective = _event_effective_session(event, calendar)
            distance = _session_distance(calendar, as_of, effective)
            event_at_raw = event.get("event_at_utc")
            if event_at_raw:
                event_at = datetime.fromisoformat(str(event_at_raw).replace("Z", "+00:00"))
                local_now = generated_at.astimezone(local_tz)
                local_event = event_at.astimezone(local_tz)
                local_day = pd.Timestamp(local_now.date())
                if (
                    local_event.date() == local_now.date()
                    and event_at > generated_at
                    and calendar.is_session(local_day)
                    and event_at <= calendar.session_close(calendar.date_to_session(local_day)).to_pydatetime()
                ):
                    # A still-upcoming release in today's open session is day 0,
                    # even though the formal as-of anchor is the prior close.
                    distance = 0
        except (CalendarError, ValueError, KeyError, TypeError, pd.errors.OutOfBoundsDatetime):
            event["sessions_until"] = None
            event["effective_session"] = None
            event_list.append(event)
            continue
        event["sessions_until"] = distance
        event["effective_session"] = effective.date().isoformat()
        if distance >= 0:
            future_families.add(str(event.get("family")))
            event_list.append(event)

    feed_failures = [feed["provider"] for feed in feed_list if feed.get("status") != "OK"]
    missing = sorted(required - future_families)
    invalid_events = any(event.get("sessions_until") is None for event in event_list)
    if feed_failures or missing or invalid_events:
        reasons = []
        if feed_failures:
            reasons.append("REQUIRED_FEED_INCOMPLETE")
        if missing:
            reasons.append("REQUIRED_FAMILY_MISSING")
        if invalid_events:
            reasons.append("EVENT_TIME_INVALID")
        decision = {
            "status": "DATA_INCOMPLETE",
            "level": "UNKNOWN",
            "reason_codes": reasons,
            "event_ids": [],
            "missing_families": missing,
            "failed_providers": sorted(feed_failures),
        }
    else:
        blocking = sorted(
            (
                event
                for event in event_list
                if event.get("family") in required
                and isinstance(event.get("sessions_until"), int)
                and 0 <= int(event["sessions_until"]) <= window
            ),
            key=lambda event: (int(event["sessions_until"]), str(event["event_at_utc"])),
        )
        if blocking:
            decision = {
                "status": "WAIT",
                "level": "HIGH",
                "reason_codes": ["HIGH_IMPACT_MACRO_WITHIN_WINDOW"],
                "event_ids": [event["event_id"] for event in blocking],
                "next_event": blocking[0],
            }
        else:
            decision = {
                "status": "CLEAR",
                "level": "LOW",
                "reason_codes": ["NO_HIGH_IMPACT_MACRO_WITHIN_WINDOW"],
                "event_ids": [],
            }
    return decision, as_of.date().isoformat(), sorted(
        event_list,
        key=lambda event: (
            event.get("sessions_until") is None,
            event.get("sessions_until") if event.get("sessions_until") is not None else 10**9,
            str(event.get("event_at_utc") or event.get("event_date_local")),
        ),
    )


def build_event_risk_overlay(
    config: Optional[Mapping[str, Any]],
    candidate_tickers: Iterable[str] = (),
    generated_at: Optional[datetime] = None,
    http_get: Callable[..., Any] = requests.get,
    calendar: Optional[Any] = None,
) -> Dict[str, Any]:
    """Fetch, normalize, and evaluate the immutable Friday event overlay."""
    policy = {**DEFAULT_POLICY, **dict(config or {})}
    policy["providers"] = {
        **DEFAULT_POLICY["providers"],
        **dict((config or {}).get("providers", {})),
    }
    generated_at = generated_at or datetime.now(timezone.utc)
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=timezone.utc)
    max_age = int(policy.get("max_fetch_age_hours", 36))
    public_policy = {
        key: policy[key]
        for key in (
            "enabled",
            "mode",
            "timezone",
            "exchange_calendar",
            "macro_window_sessions",
            "required_macro_families",
            "max_fetch_age_hours",
            "require_earnings_for_new_buys",
            "allow_event_only_sell",
            "providers",
        )
    }
    if not bool(policy.get("enabled", True)):
        decision = {
            "status": "DATA_INCOMPLETE",
            "level": "UNKNOWN",
            "reason_codes": ["EVENT_RISK_DISABLED"],
            "event_ids": [],
        }
        as_of_session = None
        normalized_events: List[Dict[str, Any]] = []
        feeds: List[Dict[str, Any]] = []
    else:
        events, feeds = _load_official_events(policy, generated_at, http_get)
        decision, as_of_session, normalized_events = evaluate_macro_events(
            events, feeds, policy, generated_at, calendar=calendar
        )
    ticker_decisions = {
        str(ticker).upper(): {
            "status": decision["status"],
            "reason_codes": list(decision.get("reason_codes", [])),
            "event_ids": list(decision.get("event_ids", [])),
            "earnings_status": "NOT_CONFIGURED",
            "enforced": False,
        }
        for ticker in sorted({str(value).upper() for value in candidate_tickers if value})
    }
    return {
        "schema_version": str(policy.get("schema_version", SCHEMA_VERSION)),
        "policy_version": str(policy.get("policy_version", DEFAULT_POLICY_VERSION)),
        "mode": str(policy.get("mode", "shadow")),
        "generated_at_utc": _utc_iso(generated_at),
        "as_of_session": as_of_session,
        "valid_until_utc": _utc_iso(generated_at + timedelta(hours=max_age)),
        "policy": public_policy,
        "feeds": feeds,
        "events": normalized_events,
        "market_decision": {**decision, "enforced": False},
        "ticker_decisions": ticker_decisions,
    }


def format_event_risk_text(overlay: Mapping[str, Any]) -> str:
    decision = overlay.get("market_decision", {}) if isinstance(overlay, Mapping) else {}
    status = decision.get("status", "DATA_INCOMPLETE")
    mode = str(overlay.get("mode", "shadow")).upper() if isinstance(overlay, Mapping) else "SHADOW"
    lines = ["", "=" * 60, f"EVENT RISK ({mode})", "=" * 60]
    lines.append(f"Macro event status: {status} (not enforced)")
    next_event = decision.get("next_event") if isinstance(decision, Mapping) else None
    if isinstance(next_event, Mapping):
        lines.append(
            f"Next event: {next_event.get('family')} on {next_event.get('event_date_local')} "
            f"({next_event.get('sessions_until')} sessions)"
        )
    if decision.get("reason_codes"):
        lines.append("Reasons: " + ", ".join(str(code) for code in decision["reason_codes"]))
    return "\n".join(lines)
