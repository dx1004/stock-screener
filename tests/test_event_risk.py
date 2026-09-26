"""Offline contract tests for the Friday macro event-risk overlay."""

from datetime import datetime, timezone

import exchange_calendars as xcals
import pytest
import requests

import scripts.run_quant_engine as quant_runner
from scripts.saturday_weekly_review import ReportSource, build_structured_review
from src.screening.event_risk import (
    build_event_risk_overlay,
    evaluate_macro_events,
    parse_bls_html_events,
    parse_fed_events,
    parse_ics_events,
)


def _event(family: str, at_utc: str, local_date: str, source: str = "test"):
    return {
        "event_id": f"{source}:{family}:{local_date}",
        "family": family,
        "name": family,
        "symbol": None,
        "event_at_utc": at_utc,
        "event_date_local": local_date,
        "source_timezone": "America/New_York",
        "time_precision": "EXACT",
        "timing": "EXACT",
        "confirmation_status": "SCHEDULED",
        "source_id": source,
        "source_url": "https://example.invalid/fixture",
    }


def _feeds(status: str = "OK"):
    return [
        {"provider": provider, "status": status, "coverage_verified": status == "OK"}
        for provider in ("fed", "bls", "bea")
    ]


def _policy(window: int = 2):
    return {
        "timezone": "America/New_York",
        "exchange_calendar": "XNYS",
        "macro_window_sessions": window,
        "required_macro_families": [
            "FOMC_DECISION",
            "CPI",
            "PCE",
            "EMPLOYMENT_SITUATION",
        ],
    }


def _complete_events():
    return [
        _event("FOMC_DECISION", "2026-10-28T18:00:00Z", "2026-10-28"),
        _event("CPI", "2026-10-13T12:30:00Z", "2026-10-13"),
        _event("PCE", "2026-09-30T12:30:00Z", "2026-09-30"),
        _event("EMPLOYMENT_SITUATION", "2026-10-02T12:30:00Z", "2026-10-02"),
    ]


def test_fed_parser_keeps_meeting_only():
    payload = {
        "events": [
            {"type": "FOMC", "title": "FOMC Meeting", "month": "2026-10", "days": "28", "time": "2:00 p.m."},
            {"type": "FOMC", "title": "FOMC Minutes", "month": "2026-11", "days": "18", "time": "2:00 p.m."},
            {"type": "FOMC", "title": "FOMC Press Conference", "month": "2026-10", "days": "28", "time": "2:30 p.m."},
        ]
    }

    events = parse_fed_events(payload)

    assert [event["family"] for event in events] == ["FOMC_DECISION"]
    assert events[0]["event_at_utc"] == "2026-10-28T18:00:00Z"


def test_ics_parser_maps_bls_and_bea_families():
    bls = b"""BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
UID:cpi-1\r
DTSTART;TZID=US-Eastern:20261013T083000\r
SUMMARY:Consumer Price Index\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:jobs-1\r
DTSTART;TZID=US-Eastern:20261002T083000\r
SUMMARY:Employment Situation\r
END:VEVENT\r
END:VCALENDAR\r
"""
    bea = b"""BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
UID:pce-1\r
DTSTART:20260930T123000Z\r
SUMMARY:Personal Income Outlays\\, August 2026\r
END:VEVENT\r
END:VCALENDAR\r
"""

    assert {event["family"] for event in parse_ics_events(bls, "bls")} == {
        "CPI",
        "EMPLOYMENT_SITUATION",
    }
    pce = parse_ics_events(bea, "bea")
    assert [event["family"] for event in pce] == ["PCE"]
    assert pce[0]["event_at_utc"] == "2026-09-30T12:30:00Z"


def test_bls_official_html_fallback_parser_maps_release_rows():
    html = b"""<!doctype html><html><body><table>
    <tr><th>Date</th><th>Time</th><th>Release</th></tr>
    <tr><td>Friday, October 2, 2026</td><td>08:30 AM</td>
        <td>Employment Situation for September 2026</td></tr>
    <tr><td>Wednesday, October 14, 2026</td><td>08:30 AM</td>
        <td>Consumer Price Index for September 2026</td></tr>
    </table></body></html>"""

    events = parse_bls_html_events(
        html, "https://www.bls.gov/schedule/2026/home.htm"
    )

    assert [event["family"] for event in events] == [
        "EMPLOYMENT_SITUATION",
        "CPI",
    ]
    assert events[0]["event_at_utc"] == "2026-10-02T12:30:00Z"


def test_near_event_waits_and_outside_window_clears():
    calendar = xcals.get_calendar("XNYS")
    generated = datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc)  # Friday after close
    events = _complete_events()
    events[0] = _event("FOMC_DECISION", "2026-09-29T18:00:00Z", "2026-09-29")

    decision, as_of, normalized = evaluate_macro_events(events, _feeds(), _policy(), generated, calendar)
    assert as_of == "2026-09-25"
    assert decision["status"] == "WAIT"
    assert decision["next_event"]["sessions_until"] == 2

    outside, _, _ = evaluate_macro_events(_complete_events(), _feeds(), _policy(), generated, calendar)
    assert outside["status"] == "CLEAR"
    assert all(event["sessions_until"] > 2 for event in normalized if event["family"] != "FOMC_DECISION")


def test_same_session_future_is_day_zero_and_good_friday_prices_monday():
    calendar = xcals.get_calendar("XNYS")
    same_day = _complete_events()
    same_day[0] = _event("FOMC_DECISION", "2026-09-29T18:00:00Z", "2026-09-29")
    decision, _, _ = evaluate_macro_events(
        same_day,
        _feeds(),
        _policy(),
        datetime(2026, 9, 29, 17, 0, tzinfo=timezone.utc),
        calendar,
    )
    assert decision["next_event"]["sessions_until"] == 0

    good_friday = _complete_events()
    good_friday[-1] = _event("EMPLOYMENT_SITUATION", "2026-04-03T12:30:00Z", "2026-04-03")
    good_friday[0] = _event("FOMC_DECISION", "2026-05-06T18:00:00Z", "2026-05-06")
    good_friday[1] = _event("CPI", "2026-04-10T12:30:00Z", "2026-04-10")
    good_friday[2] = _event("PCE", "2026-04-30T12:30:00Z", "2026-04-30")
    decision, _, _ = evaluate_macro_events(
        good_friday,
        _feeds(),
        _policy(),
        datetime(2026, 4, 2, 22, 0, tzinfo=timezone.utc),
        calendar,
    )
    assert decision["status"] == "WAIT"
    assert decision["next_event"]["family"] == "EMPLOYMENT_SITUATION"
    assert decision["next_event"]["effective_session"] == "2026-04-06"
    assert decision["next_event"]["sessions_until"] == 1


def test_missing_required_family_or_failed_feed_is_incomplete():
    calendar = xcals.get_calendar("XNYS")
    generated = datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc)
    missing_pce = [event for event in _complete_events() if event["family"] != "PCE"]

    decision, _, _ = evaluate_macro_events(missing_pce, _feeds(), _policy(), generated, calendar)
    assert decision["status"] == "DATA_INCOMPLETE"
    assert decision["missing_families"] == ["PCE"]

    failed = _feeds()
    failed[1]["status"] = "DATA_INCOMPLETE"
    decision, _, _ = evaluate_macro_events(_complete_events(), failed, _policy(), generated, calendar)
    assert decision["status"] == "DATA_INCOMPLETE"
    assert "bls" in decision["failed_providers"]


def test_overlay_is_shadow_and_does_not_change_trading_payload():
    fed = b'{"events":[{"type":"FOMC","title":"FOMC Meeting","month":"2026-09","days":"29","time":"2:00 p.m."}]}'
    bls = b"""BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
UID:cpi\r
DTSTART:20261013T123000Z\r
SUMMARY:Consumer Price Index\r
END:VEVENT\r
BEGIN:VEVENT\r
UID:jobs\r
DTSTART:20261002T123000Z\r
SUMMARY:Employment Situation\r
END:VEVENT\r
END:VCALENDAR\r
"""
    bea = b"""BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
UID:pce\r
DTSTART:20260930T123000Z\r
SUMMARY:Personal Income and Outlays\r
END:VEVENT\r
END:VCALENDAR\r
"""
    bodies = {"calendar.json": fed, "bls.ics": bls, "online-calendar-subscription.ics": bea}

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    def fake_get(url, **_kwargs):
        return Response(next(body for suffix, body in bodies.items() if url.endswith(suffix)))

    trading_payload = {
        "qualified_buys": [{"ticker": "NVDA"}],
        "holdings_actions": [{"ticker": "ABBV", "action": "HOLD"}],
    }
    before = repr(trading_payload)
    overlay = build_event_risk_overlay(
        {"mode": "shadow"},
        candidate_tickers=["NVDA"],
        generated_at=datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc),
        http_get=fake_get,
    )

    assert overlay["market_decision"]["status"] == "WAIT"
    assert overlay["market_decision"]["enforced"] is False
    assert overlay["ticker_decisions"]["NVDA"]["earnings_status"] == "NOT_CONFIGURED"
    assert repr(trading_payload) == before


def test_overlay_uses_official_bls_html_when_ics_is_blocked():
    fed = b'{"events":[{"type":"FOMC","title":"FOMC Meeting","month":"2026-10","days":"28","time":"2:00 p.m."}]}'
    bea = b"""BEGIN:VCALENDAR\r
VERSION:2.0\r
BEGIN:VEVENT\r
UID:pce\r
DTSTART:20260930T123000Z\r
SUMMARY:Personal Income and Outlays\r
END:VEVENT\r
END:VCALENDAR\r
"""
    bls_html = b"""<table>
    <tr><td>Friday, October 2, 2026</td><td>08:30 AM</td><td>Employment Situation for September 2026</td></tr>
    <tr><td>Wednesday, October 14, 2026</td><td>08:30 AM</td><td>Consumer Price Index for September 2026</td></tr>
    </table>"""

    class Response:
        def __init__(self, content=b"", error=None):
            self.content = content
            self.error = error

        def raise_for_status(self):
            if self.error:
                raise self.error

    def fake_get(url, **_kwargs):
        if url.endswith("bls.ics"):
            return Response(error=requests.HTTPError("403"))
        if url.endswith("/2026/home.htm"):
            return Response(bls_html)
        if url.endswith("calendar.json"):
            return Response(fed)
        return Response(bea)

    overlay = build_event_risk_overlay(
        {"mode": "shadow"},
        generated_at=datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc),
        http_get=fake_get,
    )

    bls_feed = next(feed for feed in overlay["feeds"] if feed["provider"] == "bls")
    assert bls_feed["status"] == "OK"
    assert bls_feed["source_format"] == "HTML"
    assert bls_feed["primary_error_code"] == "HTTP_ERROR"
    assert overlay["market_decision"]["status"] == "CLEAR"


def test_disabled_overlay_does_not_fetch_and_unexpected_errors_are_not_hidden():
    def must_not_fetch(*_args, **_kwargs):
        raise AssertionError("disabled overlay attempted network access")

    disabled = build_event_risk_overlay(
        {"enabled": False, "mode": "shadow"},
        generated_at=datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc),
        http_get=must_not_fetch,
    )
    assert disabled["market_decision"]["status"] == "DATA_INCOMPLETE"
    assert disabled["market_decision"]["reason_codes"] == ["EVENT_RISK_DISABLED"]

    def programmer_error(*_args, **_kwargs):
        raise RuntimeError("programming defect")

    with pytest.raises(RuntimeError, match="programming defect"):
        build_event_risk_overlay(
            {"mode": "shadow"},
            generated_at=datetime(2026, 9, 25, 22, 0, tzinfo=timezone.utc),
            http_get=programmer_error,
        )


def test_saturday_passes_overlay_through_and_respects_explicit_empty_qualified_list():
    overlay = {
        "schema_version": "1.0",
        "policy_version": "test",
        "mode": "shadow",
        "market_decision": {"status": "WAIT", "reason_codes": ["TEST"], "enforced": False},
    }
    payload = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qualified_buys": [],
        "buys": [{"ticker": "NVDA", "score": 90}],
        "sells": [],
        "holdings_actions": [{"ticker": "ABBV", "action": "HOLD", "reason": "test"}],
        "event_risk": overlay,
    }
    source = ReportSource(source="test", payload=payload, timestamp=payload["timestamp"])

    review = build_structured_review(payload, source)

    assert review["event_risk"] is overlay
    assert review["buy_candidates"] == []
    assert review["holdings_actions"] == payload["holdings_actions"]
    assert "事件风险：WAIT（SHADOW" in review["summary_text"]


def test_friday_runner_attaches_same_overlay_before_json_save(monkeypatch):
    payload = {"qualified_buys": [{"ticker": "NVDA"}]}
    overlay = {
        "mode": "shadow",
        "market_decision": {"status": "WAIT", "reason_codes": ["TEST"], "enforced": False},
    }
    captured = {}

    class Engine:
        def __init__(self, **_kwargs):
            pass

        def run_report(self, _tickers):
            return "BASE REPORT", payload

    monkeypatch.setattr(quant_runner, "QuantAnalysisEngine", Engine)
    monkeypatch.setattr(
        quant_runner,
        "load_config",
        lambda _path: {
            "stock_universe": ["NVDA"],
            "parameters": {},
            "risk_control": {},
            "holdings": [],
            "event_risk": {"mode": "shadow"},
            "output": {"output_dir": "/tmp/not-used"},
        },
    )
    monkeypatch.setattr(quant_runner, "build_event_risk_overlay", lambda *_args, **_kwargs: overlay)
    monkeypatch.setattr(quant_runner, "format_event_risk_text", lambda value: f"\nEVENT {value['market_decision']['status']}")
    monkeypatch.setattr(quant_runner, "save_results", lambda report, _path: captured.setdefault("report", report))
    monkeypatch.setattr(quant_runner, "save_json_result", lambda result, _path: captured.setdefault("payload", result))
    monkeypatch.setattr("sys.argv", ["run_quant_engine.py", "--tickers", "NVDA"])

    quant_runner.main()

    assert captured["payload"]["event_risk"] is overlay
    assert captured["report"].endswith("EVENT WAIT")


def test_legacy_report_remains_readable_with_incomplete_event_status():
    payload = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "qualified_buys": [],
        "buys": [],
        "sells": [],
        "holdings_actions": [],
    }
    review = build_structured_review(
        payload,
        ReportSource(source="legacy", payload=payload, timestamp=payload["timestamp"]),
    )

    assert review["event_risk"]["market_decision"]["status"] == "DATA_INCOMPLETE"
    assert "EVENT_RISK_OVERLAY_MISSING" in review["event_risk"]["market_decision"]["reason_codes"]
