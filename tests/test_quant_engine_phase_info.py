"""Regression coverage for phase metadata in rendered buy signals."""

from unittest.mock import Mock, patch

import pandas as pd

from src.screening.quant_engine import QuantAnalysisEngine


def _screening_results(buy):
    return {
        "total_analyzed": 1,
        "fundamentals_coverage": {
            "sec_edgar": 0,
            "sec_edgar_ifrs": 0,
            "yahoo_fallback": 0,
            "unavailable": 0,
        },
        "spy_analysis": {},
        "breadth": {},
        "buys": [buy],
        "sells": [],
        "signal_recommendation": {},
    }


def _phase_two_buy(phase_info=None):
    buy = {
        "ticker": "ACME",
        "score": 75,
        "phase": "Phase 2: Advancing",
        "breakout_price": 100.0,
        "details": {},
        "reasons": ["Qualified breakout"],
    }
    if phase_info is not None:
        buy["phase_info"] = phase_info
    return buy


def test_accepted_buy_retains_phase_info():
    engine = QuantAnalysisEngine()
    phase_info = {"phase": "Phase 2: Advancing", "distance_from_50sma": 4.25}
    analysis = {
        "ticker": "ACME",
        "price_data": pd.DataFrame(),
        "current_price": 100.0,
        "phase_info": phase_info,
        "rs_series": pd.Series(dtype=float),
        "fundamental_analysis": {},
        "quarterly_data": pd.DataFrame(),
    }
    buy_signal = {
        "is_buy": True,
        "ticker": "ACME",
        "score": 75,
        "phase": "Phase 2: Advancing",
        "breakout_price": 100.0,
        "details": {},
        "reasons": [],
    }
    engine.spy_data = pd.DataFrame({"Close": [100.0]})
    engine.spy_price = 100.0

    with (
        patch("src.screening.quant_engine.analyze_spy_trend", return_value={}),
        patch("src.screening.quant_engine.calculate_market_breadth", return_value={}),
        patch(
            "src.screening.quant_engine.should_generate_signals",
            return_value={"should_generate_buys": True, "should_generate_sells": False},
        ),
        patch.object(engine, "analyze_stock", return_value=analysis),
        patch("src.screening.quant_engine.score_buy_signal", return_value=buy_signal),
        patch("src.screening.quant_engine.create_fundamental_snapshot", return_value="snapshot"),
    ):
        results = engine.screen_stocks(["ACME"])

    assert results["buys"][0]["phase_info"] == phase_info


def test_phase_two_buy_renders_distance_from_50_sma():
    engine = QuantAnalysisEngine()
    engine.screen_stocks = Mock(
        return_value=_screening_results(
            _phase_two_buy({"phase": "Phase 2: Advancing", "distance_from_50sma": 4.25})
        )
    )

    with patch("src.screening.quant_engine.format_benchmark_summary", return_value="benchmark"):
        report = engine.run(["ACME"])

    assert "Distance from 50 SMA: 4.2%" in report


def test_phase_two_buy_without_distance_does_not_render_distance():
    engine = QuantAnalysisEngine()
    engine.screen_stocks = Mock(return_value=_screening_results(_phase_two_buy()))

    with patch("src.screening.quant_engine.format_benchmark_summary", return_value="benchmark"):
        report = engine.run(["ACME"])

    assert "Distance from 50 SMA:" not in report
