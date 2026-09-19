"""Regression coverage for the weekly review risk-control fixes."""

from datetime import datetime
from unittest.mock import patch

import pandas as pd

from scripts.saturday_weekly_review import ReportSource, build_structured_review
from src.screening.quant_engine import QuantAnalysisEngine
from src.screening.risk_management import evaluate_buy_risk


def _price_frame(high: float, low: float, volume: int = 1_000_000) -> pd.DataFrame:
    index = pd.date_range(end=pd.Timestamp.now().normalize(), periods=80, freq="D")
    frame = pd.DataFrame(
        {
            "High": high,
            "Low": low,
            "Close": 100.0,
            "Volume": volume,
        },
        index=index,
    )
    return frame


def test_r5_scales_position_instead_of_rejecting_valid_setup():
    stock = _price_frame(high=102.0, low=98.0)
    # An older pivot supplies a plausible 1-4 month target without changing
    # the latest 14-day ATR used by the risk comparison.
    stock.iloc[30, stock.columns.get_loc("High")] = 130.0
    spy = _price_frame(high=100.5, low=99.5)

    result = evaluate_buy_risk(
        candidate={"entry_price": 100.0, "stop_loss": 92.0},
        analysis={"price_data": stock},
        spy_context={"price_data": spy, "current_price": 100.0},
        benchmark_regime="RISK-ON (Strong)",
        risk_policy={
            "atr_period": 14,
            "spy_atr_period": 14,
            "spy_atr_multiple": 2.0,
            "min_atr_distance_multiple": 1.5,
            "max_atr_distance_multiple": 3.0,
            "max_stop_distance_pct": 0.08,
            "min_rr": 2.5,
            "max_data_staleness_days": 3,
            "min_entry_price": 10.0,
            "min_median_20d_dollar_volume": 20_000_000,
            "base_risk_budget_pct": 1.0,
            "default_capital": 10_000.0,
            "max_notional_pct": 10.0,
        },
    )

    assert result["status"] == "PASS"
    assert result["risk"]["spy_risk_grade"] == "R5"
    assert result["risk"]["spy_risk_ratio"] == 4.0
    assert result["risk"]["risk_budget_per_trade_pct"] == 0.25
    assert result["risk"]["stock_atr_risk_pct"] == 0.08
    assert result["risk"]["spy_atr_risk_pct"] == 0.02
    assert any(
        reason == "风险等级（以S&P 500指数风险作为标准）：R5"
        for reason in result["reasons"]
    )


def test_quant_engine_passes_raw_quarterly_fields_to_signal_scorer():
    engine = QuantAnalysisEngine()
    raw_quarterly = {
        "quarterly_revenue": {"2026-Q1": 100, "2026-Q2": 110},
        "revenue_yoy_change": 18.0,
        "eps_yoy_change": 25.0,
    }
    normalized = {"revenue_trend": "growing", "eps_trend": "accelerating"}
    analysis = {
        "ticker": "ACME",
        "price_data": _price_frame(high=102.0, low=98.0),
        "current_price": 100.0,
        "phase_info": {"phase": 2},
        "rs_series": pd.Series(dtype=float),
        "quarterly_data": raw_quarterly,
        "fundamental_analysis": normalized,
        "analysis_error": None,
    }
    buy_signal = {"ticker": "ACME", "is_buy": True, "score": 75.0}

    with (
        patch.object(engine, "fetch_spy_data", return_value=True),
        patch.object(engine, "analyze_stock", return_value=analysis),
        patch("src.screening.quant_engine.calculate_market_breadth", return_value={}),
        patch(
            "src.screening.quant_engine.should_generate_signals",
            return_value={"should_generate_buys": True, "regime": "RISK-ON (Strong)"},
        ),
        patch("src.screening.quant_engine.score_buy_signal", return_value=buy_signal) as scorer,
        patch("src.screening.quant_engine.create_fundamental_snapshot", return_value="snapshot"),
        patch.object(
            engine,
            "_enrich_buy_with_risk",
            return_value={**buy_signal, "qualified": False},
        ),
    ):
        engine.screen_stocks(["ACME"])

    assert scorer.call_args.kwargs["fundamentals"] is raw_quarterly


def test_weekly_review_preserves_upstream_rejection_reason():
    timestamp = datetime.now().isoformat()
    reason = "入场到ATR止损距离不足（1.43x ATR < 1.5x ATR）"
    payload = {
        "timestamp": timestamp,
        "status": "ok",
        "breadth": {},
        "buys": [
            {
                "ticker": "XOM",
                "score": 81.5,
                "risk_assessment": {
                    "status": "REJECT",
                    "reasons": [reason],
                    "risk": {},
                    "sizing": {},
                },
            }
        ],
        "holdings_actions": [],
        "sells": [],
    }
    review = build_structured_review(
        payload,
        ReportSource(source="test", payload=payload, timestamp=timestamp),
    )

    assert review["rejected_candidates"] == [
        {"ticker": "XOM", "status": "REJECT", "reasons": [reason]}
    ]
    assert reason in review["summary_text"]
    assert "风险状态=REJECT（非PASS）" not in review["summary_text"]
