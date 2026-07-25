"""Regression tests for nullable provider fundamentals."""

from src.data.fundamentals_fetcher import analyze_fundamentals_for_signal


def test_nullable_numeric_fundamentals_are_treated_as_neutral():
    result = analyze_fundamentals_for_signal(
        {
            "revenue_yoy_change": None,
            "revenue_qoq_change": None,
            "eps_yoy_change": None,
            "inventory_qoq_change": None,
        }
    )

    assert result["revenue_trend"] == "flat"
    assert result["eps_trend"] == "flat"
    assert result["inventory_signal"] == "neutral"
    assert result["sequential_revenue_declining"] is False
