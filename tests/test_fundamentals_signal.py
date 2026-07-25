"""Regression tests for nullable provider fundamentals."""

from src.data.fundamentals_fetcher import (
    analyze_fundamentals_for_signal,
    create_fundamental_snapshot,
)


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


def test_nullable_numeric_fundamentals_render_snapshot():
    snapshot = create_fundamental_snapshot(
        "ACME",
        {
            "revenue_yoy_change": None,
            "revenue_qoq_change": None,
            "eps_yoy_change": None,
            "margin_change": None,
            "inventory_qoq_change": None,
            "inventory_to_sales_ratio": None,
            "gross_margin": 50.0,
        },
    )

    assert "FUNDAMENTAL SNAPSHOT - ACME" in snapshot
    assert "Revenue: Data not available" in snapshot
    assert "EPS: Data not available" in snapshot
