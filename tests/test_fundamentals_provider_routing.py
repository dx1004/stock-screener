"""Routing tests for EDGAR-first quarterly fundamentals."""

from unittest.mock import Mock, patch

import pandas as pd

import src.data.fundamentals_fetcher as fundamentals


def test_fetch_quarterly_financials_prefers_sec_edgar(monkeypatch):
    expected = {"ticker": "ACME", "data_source": "sec_edgar", "quarterly_revenue": {}}
    sec_fetcher = Mock()
    sec_fetcher.fetch_quarterly_financials.return_value = expected
    monkeypatch.setattr(fundamentals, "_sec_edgar_fetcher", sec_fetcher)

    with patch("yfinance.Ticker") as yahoo_ticker:
        result = fundamentals.fetch_quarterly_financials("ACME")

    assert result == expected
    yahoo_ticker.assert_not_called()


def test_fetch_quarterly_financials_falls_back_to_yahoo(monkeypatch):
    sec_fetcher = Mock()
    sec_fetcher.fetch_quarterly_financials.return_value = None
    monkeypatch.setattr(fundamentals, "_sec_edgar_fetcher", sec_fetcher)
    income = pd.DataFrame(
        {pd.Timestamp("2025-06-30"): [100, 1]},
        index=["Total Revenue", "Diluted EPS"],
    )

    with patch("yfinance.Ticker") as yahoo_ticker:
        yahoo_ticker.return_value.quarterly_financials = income
        yahoo_ticker.return_value.quarterly_balance_sheet = pd.DataFrame()
        yahoo_ticker.return_value.quarterly_cashflow = pd.DataFrame()
        result = fundamentals.fetch_quarterly_financials("ACME")

    assert result["ticker"] == "ACME"
    assert "quarterly_revenue" in result
    yahoo_ticker.assert_called_once_with("ACME")
