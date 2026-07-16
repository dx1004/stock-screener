"""Offline contract tests for SEC EDGAR quarterly fundamentals."""

from unittest.mock import Mock

import pytest

from src.data.sec_edgar_fetcher import SECEdgarFundamentalsFetcher


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def _duration_entries(values):
    dates = [
        ("2024-01-01", "2024-03-31", 2024, "Q1"),
        ("2024-04-01", "2024-06-30", 2024, "Q2"),
        ("2024-07-01", "2024-09-30", 2024, "Q3"),
        ("2025-01-01", "2025-03-31", 2025, "Q1"),
        ("2025-04-01", "2025-06-30", 2025, "Q2"),
    ]
    return [
        {
            "start": start,
            "end": end,
            "val": value,
            "fy": fiscal_year,
            "fp": fiscal_period,
            "form": "10-Q",
            "filed": "2025-08-01",
        }
        for (start, end, fiscal_year, fiscal_period), value in zip(dates, values)
    ]


def _instant_entries(values):
    return [
        {
            "end": end,
            "val": value,
            "fy": fiscal_year,
            "fp": fiscal_period,
            "form": "10-Q",
            "filed": "2025-08-01",
        }
        for (_, end, fiscal_year, fiscal_period), value in zip(
            [
                ("", "2024-03-31", 2024, "Q1"),
                ("", "2024-06-30", 2024, "Q2"),
                ("", "2024-09-30", 2024, "Q3"),
                ("", "2025-03-31", 2025, "Q1"),
                ("", "2025-06-30", 2025, "Q2"),
            ],
            values,
        )
    ]


def _company_facts():
    return {
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {"units": {"USD": _duration_entries([100, 120, 140, 110, 132])}},
                "EarningsPerShareDiluted": {"units": {"USD/shares": _duration_entries([1, 1.2, 1.4, 1.1, 1.32])}},
                "GrossProfit": {"units": {"USD": _duration_entries([40, 48, 56, 44, 52.8])}},
                "OperatingIncomeLoss": {"units": {"USD": _duration_entries([20, 24, 28, 22, 26.4])}},
                "InventoryNet": {"units": {"USD": _instant_entries([50, 60, 70, 55, 66])}},
            }
        }
    }


def test_fetch_quarterly_financials_normalizes_company_facts(tmp_path):
    session = Mock()
    session.get.side_effect = [
        _response({"0": {"ticker": "ACME", "cik_str": 1234}}),
        _response(_company_facts()),
    ]
    fetcher = SECEdgarFundamentalsFetcher(
        user_agent="Example Owner owner@example.com",
        cache_dir=str(tmp_path),
        session=session,
    )

    result = fetcher.fetch_quarterly_financials("ACME")

    assert result["data_source"] == "sec_edgar"
    assert result["quarterly_revenue"]["2025-06-30"] == 132
    assert result["quarterly_eps"]["2025-06-30"] == 1.32
    assert result["revenue_qoq_change"] == 20
    assert result["revenue_yoy_change"] == 10
    assert result["eps_qoq_change"] == pytest.approx(20)
    assert result["eps_yoy_change"] == pytest.approx(10)
    assert result["gross_margin"] == 40
    assert result["operating_margin"] == 20
    assert result["inventory_qoq_change"] == 20
    assert result["inventory_to_sales_ratio"] == 0.5
    assert session.get.call_count == 2
    assert session.get.call_args_list[0].kwargs["headers"]["User-Agent"] == "Example Owner owner@example.com"


def test_fetch_quarterly_financials_requires_an_identifying_user_agent(tmp_path, monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    session = Mock()
    fetcher = SECEdgarFundamentalsFetcher(user_agent="", cache_dir=str(tmp_path), session=session)

    assert fetcher.fetch_quarterly_financials("ACME") is None
    session.get.assert_not_called()


def test_fetch_quarterly_financials_uses_ifrs_when_us_gaap_is_absent(tmp_path):
    session = Mock()
    ifrs_facts = _company_facts()["facts"]["us-gaap"]
    ifrs_facts = {
        "Revenue": ifrs_facts["RevenueFromContractWithCustomerExcludingAssessedTax"],
        "DilutedEarningsLossPerShare": ifrs_facts["EarningsPerShareDiluted"],
        "GrossProfit": ifrs_facts["GrossProfit"],
        "ProfitLossFromOperatingActivities": ifrs_facts["OperatingIncomeLoss"],
        "Inventories": ifrs_facts["InventoryNet"],
    }
    session.get.side_effect = [
        _response({"0": {"ticker": "ACME", "cik_str": 1234}}),
        _response({"facts": {"ifrs-full": ifrs_facts}}),
    ]
    fetcher = SECEdgarFundamentalsFetcher(
        user_agent="Example Owner owner@example.com", cache_dir=str(tmp_path), session=session
    )

    result = fetcher.fetch_quarterly_financials("ACME")

    assert result["data_source"] == "sec_edgar_ifrs"
    assert result["quarterly_revenue"]["2025-06-30"] == 132


def test_fetch_quarterly_financials_returns_none_for_incomplete_facts(tmp_path):
    session = Mock()
    session.get.side_effect = [
        _response({"0": {"ticker": "ACME", "cik_str": 1234}}),
        _response({"facts": {"us-gaap": {}}}),
    ]
    fetcher = SECEdgarFundamentalsFetcher(
        user_agent="Example Owner owner@example.com",
        cache_dir=str(tmp_path),
        session=session,
    )

    assert fetcher.fetch_quarterly_financials("ACME") is None
