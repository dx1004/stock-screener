"""Unit tests for candidate-only FMP backfill controls."""

from unittest.mock import Mock

from src.data.enhanced_fundamentals import EnhancedFundamentalsFetcher


def test_fmp_snapshot_fetches_one_selected_candidate_once():
    fetcher = EnhancedFundamentalsFetcher()
    fetcher.fmp_available = True
    fetcher.fmp_fetcher = Mock()
    fmp_data = {"ticker": "ACME", "income_statement": [{"revenue": 1}]}
    fetcher.fmp_fetcher.fetch_comprehensive_fundamentals.return_value = fmp_data
    fetcher.fmp_fetcher.create_enhanced_snapshot.return_value = "FMP snapshot"

    result = fetcher.create_snapshot("ACME", quarterly_data={"ticker": "ACME"}, use_fmp=True)

    assert result == "FMP snapshot"
    assert fetcher.fmp_call_count == 4
    fetcher.fmp_fetcher.fetch_comprehensive_fundamentals.assert_called_once_with("ACME")


def test_fmp_backfill_stops_before_exceeding_the_daily_quota():
    fetcher = EnhancedFundamentalsFetcher()
    fetcher.fmp_available = True
    fetcher.fmp_fetcher = Mock()
    fetcher.fmp_call_count = 248

    assert fetcher._fetch_fmp_data("ACME") is None
    fetcher.fmp_fetcher.fetch_comprehensive_fundamentals.assert_not_called()
