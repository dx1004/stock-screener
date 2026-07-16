"""Coverage for Alpaca-first routing in scheduled price fetchers."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.data.git_storage_fetcher import GitStorageFetcher
from src.data.smart_fetcher import SmartDataFetcher


def _price_history():
    return pd.DataFrame(
        {
            "Open": [10.0, 11.0],
            "High": [11.0, 12.0],
            "Low": [9.0, 10.0],
            "Close": [10.5, 11.5],
            "Volume": [100, 110],
        },
        index=pd.date_range("2025-01-02", periods=2, freq="D", name="Date"),
    )


def test_git_storage_fetcher_prefers_alpaca_prices(tmp_path):
    fetcher = GitStorageFetcher(fundamentals_dir=str(tmp_path / "fundamentals"))
    expected = _price_history()
    fetcher.alpaca_fetcher.fetch_price_history = Mock(return_value=expected)

    with patch("yfinance.Ticker") as yahoo_ticker:
        result = fetcher.fetch_price_fresh("AAPL")

    pd.testing.assert_frame_equal(result, expected)
    yahoo_ticker.assert_not_called()
    fetcher.alpaca_fetcher.fetch_price_history.assert_called_once_with("AAPL", period="1y")


def test_git_storage_fetcher_falls_back_to_yahoo(tmp_path):
    fetcher = GitStorageFetcher(fundamentals_dir=str(tmp_path / "fundamentals"))
    expected = _price_history()
    fetcher.alpaca_fetcher.fetch_price_history = Mock(return_value=None)

    with patch("yfinance.Ticker") as yahoo_ticker:
        yahoo_ticker.return_value.history.return_value = expected
        result = fetcher.fetch_price_fresh("AAPL")

    pd.testing.assert_frame_equal(result, expected)
    yahoo_ticker.return_value.history.assert_called_once_with(period="1y", interval="1d")


def test_smart_fetcher_prefers_alpaca_for_a_full_refresh(tmp_path):
    fetcher = SmartDataFetcher(cache_dir=str(tmp_path / "cache"))
    expected = _price_history()
    fetcher.alpaca_fetcher.fetch_price_history = Mock(return_value=expected)

    with patch("yfinance.Ticker") as yahoo_ticker:
        result = fetcher.fetch_price_incremental("AAPL", required_days=2)

    pd.testing.assert_frame_equal(result, expected)
    yahoo_ticker.assert_not_called()
    fetcher.alpaca_fetcher.fetch_price_history.assert_called_once_with("AAPL", period="1y")


def test_workflow_injects_alpaca_secrets_only_for_the_screening_step():
    workflow = (Path(__file__).parents[1] / ".github/workflows/daily_screening_git_storage.yml").read_text()

    before_screen, remaining_workflow = workflow.split("      - name: Run weekend quant screen", 1)
    screen_step, after_screen = remaining_workflow.split("      - name: Commit generated screening data", 1)
    assert "ALPACA_API_KEY: ${{ secrets.ALPACA_API_KEY }}" in screen_step
    assert "ALPACA_API_SECRET: ${{ secrets.ALPACA_API_SECRET }}" in screen_step
    assert "SEC_EDGAR_USER_AGENT: ${{ vars.SEC_EDGAR_USER_AGENT }}" in screen_step
    assert "ALPACA_API_" not in before_screen + after_screen
    assert "SEC_EDGAR_USER_AGENT" not in before_screen + after_screen
    assert "set -o pipefail\n          ./run_screen.sh" in screen_step
