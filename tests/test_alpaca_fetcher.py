"""Unit tests for Alpaca daily-bar retrieval without live credentials."""

from unittest.mock import Mock

import pandas as pd
import pytest
import requests

from src.data.alpaca_fetcher import AlpacaPriceFetcher


@pytest.fixture(autouse=True)
def clear_alpaca_credentials(monkeypatch):
    """Ensure these unit tests never use local or CI credentials."""
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)


def _response(payload):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_fetch_price_history_returns_normalized_daily_bars():
    session = Mock()
    session.get.return_value = _response({
        "bars": [
            {"t": "2025-01-03T05:00:00Z", "o": 10.0, "h": 12.0, "l": 9.0, "c": 11.0, "v": 100},
            {"t": "2025-01-02T05:00:00Z", "o": 9.0, "h": 11.0, "l": 8.0, "c": 10.0, "v": 90},
        ],
        "next_page_token": None,
    })
    fetcher = AlpacaPriceFetcher(api_key="key", api_secret="secret", session=session)

    result = fetcher.fetch_price_history("AAPL", period="1mo")

    assert isinstance(result.index, pd.DatetimeIndex)
    assert result.index.name == "Date"
    assert result.columns.tolist() == ["Open", "High", "Low", "Close", "Volume"]
    assert result.index.is_monotonic_increasing
    assert result.iloc[-1]["Close"] == 11.0
    assert session.get.call_count == 1
    request = session.get.call_args
    assert request.args[0].endswith("/AAPL/bars")
    assert request.kwargs["headers"]["APCA-API-KEY-ID"] == "key"
    assert request.kwargs["params"]["feed"] == "sip"


def test_fetch_price_history_follows_pages():
    session = Mock()
    session.get.side_effect = [
        _response({"bars": [{"t": "2025-01-02T05:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 1}], "next_page_token": "next"}),
        _response({"bars": [{"t": "2025-01-03T05:00:00Z", "o": 2, "h": 3, "l": 2, "c": 3, "v": 2}], "next_page_token": None}),
    ]
    fetcher = AlpacaPriceFetcher(api_key="key", api_secret="secret", session=session)

    result = fetcher.fetch_price_history("AAPL", period="1mo")

    assert len(result) == 2
    assert session.get.call_count == 2
    assert session.get.call_args_list[1].kwargs["params"]["page_token"] == "next"


def test_fetch_price_history_returns_none_without_credentials_or_for_non_daily_interval():
    session = Mock()
    fetcher = AlpacaPriceFetcher(session=session)

    assert fetcher.fetch_price_history("AAPL") is None
    assert session.get.call_count == 0

    configured = AlpacaPriceFetcher(api_key="key", api_secret="secret", session=session)
    assert configured.fetch_price_history("AAPL", interval="1h") is None
    assert session.get.call_count == 0


def test_fetch_price_history_returns_none_after_request_failure():
    session = Mock()
    session.get.side_effect = requests.RequestException("unavailable")
    fetcher = AlpacaPriceFetcher(api_key="key", api_secret="secret", session=session)

    assert fetcher.fetch_price_history("AAPL") is None


def test_fetch_price_history_returns_none_for_an_unexpected_payload():
    session = Mock()
    session.get.return_value = _response([])
    fetcher = AlpacaPriceFetcher(api_key="key", api_secret="secret", session=session)

    assert fetcher.fetch_price_history("AAPL") is None
