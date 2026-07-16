"""Daily US equity price history from Alpaca Market Data.

The client is deliberately small and returns ``None`` when Alpaca is not
configured or cannot serve a request.  Callers can then use their configured
fallback provider without exposing credentials or treating provider outages as
symbol failures.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


class AlpacaPriceFetcher:
    """Fetch adjusted daily OHLCV bars from Alpaca's historical API."""

    BASE_URL = "https://data.alpaca.markets/v2/stocks"
    _PERIOD_DAYS = {
        "1d": 3,
        "5d": 10,
        "1mo": 35,
        "3mo": 100,
        "6mo": 190,
        "1y": 380,
        "2y": 760,
        "5y": 1900,
        "10y": 3800,
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ALPACA_API_KEY")
        self.api_secret = api_secret or os.getenv("ALPACA_API_SECRET")
        self.session = session or requests.Session()

    @property
    def is_configured(self) -> bool:
        """Whether credentials are available for an authenticated request."""
        return bool(self.api_key and self.api_secret)

    def fetch_price_history(
        self,
        ticker: str,
        period: str = "5y",
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Return adjusted daily bars, or ``None`` when a fallback is needed.

        Alpaca's historical endpoint is used only for daily bars because the
        screener's public fetcher contract supports yfinance-specific intraday
        intervals as well.  Those requests continue directly to the fallback.
        """
        if not self.is_configured or interval != "1d":
            return None

        start = self._period_start(period)
        if start is None:
            logger.info("Alpaca does not support period=%s; using fallback", period)
            return None

        headers = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
        }
        params = {
            "timeframe": "1Day",
            "start": start.isoformat(),
            # Basic Alpaca accounts can query SIP history when the requested
            # end time is at least 15 minutes old.  This preserves access to
            # the consolidated daily bars used by the scheduled screener.
            "end": (datetime.now(timezone.utc) - timedelta(minutes=16)).isoformat(),
            "adjustment": "all",
            "feed": "sip",
            "limit": 10000,
        }
        bars = []

        try:
            while True:
                response = self.session.get(
                    f"{self.BASE_URL}/{ticker}/bars",
                    headers=headers,
                    params=params,
                    timeout=15,
                )
                response.raise_for_status()
                payload = response.json()
                bars.extend(payload.get("bars", []))

                page_token = payload.get("next_page_token")
                if not page_token:
                    break
                params["page_token"] = page_token
        except (requests.RequestException, ValueError, AttributeError, TypeError) as exc:
            logger.warning("Alpaca price history unavailable for %s: %s", ticker, exc)
            return None

        if not bars:
            logger.warning("Alpaca returned no price history for %s", ticker)
            return None

        frame = pd.DataFrame(bars)
        required_columns = {"t", "o", "h", "l", "c", "v"}
        if not required_columns.issubset(frame.columns):
            logger.warning("Alpaca returned an incomplete bar payload for %s", ticker)
            return None

        frame = frame.rename(
            columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"}
        )
        index = pd.to_datetime(frame.pop("t"), utc=True)
        frame.index = index.dt.tz_convert("America/New_York").dt.normalize()
        frame.index.name = "Date"
        return frame[["Open", "High", "Low", "Close", "Volume"]].sort_index()

    def _period_start(self, period: str) -> Optional[datetime]:
        """Translate the shared yfinance-style period into an Alpaca start time."""
        now = datetime.now(timezone.utc)
        if period == "ytd":
            return datetime(now.year, 1, 1, tzinfo=timezone.utc)
        if period == "max":
            return datetime(2016, 1, 1, tzinfo=timezone.utc)

        days = self._PERIOD_DAYS.get(period)
        return now - timedelta(days=days) if days else None
