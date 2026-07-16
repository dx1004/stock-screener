"""SEC EDGAR Company Facts client for normalized quarterly fundamentals."""

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests
from dotenv import load_dotenv


load_dotenv()

logger = logging.getLogger(__name__)


class SECEdgarFundamentalsFetcher:
    """Fetch comparable quarterly US-GAAP facts while respecting SEC fair access."""

    COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    REQUESTS_PER_SECOND = 9
    CACHE_EXPIRY = timedelta(days=1)

    _TAXONOMY_TAGS = {
        "us-gaap": {
            "revenue": (
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "Revenues",
            ),
            "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
            "gross_profit": ("GrossProfit",),
            "operating_income": ("OperatingIncomeLoss",),
            "inventory": ("InventoryNet", "InventoryGross"),
        },
        "ifrs-full": {
            "revenue": ("Revenue",),
            "eps": (
                "DilutedEarningsLossPerShare",
                "BasicAndDilutedEarningsLossPerShare",
                "BasicEarningsLossPerShare",
            ),
            "gross_profit": ("GrossProfit",),
            "operating_income": ("ProfitLossFromOperatingActivities",),
            "inventory": ("Inventories",),
        },
    }

    def __init__(
        self,
        user_agent: Optional[str] = None,
        cache_dir: str = "./data/cache/sec_edgar",
        session: Optional[requests.Session] = None,
    ) -> None:
        self.user_agent = user_agent or os.getenv("SEC_EDGAR_USER_AGENT")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ticker_cache_path = self.cache_dir / "company_tickers.json"
        self.session = session or requests.Session()
        self._ticker_to_cik: Optional[Dict[str, int]] = None
        self._last_request_at = 0.0
        self._request_lock = threading.Lock()
        self._ticker_lock = threading.Lock()

    @property
    def is_configured(self) -> bool:
        """Whether a compliant identifying User-Agent was supplied."""
        return bool(self.user_agent)

    def fetch_quarterly_financials(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Return the existing fundamentals contract, or ``None`` for fallback."""
        if not self.is_configured:
            logger.debug("SEC_EDGAR_USER_AGENT is not set; using Yahoo fundamentals fallback")
            return None

        cik = self._resolve_cik(ticker)
        if cik is None:
            logger.warning("SEC EDGAR has no CIK mapping for %s", ticker)
            return None

        facts = self._request_json(self.COMPANY_FACTS_URL.format(cik=cik))
        if not facts:
            return None

        result = self._normalize_company_facts(ticker, facts)
        if result is None:
            logger.warning("SEC EDGAR returned incomplete quarterly facts for %s", ticker)
        return result

    def _resolve_cik(self, ticker: str) -> Optional[int]:
        with self._ticker_lock:
            if self._ticker_to_cik is None:
                self._ticker_to_cik = self._load_ticker_mapping()
        return self._ticker_to_cik.get(ticker.upper())

    def _load_ticker_mapping(self) -> Dict[str, int]:
        payload = self._load_cached_tickers()
        if payload is None:
            payload = self._request_json(self.COMPANY_TICKERS_URL)
            if not payload:
                return {}
            self._save_ticker_cache(payload)

        return {
            entry["ticker"].upper(): int(entry["cik_str"])
            for entry in payload.values()
            if entry.get("ticker") and entry.get("cik_str")
        }

    def _load_cached_tickers(self) -> Optional[Dict[str, Dict[str, Any]]]:
        if not self.ticker_cache_path.exists():
            return None
        if datetime.now() - datetime.fromtimestamp(self.ticker_cache_path.stat().st_mtime) > self.CACHE_EXPIRY:
            return None
        try:
            with self.ticker_cache_path.open() as handle:
                return json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Could not load SEC ticker cache: %s", exc)
            return None

    def _save_ticker_cache(self, payload: Dict[str, Dict[str, Any]]) -> None:
        try:
            with self.ticker_cache_path.open("w") as handle:
                json.dump(payload, handle)
        except OSError as exc:
            logger.warning("Could not save SEC ticker cache: %s", exc)

    def _request_json(self, url: str) -> Optional[Dict[str, Any]]:
        with self._request_lock:
            delay = (1 / self.REQUESTS_PER_SECOND) - (time.monotonic() - self._last_request_at)
            if delay > 0:
                time.sleep(delay)

            try:
                response = self.session.get(
                    url,
                    headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
                    timeout=20,
                )
                self._last_request_at = time.monotonic()
                response.raise_for_status()
                payload = response.json()
                return payload if isinstance(payload, dict) else None
            except (requests.RequestException, ValueError) as exc:
                logger.warning("SEC EDGAR request failed: %s", exc)
                return None

    def _normalize_company_facts(
        self, ticker: str, company_facts: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        all_facts = company_facts.get("facts", {})
        for taxonomy, data_source in (("us-gaap", "sec_edgar"), ("ifrs-full", "sec_edgar_ifrs")):
            result = self._normalize_taxonomy_facts(
                ticker, all_facts.get(taxonomy, {}), self._TAXONOMY_TAGS[taxonomy], data_source
            )
            if result is not None:
                return result
        return None

    def _normalize_taxonomy_facts(
        self,
        ticker: str,
        facts: Dict[str, Any],
        tags: Dict[str, Iterable[str]],
        data_source: str,
    ) -> Optional[Dict[str, Any]]:
        revenues = self._quarterly_records(facts, tags["revenue"])
        eps = self._quarterly_records(facts, tags["eps"])
        if not revenues or not eps:
            return None

        gross_profit = self._quarterly_records(facts, tags["gross_profit"])
        operating_income = self._quarterly_records(facts, tags["operating_income"])
        inventory = self._instant_records(facts, tags["inventory"])

        result: Dict[str, Any] = {
            "ticker": ticker,
            "fetch_date": datetime.now().isoformat(),
            "data_source": data_source,
            "quarterly_revenue": self._record_values(revenues),
            "quarterly_eps": self._record_values(eps),
            "inventory_breakdown_available": False,
        }
        self._add_growth_metrics(result, "revenue", revenues)
        self._add_growth_metrics(result, "eps", eps)

        latest_revenue = revenues[-1]
        gross_latest = self._value_for_end(gross_profit, latest_revenue["end"])
        gross_previous = self._value_for_end(gross_profit, revenues[-2]["end"])
        if gross_latest is not None:
            result["gross_margin"] = round(gross_latest / latest_revenue["value"] * 100, 2)
            if gross_previous is not None and revenues[-2]["value"]:
                previous_margin = gross_previous / revenues[-2]["value"] * 100
                result["margin_change"] = round(result["gross_margin"] - previous_margin, 2)

        operating_latest = self._value_for_end(operating_income, latest_revenue["end"])
        if operating_latest is not None:
            result["operating_margin"] = round(operating_latest / latest_revenue["value"] * 100, 2)

        if inventory:
            result["quarterly_inventory"] = self._record_values(inventory)
            latest_inventory = self._latest_on_or_before(inventory, latest_revenue["end"])
            previous_inventory = self._latest_on_or_before(inventory, revenues[-2]["end"])
            if latest_inventory is not None and previous_inventory is not None and previous_inventory["value"]:
                result["inventory_qoq_change"] = round(
                    (latest_inventory["value"] - previous_inventory["value"])
                    / previous_inventory["value"]
                    * 100,
                    2,
                )
            if latest_inventory is not None and latest_revenue["value"]:
                result["inventory_to_sales_ratio"] = round(
                    latest_inventory["value"] / latest_revenue["value"], 3
                )
        return result

    def _quarterly_records(
        self, facts: Dict[str, Any], tags: Iterable[str]
    ) -> List[Dict[str, Any]]:
        for tag in tags:
            records = self._records_from_tag(facts.get(tag), duration_only=True)
            if records:
                return records
        return []

    def _instant_records(
        self, facts: Dict[str, Any], tags: Iterable[str]
    ) -> List[Dict[str, Any]]:
        for tag in tags:
            records = self._records_from_tag(facts.get(tag), duration_only=False)
            if records:
                return records
        return []

    def _records_from_tag(
        self, fact: Optional[Dict[str, Any]], duration_only: bool
    ) -> List[Dict[str, Any]]:
        if not fact:
            return []

        entries = []
        for unit_entries in fact.get("units", {}).values():
            for entry in unit_entries:
                if entry.get("form") not in {"10-Q", "10-K"} or not isinstance(entry.get("val"), (int, float)):
                    continue
                if not entry.get("end") or not entry.get("fy") or not entry.get("fp"):
                    continue
                if duration_only:
                    if not entry.get("start"):
                        continue
                    duration = (datetime.fromisoformat(entry["end"]) - datetime.fromisoformat(entry["start"])).days
                    if not 70 <= duration <= 110:
                        continue
                entries.append({
                    "end": entry["end"],
                    "value": entry["val"],
                    "fy": int(entry["fy"]),
                    "fp": entry["fp"],
                    "filed": entry.get("filed", ""),
                })

        latest_by_period: Dict[tuple, Dict[str, Any]] = {}
        for entry in entries:
            key = (entry["end"], entry["fp"])
            if key not in latest_by_period or entry["filed"] > latest_by_period[key]["filed"]:
                latest_by_period[key] = entry
        return sorted(latest_by_period.values(), key=lambda entry: entry["end"])

    @staticmethod
    def _record_values(records: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {record["end"]: record["value"] for record in records}

    @staticmethod
    def _value_for_end(records: List[Dict[str, Any]], end: str) -> Optional[float]:
        for record in reversed(records):
            if record["end"] == end:
                return record["value"]
        return None

    @staticmethod
    def _latest_on_or_before(records: List[Dict[str, Any]], end: str) -> Optional[Dict[str, Any]]:
        return next((record for record in reversed(records) if record["end"] <= end), None)

    @staticmethod
    def _add_growth_metrics(result: Dict[str, Any], metric: str, records: List[Dict[str, Any]]) -> None:
        latest = records[-1]
        previous = records[-2] if len(records) >= 2 else None
        year_ago = next(
            (
                record
                for record in reversed(records[:-1])
                if record["fp"] == latest["fp"] and record["fy"] == latest["fy"] - 1
            ),
            None,
        )
        if previous and previous["value"]:
            denominator = abs(previous["value"]) if metric == "eps" else previous["value"]
            if denominator:
                result[f"{metric}_qoq_change"] = (latest["value"] - previous["value"]) / denominator * 100
        if year_ago and year_ago["value"]:
            denominator = abs(year_ago["value"]) if metric == "eps" else year_ago["value"]
            if denominator:
                result[f"{metric}_yoy_change"] = (latest["value"] - year_ago["value"]) / denominator * 100
