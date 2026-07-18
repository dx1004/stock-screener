"""Main Quant Analysis & Execution Engine.

This engine:
1. Fetches data (price history + fundamentals)
2. Detects phase and relative strength context
3. Scores buy/sell candidates
4. Applies conservative risk controls
5. Builds both readable and structured weekly reports
"""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

import pandas as pd

from src.data.fetcher import YahooFinanceFetcher
from src.data.fundamentals_fetcher import (
    create_fundamental_snapshot,
    analyze_fundamentals_for_signal,
    fetch_quarterly_financials,
    get_fundamentals_coverage,
    reset_fundamentals_coverage,
)
from .benchmark import (
    analyze_spy_trend,
    calculate_market_breadth,
    format_benchmark_summary,
    should_generate_signals,
)
from .phase_indicators import classify_phase, calculate_relative_strength
from .risk_management import evaluate_buy_risk
from .signal_engine import score_buy_signal, score_sell_signal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QuantAnalysisEngine:
    """Autonomous Quant Analysis & Execution Engine."""

    DEFAULT_RISK_CONTROL = {
        "atr_period": 14,
        "spy_atr_period": 14,
        "spy_atr_multiple": 2.0,
        "min_atr_distance_multiple": 2.0,
        "max_stop_distance_pct": 0.08,
        "min_rr": 2.5,
        "max_data_staleness_days": 3,
        "min_entry_price": 10.0,
        "min_median_20d_dollar_volume": 20000000,
        "base_risk_budget_pct": 1.0,
        "max_notional_pct": 10.0,
        "max_new_buys": 2,
    }

    def __init__(
        self,
        cache_dir: str = "./data/cache",
        risk_policy: Optional[Dict[str, Any]] = None,
        min_buy_score: Optional[float] = 70,
        min_sell_score: Optional[float] = 60,
        min_phase2_pct: Optional[float] = 15.0,
        holdings: Optional[List[str]] = None,
    ):
        """Initialize the engine.

        Args:
            cache_dir: Directory for caching data
            risk_policy: Optional risk policy overrides
            min_buy_score: Buy threshold in score scale
            min_sell_score: Sell threshold in score scale
            holdings: Current holdings to evaluate for HOLD/SELL actions
        """
        self.fetcher = YahooFinanceFetcher(cache_dir=cache_dir)
        self.spy_data = None
        self.spy_price = None
        self.spy_analysis = {}
        self.risk_policy = {**self.DEFAULT_RISK_CONTROL, **(risk_policy or {})}
        self.min_buy_score = float(min_buy_score or 70)
        self.min_sell_score = float(min_sell_score or 60)
        self.min_phase2_pct = float(min_phase2_pct or 15.0)
        self.holdings = [h.upper() for h in (holdings or [])]
        self.latest_report_payload: Dict[str, Any] = {}
        logger.info("QuantAnalysisEngine initialized")

    def _unique_tickers(self, tickers: List[str]) -> List[str]:
        seen = set()
        ordered = []
        for ticker in tickers:
            ticker = ticker.upper().strip()
            if not ticker or ticker in seen:
                continue
            seen.add(ticker)
            ordered.append(ticker)
        return ordered

    def fetch_spy_data(self) -> bool:
        """Fetch SPY benchmark data.

        Returns:
            True if successful
        """
        try:
            logger.info("Fetching SPY data...")
            spy_hist = self.fetcher.fetch_price_history('SPY', period='2y')

            if spy_hist.empty:
                logger.error("Failed to fetch SPY data")
                return False

            self.spy_data = spy_hist
            self.spy_price = float(spy_hist['Close'].iloc[-1])
            self.spy_analysis = analyze_spy_trend(self.spy_data, self.spy_price)
            logger.info(f"SPY data fetched: {len(spy_hist)} days, current price: ${self.spy_price:.2f}")
            return True

        except Exception as e:
            logger.error(f"Error fetching SPY data: {e}")
            return False

    def analyze_stock(self, ticker: str) -> Optional[Dict]:
        """Analyze a single stock."""
        try:
            logger.info(f"Analyzing {ticker}...")

            price_data = self.fetcher.fetch_price_history(ticker, period='2y')
            if price_data.empty or len(price_data) < 200:
                logger.warning(f"{ticker}: Insufficient price data ({len(price_data)} days)")
                return None

            current_price = float(price_data['Close'].iloc[-1])
            phase_info = classify_phase(price_data, current_price)

            rs_series = calculate_relative_strength(
                price_data['Close'],
                self.spy_data['Close'],
                period=63
            )

            quarterly_data = fetch_quarterly_financials(ticker)
            fundamental_analysis = analyze_fundamentals_for_signal(quarterly_data)

            return {
                "ticker": ticker,
                "price_data": price_data,
                "current_price": current_price,
                "phase_info": phase_info,
                "rs_series": rs_series,
                "quarterly_data": quarterly_data,
                "fundamental_analysis": fundamental_analysis,
                "analysis_error": None,
            }

        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return {
                "ticker": ticker,
                "price_data": pd.DataFrame(),
                "current_price": None,
                "phase_info": None,
                "rs_series": pd.Series(dtype=float),
                "quarterly_data": pd.DataFrame(),
                "fundamental_analysis": None,
                "analysis_error": str(e),
            }

    def _enrich_buy_with_risk(
        self,
        buy_signal: Dict,
        analysis: Dict,
        benchmark_regime: str,
        rejected_candidates: List[Dict],
    ) -> Dict:
        if not isinstance(buy_signal, dict):
            return {
                "source": buy_signal,
                "qualified": False,
                "status": "REJECT",
                "reject_reasons": ["Invalid buy signal payload"]
            }

        # Evaluate conservative risk gates only for screened buy candidates.
        risk_assessment = evaluate_buy_risk(
            candidate=buy_signal,
            analysis=analysis,
            spy_context={
                "price_data": self.spy_data,
                "current_price": self.spy_price,
            },
            benchmark_regime=benchmark_regime,
            risk_policy=self.risk_policy,
        )

        enriched = dict(buy_signal)
        enriched["risk_assessment"] = risk_assessment
        enriched["risk_grade"] = risk_assessment.get("risk", {}).get("spy_risk_grade", "RX")
        enriched["risk_status"] = risk_assessment["status"]
        enriched["risk_reasons"] = risk_assessment.get("reasons", [])
        enriched["sizing_guidance"] = risk_assessment.get("sizing", {})
        enriched["qualified"] = risk_assessment["status"] == "PASS"
        enriched["entry_price"] = enriched.get("entry_price", float(analysis["current_price"]))

        if not enriched["qualified"]:
            rejected_candidates.append({
                "ticker": enriched["ticker"],
                "score": enriched.get("score"),
                "status": risk_assessment["status"],
                "reject_reasons": enriched["risk_reasons"],
            })

        return enriched

    def screen_stocks(self, tickers: List[str]) -> Dict[str, Any]:
        """Screen a list of stocks for buy/sell signals and holding actions."""
        raw_tickers = self._unique_tickers((tickers or []) + self.holdings)
        logger.info(f"Screening {len(raw_tickers)} stocks...")
        reset_fundamentals_coverage()

        if self.spy_data is None and not self.fetch_spy_data():
            logger.error("Cannot proceed without SPY data")
            return {
                "status": "incomplete",
                "error": "Failed to fetch SPY data",
                "timestamp": datetime.now().isoformat(),
                "total_analyzed": 0,
                "completeness_errors": ["SPY 数据缺失"],
            }

        # Analyze all stocks
        all_analyses: List[Dict] = []
        phase_results: List[Dict] = []
        analysis_by_ticker: Dict[str, Dict] = {}
        completeness_errors = []

        for ticker in raw_tickers:
            analysis = self.analyze_stock(ticker)
            if analysis:
                all_analyses.append(analysis)
                analysis_by_ticker[ticker] = analysis
                phase = None
                if isinstance(analysis.get("phase_info"), dict):
                    phase = analysis["phase_info"].get("phase")
                phase_results.append({
                    "ticker": ticker,
                    "phase": phase,
                })
            else:
                completeness_errors.append(f"{ticker}: 股票数据不足或抓取失败，无法分析")
                logger.debug(f"Skipping {ticker}: analyze_stock returned None")

        logger.info(f"Successfully analyzed {len(all_analyses)}/{len(raw_tickers)} stocks")

        breadth = calculate_market_breadth(phase_results)
        signal_recommendation = should_generate_signals(
            self.spy_analysis,
            breadth,
            min_phase2_pct=self.min_phase2_pct
        )

        report_status = "ok"

        # Score buy signals
        screened_buys: List[Dict] = []
        all_buys: List[Dict] = []
        rejected_candidates: List[Dict] = []

        if signal_recommendation.get("should_generate_buys"):
            for analysis in all_analyses:
                if analysis.get("analysis_error"):
                    completeness_errors.append(f"{analysis['ticker']}: {analysis['analysis_error']}")
                    continue

                buy_signal = score_buy_signal(
                    ticker=analysis["ticker"],
                    price_data=analysis["price_data"],
                    current_price=analysis["current_price"],
                    phase_info=analysis["phase_info"],
                    rs_series=analysis["rs_series"],
                    fundamentals=analysis["fundamental_analysis"],
                )

                if not buy_signal.get("is_buy", False):
                    continue

                score = float(buy_signal.get("score", 0))
                if score <= self.min_buy_score:
                    rejected_candidates.append({
                        "ticker": analysis["ticker"],
                        "score": score,
                        "status": "REJECT",
                        "reject_reasons": [f"买入评分未达阈值（需要 > {self.min_buy_score}）"],
                    })
                    continue

                buy_signal["phase_info"] = analysis["phase_info"]
                buy_signal["fundamental_snapshot"] = create_fundamental_snapshot(
                    analysis["ticker"],
                    analysis["quarterly_data"]
                )
                enriched = self._enrich_buy_with_risk(
                    buy_signal,
                    analysis,
                    signal_recommendation.get("regime", ""),
                    rejected_candidates
                )
                all_buys.append(enriched)
                if enriched.get("qualified"):
                    screened_buys.append(enriched)

        else:
            report_status = "complete"
            completeness_errors.append("当前市场环境不支持新买入（市场条件不足）")

        all_buys = sorted(
            all_buys,
            key=lambda x: (
                float(x.get("score", 0)),
                x.get("risk_assessment", {}).get("risk", {}).get("risk_reward_ratio", 0)
            ),
            reverse=True
        )

        screened_buys = sorted(
            screened_buys,
            key=lambda x: (
                float(x.get("score", 0)),
                x.get("risk_assessment", {}).get("risk", {}).get("risk_reward_ratio", 0)
            ),
            reverse=True
        )
        max_new_buys = min(2, int(self.risk_policy.get("max_new_buys", 2)))
        qualified_buys = screened_buys[:max_new_buys]

        if completeness_errors:
            qualified_buys = []

        # Score sell signals
        sell_candidates = []
        if signal_recommendation.get("should_generate_sells", True):
            for analysis in all_analyses:
                if analysis.get("analysis_error"):
                    continue
                sell_signal = score_sell_signal(
                    ticker=analysis["ticker"],
                    price_data=analysis["price_data"],
                    current_price=analysis["current_price"],
                    phase_info=analysis["phase_info"],
                    rs_series=analysis["rs_series"],
                    previous_phase=None,
                )
                if sell_signal.get("is_sell") and sell_signal.get("score", 0) >= self.min_sell_score:
                    sell_candidates.append(sell_signal)
        sell_candidates = sorted(sell_candidates, key=lambda x: x.get("score", 0), reverse=True)

        sell_by_ticker = {s["ticker"]: s for s in sell_candidates}
        holdings_actions = []
        for holding in self.holdings:
            analysis = analysis_by_ticker.get(holding)
            if not analysis or analysis.get("analysis_error"):
                holdings_actions.append({
                    "ticker": holding,
                    "action": "DATA_INCOMPLETE",
                    "reason": "持仓股票分析数据不完整，无法执行自动风控判断",
                })
                continue

            sell_signal = sell_by_ticker.get(holding)
            if sell_signal:
                holdings_actions.append({
                    "ticker": holding,
                    "action": "SELL",
                    "reason": f"触发卖出警告（分数 {sell_signal['score']}）",
                    "sell_score": sell_signal.get("score"),
                    "sell_severity": sell_signal.get("severity"),
                })
            else:
                holdings_actions.append({
                    "ticker": holding,
                    "action": "HOLD",
                    "reason": "未触发高可信卖出警告（买入候选评分与风控不替代平时持仓执行）",
                })

        return {
            "schema_version": "1.0",
            "status": report_status if not completeness_errors else "incomplete",
            "timestamp": datetime.now().isoformat(),
            "error": None,
            "spy_analysis": self.spy_analysis,
            "breadth": breadth,
            "signal_recommendation": signal_recommendation,
            "buys": all_buys,
            "qualified_buys": qualified_buys,
            "rejected_candidates": rejected_candidates,
            "sells": sell_candidates,
            "holdings_actions": holdings_actions,
            "risk_policy": self.risk_policy,
            "total_analyzed": len(all_analyses),
            "fundamentals_coverage": get_fundamentals_coverage(),
            "completeness_errors": completeness_errors,
            "all_analyses": {
                t: {
                    "has_data": bool(v.get("price_data") is not None and not v["price_data"].empty),
                    "analysis_error": v.get("analysis_error"),
                }
                for t, v in analysis_by_ticker.items()
            },
        }

    def _format_buy_output(self, candidates: List[Dict]) -> List[str]:
        lines = []
        for i, buy in enumerate(candidates, 1):
            risk = buy.get("risk_assessment", {}).get("risk", {})
            lines.extend([
                f"\n{'#'*60}",
                f"BUY #{i}: {buy['ticker']} | Score: {buy['score']}/125 | Phase {buy['phase']}",
                f"{'#'*60}",
                f"Entry: ${risk.get('entry_price', buy.get('current_price', 0)):.2f}",
            ])
            if risk.get("stop_loss") is not None:
                lines.append(f"Structured Stop: ${risk['stop_loss']:.2f}")
            if risk.get("stop_distance_atr_multiple") is not None:
                lines.append(f"Entry→ATR Stop Distance: {risk['stop_distance_atr_multiple']:.2f}x ATR")
            if risk.get("risk_reward_ratio") is not None:
                lines.append(f"S/R 1-4月目标RR: {risk['risk_reward_ratio']:.2f}:1")
            if risk.get("swing_target_1_4m") is not None:
                lines.append(f"1-4月目标位: ${risk['swing_target_1_4m']:.2f}")
            if risk.get("spy_risk_grade") is not None:
                lines.append(f"风险等级: {risk['spy_risk_grade']} (以 S&P 500 指数风险为标准)")
            if risk.get("risk_grade") is not None:
                lines.append(f"风险说明: {risk['risk_grade']}")
            sizing = buy.get("sizing_guidance", {})
            if sizing:
                lines.append(f"仓位公式: {sizing.get('risk_basis_symbolic', '')}")
                lines.append(f"单股风险: {sizing.get('max_shares_symbolic', '')}")
                lines.append(f"仓位上限: {sizing.get('notional_cap_symbolic', '')}")
            if isinstance(buy.get("phase_info"), dict) and 'distance_from_50sma' in buy["phase_info"]:
                lines.append(f"Distance from 50 SMA: {buy['phase_info']['distance_from_50sma']:.1f}%")

            lines.append("\nReasons:")
            for reason in buy.get("risk_reasons", []):
                lines.append(f"  • {reason}")
            for reason in buy.get("reasons", []):
                lines.append(f"  • {reason}")

            if buy.get("fundamental_snapshot"):
                lines.append(buy["fundamental_snapshot"])

        return lines

    def _format_sell_output(self, sells: List[Dict], holdings_actions: List[Dict]) -> List[str]:
        lines = []
        if sells:
            lines.append(f"\nFound {len(sells)} SELL warnings (score >= {int(self.min_sell_score)}):\n")
            for i, sell in enumerate(sells, 1):
                lines.extend([
                    f"\n{'#'*60}",
                    f"SELL #{i}: {sell['ticker']} | Score: {sell['score']}/100 | Severity: {sell['severity'].upper()}",
                    f"{'#'*60}",
                    f"Phase: {sell['phase']}",
                    f"Breakdown Level: ${sell['breakdown_level']:.2f}" if sell.get("breakdown_level") else "Breakdown Level: N/A",
                ])
                details = sell.get("details", {})
                if "rs_slope" in details:
                    lines.append(f"RS Rollover: {details['rs_slope']:.3f}")
                if "volume_ratio" in details:
                    lines.append(f"Volume vs Avg: {details['volume_ratio']:.1f}x")
                lines.append("\nReasons:")
                for reason in sell.get("reasons", []):
                    lines.append(f"  • {reason}")

        else:
            lines.append("\n✗ NO SELL WARNINGS TODAY")

        lines.append("")
        lines.append("HOLDINGS ACTIONS")
        lines.append("-"*40)
        if holdings_actions:
            for action in holdings_actions:
                lines.append(
                    f"{action['ticker']}: {action['action']} | {action['reason']}"
                )
        else:
            lines.append("No holdings configured.")
        return lines

    def _format_legacy_buy_output(self, candidates: List[Dict]) -> List[str]:
        lines = []
        for i, buy in enumerate(candidates, 1):
            lines.extend([
                f"\n{'#'*60}",
                f"BUY #{i}: {buy['ticker']} | Score: {buy['score']}/100",
                f"{'#'*60}",
                f"Phase: {buy['phase']}"
            ])
            if buy.get('breakout_price'):
                lines.append(f"Breakout Price: ${buy['breakout_price']:.2f}")
            details = buy.get("details", {})
            if 'rs_slope' in details:
                lines.append(f"RS Slope (3-week): {details['rs_slope']:.3f}")
            if 'volume_ratio' in details:
                lines.append(f"Volume vs Avg: {details['volume_ratio']:.1f}x")
            phase_info = buy.get('phase_info', {})
            if isinstance(phase_info, dict) and 'distance_from_50sma' in phase_info:
                lines.append(f"Distance from 50 SMA: {phase_info['distance_from_50sma']:.1f}%")

            lines.append("\nReasons:")
            for reason in buy.get("reasons", []):
                lines.append(f"  • {reason}")

            if buy.get("fundamental_snapshot"):
                lines.append(buy["fundamental_snapshot"])
        return lines

    def run_report(self, tickers: List[str]) -> (str, Dict[str, Any]):
        """Run screening and return report text and structured payload."""
        logger.info("="*60)
        logger.info("QUANT ANALYSIS & EXECUTION ENGINE - STARTING")
        logger.info("="*60)

        results = self.screen_stocks(tickers)
        self.latest_report_payload = dict(results)
        if results.get("error"):
            error_text = f"ERROR: {results['error']}"
            return error_text, results

        # Backward-compatible rendering entry points for text report
        output: List[str] = []
        output.append("\n" + "="*60)
        output.append("QUANT ANALYSIS & EXECUTION ENGINE")
        output.append(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"Stocks Analyzed: {results['total_analyzed']}")
        coverage = results["fundamentals_coverage"]
        output.append(
            "Fundamentals: "
            f"SEC US-GAAP {coverage['sec_edgar']}, SEC IFRS {coverage['sec_edgar_ifrs']}, "
            f"Yahoo fallback {coverage['yahoo_fallback']}, unavailable {coverage['unavailable']}"
        )
        output.append("="*60)

        output.append(format_benchmark_summary(results["spy_analysis"], results["breadth"]))

        output.append("\n" + "="*60)
        output.append(f"BUY LIST (Score > {int(self.min_buy_score)} + Risk gates)")
        output.append("="*60)

        if results.get("completeness_errors"):
            output.append("\n⚠ Market/Inputs completeness issues:")
            for err in results["completeness_errors"]:
                output.append(f"  • {err}")

        qualified_buys = results.get("qualified_buys", [])
        all_buys = results.get("buys", [])
        rejected_candidates = results.get("rejected_candidates", [])
        has_risk_payload = all(
            isinstance(b, dict) and b.get("risk_assessment") is not None for b in all_buys
        )
        if not qualified_buys and all_buys and not has_risk_payload:
            # Compatibility path for legacy/mock structures.
            qualified_buys = all_buys

        if qualified_buys:
            output.append(f"\nFound {len(qualified_buys)} QUALIFIED BUY candidates (max {min(2, len(qualified_buys))} shown):\n")
            if has_risk_payload:
                output.extend(self._format_buy_output(qualified_buys))
            else:
                output.extend(self._format_legacy_buy_output(qualified_buys))
        else:
            if results.get("buys") and not has_risk_payload:
                qualified_buys = results.get("buys", [])
                output.append(f"\nFound {len(qualified_buys)} QUALIFIED BUY candidates (legacy payload fallback):\n")
                output.extend(self._format_legacy_buy_output(qualified_buys))
            else:
                output.append("\n✗ NO QUALIFIED BUY CANDIDATES")

        if rejected_candidates:
            output.append("\n\nREJECTED CANDIDATES (Raw score/规则原因):")
            for item in rejected_candidates:
                output.append(
                    f"{item['ticker']} | score={item.get('score')} | status={item.get('status')} | "
                    f"reasons={', '.join(item.get('reject_reasons', []))}"
                )

        output.append("\n" + "="*60)
        output.append(f"SELL LIST (Score >= {int(self.min_sell_score)})")
        output.append("="*60)
        output.extend(
            self._format_sell_output(results.get("sells", []), results.get("holdings_actions", []))
        )

        output.append("\n" + "="*60)
        output.append("END OF REPORT")
        output.append("="*60)

        # Ensure no buy actions for stale/missing data states
        if results.get("status", "ok") != "ok":
            output.append("\nNote: Data completeness或市场状态不满足周报条件，本周不触发新增买入建议。")

        report = "\n".join(output)
        return report, results

    def run(self, tickers: List[str]) -> str:
        """Backward-compatible API: return report text only."""
        return self.run_report(tickers)[0]
