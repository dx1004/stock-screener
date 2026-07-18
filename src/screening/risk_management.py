"""Risk-control helpers for weekly screening.

This module contains the conservative, portfolio-level risk gates requested for the
weekly screening report. It intentionally avoids any order execution semantics and
only outputs analysis-grade sizing guidance expressed with symbolic capital `C`.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, Optional

import pandas as pd


DEFAULT_SPY_GRADE_LABEL = "RX"


def calculate_atr(price_data: pd.DataFrame, period: int = 14) -> Optional[float]:
    """Calculate Wilder ATR for a price dataframe.

    Args:
        price_data: OHLCV dataframe.
        period: ATR period.
    """
    required = {"High", "Low", "Close"}
    if price_data is None or price_data.empty:
        return None
    if not required.issubset(price_data.columns):
        return None

    highs = price_data["High"].astype(float)
    lows = price_data["Low"].astype(float)
    closes = price_data["Close"].astype(float)
    if len(price_data) < period + 1:
        return None

    prev_close = closes.shift(1)
    tr = pd.concat([
        highs - lows,
        (highs - prev_close).abs(),
        (lows - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period, min_periods=period).mean().iloc[-1]
    if pd.isna(atr) or atr <= 0:
        return None
    return float(atr)


def _market_regime_multiplier(regime: str) -> float:
    """Map benchmark regime string to risk multiplier."""
    if "RISK-ON (Strong)" in regime:
        return 1.0
    if "RISK-ON (Moderate)" in regime:
        return 0.75
    if "RISK-ON (Weak)" in regime:
        return 0.5
    if "TRANSITIONAL" in regime:
        return 0.25
    if "RISK-OFF" in regime:
        return 0.0
    return 0.5


def _spy_risk_grade(spy_risk_ratio: Optional[float]) -> str:
    """Classify SPY-relative risk grade."""
    if spy_risk_ratio is None or not math.isfinite(spy_risk_ratio):
        return DEFAULT_SPY_GRADE_LABEL

    if spy_risk_ratio <= 0.75:
        return "R1"
    if spy_risk_ratio < 0.90:
        return "R2"
    if spy_risk_ratio <= 1.10:
        return "R3"
    if spy_risk_ratio <= 1.50:
        return "R4"
    return "R5"


def _grade_multiplier(grade: str) -> float:
    return {
        "R1": 1.0,
        "R2": 1.0,
        "R3": 0.8,
        "R4": 0.5,
        "R5": 0.0,
        DEFAULT_SPY_GRADE_LABEL: 0.0,
    }.get(grade, 0.0)


def classify_bearish_staleness(latest_index, max_age_days: int = 3) -> bool:
    """Return True when data is stale."""
    if latest_index is None or pd.isna(latest_index):
        return True

    if hasattr(latest_index, "tzinfo") and latest_index.tzinfo is not None:
        latest_index = latest_index.tz_localize(None)

    now = datetime.now()
    try:
        age = now - latest_index.to_pydatetime().replace(tzinfo=None)
    except AttributeError:
        age = now - datetime.fromtimestamp(latest_index.timestamp())
    return age > timedelta(days=max_age_days)


def evaluate_buy_risk(
    candidate: Dict,
    analysis: Dict,
    spy_context: Dict,
    benchmark_regime: str,
    risk_policy: Dict,
) -> Dict:
    """Return risk gate decisions for a buy candidate.

    Returns dict with:
      - status: PASS / REJECT / DATA_INCOMPLETE
      - reasons: list
      - risk: risk metrics
      - sizing: symbolic sizing guidance
    """
    reasons = []
    risk = {}
    sizing = {}

    price_data = analysis.get("price_data")
    if price_data is None or price_data.empty or len(price_data) < 2:
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少足够日K价格数据（<2 rows）"],
            "risk": risk,
            "sizing": sizing,
        }

    if "Close" not in price_data.columns:
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少Close列，无法确认入场点与止损距离"],
            "risk": risk,
            "sizing": sizing,
        }

    latest_price = float(price_data["Close"].iloc[-1])
    entry_price = float(candidate.get("entry_price", latest_price))
    stop_loss = candidate.get("stop_loss")
    if stop_loss is None or not (isinstance(stop_loss, (int, float)) and math.isfinite(stop_loss)):
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少可计算止损价（入场至ATR止损距离缺失）"],
            "risk": risk,
            "sizing": sizing,
        }

    stop_loss = float(stop_loss)
    stop_distance = entry_price - stop_loss
    if stop_distance <= 0:
        return {
            "status": "REJECT",
            "reasons": ["结构性止损在入场上方/平价，风险失效"],
            "risk": risk,
            "sizing": sizing,
        }

    atr = calculate_atr(price_data, period=int(risk_policy.get("atr_period", 14)))
    if not atr:
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少可用ATR（ATR计算失败）"],
            "risk": risk,
            "sizing": sizing,
        }

    stop_distance_atr_multiple = stop_distance / atr
    stop_distance_pct = (stop_distance / entry_price) if entry_price > 0 else 0.0

    # Liquidity gates
    min_price = float(risk_policy.get("min_entry_price", 10.0))
    if latest_price < min_price:
        return {
            "status": "REJECT",
            "reasons": [f"最新价低于流动性门槛（<{min_price:.2f}）"],
            "risk": {"entry_price": round(entry_price, 4), "latest_price": round(latest_price, 4)},
            "sizing": sizing,
        }

    if "Volume" not in price_data.columns:
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少成交量列，无法通过保守流动性筛选"],
            "risk": risk,
            "sizing": sizing,
        }

    dollar_volume = (price_data["Close"] * price_data["Volume"]).astype(float)
    median_20d_dollar_volume = dollar_volume.tail(20).median()
    min_median_20d_dollar_volume = float(risk_policy.get("min_median_20d_dollar_volume", 20000000))
    if pd.isna(median_20d_dollar_volume) or median_20d_dollar_volume < min_median_20d_dollar_volume:
        return {
            "status": "REJECT",
            "reasons": ["保守流动性不达标（20日中位数成交额不足）"],
            "risk": {
                "median_20d_dollar_volume": round(float(median_20d_dollar_volume) if pd.notna(median_20d_dollar_volume) else 0.0, 2),
            },
            "sizing": sizing,
        }

    if len(price_data.index) == 0:
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少K线时间轴信息"],
            "risk": risk,
            "sizing": sizing,
        }

    latest_index = price_data.index[-1]
    stale_days = int(risk_policy.get("max_data_staleness_days", 3))
    if classify_bearish_staleness(latest_index, stale_days):
        return {
            "status": "REJECT",
            "reasons": [f"行情数据过期（最新日期不在{stale_days}日内）"],
            "risk": risk,
            "sizing": sizing,
        }

    # 1-4 month swing RR target: conservatively require min RR with pivot-range estimate
    prior_window = price_data.iloc[-61:-1] if len(price_data) > 61 else price_data.iloc[:-1]
    if prior_window.empty or len(prior_window) < 20:
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["缺少完整 swing-window（不足以构造1-4月目标位）"],
            "risk": risk,
            "sizing": sizing,
        }

    pivot = float(prior_window["High"].max())
    base_low = float(prior_window["Low"].min())
    if not (pivot > 0 and base_low > 0 and not pd.isna(pivot) and not pd.isna(base_low)):
        return {
            "status": "DATA_INCOMPLETE",
            "reasons": ["无法稳定构建Pivot/Support用于风险回报预估"],
            "risk": risk,
            "sizing": sizing,
        }

    range_1 = pivot - base_low
    target = min(entry_price * 1.25, pivot + range_1)
    if target <= entry_price:
        target = entry_price * 1.25

    rr_ratio = (target - entry_price) / stop_distance if stop_distance > 0 else 0.0
    min_rr = float(risk_policy.get("min_rr", 2.5))
    if rr_ratio < min_rr:
        return {
            "status": "REJECT",
            "reasons": [f"风险回报不足（R/R={rr_ratio:.2f}，低于{min_rr}）"],
            "risk": {},
            "sizing": sizing,
        }

    # Stop distance cap by entry price and ATR sanity
    max_stop_pct = float(risk_policy.get("max_stop_distance_pct", 0.08))
    if stop_distance_pct > max_stop_pct:
        return {
            "status": "REJECT",
            "reasons": [f"止损距离过大（{stop_distance_pct:.2%} > {max_stop_pct:.2%}）"],
            "risk": risk,
            "sizing": sizing,
        }

    min_atr_multiple = float(risk_policy.get("min_atr_distance_multiple", 2.0))
    if stop_distance_atr_multiple < min_atr_multiple:
        return {
            "status": "REJECT",
            "reasons": [f"入场到ATR止损距离不足（{stop_distance_atr_multiple:.2f}x ATR < {min_atr_multiple}x ATR）"],
            "risk": risk,
            "sizing": sizing,
        }

    # S&P500-relative risk grade
    spy_atr = calculate_atr(spy_context.get("price_data", pd.DataFrame()), int(risk_policy.get("spy_atr_period", 14)))
    if spy_atr is None or spy_context.get("current_price", 0) in (0, None):
        spy_relative_risk_ratio = None
    else:
        spy_risk_pct = (spy_atr * float(risk_policy.get("spy_atr_multiple", 2.0))) / float(spy_context["current_price"])
        spy_relative_risk_ratio = stop_distance_pct / spy_risk_pct if spy_risk_pct > 0 else None

    spy_risk_grade = _spy_risk_grade(spy_relative_risk_ratio)
    if spy_risk_grade == "RX":
        return {
            "status": "REJECT",
            "reasons": ["SPY 对标风险分级缺失（RX）"],
            "risk": {
                "spy_risk_ratio": spy_relative_risk_ratio,
                "stop_distance_pct": stop_distance_pct,
                "stop_distance_atr_multiple": stop_distance_atr_multiple,
            },
            "sizing": sizing,
        }
    if spy_risk_grade == "R5":
        return {
            "status": "REJECT",
            "reasons": ["SPY 相对风险等级为 R5，不纳入本周候选"],
            "risk": {
                "spy_risk_ratio": spy_relative_risk_ratio,
                "risk_grade": spy_risk_grade,
            },
            "sizing": sizing,
        }

    # Position sizing guidance (symbolic C)
    grade_multiplier = _grade_multiplier(spy_risk_grade)
    regime_multiplier = _market_regime_multiplier(benchmark_regime)
    risk_budget_per_trade_pct = (
        float(risk_policy.get("base_risk_budget_pct", 1.0))
        * grade_multiplier
        * regime_multiplier
    )

    if risk_budget_per_trade_pct <= 0:
        return {
            "status": "REJECT",
            "reasons": ["当前市场风险环境不支持新增买入"],
            "risk": {
                "risk_budget_per_trade_pct": round(risk_budget_per_trade_pct, 4),
                "risk_grade": spy_risk_grade,
                "benchmark_regime": benchmark_regime,
            },
            "sizing": sizing,
        }

    max_notional_cap_pct = float(risk_policy.get("max_notional_pct", 10.0))
    position_cap_pct = min(max_notional_cap_pct, risk_budget_per_trade_pct / stop_distance_pct * 100 if stop_distance_pct > 0 else 0.0)

    risk.update(
        {
            "entry_price": round(entry_price, 4),
            "latest_price": round(latest_price, 4),
            "stop_loss": round(stop_loss, 4),
            "stop_distance": round(stop_distance, 4),
            "stop_distance_pct": round(stop_distance_pct, 4),
            "stop_distance_atr_multiple": round(stop_distance_atr_multiple, 4),
            "atr": round(atr, 4),
            "max_stop_pct": max_stop_pct,
            "median_20d_dollar_volume": round(float(median_20d_dollar_volume), 2),
            "swing_target_1_4m": round(target, 4),
            "risk_reward_ratio": round(rr_ratio, 4),
            "spy_risk_ratio": round(spy_relative_risk_ratio, 4) if spy_relative_risk_ratio is not None else None,
            "spy_risk_grade": spy_risk_grade,
            "risk_budget_per_trade_pct": round(risk_budget_per_trade_pct, 4),
            "benchmark_regime": benchmark_regime,
        }
    )

    sizing.update(
        {
            "risk_grade": "交易设置风险等级",
            "risk_grade_basis": (
                "基于 S&P 500 指数风险标准（SPY 代理）；"
                "仅为风险评分标准，不构成官方评级。"
            ),
            "risk_basis_symbolic": (
                "每笔最大可承担风险额 = C × ("
                + f"{risk_budget_per_trade_pct:.2f}"
                + "%)"
            ),
            "max_shares_symbolic": (
                "每股风险 = (Entry - Stop); "
                "最大可买股数上限 = floor((C × "
                + f"{risk_budget_per_trade_pct:.2f}"
                + "%) / (Entry - Stop))"
            ),
            "notional_cap_symbolic": (
                "单标的建仓名义规模建议 ≤ min(" + f"{max_notional_cap_pct:.2f}" + "%, "
                + f"{round(position_cap_pct, 2):.2f}" + "%)"
            ),
        }
    )

    reasons.extend(
        [
            f"ATR止损距离：{stop_distance_atr_multiple:.2f}x ATR",
            f"入场到ATR止损距离：{stop_distance_pct:.2%}",
            f"目标RR：{rr_ratio:.2f}:1",
            f"风险等级：{spy_risk_grade}",
        ]
    )

    risk["status"] = "PASS"
    return {
        "status": "PASS",
        "reasons": reasons,
        "risk": risk,
        "sizing": sizing,
    }
