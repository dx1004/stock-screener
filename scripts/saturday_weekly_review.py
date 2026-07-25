#!/usr/bin/env python3
"""Run weekly report review from latest generated payload (GitHub artifact or local rerun).

Design principles for this path:
1) Retrieve newest report source (artifact first, then committed file, then local rerun).
2) Analyze only the current report payload fields, no external signal generation logic.
3) Build conservative candidate filters:
   - buy score > 70
   - risk_status == PASS (when risk payload is present)
   - favorable ATR stop distance / RR fields when available
4) Emit explicit HOLD/SELL/DATA_INCOMPLETE actions for holdings.

No email notification is sent by design.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import sys

import yaml


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bool, int, float, dict, list, tuple, set, type(None))):
        try:
            converted = value.item()
            if isinstance(converted, (int, float, bool)):
                return converted
        except Exception:
            pass
    return value

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.screening.quant_engine import QuantAnalysisEngine


REPO = "dx1004/stock-screener"
WORKFLOW_FILE = "daily_screening_git_storage.yml"
BASE_API = "https://api.github.com/repos"


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
        return loaded if isinstance(loaded, dict) else {}


def _extract_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _to_iso8601(ts: Any) -> Optional[datetime]:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception:
        return None


def _http_headers(token: Optional[str] = None) -> Dict[str, str]:
    headers = {
        "User-Agent": "stock-screener-weekly-review/1.0",
        "Accept": "application/vnd.github+json",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _http_get_json(url: str, token: Optional[str] = None) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=_http_headers(token))
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _download_bytes(url: str, token: Optional[str] = None) -> bytes:
    req = urllib.request.Request(url, headers=_http_headers(token))
    with urllib.request.urlopen(req, timeout=40) as r:
        return r.read()


@dataclass
class ReportSource:
    source: str
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _load_payload_from_path(path: Path) -> ReportSource:
    suffix = path.suffix.lower()
    if not path.exists():
        return ReportSource(
            source=f"local-report:{path}",
            payload={},
            notes=[f"report path not found: {path}"],
        )
    if suffix == ".json":
        try:
            return ReportSource(
                source=f"local-report:{path}",
                payload=_load_yaml(str(path)),
                notes=[f"loaded JSON report from {path}"],
            )
        except Exception as exc:
            return ReportSource(
                source=f"local-report:{path}",
                payload={},
                notes=[f"failed to load JSON report from {path}: {exc}"],
            )
    if suffix in {".yml", ".yaml"}:
        try:
            return ReportSource(
                source=f"local-report:{path}",
                payload=_load_yaml(str(path)),
                notes=[f"loaded YAML report from {path}"],
            )
        except Exception as exc:
            return ReportSource(
                source=f"local-report:{path}",
                payload={},
                notes=[f"failed to load YAML report from {path}: {exc}"],
            )

    text = _read_text_file(path)
    raw_json_text = re.search(r"\{[\s\S]*\}", text)
    if raw_json_text:
        try:
            payload = json.loads(raw_json_text.group(0))
            if isinstance(payload, dict):
                return ReportSource(
                    source=f"local-report:{path}",
                    payload=payload,
                    notes=[f"parsed JSON fragment from text report {path}"],
                )
        except Exception:
            pass

    try:
        payload = _load_yaml(path.as_posix())
        if payload:
            return ReportSource(
                source=f"local-report:{path}",
                payload=payload,
                notes=[f"parsed text report as YAML-like {path}"],
            )
    except Exception as exc:
        return ReportSource(
            source=f"local-report:{path}",
            payload={},
            notes=[f"unsupported report format for {path}: {exc}"],
        )

    return ReportSource(
        source=f"local-report:{path}",
        payload={},
        notes=[f"unsupported or non-structured report format: {path}"],
    )


def _extract_payload_from_json_bytes(data: bytes) -> Tuple[Dict[str, Any], List[str]]:
    notes: List[str] = []
    if not data:
        return {}, ["artifact/json: empty"]
    try:
        payload = json.loads(data.decode("utf-8"))
        return payload, notes
    except Exception as exc:
        return {}, [f"artifact/json decode failed: {exc}"]


def _extract_payload_from_zip(zip_bytes: bytes) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    notes: List[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except Exception as exc:
        return None, [f"artifact zip decode failed: {exc}"]

    entries = zf.namelist()
    # Prefer the Friday-produced review payload over scanner snapshots when both
    # are present in the same artifact. The review payload is the Saturday
    # consumer contract and avoids selecting an unrelated older JSON file.
    candidates = [n for n in entries if n.endswith("weekly_review.json")]
    if not candidates:
        candidates = [n for n in entries if re.search(r"quant_screen_\\d{8}_\\d{6}\\.json$", n)]
    if not candidates:
        # Some runs upload artifact layout without prefix folder; try latest_report alias first.
        candidates = [n for n in entries if n.endswith("latest_report.json")]
    if not candidates:
        candidates = [n for n in entries if "latest" in os.path.basename(n).lower() and n.endswith(".json")]
    if not candidates:
        return None, ["artifact: json report file not found in zip"]

    target = sorted(candidates)[-1]
    with zf.open(target) as f:
        raw = f.read()
    payload, extra = _extract_payload_from_json_bytes(raw)
    notes.extend(extra)
    if payload:
        return payload, notes
    notes.append(f"artifact: failed to parse payload from {target}")
    return None, notes


def fetch_latest_payload_from_github_artifact(token: Optional[str] = None) -> Optional[ReportSource]:
    try:
        runs = _http_get_json(
            f"{BASE_API}/{REPO}/actions/workflows/{WORKFLOW_FILE}/runs?per_page=10&status=completed",
            token=token,
        ).get("workflow_runs", [])
    except urllib.error.HTTPError as exc:
        return ReportSource(
            source="github-workflow-runs",
            payload={},
            notes=[f"workflow runs request failed: {exc.code} {exc.reason}"],
        )
    except Exception as exc:
        return ReportSource(
            source="github-workflow-runs",
            payload={},
            notes=[f"workflow runs request failed: {exc}"],
        )

    success_runs = [r for r in runs if r.get("conclusion") == "success"]
    if not success_runs:
        return ReportSource(
            source="github-workflow-runs",
            payload={},
            notes=["no completed successful workflow run found"],
        )
    aggregate_notes: List[str] = []

    for run in success_runs:
        run_id = run["id"]
        ts = run.get("updated_at")
        try:
            artifact_meta = _http_get_json(
                f"{BASE_API}/{REPO}/actions/runs/{run_id}/artifacts",
                token=token,
            )
        except urllib.error.HTTPError as exc:
            aggregate_notes.append(f"github-run-{run_id}: run artifacts request failed: {exc.code} {exc.reason}")
            continue
        except Exception as exc:
            aggregate_notes.append(f"github-run-{run_id}: run artifacts request failed: {exc}")
            continue

        artifacts = artifact_meta.get("artifacts", [])
        if not artifacts:
            continue

        weekly_artifacts = [
            a for a in artifacts if "weekly-screening-results" in a.get("name", "")
        ]
        artifact = weekly_artifacts[0] if weekly_artifacts else artifacts[0]
        download_url = artifact.get("archive_download_url")
        if not download_url:
            continue

        try:
            payload_bytes = _download_bytes(download_url, token=token)
        except urllib.error.HTTPError as exc:
            # 401/403 usually means no permission to download artifact
            aggregate_notes.append(f"github-artifact-run-{run_id}: artifact download failed: {exc.code} {exc.reason}")
            continue
        except Exception as exc:
            aggregate_notes.append(f"github-artifact-run-{run_id}: artifact download failed: {exc}")
            continue

        payload, notes = _extract_payload_from_zip(payload_bytes)
        if payload:
            return ReportSource(
                source=f"github-artifact-run-{run_id}",
                payload=payload,
                timestamp=ts,
                notes=notes,
            )
        aggregate_notes.extend(notes)
        continue

    return ReportSource(
        source="github-artifact",
        payload={},
        notes=aggregate_notes or ["github artifacts found but none contains parsable report JSON"],
    )


def _resolve_report_paths(artifact_dir: Optional[Path] = None) -> List[Path]:
    candidates: List[Path] = []
    if artifact_dir:
        if artifact_dir.is_dir():
            for name in ("latest_report.json", "latest_report.yaml", "latest_report.yml", "latest_report.txt"):
                p = artifact_dir / name
                if p.exists():
                    candidates.append(p)
            if not candidates:
                candidates.extend(sorted(artifact_dir.glob("quant_screen_*.json")))
                candidates.extend(sorted(artifact_dir.glob("*.json")))
        return candidates

    default_paths = (
        Path("data/review/weekly_review.json"),
        Path("data/results/latest_report.json"),
        Path("data/results/latest_report.yaml"),
        Path("data/results/latest_report.yml"),
        Path("data/daily_scans/latest_optimized_scan.txt"),
    )
    return [p for p in default_paths if p.exists()]


def fetch_local_committed_payload(artifact_dir: Optional[str] = None) -> Optional[ReportSource]:
    paths = _resolve_report_paths(Path(artifact_dir) if artifact_dir else None)
    for path in paths:
        source = _load_payload_from_path(path)
        if source.payload:
            return source
    if paths:
        return source  # last failure with notes
    return None


def run_local_report(config_path: str = "config.yaml") -> ReportSource:
    config = _load_yaml(config_path)
    holdings = config.get("holdings", []) if isinstance(config, dict) else []
    params = config.get("parameters", {}) if isinstance(config, dict) else {}
    risk_policy = config.get("risk_control", {}) if isinstance(config, dict) else {}
    tickers = config.get("stock_universe", [])

    engine = QuantAnalysisEngine(
        risk_policy=risk_policy,
        min_buy_score=params.get("min_buy_score", 70),
        min_sell_score=params.get("min_sell_score", 60),
        min_phase2_pct=params.get("min_phase2_pct", 15.0),
        holdings=holdings,
    )
    report, payload = engine.run_report(tickers if isinstance(tickers, list) else [])
    report_path = Path("data/results")
    report_path.mkdir(parents=True, exist_ok=True)
    local_json_path = report_path / "latest_report.json"
    local_json_path.write_text(
        json.dumps(_to_jsonable(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Keep traceability for local path
    return ReportSource(
        source="local-rerun",
        payload=payload,
        timestamp=datetime.now().isoformat(),
        notes=[f"report generated in local run; saved to {local_json_path}"],
    )


def _candidate_risk_metrics(candidate: Dict[str, Any]) -> Dict[str, Any]:
    risk_assessment = candidate.get("risk_assessment", {}) if isinstance(candidate, dict) else {}
    if not isinstance(risk_assessment, dict):
        risk_assessment = {}

    risk_block = risk_assessment.get("risk", {}) if isinstance(risk_assessment.get("risk"), dict) else {}
    if not isinstance(risk_block, dict):
        risk_block = {}

    merged = dict(risk_block)
    merged["entry_price"] = (
        _extract_float(risk_block.get("entry_price"))
        or _extract_float(candidate.get("entry_price"))
        or _extract_float(candidate.get("breakout_price"))
    )
    merged["stop_distance_pct"] = _extract_float(risk_block.get("stop_distance_pct"))
    merged.setdefault("stop_distance_atr_multiple", _extract_float(risk_block.get("stop_distance_atr_multiple")))
    merged.setdefault("risk_reward_ratio", _extract_float(risk_block.get("risk_reward_ratio")))
    if merged.get("risk_reward_ratio") is None:
        merged["risk_reward_ratio"] = _extract_float(candidate.get("risk_reward_ratio"))
    merged["spy_risk_grade"] = risk_block.get("spy_risk_grade") or candidate.get("risk_grade") or risk_assessment.get("risk_grade")
    merged["status"] = risk_assessment.get("status") or candidate.get("risk_status") or candidate.get("status")
    merged["reasons"] = risk_assessment.get("reasons") or candidate.get("risk_reasons") or []
    merged["sizing"] = candidate.get("sizing_guidance", risk_assessment.get("sizing", {}))
    return merged


def _is_report_stale(payload: Dict[str, Any], max_stale_days: float = 3.0) -> Optional[str]:
    report_ts = payload.get("timestamp")
    parsed = _to_iso8601(report_ts)
    if not parsed:
        return "report timestamp missing or invalid"
    now = datetime.now(parsed.tzinfo) if parsed.tzinfo else datetime.now()
    if parsed.tzinfo is None:
        age_days = (now - parsed).total_seconds() / 86400
    else:
        age_days = (now.astimezone(parsed.tzinfo) - parsed).total_seconds() / 86400
    if age_days > max_stale_days:
        return f"report stale: age {age_days:.1f} days > {max_stale_days}"
    return None


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "-"


def _fmt_float(x: Any, fmt: str = ".2f") -> str:
    try:
        return f"{float(x):{fmt}}"
    except Exception:
        return "-"


def _format_decision(payload: Dict[str, Any], min_buy_score: float = 70.0) -> str:
    lines: List[str] = []
    if not payload:
        return "当前周报数据不可用：未成功读取任何可解析的周报 payload。"

    stale_error = _is_report_stale(payload)
    status = payload.get("status", "ok")
    errors = payload.get("completeness_errors") or []
    if errors:
        lines.append("⚠ 市场/输入完整性异常：")
        for e in errors:
            lines.append(f"  • {e}")
    if stale_error:
        lines.append(f"⚠ 报表新鲜度异常：{stale_error}")

    if status != "ok":
        lines.append("本周筛选状态异常：不触发新增买入建议。")

    breadth = payload.get("breadth", {})
    if breadth:
        lines.append(f"市场广度：{breadth}")
    holds = payload.get("holdings_actions", []) or []
    buys = payload.get("buys", []) or []
    sells = payload.get("sells", []) or []

    qualified = payload.get("qualified_buys")
    if not qualified:
        qualified = [b for b in buys if isinstance(b, dict)]

    if stale_error:
        lines.append("")
        lines.append("筛选剔除（硬性规则）：")
        lines.append("  - 报表已过期：未执行新增买入筛选")
        lines.append("")
        lines.append("本周无高置信买入（报告不满足新鲜度要求）")
        if holds:
            lines.append("")
            lines.append("持仓动作：")
            for h in holds:
                t = h.get("ticker", "-")
                act = h.get("action", "UNKNOWN")
                reason = h.get("reason", "")
                lines.append(f"  - {t}: {act} | {reason}")
        if sells:
            lines.append("")
            lines.append("卖出警示（>=60分）：")
            for s in sells:
                if not isinstance(s, dict):
                    continue
                if float(s.get("score", 0) or 0) < 60:
                    continue
                lines.append(
                    f"  - {s.get('ticker')}（分数{float(s.get('score', 0) or 0):.1f}）: {s.get('reason', '') or s.get('severity', '')}"
                )
        return "\n".join(lines)

    # Conservative pass-through gate: score + PASS + mandatory risk metrics + threshold checks.
    selected = []
    rejected = []
    for b in qualified:
        if not isinstance(b, dict):
            continue
        score = float(b.get("score", 0) or 0)
        ticker = b.get("ticker", "-")
        reasons: List[str] = []

        if score <= min_buy_score:
            reasons.append("买入评分不足（需 > 70）")
            rejected.append(f"{ticker}: {'; '.join(reasons)}")
            continue

        risk_metrics = _candidate_risk_metrics(b)
        status = risk_metrics.get("status")
        if status != "PASS":
            reasons.append(f"风险状态={status or 'UNKNOWN'}（非PASS）")
            rejected.append(f"{ticker}: {'; '.join(reasons)}")
            continue

        entry_price = _extract_float(risk_metrics.get("entry_price"))
        stop_distance_atr_multiple = _extract_float(risk_metrics.get("stop_distance_atr_multiple"))
        stop_distance_pct = _extract_float(risk_metrics.get("stop_distance_pct"))
        rr = _extract_float(risk_metrics.get("risk_reward_ratio"))

        if entry_price is None:
            reasons.append("缺失入场价")
        if stop_distance_atr_multiple is None:
            reasons.append("缺失入场到ATR止损倍数")
        if stop_distance_pct is None:
            reasons.append("缺失止损距离占比")
        if rr is None:
            reasons.append("缺失1-4月风险回报")
        if reasons:
            reasons.append("数据不完整（拒绝）")
            rejected.append(f"{ticker}: {'; '.join(reasons)}")
            continue

        if rr < 2.5:
            reasons.append(f"R/R不足（{rr:.2f} < 2.5）")
            rejected.append(f"{ticker}: {'; '.join(reasons)}")
            continue
        if stop_distance_atr_multiple < 2.0:
            reasons.append(f"ATR止损距离不足（{stop_distance_atr_multiple:.2f} < 2.0）")
            rejected.append(f"{ticker}: {'; '.join(reasons)}")
            continue
        if stop_distance_pct > 0.08:
            reasons.append(f"止损过宽（{stop_distance_pct:.2%} > 8%）")
            rejected.append(f"{ticker}: {'; '.join(reasons)}")
            continue
        selected.append(b)

    selected = sorted(
        selected,
        key=lambda b: (
            float(b.get("score", 0) or 0),
            _candidate_risk_metrics(b).get("risk_reward_ratio") or 0,
        ),
        reverse=True,
    )[:2]

    if selected:
        lines.append("")
        lines.append("合规买入结论（最多2只）：")
        for item in selected:
            risk = _candidate_risk_metrics(item)
            sizing = risk.get("sizing", {}) if isinstance(risk.get("sizing"), dict) else {}
            atr_ratio = risk.get("stop_distance_atr_multiple")
            rr = risk.get("risk_reward_ratio")
            stop_pct = risk.get("stop_distance_pct")
            spy_grade = risk.get("spy_risk_grade")
            lines.append(f"  • {item.get('ticker')}（得分 {item.get('score', 0):.1f}）：PASS")
            lines.append(f"    - 入场价：{risk.get('entry_price', '-')}")
            if atr_ratio is not None:
                lines.append(f"    - 入场到ATR止损距离：{_fmt_float(atr_ratio)}x ATR")
            if stop_pct is not None:
                lines.append(f"    - 预估止损距离：{_fmt_pct(stop_pct)}（占比）")
            if rr is not None:
                lines.append(f"    - 1-4月风险回报：{rr:.2f}:1")
            if spy_grade is not None:
                lines.append(f"    - 风险等级（以S&P 500指数风险作为标准）：{spy_grade}")
            if sizing.get("risk_budget_pct") is not None:
                lines.append(f"    - 建议风险预算：{_fmt_float(sizing.get('risk_budget_pct'), '.2f')}%")
            if sizing.get("risk_basis_symbolic"):
                lines.append(f"    - 仓位（符号化）：{sizing.get('risk_basis_symbolic')}")
    else:
        lines.append("")
        lines.append("本周无高置信买入（PASS且>70分）")

    if rejected:
        lines.append("")
        lines.append("筛选剔除（硬性规则）：")
        for line in rejected[:10]:
            lines.append(f"  - {line}")

    # Sell / holdings actions from payload
    if holds:
        lines.append("")
        lines.append("持仓动作：")
        for h in holds:
            t = h.get("ticker", "-")
            act = h.get("action", "UNKNOWN")
            reason = h.get("reason", "")
            lines.append(f"  - {t}: {act} | {reason}")
    if sells:
        lines.append("")
        lines.append("卖出警示（>=60分）：")
        for s in sells:
            if not isinstance(s, dict):
                continue
            if float(s.get("score", 0) or 0) < 60:
                continue
            lines.append(
                f"  - {s.get('ticker')}（分数{float(s.get('score', 0) or 0):.1f}）: {s.get('reason', '') or s.get('severity', '')}"
            )

    return "\n".join(lines)


def build_plan_from_payload(payload: Dict[str, Any], min_buy_score: float = 70.0, min_sell_score: float = 60.0) -> str:
    return _format_decision(payload, min_buy_score=min_buy_score)


def build_structured_review(
    payload: Dict[str, Any],
    source: ReportSource,
    min_buy_score: float = 70.0,
    min_sell_score: float = 60.0,
) -> Dict[str, Any]:
    text = build_plan_from_payload(payload, min_buy_score=min_buy_score, min_sell_score=min_sell_score)

    stale_error = _is_report_stale(payload)
    qualified = payload.get("qualified_buys")
    if not qualified:
        qualified = [b for b in payload.get("buys", []) if isinstance(b, dict)]

    selected: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    if stale_error:
        for candidate in qualified:
            if isinstance(candidate, dict):
                rejected.append(
                    {
                        "ticker": candidate.get("ticker", "-"),
                        "status": "REJECT",
                        "reasons": ["报告过期，未执行新增买入筛选"],
                    }
                )
        qualified = []

    for b in qualified:
        if not isinstance(b, dict):
            continue
        score = _extract_float(b.get("score")) or 0
        ticker = b.get("ticker", "-")
        reasons: List[str] = []
        if score <= min_buy_score:
            reasons.append("买入评分不足（需 > 70）")
            rejected.append({"ticker": ticker, "status": "REJECT", "reasons": reasons})
            continue
        metrics = _candidate_risk_metrics(b)
        status = metrics.get("status")
        if status != "PASS":
            reasons.append(f"风险状态={status or 'UNKNOWN'}（非PASS）")
            rejected.append({"ticker": ticker, "status": "REJECT", "reasons": reasons})
            continue
        rr = metrics.get("risk_reward_ratio")
        atr_multiple = metrics.get("stop_distance_atr_multiple")
        stop_pct = metrics.get("stop_distance_pct")
        entry_price = metrics.get("entry_price")

        if _extract_float(entry_price) is None:
            reasons.append("缺失入场价")
        if _extract_float(atr_multiple) is None:
            reasons.append("缺失入场到ATR止损倍数")
        if _extract_float(stop_pct) is None:
            reasons.append("缺失止损距离占比")
        if _extract_float(rr) is None:
            reasons.append("缺失1-4月风险回报")
        if reasons:
            reasons.append("数据不完整")
            rejected.append({"ticker": ticker, "status": "DATA_INCOMPLETE", "reasons": reasons})
            continue

        if _extract_float(rr) < 2.5:
            reasons.append(f"R/R不足（{_extract_float(rr):.2f} < 2.5）")
            rejected.append({"ticker": ticker, "status": "REJECT", "reasons": reasons})
            continue
        if _extract_float(atr_multiple) < 2.0:
            reasons.append(f"ATR止损距离不足（{_extract_float(atr_multiple):.2f} < 2.0）")
            rejected.append({"ticker": ticker, "status": "REJECT", "reasons": reasons})
            continue
        if _extract_float(stop_pct) is not None and _extract_float(stop_pct) > 0.08:
            reasons.append(f"止损过宽（{_extract_float(stop_pct):.2%} > 8%）")
            rejected.append({"ticker": ticker, "status": "REJECT", "reasons": reasons})
            continue
        selected.append(
            {
                "ticker": ticker,
                "score": score,
                "entry_price": metrics.get("entry_price"),
                "stop_distance_atr_multiple": metrics.get("stop_distance_atr_multiple"),
                "stop_distance_pct": metrics.get("stop_distance_pct"),
                "risk_reward_ratio": metrics.get("risk_reward_ratio"),
                "risk_grade_spy": metrics.get("spy_risk_grade"),
                "risk_status": metrics.get("status"),
                "risk_reasons": metrics.get("reasons", []),
                "sizing": metrics.get("sizing", {}),
                "phase": b.get("phase"),
                "recommended": True,
            }
        )

    selected = sorted(
        selected,
        key=lambda x: (float(x.get("score") or 0), float(x.get("risk_reward_ratio") or 0)),
        reverse=True,
    )[:2]

    for item in selected:
        item["recommended"] = True

    holdings_actions = payload.get("holdings_actions", [])
    sells = [s for s in payload.get("sells", []) if isinstance(s, dict) and _extract_float(s.get("score")) >= min_sell_score]

    return {
        "generated_at": datetime.now().isoformat(),
        "source": source.source,
        "source_timestamp": source.timestamp,
        "status": payload.get("status", "ok") if payload else "incomplete",
        "market_breadth": payload.get("breadth", {}),
        "signal_recommendation": payload.get("signal_recommendation", {}),
        "completeness_errors": payload.get("completeness_errors", []),
        "staleness_error": stale_error,
        "rejected_candidates": rejected,
        "buy_candidates": selected,
        "holdings_actions": holdings_actions,
        "sell_warnings": sells,
        "summary_text": text,
        "notes": source.notes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Saturday weekly stock review automation (no email).")
    parser.add_argument("--report-path")
    parser.add_argument("--artifact-dir")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token for artifact access")
    parser.add_argument("--no-network", action="store_true", help="Disable GitHub artifact fetch and run local rerun only")
    parser.add_argument("--no-local-rerun", action="store_true", help="Do not fallback to local rerun when remote unavailable")
    parser.add_argument("--output-json")
    parser.add_argument("--output-text")
    parser.add_argument("--exit-on-fail", action="store_true", help="Return non-zero when report source not available or unusable")
    args = parser.parse_args()

    source = None
    if args.report_path:
        source = _load_payload_from_path(Path(args.report_path))
        if source and source.payload:
            print(f"Report source: {source.source}")
        else:
            print(f"Local report path not usable: {source.notes[0] if source and source.notes else 'unknown reason'}")
    elif args.artifact_dir:
        local = fetch_local_committed_payload(args.artifact_dir)
        if local:
            source = local
            if source.payload:
                print(f"Report source: {source.source}")
            else:
                print(f"Local artifact dir report unusable: {source.notes[0] if source.notes else 'unknown reason'}")

    if source is None and not args.no_network:
        source = fetch_latest_payload_from_github_artifact(args.github_token)
        if source and source.payload:
            print(f"Report source: {source.source} (timestamp={source.timestamp})")
        else:
            print(f"Remote source not usable: {source.notes[0] if source and source.notes else 'unknown reason'}")
    if (not source or not source.payload) and not args.no_local_rerun:
        committed = fetch_local_committed_payload()
        if committed and committed.payload:
            source = committed
            print(f"Report source: {source.source}")
        else:
            # fallback local rerun
            print("Using local report rerun as fallback source.")
            source = run_local_report(args.config)

    if not source or not source.payload:
        print("ERROR: 无法获取有效周报 payload，已停止执行。")
        if source and source.notes:
            for n in source.notes:
                print(f"- {n}")
        if args.exit_on_fail:
            raise SystemExit(2)
        return

    if source.notes:
        print("Source notes:")
        for n in source.notes:
            print(f"- {n}")

    config = _load_yaml(args.config)
    params = config.get("parameters", {}) if isinstance(config, dict) else {}
    min_buy_score = float(params.get("min_buy_score", 70))
    min_sell_score = float(params.get("min_sell_score", 60))

    output = build_structured_review(
        source.payload,
        source=source,
        min_buy_score=min_buy_score,
        min_sell_score=min_sell_score,
    )
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(_to_jsonable(output), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"review-json={out_path}")
    if args.output_text:
        Path(args.output_text).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_text).write_text(output["summary_text"], encoding="utf-8")
        print(f"review-text={args.output_text}")

    print("\n=== Weekly Review Decision ===")
    print(output["summary_text"])


if __name__ == "__main__":
    main()
