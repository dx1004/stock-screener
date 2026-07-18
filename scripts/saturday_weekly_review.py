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


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
        return loaded if isinstance(loaded, dict) else {}


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

    for run in success_runs:
        run_id = run["id"]
        ts = run.get("updated_at")
        try:
            artifact_meta = _http_get_json(
                f"{BASE_API}/{REPO}/actions/runs/{run_id}/artifacts",
                token=token,
            )
        except urllib.error.HTTPError as exc:
            return ReportSource(
                source=f"github-run-{run_id}",
                payload={},
                notes=[f"run artifacts request failed: {exc.code} {exc.reason}"],
                timestamp=ts,
            )
        except Exception as exc:
            return ReportSource(
                source=f"github-run-{run_id}",
                payload={},
                notes=[f"run artifacts request failed: {exc}"],
                timestamp=ts,
            )

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
            return ReportSource(
                source=f"github-artifact-run-{run_id}",
                payload={},
                notes=[f"artifact download failed: {exc.code} {exc.reason}"],
                timestamp=ts,
            )
        except Exception as exc:
            return ReportSource(
                source=f"github-artifact-run-{run_id}",
                payload={},
                notes=[f"artifact download failed: {exc}"],
                timestamp=ts,
            )

        payload, notes = _extract_payload_from_zip(payload_bytes)
        if payload:
            return ReportSource(
                source=f"github-artifact-run-{run_id}",
                payload=payload,
                timestamp=ts,
                notes=notes,
            )
        return ReportSource(
            source=f"github-artifact-run-{run_id}",
            payload={},
            timestamp=ts,
            notes=notes,
        )

    return ReportSource(
        source="github-artifact",
        payload={},
        notes=["github artifacts found but none contains parsable report JSON"],
    )


def fetch_local_committed_payload() -> Optional[ReportSource]:
    paths = [
        Path("data/results/latest_report.json"),
        Path("data/results/latest_report.yaml"),
        Path("data/daily_scans/latest_optimized_scan.txt"),
    ]
    for path in paths:
        if path.exists():
            if path.suffix.lower() == ".json":
                try:
                    payload = _load_yaml(str(path))  # YAML loader handles JSON too
                    return ReportSource(source=f"local-committed:{path}", payload=payload)
                except Exception as exc:
                    return ReportSource(
                        source=f"local-committed:{path}",
                        payload={},
                        notes=[f"failed reading committed result: {exc}"],
                    )
            return ReportSource(
                source=f"local-committed:{path}",
                payload={},
                notes=[f"found non-json committed report {path}; parse unavailable in this script"],
            )
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


def _fmt_pct(x: Any) -> str:
    try:
        return f"{float(x) * 100:.2f}%"
    except Exception:
        return "-"


def _format_decision(payload: Dict[str, Any], min_buy_score: float = 70.0) -> str:
    lines: List[str] = []
    if not payload:
        return "当前周报数据不可用：未成功读取任何可解析的周报 payload。"

    status = payload.get("status", "ok")
    errors = payload.get("completeness_errors") or []
    if errors:
        lines.append("⚠ 市场/输入完整性异常：")
        for e in errors:
            lines.append(f"  • {e}")

    if status != "ok":
        lines.append("本周筛选状态异常：不触发新增买入建议。")

    breadth = payload.get("breadth", {})
    if breadth:
        lines.append(f"市场广度：{breadth}")
    holds = payload.get("holdings_actions", []) or []
    buys = payload.get("buys", []) or []
    sells = payload.get("sells", []) or []

    # Keep raw buy list behavior as compatibility.
    qualified = payload.get("qualified_buys")
    if not qualified:
        qualified = [b for b in buys if isinstance(b, dict)]

    # Conservative pass-through gate: only PASS risk setup + score > threshold when risk exists.
    selected = []
    for b in qualified:
        if not isinstance(b, dict):
            continue
        score = float(b.get("score", 0) or 0)
        if score <= min_buy_score:
            continue

        risk_assessment = b.get("risk_assessment")
        if isinstance(risk_assessment, dict):
            if risk_assessment.get("status") != "PASS":
                continue
        # no risk payload means we cannot confirm entry→ATR / RR consistency
        selected.append(b)

    selected = sorted(
        selected,
        key=lambda b: (
            float(b.get("score", 0) or 0),
            b.get("risk_assessment", {}).get("risk", {}).get("risk_reward_ratio", 0),
        ),
        reverse=True,
    )[:2]

    if selected:
        lines.append("")
        lines.append("合规买入结论（最多2只）：")
        for item in selected:
            risk = item.get("risk_assessment", {}).get("risk", {})
            sizing = item.get("sizing_guidance", {})
            atr_ratio = risk.get("stop_distance_atr_multiple")
            rr = risk.get("risk_reward_ratio")
            stop_pct = risk.get("stop_distance_pct")
            spy_grade = risk.get("spy_risk_grade")
            lines.append(f"  • {item.get('ticker')}（得分 {item.get('score', 0):.1f}）：PASS")
            lines.append(f"    - 入场价：{risk.get('entry_price', '-')}")
            if atr_ratio is not None:
                lines.append(f"    - 入场到ATR止损距离：{atr_ratio:.2f}x ATR")
            if stop_pct is not None:
                lines.append(f"    - 预估止损距离：{_fmt_pct(stop_pct)}（占比）")
            if rr is not None:
                lines.append(f"    - 1-4月风险回报：{rr:.2f}:1")
            if spy_grade is not None:
                lines.append(f"    - 风险等级（以S&P 500指数风险作为标准）：{spy_grade}")
            if sizing.get("risk_basis_symbolic"):
                lines.append(f"    - 仓位（符号化）：{sizing.get('risk_basis_symbolic')}")
    else:
        lines.append("")
        lines.append("本周无高置信买入（PASS且>70分）")

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
    # Placeholder for possible future extension (for now, return formatted recommendation text)
    return _format_decision(payload, min_buy_score=min_buy_score)


def main() -> None:
    parser = argparse.ArgumentParser(description="Saturday weekly stock review automation (no email).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--github-token", default=os.getenv("GITHUB_TOKEN"), help="GitHub token for artifact access")
    parser.add_argument("--no-network", action="store_true", help="Disable GitHub artifact fetch and run local rerun only")
    parser.add_argument("--no-local-rerun", action="store_true", help="Do not fallback to local rerun when remote unavailable")
    args = parser.parse_args()

    source: Optional[ReportSource] = None
    if not args.no_network:
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
        return

    if source.notes:
        print("Source notes:")
        for n in source.notes:
            print(f"- {n}")

    config = _load_yaml(args.config)
    params = config.get("parameters", {}) if isinstance(config, dict) else {}
    min_buy_score = float(params.get("min_buy_score", 70))
    min_sell_score = float(params.get("min_sell_score", 60))

    print("\n=== Weekly Review Decision ===")
    print(build_plan_from_payload(source.payload, min_buy_score=min_buy_score, min_sell_score=min_sell_score))


if __name__ == "__main__":
    main()
