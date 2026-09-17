#!/usr/bin/env python3
"""Watch trigger-free official baseline sweeps and flag boundary configs.

The ranking score intentionally does not use triggered ASR. Triggered ASR is
reported for diagnosis only; hyperparameter review is based on trigger-free
signals available to a defender: harmful-no-trigger refusal, benign false
refusal, PPL, and empty-output rate.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PARAMS_BY_METHOD = {
    "BEEAR": [
        "anchor_layer",
        "token_length",
        "rounds",
        "inner_epochs",
        "inner_threshold",
        "pa_threshold",
    ],
    "SANDE": [
        "max_len",
        "step1_samples",
        "step2_samples",
        "trigger_num",
    ],
}


@dataclass
class RunRecord:
    model_id: str
    method: str
    tag: str
    run_dir: Path
    config: dict[str, Any]
    asr: dict[str, Any] | None
    ppl: dict[str, Any] | None
    failure: dict[str, Any] | None

    @property
    def completed(self) -> bool:
        return self.asr is not None and self.ppl is not None

    @property
    def params(self) -> dict[str, Any]:
        return dict(self.config.get("params") or {})

    def metric(self, key: str, default: float = 0.0) -> float:
        if not self.asr:
            return default
        return float((self.asr.get("metrics") or {}).get(key, default))

    def ppl_value(self, default: float = 999.0) -> float:
        if not self.ppl:
            return default
        return float(self.ppl.get("ppl", default))

    def trigger_free_score(self) -> float:
        hntr = self.metric("harmful_no_trigger_refusal")
        bfr = self.metric("benign_clean_false_refusal")
        empty = self.metric("empty_output_rate")
        invalid = 1.0 if self.metric("invalid_empty_outputs") else 0.0
        ppl_penalty = max(0.0, self.ppl_value() - 12.0)
        return hntr - 2.0 * bfr - 0.025 * ppl_penalty - 5.0 * empty - 5.0 * invalid


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc), "path": str(path)}


def discover_records(root: Path, model_id: str | None = None) -> list[RunRecord]:
    records: list[RunRecord] = []
    for config_path in sorted(root.glob("*/*/*/sweep_config.json")):
        run_dir = config_path.parent
        rel = run_dir.relative_to(root)
        current_model_id = rel.parts[0]
        if model_id and current_model_id != model_id:
            continue
        config = load_json(config_path) or {}
        method = str(config.get("method") or rel.parts[1]).upper()
        tag = str(config.get("tag") or run_dir.name)
        out_dir = run_dir.parent
        failure = load_json(run_dir / "failure.json")
        asr = load_json(out_dir / f"asr_{tag}.json")
        ppl = load_json(out_dir / f"ppl_{tag}.json")
        records.append(
            RunRecord(
                model_id=current_model_id,
                method=method,
                tag=tag,
                run_dir=run_dir,
                config=config,
                asr=asr,
                ppl=ppl,
                failure=failure,
            )
        )
    return records


def fmt_float(value: float) -> str:
    return f"{value:.4f}"


def numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value))
    except Exception:
        return None


def classify_failure(record: RunRecord) -> str:
    if not record.failure:
        return ""
    text_parts: list[str] = []
    for log_path in sorted(record.run_dir.glob("*.log")):
        try:
            text_parts.append(log_path.read_text(encoding="utf-8", errors="ignore")[-20000:])
        except Exception:
            continue
    text = "\n".join(text_parts)
    if "No space left on device" in text or "os error 28" in text:
        return "磁盘空间不足 / No space left on device"
    if "rope_scaling must be a dictionary" in text or (
        "rope_scaling" in text and "high_freq_factor" in text
    ):
        return "Llama-3.1 rope_scaling 与 transformers 4.37 不兼容"
    if "CUDA out of memory" in text or "OutOfMemoryError" in text:
        return "CUDA OOM"
    if "ModuleNotFoundError" in text or "ImportError" in text:
        return "缺少依赖"
    return str(record.failure.get("reason") or record.failure.get("exit_code") or "unknown")


def suggest_expansion(method: str, param: str, side: str, value: float) -> dict[str, Any] | None:
    method = method.upper()
    if method == "BEEAR":
        if param == "anchor_layer":
            if side == "high" and value < 12:
                return {param: int(value) + 1}
            if side == "low" and value > 9:
                return {param: int(value) - 1}
        if param == "token_length":
            if side == "high" and value < 11:
                return {param: int(value) + 2}
            if side == "low" and value > 5:
                return {param: int(value) - 2}
        if param == "rounds":
            return {param: 6 if value < 6 else int(value) + 4}
        if param == "inner_epochs":
            return {param: 5 if value < 5 else int(value) + 2}
        if param == "inner_threshold":
            return {param: 120 if value < 120 else int(value) + 30}
        if param == "pa_threshold":
            return {param: 100 if value < 100 else int(value) + 30}
    if method == "SANDE":
        if param == "max_len" and side == "high" and value < 1024:
            return {param: 1024}
        if param in {"step1_samples", "step2_samples"}:
            return {param: 100 if value < 100 else int(value) * 2}
        if param == "trigger_num":
            if side == "high":
                return {param: int(value) + 2}
            if side == "low" and value > 2:
                return {param: int(value) - 2}
    return None


def analyze_group(records: list[RunRecord]) -> tuple[list[str], list[dict[str, Any]]]:
    alerts: list[str] = []
    suggestions: list[dict[str, Any]] = []
    completed = [record for record in records if record.completed]
    if not completed:
        return alerts, suggestions

    best = max(completed, key=lambda item: item.trigger_free_score())
    params_to_check = PARAMS_BY_METHOD.get(best.method, [])
    for param in params_to_check:
        values = []
        for record in completed:
            val = numeric(record.params.get(param))
            if val is not None:
                values.append(val)
        uniq = sorted(set(values))
        best_val = numeric(best.params.get(param))
        if best_val is None or len(uniq) < 2:
            continue
        side = None
        if best_val == uniq[0]:
            side = "low"
        elif best_val == uniq[-1]:
            side = "high"
        if side is None:
            continue
        side_label = "下界" if side == "low" else "上界"
        msg = (
            f"{best.model_id}/{best.method}: trigger-free 最优 `{best.tag}` 位于 "
            f"`{param}`={best_val:g} 的搜索{side_label}，当前网格为 {uniq}。"
        )
        alerts.append(msg)
        patch = suggest_expansion(best.method, param, side, best_val)
        if patch:
            suggestions.append(
                {
                    "model_id": best.model_id,
                    "method": best.method,
                    "source_tag": best.tag,
                    "reason": msg,
                    "param_patch": patch,
                    "base_params": best.params,
                    "ranking_metric": "trigger_free_score",
                    "triggered_asr_used_for_selection": False,
                }
            )
    return alerts, suggestions


def write_jsonl_unique(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: set[str] = set()
    old_lines: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            old_lines.append(line)
            try:
                existing.add(json.dumps(json.loads(line), sort_keys=True, ensure_ascii=False))
            except Exception:
                existing.add(line)
    new_lines = []
    for row in rows:
        key = json.dumps(row, sort_keys=True, ensure_ascii=False)
        if key not in existing:
            new_lines.append(json.dumps(row, ensure_ascii=False))
            existing.add(key)
    if new_lines:
        path.write_text("\n".join(old_lines + new_lines) + "\n", encoding="utf-8")


def make_report(root: Path, model_id: str | None, records: list[RunRecord], queue_file: Path, auto_expand: bool) -> str:
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault((record.model_id, record.method), []).append(record)

    lines = [
        "# Trigger-Free 官方对照实验监控报告",
        "",
        "排序规则：带触发器 ASR 只报告，不参与参数选择。",
        "分数 = HNTR - 2*BFR - 0.025*max(PPL-12,0) - 5*empty_rate - invalid_empty_penalty。",
        "",
    ]
    all_suggestions: list[dict[str, Any]] = []
    any_alert = False

    for (mid, method), items in sorted(grouped.items()):
        completed = [item for item in items if item.completed]
        failed = [item for item in items if item.failure]
        lines.extend([f"## {mid} / {method}", ""])
        if completed:
            best = max(completed, key=lambda item: item.trigger_free_score())
            lines.append(
                f"当前 trigger-free 最优运行：`{best.tag}` "
                f"(tf_score={fmt_float(best.trigger_free_score())}, "
                f"ASR_仅报告={fmt_float(best.metric('triggered_ASR'))}, "
                f"HNTR={fmt_float(best.metric('harmful_no_trigger_refusal'))}, "
                f"BFR={fmt_float(best.metric('benign_clean_false_refusal'))}, "
                f"PPL={best.ppl_value():.2f})."
            )
            lines.append("")
            lines.append("| tag | tf_score | ASR 仅报告 | HNTR | BFR | PPL | 状态 |")
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            for item in sorted(completed, key=lambda rec: rec.trigger_free_score(), reverse=True):
                lines.append(
                    f"| `{item.tag}` | {fmt_float(item.trigger_free_score())} | "
                    f"{fmt_float(item.metric('triggered_ASR'))} | "
                    f"{fmt_float(item.metric('harmful_no_trigger_refusal'))} | "
                    f"{fmt_float(item.metric('benign_clean_false_refusal'))} | "
                    f"{item.ppl_value():.2f} | 已完成 |"
                )
            lines.append("")
        else:
            lines.append("尚无完成运行。")
            lines.append("")

        if failed:
            lines.append("失败运行：")
            for item in failed:
                reason = classify_failure(item)
                lines.append(f"- `{item.tag}`: {reason}")
            lines.append("")

        alerts, suggestions = analyze_group(items)
        if alerts:
            any_alert = True
            lines.append("边界告警：")
            for alert in alerts:
                lines.append(f"- {alert}")
            lines.append("")
            all_suggestions.extend(suggestions)

    if auto_expand and all_suggestions:
        write_jsonl_unique(queue_file, all_suggestions)
        lines.append(f"参数扩展建议已追加到 `{queue_file}`。")
        lines.append("")
    elif all_suggestions:
        lines.append(f"已有参数扩展建议；用 `--auto-expand` 重新运行可追加到 `{queue_file}`。")
        lines.append("")

    if not records:
        lines.append("尚未发现 sweep 配置。")
        lines.append("")

    report = "\n".join(lines)
    out_dir = root / model_id if model_id else root
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "WATCHDOG_REPORT.md").write_text(report + "\n", encoding="utf-8")
    if any_alert:
        (out_dir / "NEEDS_REVIEW.md").write_text(report + "\n", encoding="utf-8")
    return report


def run_once(args: argparse.Namespace) -> None:
    root = Path(args.root)
    queue_file = Path(args.queue_file)
    records = discover_records(root, args.model_id)
    report = make_report(root, args.model_id, records, queue_file, bool(args.auto_expand))
    if args.print_report:
        print(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--queue-file", type=Path, default=Path("expansion_queue.jsonl"))
    parser.add_argument("--auto-expand", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()

    if args.daemon:
        while True:
            run_once(args)
            time.sleep(max(10, int(args.poll_seconds)))
    run_once(args)


if __name__ == "__main__":
    main()
