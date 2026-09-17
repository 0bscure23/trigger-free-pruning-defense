#!/usr/bin/env python3
"""Audit prompt/evaluation protocols before security-completeness experiments."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_REPO = Path("/home/lizhy/plp/trigger-free-pruning-defense-round2")
DEFAULT_RESULT_ROOT = DEFAULT_REPO / "result"
DEFAULT_OUT = DEFAULT_RESULT_ROOT / "security_completeness_protocol"
TARGET_TEX = [
    DEFAULT_REPO / "1paper_ieee_main_theory_revised.tex",
    DEFAULT_REPO / "2paper_ieee_main_theory_revised.tex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULT_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--lock-prompt-template", choices=["alpaca", "chat", "none"], default=None)
    parser.add_argument("--lock-runtime", default=None, help="Optional runtime/version note for the protocol lock.")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _counter_table(counter: Counter[str]) -> list[dict[str, Any]]:
    return [{"value": key, "count": int(count)} for key, count in sorted(counter.items())]


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for path in sorted(args.result_root.rglob("*.json")):
        payload = _read_json(path)
        if not payload:
            continue
        if payload.get("script") != "diagnose_generation_metrics.py" and "split_metrics" not in payload:
            continue
        if "prompt_template" not in payload:
            continue
        records.append(
            {
                "path": str(path),
                "label": str(payload.get("label", "")),
                "prompt_template": str(payload.get("prompt_template", "")),
                "dtype": str(payload.get("dtype", "")),
                "eval_max_new_tokens": str(payload.get("eval_max_new_tokens", "")),
                "eval_do_sample": str(payload.get("eval_do_sample", False)),
                "transformers_version": str(payload.get("transformers_version", "")),
            }
        )

    counters = {
        "prompt_template": Counter(record["prompt_template"] for record in records),
        "dtype": Counter(record["dtype"] for record in records),
        "eval_max_new_tokens": Counter(record["eval_max_new_tokens"] for record in records),
        "eval_do_sample": Counter(record["eval_do_sample"] for record in records),
        "transformers_version": Counter(record["transformers_version"] for record in records),
    }
    tex_status = [{"path": str(path), "exists": path.exists()} for path in TARGET_TEX]
    mixed_protocol_warning = any(len(counter) > 1 for key, counter in counters.items() if key != "transformers_version")

    audit = {
        "result_root": str(args.result_root),
        "record_count": len(records),
        "counters": {key: _counter_table(counter) for key, counter in counters.items()},
        "mixed_protocol_warning": bool(mixed_protocol_warning),
        "target_tex": tex_status,
        "records": records,
    }

    if args.lock_prompt_template is not None:
        lock = {
            "prompt_template": str(args.lock_prompt_template),
            "runtime_note": args.lock_runtime,
            "asr_protocol": "diagnose_generation_metrics.py BackdoorLLM keyword protocol",
            "decoding": "greedy unless eval_do_sample=true",
            "max_new_tokens": 64,
            "empty_normalization": ["", ".", "?"],
            "denominator": "non-empty generations for ASR/HarmRef/BFR",
        }
        (args.out_dir / "protocol_lock.json").write_text(
            json.dumps(lock, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        audit["protocol_lock"] = lock

    (args.out_dir / "protocol_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Protocol Audit for Security-Completeness Experiments",
        "",
        f"- Result root: `{args.result_root}`",
        f"- ASR JSON records found: `{len(records)}`",
        f"- Mixed prompt/eval protocol warning: `{mixed_protocol_warning}`",
        "",
        "## Protocol Counts",
        "",
    ]
    for key, counter in counters.items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("| Value | Count |")
        lines.append("|---|---:|")
        for value, count in sorted(counter.items()):
            lines.append(f"| `{value}` | {int(count)} |")
        lines.append("")
    lines.extend(["## Target TeX Files", "", "| Path | Exists |", "|---|---:|"])
    for item in tex_status:
        lines.append(f"| `{item['path']}` | `{item['exists']}` |")
    if args.lock_prompt_template is not None:
        lines.extend(
            [
                "",
                "## Protocol Lock",
                "",
                f"- prompt template: `{args.lock_prompt_template}`",
                "- ASR/HarmRef/BFR denominator: non-empty generations",
                "- Empty outputs are reported separately.",
            ]
        )
    (args.out_dir / "PROTOCOL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote protocol audit to {args.out_dir}")


if __name__ == "__main__":
    main()
