#!/usr/bin/env python3
"""Summarize and audit security-completeness experiment outputs."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/home/lizhy/plp/trigger-free-pruning-defense-round2/result/security_completeness_experiments")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, default=None)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _required_asr_fields(payload: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    metrics = payload.get("metrics", {})
    for key in ("total", "valid_total", "empty_count", "empty_output_rate", "avg_output_tokens", "median_output_tokens"):
        if key not in metrics:
            missing.append(f"metrics.{key}")
    for split_name in ("triggered_asr", "harmful_no_trigger_refusal", "benign_clean_false_refusal"):
        split = payload.get("split_metrics", {}).get(split_name, {})
        for key in ("total", "valid_total", "empty_count", "avg_output_tokens", "median_output_tokens"):
            if key not in split:
                missing.append(f"split_metrics.{split_name}.{key}")
    return missing


def main() -> None:
    args = parse_args()
    out_md = args.out_md or (args.root / "SECURITY_COMPLETENESS_AUDIT.md")
    out_json = args.out_json or (args.root / "security_completeness_audit.json")
    args.root.mkdir(parents=True, exist_ok=True)

    runs: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    leftover_checkpoints: list[str] = []

    for run_dir in sorted((args.root / "runs").glob("*")) if (args.root / "runs").exists() else []:
        if not run_dir.is_dir():
            continue
        config = _read_json(run_dir / "config.json") or {}
        asr = _read_json(run_dir / "asr.json") or {}
        ppl = _read_json(run_dir / "ppl.json") or {}
        plan = _read_json(run_dir / "pruning_plan.json") or {}
        for subdir in ("pruned_model", "recovered_model"):
            if (run_dir / subdir).exists():
                leftover_checkpoints.append(str(run_dir / subdir))
        missing_asr = _required_asr_fields(asr) if asr else ["asr.json"]
        missing_plan = [
            key for key in ("pruned_total", "pruned_heads", "pruned_channels")
            if key not in plan
        ] if plan else ["pruning_plan.json"]
        if missing_asr or missing_plan or not ppl:
            issues.append(
                {
                    "run": run_dir.name,
                    "missing_asr_fields": missing_asr,
                    "missing_plan_fields": missing_plan,
                    "missing_ppl": not bool(ppl),
                }
            )
        metrics = asr.get("metrics", {})
        runs.append(
            {
                "run": run_dir.name,
                "family": config.get("family", ""),
                "seed": config.get("seed", ""),
                "requested_budget": config.get("requested_budget_units", plan.get("requested_budget", "")),
                "gate": config.get("gate", plan.get("gate_label", "")),
                "actual_pruned": plan.get("pruned_total", ""),
                "heads": plan.get("pruned_heads", ""),
                "mlp_channels": plan.get("pruned_channels", ""),
                "ASR": metrics.get("triggered_ASR", ""),
                "HarmRef": metrics.get("HarmRef", metrics.get("harmful_no_trigger_refusal", "")),
                "BFR": metrics.get("BFR", metrics.get("benign_clean_false_refusal", "")),
                "Empty": metrics.get("empty_output_rate", ""),
                "avg_output_tokens": metrics.get("avg_output_tokens", ""),
                "median_output_tokens": metrics.get("median_output_tokens", ""),
                "PPL": ppl.get("ppl", ""),
                "prompt_template": asr.get("prompt_template", config.get("prompt_template", "")),
                "transformers_version": asr.get("transformers_version", ""),
            }
        )

    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_family[str(run["family"])].append(run)

    seed_summary: dict[str, dict[str, dict[str, float]]] = {}
    for family, family_runs in by_family.items():
        if family not in {"full_pipeline_seed", "fixed_plan_recovery_seed"}:
            continue
        family_summary: dict[str, dict[str, float]] = {}
        for metric in ("ASR", "HarmRef", "BFR", "Empty", "PPL"):
            values: list[float] = []
            for run in family_runs:
                try:
                    values.append(float(run[metric]))
                except Exception:
                    pass
            mean, std = _mean_std(values)
            family_summary[metric] = {"mean": mean, "std": std, "n": len(values)}
        seed_summary[family] = family_summary

    audit = {
        "root": str(args.root),
        "run_count": len(runs),
        "runs": runs,
        "issues": issues,
        "leftover_checkpoints": leftover_checkpoints,
        "seed_summary": seed_summary,
        "required_fields": [
            "total",
            "valid_total",
            "empty_count",
            "empty_output_rate",
            "avg_output_tokens",
            "median_output_tokens",
            "pruned_heads",
            "pruned_channels",
        ],
    }
    out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# Security-Completeness Experiment Audit",
        "",
        f"- Root: `{args.root}`",
        f"- Runs found: `{len(runs)}`",
        f"- Runs with issues: `{len(issues)}`",
        f"- Leftover temporary checkpoints: `{len(leftover_checkpoints)}`",
        "",
    ]
    if seed_summary:
        for family, family_summary in seed_summary.items():
            title = "Full-Pipeline Seed Summary" if family == "full_pipeline_seed" else "Fixed-Plan Recovery Seed Summary"
            lines.extend([f"## {title}", "", "| Metric | n | Mean | Std |", "|---|---:|---:|---:|"])
            for metric, stats in family_summary.items():
                lines.append(f"| {metric} | {int(stats['n'])} | {stats['mean']:.6g} | {stats['std']:.6g} |")
            lines.append("")

    lines.extend(
        [
            "## Run Table",
            "",
            "| Run | Family | Seed | Requested | Actual | #Heads | #MLP channels | ASR | HarmRef | BFR | Empty | Avg tok | Median tok | PPL |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run in runs:
        lines.append(
            "| {run} | {family} | {seed} | {requested_budget} | {actual_pruned} | {heads} | {mlp_channels} | "
            "{ASR} | {HarmRef} | {BFR} | {Empty} | {avg_output_tokens} | {median_output_tokens} | {PPL} |".format(**run)
        )
    lines.append("")
    if issues:
        lines.extend(["## Issues", "", "| Run | Missing ASR fields | Missing plan fields | Missing PPL |", "|---|---|---|---:|"])
        for issue in issues:
            lines.append(
                f"| `{issue['run']}` | `{', '.join(issue['missing_asr_fields'])}` | "
                f"`{', '.join(issue['missing_plan_fields'])}` | `{issue['missing_ppl']}` |"
            )
        lines.append("")
    if leftover_checkpoints:
        lines.extend(["## Leftover Temporary Checkpoints", ""])
        for path in leftover_checkpoints:
            lines.append(f"- `{path}`")
        lines.append("")

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote audit to {out_md}")


if __name__ == "__main__":
    main()
