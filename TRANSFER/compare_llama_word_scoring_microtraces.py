#!/usr/bin/env python3
"""Compare two Llama-word scoring microtrace JSON files.

Use this after running dump_llama_word_scoring_microtrace.py on two environments.
It reports the earliest observable divergence: prompt fingerprints, losses,
FGSM perturbation signs, aggregate unit fields, and per-sample unit fields.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_abs(a: float, b: float) -> tuple[float, float]:
    absd = abs(a - b)
    denom = max(abs(a), abs(b), 1e-12)
    return absd, absd / denom


def compare_scalar_rows(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    path: str,
    fields: list[str],
    abs_tol: float,
    rel_tol: float,
) -> list[dict[str, Any]]:
    diffs = []
    for field in fields:
        if field not in left or field not in right:
            continue
        if not isinstance(left[field], (int, float)) or not isinstance(right[field], (int, float)):
            continue
        absd, reld = rel_abs(float(left[field]), float(right[field]))
        if absd > abs_tol and reld > rel_tol:
            diffs.append(
                {
                    "path": f"{path}.{field}",
                    "left": left[field],
                    "right": right[field],
                    "abs_diff": absd,
                    "rel_diff": reld,
                }
            )
    return diffs


def first_prompt_mismatches(left_records: list[dict[str, Any]], right_records: list[dict[str, Any]], kind: str) -> list[str]:
    out = []
    for i, (lrow, rrow) in enumerate(zip(left_records, right_records)):
        for field in ("prompt_text_sha256", "input_ids_sha256", "input_len"):
            lv = lrow["prompt"].get(field)
            rv = rrow["prompt"].get(field)
            if lv != rv:
                out.append(f"{kind}[{i}].prompt.{field}: {lv} != {rv}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True, help="Reference microtrace JSON, e.g. old server")
    parser.add_argument("--right", type=Path, required=True, help="Comparison microtrace JSON, e.g. current server")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--abs-tol", type=float, default=1e-7)
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    args = parser.parse_args()

    left = load(args.left)
    right = load(args.right)
    report: dict[str, Any] = {
        "left": str(args.left),
        "right": str(args.right),
        "abs_tol": args.abs_tol,
        "rel_tol": args.rel_tol,
        "config_equal": left.get("config") == right.get("config"),
        "selected_sets": {
            "left": left.get("selected_sets"),
            "right": right.get("selected_sets"),
        },
        "prompt_mismatches": [],
        "sample_meta_diffs": [],
        "aggregate_unit_diffs_top": [],
        "per_sample_unit_diffs_top": [],
    }

    report["prompt_mismatches"].extend(
        first_prompt_mismatches(left.get("clean_records", []), right.get("clean_records", []), "clean")
    )
    report["prompt_mismatches"].extend(
        first_prompt_mismatches(left.get("safe_records", []), right.get("safe_records", []), "safe")
    )

    for kind, scalar_fields in (
        ("clean_records", ["clean_lm_loss", "consistency_loss", "perturbed_lm_loss"]),
        ("safe_records", ["safe_lm_loss"]),
    ):
        for i, (lrow, rrow) in enumerate(zip(left.get(kind, []), right.get(kind, []))):
            report["sample_meta_diffs"].extend(
                compare_scalar_rows(
                    lrow,
                    rrow,
                    path=f"{kind}[{i}]",
                    fields=scalar_fields,
                    abs_tol=args.abs_tol,
                    rel_tol=args.rel_tol,
                )
            )
            if kind == "clean_records":
                for sign_field in ("positive", "negative", "zero", "numel"):
                    lv = lrow.get("perturb_sign", {}).get(sign_field)
                    rv = rrow.get("perturb_sign", {}).get(sign_field)
                    if lv != rv:
                        report["sample_meta_diffs"].append(
                            {
                                "path": f"{kind}[{i}].perturb_sign.{sign_field}",
                                "left": lv,
                                "right": rv,
                                "abs_diff": None,
                                "rel_diff": None,
                            }
                        )

    scalar_fields = [
        "clean_grad_mean",
        "proxy_grad_mean",
        "cosine",
        "safe_grad_mean",
        "protect_grad_mean",
        "clean_proxy_penalty",
        "score",
    ]
    aggregate_diffs = []
    left_units = left.get("aggregate_units", {})
    right_units = right.get("aggregate_units", {})
    for label in sorted(set(left_units) & set(right_units)):
        lu = left_units[label].get("microtrace_aggregate", {})
        ru = right_units[label].get("microtrace_aggregate", {})
        aggregate_diffs.extend(
            compare_scalar_rows(
                lu,
                ru,
                path=f"aggregate_units[{label}]",
                fields=scalar_fields,
                abs_tol=args.abs_tol,
                rel_tol=args.rel_tol,
            )
        )
    aggregate_diffs.sort(key=lambda item: -float(item.get("abs_diff") or 0.0))
    report["aggregate_unit_diffs_top"] = aggregate_diffs[:200]

    per_sample_diffs = []
    for kind, unit_fields in (
        ("clean_records", ["mag", "proxy_mag", "cosine", "penalty", "l2", "proxy_l2"]),
        ("safe_records", ["safe_mag", "safe_l2"]),
    ):
        for i, (lrow, rrow) in enumerate(zip(left.get(kind, []), right.get(kind, []))):
            common = set(lrow.get("units", {})) & set(rrow.get("units", {}))
            for label in sorted(common):
                per_sample_diffs.extend(
                    compare_scalar_rows(
                        lrow["units"][label],
                        rrow["units"][label],
                        path=f"{kind}[{i}].units[{label}]",
                        fields=unit_fields,
                        abs_tol=args.abs_tol,
                        rel_tol=args.rel_tol,
                    )
                )
    per_sample_diffs.sort(key=lambda item: -float(item.get("abs_diff") or 0.0))
    report["per_sample_unit_diffs_top"] = per_sample_diffs[:200]

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
