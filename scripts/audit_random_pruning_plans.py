#!/usr/bin/env python3
"""Aggregate auditable random pruning plans and compute pairwise Jaccard.

Use this after generating several ``--plan-only`` random controls. It reads each
run's ``pruning_plan.json`` or ``random_control_audit.json`` and reports whether
seeds actually produced distinct unit sets. In addition to the global summary,
it reports grouped summaries by match_mode and by match_mode+candidate_source so
mixed-mode audit globs are not misinterpreted as a single seed sweep.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-glob",
        action="append",
        required=True,
        help="Glob for pruning_plan.json or random_control_audit.json files; can be repeated",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _hash_units(units: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(units)).encode("utf-8")).hexdigest()


def _load_units(path: Path) -> tuple[list[str], dict[str, object]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    audit = raw.get("audit", raw) if isinstance(raw, dict) else {}
    if isinstance(audit, dict) and isinstance(audit.get("selected_units_sorted"), list):
        units = [str(x) for x in audit["selected_units_sorted"]]
        meta = dict(audit)
    elif isinstance(raw, dict) and isinstance(raw.get("to_prune"), list):
        units = []
        for item in raw["to_prune"]:
            if isinstance(item, dict):
                units.append(f"{item.get('component')}:{int(item.get('layer'))}:{int(item.get('index'))}")
        meta = raw
    else:
        raise ValueError(f"Could not find selected units in {path}")
    return sorted(units), meta


def _jaccard(a: list[str], b: list[str]) -> tuple[int, float]:
    left = set(a)
    right = set(b)
    if not left and not right:
        return 0, 1.0
    overlap = len(left & right)
    union = len(left | right)
    return overlap, float(overlap / max(1, union))


def _pairwise_for(plans: list[dict[str, Any]]) -> list[dict[str, object]]:
    pairwise = []
    for i in range(len(plans)):
        for j in range(i + 1, len(plans)):
            overlap, jac = _jaccard(plans[i]["units"], plans[j]["units"])
            pairwise.append(
                {
                    "left": plans[i]["path"],
                    "right": plans[j]["path"],
                    "left_seed": plans[i].get("seed"),
                    "right_seed": plans[j].get("seed"),
                    "overlap": int(overlap),
                    "jaccard": float(jac),
                    "same_hash": bool(plans[i]["selected_unit_sha256"] == plans[j]["selected_unit_sha256"]),
                }
            )
    return pairwise


def _summary_for(plans: list[dict[str, Any]]) -> dict[str, object]:
    hashes = [str(plan["selected_unit_sha256"]) for plan in plans]
    pairwise = _pairwise_for(plans)
    pairwise_values = [float(item["jaccard"]) for item in pairwise]
    fallback_values = [int(plan.get("fallback_count") or 0) for plan in plans]
    return {
        "plan_count": int(len(plans)),
        "unique_selected_hash_count": int(len(set(hashes))),
        "all_hashes_unique": bool(len(set(hashes)) == len(hashes)),
        "duplicate_hashes": sorted({h for h in hashes if hashes.count(h) > 1}),
        "fallback_total": int(sum(fallback_values)),
        "fallback_nonzero_plan_count": int(sum(1 for value in fallback_values if value != 0)),
        "pairwise_jaccard_count": int(len(pairwise_values)),
        "pairwise_jaccard_min": float(min(pairwise_values)) if pairwise_values else None,
        "pairwise_jaccard_mean": float(mean(pairwise_values)) if pairwise_values else None,
        "pairwise_jaccard_max": float(max(pairwise_values)) if pairwise_values else None,
        "pairwise_jaccard": pairwise,
    }


def _group(plans: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for plan in plans:
        label = "|".join(str(plan.get(key, "unknown")) for key in keys)
        grouped[label].append(plan)
    return dict(sorted(grouped.items()))


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for pattern in args.plan_glob:
        paths.extend(Path(p) for p in glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError("No plan files matched --plan-glob")

    plans: list[dict[str, Any]] = []
    for path in paths:
        units, meta = _load_units(path)
        sampling_meta = meta.get("sampling_meta", {}) if isinstance(meta.get("sampling_meta", {}), dict) else {}
        candidate_meta = meta.get("candidate_meta", {}) if isinstance(meta.get("candidate_meta", {}), dict) else {}
        plans.append(
            {
                "path": str(path),
                "unit_count": int(len(units)),
                "selected_unit_sha256": str(meta.get("selected_unit_sha256") or _hash_units(units)),
                "seed": sampling_meta.get("seed"),
                "match_mode": sampling_meta.get("match_mode"),
                "fallback_count": sampling_meta.get("fallback_count"),
                "candidate_source": candidate_meta.get("candidate_source"),
                "selected_score_min": meta.get("selected_score_min"),
                "selected_score_mean": meta.get("selected_score_mean"),
                "selected_score_max": meta.get("selected_score_max"),
                "selected_negative_count": meta.get("selected_negative_count"),
                "units": units,
            }
        )

    global_summary = _summary_for(plans)
    by_match_mode = {
        key: _summary_for(value)
        for key, value in _group(plans, ("match_mode",)).items()
    }
    by_match_mode_candidate_source = {
        key: _summary_for(value)
        for key, value in _group(plans, ("match_mode", "candidate_source")).items()
    }

    output = {
        "plan_count": global_summary["plan_count"],
        "unique_selected_hash_count": global_summary["unique_selected_hash_count"],
        "all_hashes_unique": global_summary["all_hashes_unique"],
        "duplicate_hashes": global_summary["duplicate_hashes"],
        "global_summary": global_summary,
        "summary_by_match_mode": by_match_mode,
        "summary_by_match_mode_candidate_source": by_match_mode_candidate_source,
        "plans": [{k: v for k, v in plan.items() if k != "units"} for plan in plans],
        "pairwise_jaccard": global_summary["pairwise_jaccard"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
