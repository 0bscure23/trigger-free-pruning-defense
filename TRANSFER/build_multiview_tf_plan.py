#!/usr/bin/env python3
"""Fuse multiple trigger-free pruning rankings into one plan.

This is a trigger-free screening tool: it only consumes already-built trigger-free
plans and combines their ranks by support count plus reciprocal-rank fusion.
"""
import argparse
import collections
import json
from pathlib import Path


ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, required=True)
ap.add_argument(
    "--source",
    action="append",
    required=True,
    help="Source spec: name:path:weight[:topn]. Example fgsm:/tmp/plan.json:1.0:512",
)
ap.add_argument("--rrf-c", type=float, default=60.0)
ap.add_argument("--min-support", type=int, default=2)
args = ap.parse_args()


def parse_source(spec):
    parts = spec.split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"Bad --source spec: {spec}")
    name, path, weight = parts[:3]
    topn = int(parts[3]) if len(parts) == 4 else None
    return name, Path(path), float(weight), topn


items = {}
sources_meta = []

for spec in args.source:
    name, path, weight, topn = parse_source(spec)
    plan = json.loads(path.read_text())
    units = plan.get("to_prune", [])
    if topn is not None:
        units = units[:topn]
    sources_meta.append({
        "name": name,
        "path": str(path),
        "weight": weight,
        "topn": topn,
        "used_units": len(units),
        "proxy_type": plan.get("proxy_type"),
    })
    for rank, unit in enumerate(units, start=1):
        key = (unit["component"], int(unit["layer"]), int(unit["index"]))
        rec = items.setdefault(key, {
            "component": unit["component"],
            "layer": int(unit["layer"]),
            "index": int(unit["index"]),
            "support": 0,
            "weighted_support": 0.0,
            "rrf": 0.0,
            "best_rank": rank,
            "rank_sum": 0.0,
            "sources": [],
            "source_ranks": {},
            "clean_grad_mean": float(unit.get("clean_grad_mean", 0.0) or 0.0),
            "proxy_grad_mean": float(unit.get("proxy_grad_mean", 0.0) or 0.0),
            "cosine": float(unit.get("cosine", 0.0) or 0.0),
        })
        rec["support"] += 1
        rec["weighted_support"] += weight
        rec["rrf"] += weight / (args.rrf_c + rank)
        rec["best_rank"] = min(rec["best_rank"], rank)
        rec["rank_sum"] += weight * rank
        rec["sources"].append(name)
        rec["source_ranks"][name] = rank

eligible = [rec for rec in items.values() if rec["support"] >= args.min_support]
eligible.sort(key=lambda r: (-r["support"], -r["weighted_support"], -r["rrf"], r["best_rank"]))
ranked = eligible[: args.budget]

to_prune = []
for rec in ranked:
    score = rec["support"] * 1000.0 + rec["weighted_support"] * 100.0 + rec["rrf"]
    to_prune.append({
        "component": rec["component"],
        "layer": rec["layer"],
        "index": rec["index"],
        "clean_grad_mean": rec["clean_grad_mean"],
        "proxy_grad_mean": rec["proxy_grad_mean"],
        "cosine": rec["cosine"],
        "score": float(-score),
        "multiview_support": int(rec["support"]),
        "multiview_weighted_support": float(rec["weighted_support"]),
        "multiview_rrf": float(rec["rrf"]),
        "multiview_best_rank": int(rec["best_rank"]),
        "multiview_sources": rec["sources"],
        "multiview_source_ranks": rec["source_ranks"],
    })

support_hist = collections.Counter(rec["support"] for rec in items.values())
layer_counts = collections.Counter(unit["layer"] for unit in to_prune)
component_counts = collections.Counter(unit["component"] for unit in to_prune)

out = {
    "timestamp": 0,
    "proxy_type": "trigger_free_multiview_rank_fusion",
    "signal": "support_count_then_weighted_reciprocal_rank_fusion_over_trigger_free_plans",
    "budget": args.budget,
    "max_prune_units": args.budget,
    "min_support": args.min_support,
    "rrf_c": args.rrf_c,
    "sources": sources_meta,
    "union_units": len(items),
    "eligible_units": len(eligible),
    "support_hist": dict(sorted(support_hist.items())),
    "pruned_total": len(to_prune),
    "pruned_heads": sum(1 for u in to_prune if u["component"] == "head"),
    "pruned_channels": sum(1 for u in to_prune if u["component"] == "channel"),
    "to_prune": to_prune,
}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(out, indent=2))
print(
    f"[multiview] wrote {args.out}; selected={len(to_prune)} "
    f"eligible={len(eligible)} union={len(items)} components={dict(component_counts)}"
)
print(f"  support_hist={dict(sorted(support_hist.items()))}")
print(f"  top_layers={layer_counts.most_common(10)}")
