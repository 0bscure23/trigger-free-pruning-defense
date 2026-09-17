#!/usr/bin/env python3
"""Analyze golden-vs-current Llama Word unit-score drift and post-hoc recoverability.

This is a diagnostic script. It must not be presented as the deployed
trigger-free scoring method because it uses the golden 21-unit set as labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = (
    "clean_grad_mean",
    "safe_grad_mean",
    "protect_grad_mean",
    "proxy_grad_mean",
    "cosine",
    "attack_abs_proxy_cosine",
    "score",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path) -> dict[tuple[str, int, int], dict[str, Any]]:
    obj = json.load(path.open("r", encoding="utf-8"))
    rows = obj.get("scores") or obj.get("unit_scores") or []
    out: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = (str(row["component"]), int(row["layer"]), int(row["index"]))
        copied = dict(row)
        copied["component"] = key[0]
        copied["layer"] = key[1]
        copied["index"] = key[2]
        copied["attack_abs_proxy_cosine"] = abs(float(row["proxy_grad_mean"]) * float(row["cosine"]))
        out[key] = copied
    return out


def qstats(x: np.ndarray) -> dict[str, float]:
    if x.size == 0:
        return {}
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "q01": float(np.quantile(x, 0.01)),
        "q05": float(np.quantile(x, 0.05)),
        "median": float(np.quantile(x, 0.50)),
        "q95": float(np.quantile(x, 0.95)),
        "q99": float(np.quantile(x, 0.99)),
        "max": float(np.max(x)),
    }


def topk_overlap(scores: np.ndarray, labels: np.ndarray, k: int) -> tuple[int, np.ndarray]:
    idx = np.argpartition(scores, k - 1)[:k]
    idx = idx[np.argsort(scores[idx])]
    return int(labels[idx].sum()), idx


def keys_to_json(keys: list[tuple[str, int, int]]) -> list[dict[str, Any]]:
    return [{"component": c, "layer": l, "index": i} for c, l, i in keys]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-scores", type=Path, default=Path("/home/lizhy/plp/unit_scores.json"))
    parser.add_argument(
        "--current-scores",
        type=Path,
        default=Path(
            "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_cu121_score_test/unit_scores.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_reverse_engineering"
        ),
    )
    parser.add_argument("--min-layer", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=21)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    golden = load_rows(args.golden_scores)
    current = load_rows(args.current_scores)
    common_keys = sorted(set(golden) & set(current), key=lambda k: (k[1], k[0], k[2]))
    eligible_keys = [k for k in common_keys if k[1] >= args.min_layer]
    if not eligible_keys:
        raise RuntimeError("no eligible keys")

    golden_nonpos = {k for k in eligible_keys if float(golden[k]["score"]) <= 0.0}
    current_nonpos = {k for k in eligible_keys if float(current[k]["score"]) <= 0.0}
    labels = np.array([1 if k in golden_nonpos else 0 for k in eligible_keys], dtype=np.int8)
    layers = np.array([k[1] for k in eligible_keys], dtype=np.float64)
    layer_scaled = (layers - args.min_layer) / max(1.0, float(max(layers) - args.min_layer))

    arrays: dict[str, np.ndarray] = {}
    golden_arrays: dict[str, np.ndarray] = {}
    for feat in FEATURES:
        arrays[feat] = np.array([float(current[k][feat]) for k in eligible_keys], dtype=np.float64)
        golden_arrays[feat] = np.array([float(golden[k][feat]) for k in eligible_keys], dtype=np.float64)

    # Basic drift by group.
    groups = {
        "all_eligible": np.ones(len(eligible_keys), dtype=bool),
        "golden21": labels.astype(bool),
        "current61": np.array([k in current_nonpos for k in eligible_keys], dtype=bool),
        "shared4": np.array([k in current_nonpos and k in golden_nonpos for k in eligible_keys], dtype=bool),
        "golden_only17": np.array([k in golden_nonpos and k not in current_nonpos for k in eligible_keys], dtype=bool),
        "current_only57": np.array([k in current_nonpos and k not in golden_nonpos for k in eligible_keys], dtype=bool),
    }
    group_stats: dict[str, Any] = {}
    for group_name, mask in groups.items():
        group_stats[group_name] = {"count": int(mask.sum())}
        for feat in FEATURES:
            cur = arrays[feat][mask]
            gol = golden_arrays[feat][mask]
            group_stats[group_name][feat] = {
                "golden": qstats(gol),
                "current": qstats(cur),
                "delta_current_minus_golden": qstats(cur - gol),
            }

    # Current original formula rank diagnostics.
    original_scores = arrays["score"]
    original_overlap, original_idx = topk_overlap(original_scores, labels, args.top_k)
    rank_order = np.argsort(original_scores)
    current_rank = {eligible_keys[int(i)]: int(rank + 1) for rank, i in enumerate(rank_order)}
    golden_rank_rows = [
        {
            "rank_current_score": current_rank[k],
            "key": {"component": k[0], "layer": k[1], "index": k[2]},
            "current_score": float(current[k]["score"]),
            "golden_score": float(golden[k]["score"]),
            "current_clean": float(current[k]["clean_grad_mean"]),
            "golden_clean": float(golden[k]["clean_grad_mean"]),
            "current_safe": float(current[k]["safe_grad_mean"]),
            "golden_safe": float(golden[k]["safe_grad_mean"]),
            "current_attack": float(current[k]["attack_abs_proxy_cosine"]),
            "golden_attack": float(golden[k]["attack_abs_proxy_cosine"]),
            "current_cosine": float(current[k]["cosine"]),
            "golden_cosine": float(golden[k]["cosine"]),
            "current_proxy": float(current[k]["proxy_grad_mean"]),
            "golden_proxy": float(golden[k]["proxy_grad_mean"]),
        }
        for k in sorted(golden_nonpos, key=lambda kk: current_rank[kk])
    ]

    # Grid-search interpretable affine variants on current features.
    safe_grid = np.array([-4, -2, -1, -0.5, 0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8], dtype=np.float64)
    beta_grid = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4, 6, 8, 10], dtype=np.float64)
    layer_grid = np.array([0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3], dtype=np.float64)
    grid_rows: list[dict[str, Any]] = []
    clean = arrays["clean_grad_mean"]
    safe = arrays["safe_grad_mean"]
    attack = arrays["attack_abs_proxy_cosine"]
    for safe_coef in safe_grid:
        for beta in beta_grid:
            base_score = clean + safe_coef * safe - beta * attack
            for layer_coef in layer_grid:
                score = base_score - layer_coef * layer_scaled
                overlap, idx = topk_overlap(score, labels, args.top_k)
                selected_keys = [eligible_keys[int(i)] for i in idx]
                grid_rows.append(
                    {
                        "overlap_at_21": overlap,
                        "safe_coef": float(safe_coef),
                        "beta_attack_coef": float(beta),
                        "late_layer_prior_coef": float(layer_coef),
                        "selected_count": args.top_k,
                        "selected_layers": dict(Counter(k[1] for k in selected_keys)),
                        "selected_keys": keys_to_json(selected_keys),
                    }
                )
    grid_rows.sort(key=lambda r: (-int(r["overlap_at_21"]), abs(float(r["safe_coef"]) - 0.5), abs(float(r["beta_attack_coef"]) - 1.0), float(r["late_layer_prior_coef"])))

    # Logistic oracle diagnostic. This is explicitly in-sample and post-hoc.
    X = np.column_stack(
        [
            clean,
            safe,
            arrays["protect_grad_mean"],
            arrays["proxy_grad_mean"],
            arrays["cosine"],
            attack,
            original_scores,
            layer_scaled,
            (np.array([1 if k[0] == "channel" else 0 for k in eligible_keys], dtype=np.float64)),
        ]
    )
    feature_names = [
        "clean",
        "safe",
        "protect",
        "proxy",
        "cosine",
        "attack",
        "original_score",
        "layer_scaled",
        "is_channel",
    ]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, solver="lbfgs", random_state=0),
    )
    clf.fit(X, labels)
    prob = clf.predict_proba(X)[:, 1]
    oracle_score = -prob
    oracle_overlap, oracle_idx = topk_overlap(oracle_score, labels, args.top_k)
    try:
        oracle_auc = float(roc_auc_score(labels, prob))
    except Exception:
        oracle_auc = float("nan")
    oracle_ap = float(average_precision_score(labels, prob))
    scaler = clf.named_steps["standardscaler"]
    lr = clf.named_steps["logisticregression"]
    coef_rows = [
        {
            "feature": name,
            "coef_standardized": float(coef),
            "mean": float(mean),
            "scale": float(scale),
        }
        for name, coef, mean, scale in zip(feature_names, lr.coef_[0], scaler.mean_, scaler.scale_)
    ]
    coef_rows.sort(key=lambda r: abs(float(r["coef_standardized"])), reverse=True)
    oracle_selected = [eligible_keys[int(i)] for i in oracle_idx]

    summary = {
        "inputs": {
            "golden_scores": str(args.golden_scores),
            "golden_scores_sha256": sha256(args.golden_scores),
            "current_scores": str(args.current_scores),
            "current_scores_sha256": sha256(args.current_scores),
            "min_layer": args.min_layer,
            "top_k": args.top_k,
        },
        "counts": {
            "common_units": len(common_keys),
            "eligible_units": len(eligible_keys),
            "golden_nonpos": len(golden_nonpos),
            "current_nonpos": len(current_nonpos),
            "shared_nonpos": len(golden_nonpos & current_nonpos),
        },
        "original_current_score": {
            "overlap_at_21": original_overlap,
            "selected_keys": keys_to_json([eligible_keys[int(i)] for i in original_idx]),
        },
        "group_stats": group_stats,
        "golden_units_ranked_by_current_score": golden_rank_rows,
        "best_affine_grid": grid_rows[:20],
        "oracle_logistic_in_sample": {
            "warning": "post-hoc diagnostic only; uses golden 21 labels and is not a deployable trigger-free method",
            "overlap_at_21": oracle_overlap,
            "average_precision": oracle_ap,
            "roc_auc": oracle_auc,
            "selected_keys": keys_to_json(oracle_selected),
            "coefficients": coef_rows,
        },
    }

    json_path = args.out_dir / "score_drift_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    csv_path = args.out_dir / "golden_units_ranked_by_current_score.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(golden_rank_rows[0].keys()))
        writer.writeheader()
        writer.writerows(golden_rank_rows)

    grid_csv = args.out_dir / "best_affine_grid.csv"
    with grid_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["overlap_at_21", "safe_coef", "beta_attack_coef", "late_layer_prior_coef", "selected_count", "selected_layers"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in grid_rows[:200]:
            writer.writerow({k: row[k] for k in fieldnames})

    md_path = args.out_dir / "SUMMARY.md"
    best = grid_rows[0]
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Llama Word Score Drift Reverse-Engineering Diagnostic\n\n")
        f.write("This is a post-hoc diagnostic, not a deployable trigger-free method.\n\n")
        f.write("## Counts\n\n")
        for key, value in summary["counts"].items():
            f.write(f"- {key}: `{value}`\n")
        f.write("\n## Original Current Score\n\n")
        f.write(f"- overlap@21 with golden 21: `{original_overlap}/21`\n")
        f.write("\n## Best Interpretable Grid Formula\n\n")
        f.write("Formula family: `clean + safe_coef * safe - beta * abs(proxy * cosine) - late_layer_prior_coef * layer_scaled`.\n\n")
        f.write(f"- best overlap@21: `{best['overlap_at_21']}/21`\n")
        f.write(f"- safe_coef: `{best['safe_coef']}`\n")
        f.write(f"- beta_attack_coef: `{best['beta_attack_coef']}`\n")
        f.write(f"- late_layer_prior_coef: `{best['late_layer_prior_coef']}`\n")
        f.write(f"- selected_layers: `{best['selected_layers']}`\n")
        f.write("\n## Oracle Logistic Diagnostic\n\n")
        f.write("This model uses the golden 21 labels in-sample. It is only a signal-presence diagnostic.\n\n")
        f.write(f"- overlap@21: `{oracle_overlap}/21`\n")
        f.write(f"- average precision: `{oracle_ap:.6f}`\n")
        f.write(f"- ROC-AUC: `{oracle_auc:.6f}`\n")
        f.write("\nTop standardized coefficients:\n\n")
        for row in coef_rows[:8]:
            f.write(f"- `{row['feature']}`: `{row['coef_standardized']:.6f}`\n")
        f.write("\n## Files\n\n")
        f.write(f"- JSON: `{json_path}`\n")
        f.write(f"- Golden ranks CSV: `{csv_path}`\n")
        f.write(f"- Affine grid CSV: `{grid_csv}`\n")

    print("wrote", json_path)
    print("wrote", md_path)
    print("counts", summary["counts"])
    print("original overlap@21", original_overlap)
    print("best affine", {k: best[k] for k in ("overlap_at_21", "safe_coef", "beta_attack_coef", "late_layer_prior_coef", "selected_layers")})
    print("oracle logistic", {"overlap_at_21": oracle_overlap, "ap": oracle_ap, "auc": oracle_auc})


if __name__ == "__main__":
    main()
