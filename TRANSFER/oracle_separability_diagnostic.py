#!/usr/bin/env python3
"""Diagnose whether trigger-free rankings contain the trigger-aware oracle signal.

This is an offline diagnostic only.  The oracle labels come from the
trigger-aware causal plan, so any supervised score here is not deployable.  The
goal is to test whether existing trigger-free views carry enough information to
separate oracle units at all.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


DEFAULT_ROOT = Path("/home/lizhy/plp/trigger-free-pruning-defense-round2")
DEFAULT_CONFIG = Path("/home/lizhy/plp/Mistral-3-7B_long/config.json")

PLAN_SPECS = {
    "fgsm512": "result/recovery_followup/A_512base/pruning_plan.json",
    "seqmean": "result/future_directions/plan_seqmean.json",
    "window": "result/future_directions/plan_window.json",
    "tf_refsupp4096": "result/causal_tf/plan_tf_refsupp_4096.json",
    "tf_cons4096": "result/causal_tf/plan_tf_cons_4096.json",
    "tf_eps001": "result/causal_tf/plan_tf_refsupp_eps0.01.json",
    "tf_pgd1024": "result/causal_tf/plan_tf_pgd_1024.json",
    "tf_diff4096": "result/causal_tf_diff/plan_tf_diff_refsupp_eps0.01_bw1.0_4096.json",
    "multiview1024": "result/multiview_tf/plan_mv_support_1024.json",
    "pseudo512": "result/pseudo_trigger_pool/top_score512/pruning_plan.json",
    "weightdiff4096": "result/weight_diff_ref/plan_long_vs_word_weightdiff_4096.json",
}

ORACLE_SPEC = "result/causal_direction/plan_causal_1024.json"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def unit_id(unit: dict[str, Any], layers: int, intermediate: int, heads: int) -> int:
    layer = int(unit["layer"])
    index = int(unit["index"])
    component = unit["component"]
    if layer < 0 or layer >= layers:
        raise ValueError(f"Bad layer {layer} in {unit}")
    if component == "channel":
        if index < 0 or index >= intermediate:
            raise ValueError(f"Bad channel index {index} in {unit}")
        return layer * intermediate + index
    if component == "head":
        if index < 0 or index >= heads:
            raise ValueError(f"Bad head index {index} in {unit}")
        return layers * intermediate + layer * heads + index
    raise ValueError(f"Unknown component {component!r} in {unit}")


def unit_from_id(idx: int, layers: int, intermediate: int, heads: int) -> dict[str, Any]:
    channel_count = layers * intermediate
    if idx < channel_count:
        return {
            "component": "channel",
            "layer": int(idx // intermediate),
            "index": int(idx % intermediate),
        }
    rem = idx - channel_count
    return {
        "component": "head",
        "layer": int(rem // heads),
        "index": int(rem % heads),
    }


def score_value(unit: dict[str, Any]) -> float:
    for key in (
        "attr_z",
        "diff_z",
        "weight_delta_z",
        "multiview_rrf",
        "normalized_score",
        "attribution",
        "weight_delta",
        "proxy_grad_mean",
    ):
        value = unit.get(key)
        if value is not None:
            try:
                value_f = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value_f):
                return value_f
    value = unit.get("score")
    if value is not None:
        try:
            value_f = -float(value)
        except (TypeError, ValueError):
            return 0.0
        if math.isfinite(value_f):
            return value_f
    return 0.0


def topk_overlap(y: np.ndarray, scores: np.ndarray, ks: tuple[int, ...]) -> dict[str, Any]:
    total_pos = int(y.sum())
    order = np.argsort(scores)[::-1]
    out: dict[str, Any] = {}
    for k in ks:
        kk = min(k, len(order))
        hit = int(y[order[:kk]].sum())
        out[str(k)] = {
            "hits": hit,
            "precision": hit / kk if kk else 0.0,
            "recall": hit / total_pos if total_pos else 0.0,
        }
    return out


def topk_overlap_on_subset(
    y: np.ndarray, scores: np.ndarray, indices: np.ndarray, ks: tuple[int, ...]
) -> dict[str, Any]:
    yy = y[indices]
    ss = scores[indices]
    return topk_overlap(yy, ss, ks)


def topk_overlap_from_ordered_ids(
    y: np.ndarray, ordered_ids: list[int], ks: tuple[int, ...]
) -> dict[str, Any]:
    total_pos = int(y.sum())
    out: dict[str, Any] = {}
    for k in ks:
        effective_k = min(k, len(ordered_ids))
        hit = int(y[np.array(ordered_ids[:effective_k], dtype=np.int64)].sum())
        out[str(k)] = {
            "effective_k": effective_k,
            "hits": hit,
            "precision": hit / effective_k if effective_k else 0.0,
            "recall": hit / total_pos if total_pos else 0.0,
        }
    return out


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def fit_logistic(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test = scaler.transform(X[test_idx])
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(X_train, y[train_idx])
    test_scores = clf.predict_proba(X_test)[:, 1]
    full_scores = clf.predict_proba(scaler.transform(X))[:, 1]
    return {
        "name": "logistic_balanced",
        "test_average_precision": float(average_precision_score(y[test_idx], test_scores)),
        "test_roc_auc": safe_auc(y[test_idx], test_scores),
        "test_topk": topk_overlap_on_subset(
            y, full_scores, test_idx, (64, 128, 256, 512, 1024)
        ),
        "in_sample_average_precision": float(average_precision_score(y, full_scores)),
        "in_sample_roc_auc": safe_auc(y, full_scores),
        "in_sample_topk": topk_overlap(y, full_scores, (64, 128, 256, 512, 1024, 2048)),
        "coef": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "scores": full_scores,
    }


def fit_logistic_all(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    clf.fit(X_scaled, y)
    scores = clf.predict_proba(X_scaled)[:, 1]
    return {
        "name": "logistic_fit_all_oracle_upper_bound",
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": safe_auc(y, scores),
        "topk": topk_overlap(y, scores, (64, 128, 256, 512, 1024, 2048, 4096)),
        "coef": clf.coef_[0].astype(float).tolist(),
        "intercept": float(clf.intercept_[0]),
        "scores": scores,
    }


def fit_hgb(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> dict[str, Any]:
    pos = max(1, int(y[train_idx].sum()))
    neg = max(1, len(train_idx) - pos)
    weights = np.ones(len(train_idx), dtype=np.float32)
    weights[y[train_idx] == 1] = neg / pos
    clf = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=13,
    )
    clf.fit(X[train_idx], y[train_idx], sample_weight=weights)
    test_scores = clf.predict_proba(X[test_idx])[:, 1]
    full_scores = clf.predict_proba(X)[:, 1]
    return {
        "name": "hist_gradient_boosting_balanced",
        "test_average_precision": float(average_precision_score(y[test_idx], test_scores)),
        "test_roc_auc": safe_auc(y[test_idx], test_scores),
        "test_topk": topk_overlap_on_subset(
            y, full_scores, test_idx, (64, 128, 256, 512, 1024)
        ),
        "in_sample_average_precision": float(average_precision_score(y, full_scores)),
        "in_sample_roc_auc": safe_auc(y, full_scores),
        "in_sample_topk": topk_overlap(y, full_scores, (64, 128, 256, 512, 1024, 2048)),
        "scores": full_scores,
    }


def fit_hgb_all(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    pos = max(1, int(y.sum()))
    neg = max(1, len(y) - pos)
    weights = np.ones(len(y), dtype=np.float32)
    weights[y == 1] = neg / pos
    clf = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=0.1,
        random_state=17,
    )
    clf.fit(X, y, sample_weight=weights)
    scores = clf.predict_proba(X)[:, 1]
    return {
        "name": "hist_gradient_boosting_fit_all_oracle_upper_bound",
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": safe_auc(y, scores),
        "topk": topk_overlap(y, scores, (64, 128, 256, 512, 1024, 2048, 4096)),
        "scores": scores,
    }


def feature_importance_by_permutation(
    X: np.ndarray,
    y: np.ndarray,
    base_scores: np.ndarray,
    feature_names: list[str],
    rng: np.random.Generator,
    sample_size: int = 120_000,
) -> list[dict[str, Any]]:
    indices = np.arange(len(y))
    if len(indices) > sample_size:
        pos_idx = indices[y == 1]
        neg_idx = indices[y == 0]
        keep_neg = rng.choice(
            neg_idx, size=min(sample_size - len(pos_idx), len(neg_idx)), replace=False
        )
        indices = np.concatenate([pos_idx, keep_neg])
    base_ap = average_precision_score(y[indices], base_scores[indices])
    out: list[dict[str, Any]] = []
    for j, name in enumerate(feature_names):
        shuffled = X[indices].copy()
        rng.shuffle(shuffled[:, j])
        # This helper is intentionally model-agnostic only for linear baseline.
        # The actual coefficient table below is more reliable for sign.
        corr_before = abs(np.corrcoef(X[indices, j], base_scores[indices])[0, 1])
        corr_after = abs(np.corrcoef(shuffled[:, j], base_scores[indices])[0, 1])
        if not math.isfinite(corr_before):
            corr_before = 0.0
        if not math.isfinite(corr_after):
            corr_after = 0.0
        out.append(
            {
                "feature": name,
                "base_ap": float(base_ap),
                "abs_corr_drop": float(corr_before - corr_after),
            }
        )
    return sorted(out, key=lambda x: x["abs_corr_drop"], reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_ROOT / "result/oracle_separability",
    )
    parser.add_argument("--oracle", type=str, default=ORACLE_SPEC)
    args = parser.parse_args()

    cfg = load_json(args.config)
    layers = int(cfg["num_hidden_layers"])
    heads = int(cfg["num_attention_heads"])
    intermediate = int(cfg["intermediate_size"])
    total_units = layers * intermediate + layers * heads
    channel_count = layers * intermediate

    args.out_dir.mkdir(parents=True, exist_ok=True)

    oracle_plan = load_json(args.root / args.oracle)
    oracle_ids = [
        unit_id(unit, layers, intermediate, heads) for unit in oracle_plan["to_prune"]
    ]
    y = np.zeros(total_units, dtype=np.int8)
    y[oracle_ids] = 1

    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    unsupervised: dict[str, Any] = {}
    present_columns: list[np.ndarray] = []
    rrf_columns: list[np.ndarray] = []

    for name, rel in PLAN_SPECS.items():
        path = args.root / rel
        if not path.exists():
            continue
        plan = load_json(path)
        units = plan.get("to_prune", [])
        present = np.zeros(total_units, dtype=np.float32)
        rrf = np.zeros(total_units, dtype=np.float32)
        pct = np.zeros(total_units, dtype=np.float32)
        score = np.zeros(total_units, dtype=np.float32)
        raw_values = np.array([score_value(unit) for unit in units], dtype=np.float32)
        if len(raw_values) > 1 and float(raw_values.std()) > 1e-12:
            norm_values = (raw_values - raw_values.mean()) / raw_values.std()
        else:
            norm_values = np.zeros_like(raw_values)
        for rank, unit in enumerate(units, start=1):
            idx = unit_id(unit, layers, intermediate, heads)
            present[idx] = 1.0
            rrf[idx] = 1.0 / (rank + 60.0)
            pct[idx] = 1.0 - ((rank - 1.0) / max(1.0, len(units) - 1.0))
            score[idx] = float(norm_values[rank - 1])
        feature_columns.extend([present, rrf, pct, score])
        feature_names.extend(
            [
                f"{name}:present",
                f"{name}:rrf",
                f"{name}:rank_pct",
                f"{name}:score_z",
            ]
        )
        present_columns.append(present)
        rrf_columns.append(rrf)
        ordered_ids = [unit_id(unit, layers, intermediate, heads) for unit in units]
        unsupervised[name] = {
            "path": str(path),
            "n_units": len(units),
            "topk_oracle_overlap": topk_overlap_from_ordered_ids(
                y, ordered_ids, (64, 128, 256, 512, 1024, 2048, 4096)
            ),
        }

    if not feature_columns:
        raise RuntimeError("No trigger-free plans were found.")

    support = np.sum(np.stack(present_columns, axis=1), axis=1).astype(np.float32)
    best_rrf = np.maximum.reduce(rrf_columns)
    layer_feature = np.zeros(total_units, dtype=np.float32)
    component_feature = np.zeros(total_units, dtype=np.float32)
    for idx in range(total_units):
        if idx < channel_count:
            layer_feature[idx] = (idx // intermediate) / max(1, layers - 1)
            component_feature[idx] = 0.0
        else:
            rem = idx - channel_count
            layer_feature[idx] = (rem // heads) / max(1, layers - 1)
            component_feature[idx] = 1.0

    feature_columns.extend([support, best_rrf, layer_feature, component_feature])
    feature_names.extend(["support_count", "best_rrf", "layer_norm", "is_head"])
    X = np.stack(feature_columns, axis=1).astype(np.float32)

    baseline_rate = float(y.mean())
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=13)
    train_idx, test_idx = next(splitter.split(X, y))

    logistic = fit_logistic(X, y, train_idx, test_idx)
    hgb = fit_hgb(X, y, train_idx, test_idx)
    logistic_all = fit_logistic_all(X, y)
    hgb_all = fit_hgb_all(X, y)

    logistic_scores = logistic.pop("scores")
    hgb_scores = hgb.pop("scores")
    logistic_all_scores = logistic_all.pop("scores")
    hgb_all_scores = hgb_all.pop("scores")

    coef_rows = sorted(
        [
            {"feature": name, "coef": float(coef)}
            for name, coef in zip(feature_names, logistic["coef"], strict=True)
        ],
        key=lambda row: abs(row["coef"]),
        reverse=True,
    )
    logistic["top_abs_coefficients"] = coef_rows[:30]
    logistic.pop("coef")

    coef_all_rows = sorted(
        [
            {"feature": name, "coef": float(coef)}
            for name, coef in zip(feature_names, logistic_all["coef"], strict=True)
        ],
        key=lambda row: abs(row["coef"]),
        reverse=True,
    )
    logistic_all["top_abs_coefficients"] = coef_all_rows[:30]
    logistic_all.pop("coef")

    learned_top = {
        "logistic_splitfit_top1024": [
            unit_from_id(int(idx), layers, intermediate, heads)
            for idx in np.argsort(logistic_scores)[::-1][:1024]
        ],
        "hgb_splitfit_top1024": [
            unit_from_id(int(idx), layers, intermediate, heads)
            for idx in np.argsort(hgb_scores)[::-1][:1024]
        ],
        "logistic_fitall_top1024": [
            unit_from_id(int(idx), layers, intermediate, heads)
            for idx in np.argsort(logistic_all_scores)[::-1][:1024]
        ],
        "hgb_fitall_top1024": [
            unit_from_id(int(idx), layers, intermediate, heads)
            for idx in np.argsort(hgb_all_scores)[::-1][:1024]
        ],
    }

    rng = np.random.default_rng(13)
    importance_proxy = feature_importance_by_permutation(
        X, y, logistic_scores, feature_names, rng
    )[:30]

    result = {
        "note": (
            "Offline diagnostic only: oracle labels are trigger-aware and are not "
            "available in a trigger-free deployment."
        ),
        "model_config": {
            "layers": layers,
            "attention_heads": heads,
            "intermediate_size": intermediate,
            "total_units": total_units,
        },
        "oracle": {
            "path": str(args.root / args.oracle),
            "positives": int(y.sum()),
            "positive_rate": baseline_rate,
        },
        "features": feature_names,
        "unsupervised_rankings": unsupervised,
        "models": {
            "logistic_balanced": logistic,
            "hist_gradient_boosting_balanced": hgb,
            "logistic_fit_all_oracle_upper_bound": logistic_all,
            "hist_gradient_boosting_fit_all_oracle_upper_bound": hgb_all,
        },
        "logistic_importance_proxy": importance_proxy,
        "learned_top1024": learned_top,
    }

    json_path = args.out_dir / "oracle_separability_diagnostic.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_path = args.out_dir / "VERDICT_oracle_separability.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Oracle Separability Diagnostic\n\n")
        f.write(
            "This is an offline diagnostic only. It uses trigger-aware oracle labels, "
            "so supervised numbers are upper bounds rather than deployable defenses.\n\n"
        )
        f.write(f"- Universe: {total_units} units\n")
        f.write(f"- Oracle positives: {int(y.sum())}\n")
        f.write(f"- Random AP baseline: {baseline_rate:.6f}\n\n")
        f.write("## Unsupervised Top-1024 Oracle Hits\n\n")
        for name, data in sorted(unsupervised.items()):
            hit = data["topk_oracle_overlap"]["1024"]["hits"]
            rec = data["topk_oracle_overlap"]["1024"]["recall"]
            f.write(f"- {name}: {hit}/1024 recall={rec:.4f}\n")
        f.write("\n## Supervised Upper Bounds\n\n")
        for model_name in ("logistic_balanced", "hist_gradient_boosting_balanced"):
            data = result["models"][model_name]
            test_hit = data["test_topk"]["1024"]["hits"]
            test_rec = data["test_topk"]["1024"]["recall"]
            full_hit = data["in_sample_topk"]["1024"]["hits"]
            full_rec = data["in_sample_topk"]["1024"]["recall"]
            f.write(
                f"- {model_name}: test AP={data['test_average_precision']:.6f}, "
                f"test AUC={data['test_roc_auc']:.4f}, "
                f"test top1024 hits={test_hit} recall={test_rec:.4f}; "
                f"in-sample top1024 hits={full_hit} recall={full_rec:.4f}\n"
            )
        f.write("\n## Oracle Fit-All Upper Bounds\n\n")
        for model_name in (
            "logistic_fit_all_oracle_upper_bound",
            "hist_gradient_boosting_fit_all_oracle_upper_bound",
        ):
            data = result["models"][model_name]
            hit = data["topk"]["1024"]["hits"]
            rec = data["topk"]["1024"]["recall"]
            f.write(
                f"- {model_name}: AP={data['average_precision']:.6f}, "
                f"AUC={data['roc_auc']:.4f}, top1024 hits={hit} recall={rec:.4f}\n"
            )
        f.write("\n## Logistic Top Coefficients\n\n")
        for row in result["models"]["logistic_balanced"]["top_abs_coefficients"][:15]:
            f.write(f"- {row['feature']}: {row['coef']:.4f}\n")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(
        "Logistic test AP:",
        f"{result['models']['logistic_balanced']['test_average_precision']:.6f}",
    )
    print(
        "HGB test AP:",
        f"{result['models']['hist_gradient_boosting_balanced']['test_average_precision']:.6f}",
    )


if __name__ == "__main__":
    main()
