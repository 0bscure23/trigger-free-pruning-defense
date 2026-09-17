#!/usr/bin/env python3
"""Apply per-(layer, component) score normalization to an existing unit_scores.json.

This avoids re-running expensive gradient collection. Outputs normalized
unit_scores.json, pruning_plan.json, and optionally applies pruning.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from pipeline_utils import (
    DEFAULT_MODEL_PATH,
    UnitScore,
    apply_structured_prune,
    load_backdoorllm_model_and_tokenizer,
    normalize_unit_scores,
    now_ts,
    resolve_run_dir,
    save_model_and_tokenizer_safe,
)
from pruning_backend import BaseSafetyPruner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--scores-json", type=Path, required=True, help="Existing unit_scores.json with raw scores")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--score-normalization", choices=["layer_component_z", "layer_component_rank"], required=True)
    parser.add_argument("--norm-score-to-prune", type=float, default=0.0)
    parser.add_argument("--max-score-to-prune", type=float, default=None)
    parser.add_argument("--min-prune-layer", type=int, default=2)
    parser.add_argument("--max-prune-units", type=int, default=320)
    parser.add_argument("--kappa", type=float, default=1e9)
    parser.add_argument("--plan-only", action="store_true", help="Generate normalized plan without loading model")
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = resolve_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.scores_json.exists():
        raise FileNotFoundError(f"Missing scores file: {args.scores_json}")

    raw = json.loads(args.scores_json.read_text(encoding="utf-8"))
    raw_scores = raw["scores"]
    raw_config = raw.get("score_config", {})

    # Deserialize
    def _deser(item: dict) -> UnitScore:
        return UnitScore(
            component=str(item["component"]),
            layer=int(item["layer"]),
            index=int(item["index"]),
            clean_grad_mean=float(item["clean_grad_mean"]),
            proxy_grad_mean=float(item["proxy_grad_mean"]),
            cosine=float(item["cosine"]),
            score=float(item["score"]),
            safe_grad_mean=float(item.get("safe_grad_mean", 0.0) or 0.0),
            protect_grad_mean=float(item.get("protect_grad_mean", item.get("clean_grad_mean", 0.0)) or 0.0),
            harm_proxy_grad_mean=float(item.get("harm_proxy_grad_mean", 0.0) or 0.0),
            harm_proxy_cosine=float(item.get("harm_proxy_cosine", 0.0) or 0.0),
            clean_proxy_penalty=float(item.get("clean_proxy_penalty", 0.0) or 0.0),
            harm_proxy_penalty=float(item.get("harm_proxy_penalty", 0.0) or 0.0),
        )

    scores = [_deser(item) for item in raw_scores if isinstance(item, dict)]

    # Apply normalization
    scores, norm_stats = normalize_unit_scores(scores, normalization=args.score_normalization)

    # Replace score with normalized_score for pruning
    for s in scores:
        if s.normalized_score is not None:
            s.score = float(s.normalized_score)

    # Filter
    to_prune = [s for s in scores if s.score <= float(args.kappa)]
    if args.max_score_to_prune is not None:
        to_prune = [s for s in to_prune if s.score <= float(args.max_score_to_prune)]
    to_prune = [s for s in to_prune if s.score <= float(args.norm_score_to_prune)]
    if int(args.min_prune_layer) > 0:
        to_prune = [s for s in to_prune if int(s.layer) >= int(args.min_prune_layer)]
    if args.max_prune_units > 0:
        to_prune = to_prune[: int(args.max_prune_units)]

    print(f"Normalized pruning: {len(to_prune)} units selected "
          f"({sum(1 for s in to_prune if s.component == 'head')} heads, "
          f"{sum(1 for s in to_prune if s.component == 'channel')} channels)")

    # Save unit_scores
    score_config_out = dict(raw_config)
    score_config_out["score_normalization"] = args.score_normalization
    score_config_out["normalization_stats"] = norm_stats
    (args.run_dir / "unit_scores.json").write_text(
        json.dumps(
            {"score_config": score_config_out, "scores": [s.__dict__ for s in scores]},
            indent=2, ensure_ascii=False,
        ) + "\n", encoding="utf-8",
    )

    # Save pruning plan
    plan = {
        "timestamp": now_ts(),
        "scores_source": str(args.scores_json),
        "score_normalization": args.score_normalization,
        "norm_score_to_prune": float(args.norm_score_to_prune),
        "max_score_to_prune": args.max_score_to_prune,
        "min_prune_layer": int(args.min_prune_layer),
        "max_prune_units": int(args.max_prune_units),
        "kappa": float(args.kappa),
        "proxy_type": raw_config.get("proxy_type", "unknown"),
        "score_formula": raw_config.get("score_formula", "unknown"),
        "pruned_total": int(len(to_prune)),
        "pruned_heads": int(sum(1 for s in to_prune if s.component == "head")),
        "pruned_channels": int(sum(1 for s in to_prune if s.component == "channel")),
        "normalization_stats": norm_stats,
        "to_prune": [s.__dict__ for s in to_prune],
    }
    (args.run_dir / "pruning_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )

    if args.plan_only:
        print("Plan-only mode: skipping model loading and pruning")
        return

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=str(args.model_path),
        tokenizer_path=None,
        use_lora=False,
        lora_model_path=None,
        torch_dtype=dtype,
        merge_lora=False,
    )
    pruner = BaseSafetyPruner(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    num_key_value_heads = int(getattr(model.config, "num_key_value_heads", 0) or 0) or None
    head_info = pruner._infer_llama_head_dim(hidden_size)
    if head_info is None:
        raise RuntimeError("Cannot infer attention head dimension from model config")
    _, head_dim = head_info

    apply_structured_prune(pruner, to_prune=to_prune, head_dim=head_dim, num_key_value_heads=num_key_value_heads)

    output_dir = args.run_dir / "pruned_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_and_tokenizer_safe(model, tokenizer, str(output_dir))
    print(f"Wrote pruned model to {output_dir}")


if __name__ == "__main__":
    main()
