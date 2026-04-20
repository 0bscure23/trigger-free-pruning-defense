#!/usr/bin/env python3
"""Apply structured pruning from a precomputed unit-score ranking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from pipeline_utils import DEFAULT_MODEL_PATH, UnitScore, apply_structured_prune, load_backdoorllm_model_and_tokenizer, now_ts
from pruning_backend import BaseSafetyPruner


def _resolve_pruning_model_path(model_path: str, run_dir: Path) -> str:
    candidate = run_dir / "stage1_model"
    if str(model_path) == str(DEFAULT_MODEL_PATH) and candidate.exists():
        return str(candidate)
    return str(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--scores-json", type=Path, required=True, help="Path to unit_scores.json")
    parser.add_argument("--kappa", type=float, default=1_000_000_000.0)
    parser.add_argument("--max-prune-units", type=int, required=True)
    parser.add_argument(
        "--max-score-to-prune",
        type=float,
        default=None,
        help="Optional hard cap: only keep units with score <= this value",
    )
    parser.add_argument(
        "--min-prune-layer",
        type=int,
        default=0,
        help="Only allow pruning units with layer >= this index",
    )
    parser.add_argument("--proxy-epsilon", type=float, default=0.1)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def _load_scores(path: Path) -> list[UnitScore]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw_scores = raw.get("scores") if isinstance(raw, dict) else None
    if not isinstance(raw_scores, list):
        raise ValueError(f"Invalid score file format: {path}")

    scores: list[UnitScore] = []
    for item in raw_scores:
        if not isinstance(item, dict):
            continue
        scores.append(
            UnitScore(
                component=str(item["component"]),
                layer=int(item["layer"]),
                index=int(item["index"]),
                clean_grad_mean=float(item["clean_grad_mean"]),
                proxy_grad_mean=float(item["proxy_grad_mean"]),
                cosine=float(item["cosine"]),
                score=float(item["score"]),
            )
        )
    scores.sort(key=lambda score: score.score)
    return scores


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.max_prune_units <= 0:
        raise ValueError("--max-prune-units must be > 0")
    if int(args.min_prune_layer) < 0:
        raise ValueError("--min-prune-layer must be >= 0")
    if not args.scores_json.exists():
        raise FileNotFoundError(f"Missing score file: {args.scores_json}")

    scores = _load_scores(args.scores_json)
    to_prune = [score for score in scores if score.score <= float(args.kappa)]
    if args.max_score_to_prune is not None:
        to_prune = [score for score in to_prune if score.score <= float(args.max_score_to_prune)]
    if int(args.min_prune_layer) > 0:
        to_prune = [score for score in to_prune if int(score.layer) >= int(args.min_prune_layer)]
    to_prune = to_prune[: int(args.max_prune_units)]

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    effective_model_path = _resolve_pruning_model_path(str(args.model_path), args.run_dir)
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=effective_model_path,
        tokenizer_path=None,
        use_lora=False,
        lora_model_path=None,
        torch_dtype=dtype,
    )
    pruner = BaseSafetyPruner(model)

    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_info = pruner._infer_llama_head_dim(hidden_size)
    if head_info is None:
        raise RuntimeError("Cannot infer attention head dimension from model config")
    _, head_dim = head_info

    apply_structured_prune(pruner, to_prune=to_prune, head_dim=head_dim)

    (args.run_dir / "pruning_plan.json").write_text(
        json.dumps(
            {
                "timestamp": now_ts(),
                "proxy_epsilon": float(args.proxy_epsilon),
                "proxy_type": "perturbed_proxy_grad",
                "proxy_explanation": "gradient of perturbed sample loss, where the perturbation is generated from consistency-based FGSM on clean inputs",
                "score_formula": "alpha * clean_grad_mean - beta * abs(proxy_grad_mean * cosine)",
                "model_path_effective": effective_model_path,
                "kappa": float(args.kappa),
                "max_score_to_prune": None if args.max_score_to_prune is None else float(args.max_score_to_prune),
                "min_prune_layer": int(args.min_prune_layer),
                "max_prune_units": int(args.max_prune_units),
                "pruned_total": int(len(to_prune)),
                "pruned_heads": int(sum(1 for item in to_prune if item.component == "head")),
                "pruned_channels": int(sum(1 for item in to_prune if item.component == "channel")),
                "to_prune": [item.__dict__ for item in to_prune],
                "scores_source": str(args.scores_json),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    output_dir = args.run_dir / "pruned_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"Wrote pruning plan to {args.run_dir / 'pruning_plan.json'}")
    print(f"Wrote pruned model to {output_dir}")


if __name__ == "__main__":
    main()
