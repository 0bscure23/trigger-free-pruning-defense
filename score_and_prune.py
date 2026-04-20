#!/usr/bin/env python3
"""Stage 2-3: score units with the perturbation proxy and apply structured pruning."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from pipeline_utils import (
    DEFAULT_MODEL_PATH,
    apply_structured_prune,
    collect_unit_scores,
    load_backdoorllm_model_and_tokenizer,
    now_ts,
    read_prompts,
)
from pruning_backend import BaseSafetyPruner


def _resolve_scoring_model_path(model_path: str, run_dir: Path) -> str:
    candidate = run_dir / "stage1_model"
    if str(model_path) == str(DEFAULT_MODEL_PATH) and candidate.exists():
        return str(candidate)
    return str(model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path or a Hugging Face checkpoint directory")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-model-path", default=None)
    parser.add_argument("--prompt-template", choices=["alpaca", "chat", "none"], default="alpaca")
    parser.add_argument("--clean-jsonl", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--score-samples", type=int, default=8, help="Number of clean samples used for score estimation")
    parser.add_argument("--proxy-epsilon", type=float, default=0.1, help="FGSM epsilon used to build proxy gradients")
    parser.add_argument("--kappa", type=float, default=0.0, help="Prune units with score <= kappa")
    parser.add_argument("--max-prune-units", type=int, default=0, help="Maximum number of units to prune; 0 means uncapped")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.clean_jsonl.exists():
        raise FileNotFoundError(f"Missing clean JSONL: {args.clean_jsonl}")
    if args.proxy_epsilon <= 0:
        raise ValueError("--proxy-epsilon must be > 0")
    if args.score_samples <= 0:
        raise ValueError("--score-samples must be > 0")
    if int(args.min_prune_layer) < 0:
        raise ValueError("--min-prune-layer must be >= 0")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    effective_model_path = _resolve_scoring_model_path(str(args.model_path), args.run_dir)
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=effective_model_path,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
        use_lora=bool(args.use_lora),
        lora_model_path=str(args.lora_model_path) if args.lora_model_path else None,
        torch_dtype=dtype,
    )
    pruner = BaseSafetyPruner(model)

    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    intermediate_size = int(getattr(model.config, "intermediate_size", 0) or 0)
    head_info = pruner._infer_llama_head_dim(hidden_size)
    if head_info is None:
        raise RuntimeError("Cannot infer attention head dimension from model config")
    num_heads, head_dim = head_info

    clean_prompts = read_prompts(args.clean_jsonl)
    scores = collect_unit_scores(
        model=model,
        pruner=pruner,
        tokenizer=tokenizer,
        clean_prompts=clean_prompts,
        max_length=args.max_length,
        prompt_template=str(args.prompt_template),
        head_dim=head_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        intermediate_size=intermediate_size,
        score_samples=args.score_samples,
        alpha=args.alpha,
        beta=args.beta,
        eps=args.eps,
        proxy_epsilon=args.proxy_epsilon,
    )

    to_prune = [score for score in scores if score.score <= float(args.kappa)]
    if args.max_score_to_prune is not None:
        to_prune = [score for score in to_prune if score.score <= float(args.max_score_to_prune)]
    if int(args.min_prune_layer) > 0:
        to_prune = [score for score in to_prune if int(score.layer) >= int(args.min_prune_layer)]
    if args.max_prune_units > 0:
        to_prune = to_prune[: int(args.max_prune_units)]

    apply_structured_prune(pruner, to_prune=to_prune, head_dim=head_dim)

    (args.run_dir / "unit_scores.json").write_text(
        json.dumps({"scores": [score.__dict__ for score in scores]}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
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
                "pruned_heads": int(sum(1 for score in to_prune if score.component == "head")),
                "pruned_channels": int(sum(1 for score in to_prune if score.component == "channel")),
                "to_prune": [unit.__dict__ for unit in to_prune],
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

    print(f"Wrote unit scores to {args.run_dir / 'unit_scores.json'}")
    print(f"Wrote pruning plan to {args.run_dir / 'pruning_plan.json'}")
    print(f"Wrote pruned model to {output_dir}")


if __name__ == "__main__":
    main()
