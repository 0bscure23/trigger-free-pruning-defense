#!/usr/bin/env python3
"""Stage 2-3: score units with the perturbation proxy and apply structured pruning."""

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
    apply_structured_prune,
    collect_unit_scores,
    load_backdoorllm_model_and_tokenizer,
    now_ts,
    read_prompts,
    resolve_run_dir,
    save_model_and_tokenizer_safe,
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
    parser.add_argument(
        "--protect-safe-jsonl",
        type=Path,
        default=None,
        help="Optional harmful-no-trigger split whose gradients are used as a protective scoring term",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--alpha-safe",
        type=float,
        default=0.0,
        help="Weight of the harmful-no-trigger protective gradient term in the score",
    )
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--score-samples", type=int, default=8, help="Number of clean samples used for score estimation")
    parser.add_argument("--proxy-epsilon", type=float, default=0.1, help="FGSM epsilon used to build proxy gradients")
    parser.add_argument("--kappa", type=float, default=0.0, help="Prune units with score <= kappa")
    parser.add_argument("--max-prune-units", type=int, default=0, help="Maximum number of units to prune; 0 means uncapped")
    parser.add_argument("--score-only", action="store_true", help="Only compute unit scores; skip pruning and model saving")
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
    args.run_dir = resolve_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.clean_jsonl.exists():
        raise FileNotFoundError(f"Missing clean JSONL: {args.clean_jsonl}")
    if args.protect_safe_jsonl is not None and not args.protect_safe_jsonl.exists():
        raise FileNotFoundError(f"Missing protect-safe JSONL: {args.protect_safe_jsonl}")
    if args.proxy_epsilon <= 0:
        raise ValueError("--proxy-epsilon must be > 0")
    if args.score_samples <= 0:
        raise ValueError("--score-samples must be > 0")
    if int(args.min_prune_layer) < 0:
        raise ValueError("--min-prune-layer must be >= 0")
    if float(args.alpha_safe) < 0:
        raise ValueError("--alpha-safe must be >= 0")
    if float(args.alpha_safe) > 0 and args.protect_safe_jsonl is None:
        raise ValueError("--alpha-safe > 0 requires --protect-safe-jsonl")

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
    num_key_value_heads = int(getattr(model.config, "num_key_value_heads", 0) or 0) or None
    intermediate_size = int(getattr(model.config, "intermediate_size", 0) or 0)
    head_info = pruner._infer_llama_head_dim(hidden_size)
    if head_info is None:
        raise RuntimeError("Cannot infer attention head dimension from model config")
    num_heads, head_dim = head_info

    clean_prompts = read_prompts(args.clean_jsonl)
    protect_safe_prompts = read_prompts(args.protect_safe_jsonl) if args.protect_safe_jsonl is not None else None
    scores = collect_unit_scores(
        model=model,
        pruner=pruner,
        tokenizer=tokenizer,
        clean_prompts=clean_prompts,
        protect_safe_prompts=protect_safe_prompts,
        max_length=args.max_length,
        prompt_template=str(args.prompt_template),
        head_dim=head_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        score_samples=args.score_samples,
        alpha=args.alpha,
        beta=args.beta,
        alpha_safe=args.alpha_safe,
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

    if not args.score_only:
        apply_structured_prune(
            pruner,
            to_prune=to_prune,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
        )

    (args.run_dir / "unit_scores.json").write_text(
        json.dumps(
            {
                "score_config": {
                    "alpha": float(args.alpha),
                    "beta": float(args.beta),
                    "alpha_safe": float(args.alpha_safe),
                    "protect_safe_jsonl": None if args.protect_safe_jsonl is None else str(args.protect_safe_jsonl),
                    "proxy_epsilon": float(args.proxy_epsilon),
                    "num_key_value_heads": None if num_key_value_heads is None else int(num_key_value_heads),
                    "score_formula": (
                        "alpha * (clean_grad_mean + alpha_safe * safe_grad_mean) - beta * abs(proxy_grad_mean * cosine)"
                        if float(args.alpha_safe) > 0
                        else "alpha * clean_grad_mean - beta * abs(proxy_grad_mean * cosine)"
                    ),
                },
                "scores": [score.__dict__ for score in scores],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (args.run_dir / "pruning_plan.json").write_text(
        json.dumps(
            {
                "timestamp": now_ts(),
                "proxy_epsilon": float(args.proxy_epsilon),
                "proxy_type": "perturbed_proxy_grad",
                "proxy_explanation": "gradient of perturbed sample loss, where the perturbation is generated from consistency-based FGSM on clean inputs",
                "score_formula": (
                    "alpha * (clean_grad_mean + alpha_safe * safe_grad_mean) - beta * abs(proxy_grad_mean * cosine)"
                    if float(args.alpha_safe) > 0
                    else "alpha * clean_grad_mean - beta * abs(proxy_grad_mean * cosine)"
                ),
                "model_path_effective": effective_model_path,
                "kappa": float(args.kappa),
                "alpha_safe": float(args.alpha_safe),
                "protect_safe_jsonl": None if args.protect_safe_jsonl is None else str(args.protect_safe_jsonl),
                "num_key_value_heads": None if num_key_value_heads is None else int(num_key_value_heads),
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

    if not args.score_only:
        output_dir = args.run_dir / "pruned_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        save_model_and_tokenizer_safe(model, tokenizer, str(output_dir))
        print(f"Wrote pruned model to {output_dir}")

    print(f"Wrote unit scores to {args.run_dir / 'unit_scores.json'}")
    print(f"Wrote pruning plan to {args.run_dir / 'pruning_plan.json'}")


if __name__ == "__main__":
    main()
