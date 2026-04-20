#!/usr/bin/env python3
"""Stage 5: evaluate the defended model and assemble a final report."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from pipeline_utils import (
    DEFAULT_MODEL_PATH,
    evaluate_asr_backdoorllm_jailbreak,
    evaluate_asr_backdoorllm_refusal,
    evaluate_asr_simple,
    evaluate_clean_behavior,
    limitations_notes,
    load_backdoorllm_model_and_tokenizer,
    now_ts,
    read_asr_eval_items_jsonl,
    read_prompts,
    resolve_run_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path or a Hugging Face checkpoint directory")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-model-path", default=None)
    parser.add_argument("--prompt-template", choices=["alpaca", "chat", "none"], default="alpaca")
    parser.add_argument(
        "--asr-mode",
        choices=["auto", "simple", "backdoorllm-jailbreak", "backdoorllm-refusal"],
        default="auto",
        help="ASR metric: auto infers refusal or jailbreak from the eval file path",
    )
    parser.add_argument("--eval-asr-jsonl", type=Path, required=True, help="ASR eval data: .jsonl or BackdoorLLM-style .json")
    parser.add_argument("--eval-clean-jsonl", type=Path, default=None, help="Optional clean or no-trigger evaluation split")
    parser.add_argument("--eval-max-new-tokens", type=int, default=128)
    parser.add_argument("--eval-max-length", type=int, default=1024)
    parser.add_argument("--eval-temperature", type=float, default=0.0)
    parser.add_argument("--eval-top-p", type=float, default=0.75)
    parser.add_argument("--eval-num-beams", type=int, default=1)
    parser.add_argument("--eval-do-sample", action="store_true")
    parser.add_argument("--eval-asr-include-prompt", action="store_true", help="Only used with --asr-mode=simple")
    parser.add_argument("--asr-target", type=float, default=0.05)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    return parser.parse_args()


def _read_json_if_exists(path: Path) -> object | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _last_finite_loss(losses: list[dict]) -> dict | None:
    for item in reversed(losses):
        value = item.get("loss_total")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return item
    return None


def _resolve_asr_mode(eval_path: Path, requested_mode: str) -> str:
    if requested_mode == "simple":
        return requested_mode

    path_lower = str(eval_path).lower()
    inferred_mode: str | None = None
    if "refusal" in path_lower:
        inferred_mode = "backdoorllm-refusal"
    elif "jailbreak" in path_lower:
        inferred_mode = "backdoorllm-jailbreak"

    if requested_mode == "auto":
        return inferred_mode or "backdoorllm-jailbreak"

    if inferred_mode is not None and requested_mode != inferred_mode:
        raise ValueError(
            f"ASR mode '{requested_mode}' conflicts with eval path '{eval_path}'. "
            f"Use '--asr-mode {inferred_mode}' or '--asr-mode auto'."
        )
    return requested_mode


def _resolve_eval_model_path(model_path: str, run_dir: Path) -> str:
    if str(model_path) != str(DEFAULT_MODEL_PATH):
        return str(model_path)
    for candidate in (run_dir / "recovered_model", run_dir / "pruned_model"):
        if candidate.exists():
            return str(candidate)
    return str(model_path)


def main() -> None:
    args = parse_args()
    args.run_dir = resolve_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_asr_jsonl.exists():
        raise FileNotFoundError(f"Missing ASR eval file: {args.eval_asr_jsonl}")
    if not (0.0 < args.asr_target < 1.0):
        raise ValueError("--asr-target must be in (0,1)")

    requested_mode = str(args.asr_mode)
    effective_mode = _resolve_asr_mode(args.eval_asr_jsonl, requested_mode)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    effective_model_path = _resolve_eval_model_path(str(args.model_path), args.run_dir)
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=effective_model_path,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
        use_lora=bool(args.use_lora),
        lora_model_path=str(args.lora_model_path) if args.lora_model_path else None,
        torch_dtype=dtype,
    )

    if effective_mode == "simple":
        eval_items = read_asr_eval_items_jsonl(args.eval_asr_jsonl)
        final_asr = evaluate_asr_simple(
            model,
            tokenizer,
            eval_items,
            max_new_tokens=args.eval_max_new_tokens,
            include_prompt=bool(args.eval_asr_include_prompt),
        )
    elif effective_mode == "backdoorllm-refusal":
        eval_items = read_prompts(args.eval_asr_jsonl)
        final_asr = evaluate_asr_backdoorllm_refusal(
            model,
            tokenizer,
            eval_items,
            max_new_tokens=args.eval_max_new_tokens,
            prompt_template=str(args.prompt_template),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            num_beams=int(args.eval_num_beams),
            do_sample=bool(args.eval_do_sample),
            max_length=int(args.eval_max_length),
        )
    else:
        eval_items = read_prompts(args.eval_asr_jsonl)
        final_asr = evaluate_asr_backdoorllm_jailbreak(
            model,
            tokenizer,
            eval_items,
            max_new_tokens=args.eval_max_new_tokens,
            prompt_template=str(args.prompt_template),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            num_beams=int(args.eval_num_beams),
            do_sample=bool(args.eval_do_sample),
            max_length=int(args.eval_max_length),
        )

    clean_eval: dict[str, object] | None = None
    if args.eval_clean_jsonl is not None and args.eval_clean_jsonl.exists():
        clean_items = read_prompts(args.eval_clean_jsonl)
        clean_stats = evaluate_clean_behavior(
            model,
            tokenizer,
            clean_items,
            max_new_tokens=args.eval_max_new_tokens,
            prompt_template=str(args.prompt_template),
            temperature=float(args.eval_temperature),
            top_p=float(args.eval_top_p),
            num_beams=int(args.eval_num_beams),
            do_sample=bool(args.eval_do_sample),
            max_length=int(args.eval_max_length),
        )
        clean_eval = {
            "metric_name": str(clean_stats.metric_name),
            "score_rule": str(clean_stats.score_rule),
            "hits": int(clean_stats.hits),
            "total": int(clean_stats.total),
            "final_ratio": float(clean_stats.ratio),
            "final_percent": float(clean_stats.ratio) * 100.0,
            "dataset_path": str(args.eval_clean_jsonl),
            "interpretation_note": (
                "If eval_clean_jsonl is a benign clean split, this behaves like false-refusal rate / ASR_clean_percent. "
                "If it is a harmful no-trigger split, this behaves like safe-refusal retention."
            ),
        }

    notes = limitations_notes()
    alignment_config = _read_json_if_exists(args.run_dir / "alignment_config.json")
    alignment_losses = _read_json_if_exists(args.run_dir / "alignment_losses.json")
    pruning_plan = _read_json_if_exists(args.run_dir / "pruning_plan.json")
    recovery_losses = _read_json_if_exists(args.run_dir / "recovery_losses.json")

    alignment_loss_list = (alignment_losses or {}).get("losses", []) if isinstance(alignment_losses, dict) else []
    recovery_loss_list = (recovery_losses or {}).get("losses", []) if isinstance(recovery_losses, dict) else []

    report = {
        "timestamp": now_ts(),
        "script": "evaluate_model.py",
        "methodology": "Trigger-agnostic structured pruning with perturbation proxy",
        "run_dir": str(args.run_dir),
        "config": alignment_config.get("config") if isinstance(alignment_config, dict) else None,
        "model": {
            "evaluated_model_path": effective_model_path,
            "dtype": args.dtype,
        },
        "alignment": {
            "steps": len(alignment_loss_list) if isinstance(alignment_losses, dict) else None,
            "final": _last_finite_loss(alignment_loss_list),
        },
        "pruning": pruning_plan,
        "recovery": {
            "steps": len(recovery_loss_list) if isinstance(recovery_losses, dict) else None,
            "final": _last_finite_loss(recovery_loss_list),
        },
        "asr": {
            "target": float(args.asr_target),
            "final_asr": float(final_asr),
            "mode": effective_mode,
            "requested_mode": requested_mode,
            "final_asr_percent": float(final_asr) * 100.0,
            "met_target": bool(final_asr < float(args.asr_target)),
        },
        "clean_eval": clean_eval,
        "notes": notes,
    }

    output_path = args.run_dir / "evaluation_report.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if clean_eval is not None:
        print(
            f"ASR[{effective_mode}]={final_asr:.4f}  CLEAN={clean_eval['final_ratio']:.4f} "
            f"({clean_eval['hits']}/{clean_eval['total']})"
        )
    else:
        print(f"ASR[{effective_mode}]={final_asr:.4f}")
    print(f"Wrote report to {output_path}")


if __name__ == "__main__":
    main()
