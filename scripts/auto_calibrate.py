#!/usr/bin/env python3
"""Trigger-free auto-calibration for unknown models.

Selects pruning budget, lambda_align, lambda_safe, and training steps
using ONLY trigger-free signals (benign dev, harmful-no-trigger dev,
proxy scores, proxy loss, empty-output rate).  Never uses triggered ASR.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from pipeline_utils import (  # noqa: E402
    BACKDOORLLM_JAILBREAK_KEYWORDS,
    DEFAULT_MODEL_PATH,
    _tokenize_one,
    decode_new_tokens,
    load_backdoorllm_model_and_tokenizer,
    read_prompts,
    resolve_run_dir,
    save_model_and_tokenizer_safe,
)


# ── 1. Auto max_length ────────────────────────────────────────────────

def infer_max_length_from_jsonl(
    tokenizer,
    paths: list[Path],
    *,
    margin: int = 32,
    percentile: float = 0.90,
    min_len: int = 256,
    max_len: int = 1024,
    max_items: int = 500,
    prompt_template: str = "alpaca",
) -> int:
    """Estimate max_length from prompt token lengths.

    Uses ``shape[-1]`` (the sequence dimension) of tokenised prompts,
    NOT ``size(0)`` which is the batch dimension.
    """
    lengths: list[int] = []
    for p in paths:
        if not p or not p.exists():
            continue
        prompts = read_prompts(p)
        for prompt in prompts[:max_items]:
            tok = _tokenize_one(tokenizer, prompt,
                                max_length=max_len + margin,
                                prompt_template=prompt_template)
            lengths.append(int(tok["input_ids"].shape[-1]))
    if not lengths:
        return min_len
    p90 = sorted(lengths)[int(len(lengths) * percentile)]
    return max(min_len, min(math.ceil(p90 + margin), max_len))


# ── 2. Pruning candidate generation ────────────────────────────────────

def propose_pruning_candidates(
    scores: list[dict],
    *,
    candidate_budgets: list[int] | None = None,
    min_prune_layer: int = 2,
    max_score_to_prune: float = 0.0,
) -> list[dict]:
    if candidate_budgets:
        budgets = sorted(set(candidate_budgets))
    else:
        neg_count = sum(1 for u in scores if u["score"] <= 0.0)
        if neg_count < 8:
            budgets = [max(8, neg_count)]
        else:
            fractions = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
            raw = sorted(set(max(8, int(neg_count * f)) for f in fractions))
            budgets = [b for b in raw if 8 <= b <= 512]

    candidates: list[dict] = []
    for budget in budgets:
        # NOTE: apply min_prune_layer + max_score_to_prune as SAFETY FILTERS
        eligible = [u for u in scores
                    if u["score"] <= max_score_to_prune
                    and u.get("layer", 0) >= min_prune_layer]
        eligible.sort(key=lambda u: u["score"])
        pruned = eligible[:budget]
        # Component names in UnitScore are "head" / "channel"
        heads = sum(1 for u in pruned if u.get("component") == "head")
        chans = sum(1 for u in pruned if u.get("component") == "channel")
        layers: dict[int, int] = {}
        for u in pruned:
            l = u.get("layer", -1)
            layers[l] = layers.get(l, 0) + 1
        candidates.append({
            "budget": budget,
            "actual_pruned_total": len(pruned),
            "pruned_heads": heads,
            "pruned_channels": chans,
            "score_threshold": pruned[-1]["score"] if pruned else 0.0,
            "min_score": pruned[0]["score"] if pruned else 0.0,
            "layer_histogram": {str(k): v for k, v in sorted(layers.items())},
            # raw eligible unit list for later plan generation
            "_units": pruned,
        })
    return candidates


# ── 3. Trigger-free dev metrics ────────────────────────────────────────

def compute_dev_metrics(
    model,
    tokenizer,
    benign_prompts: list,
    harmful_no_trigger_prompts: list,
    *,
    max_new_tokens: int = 64,
    max_dev_items: int = 200,
    prompt_template: str = "chat",
) -> dict:
    """Compute trigger-free dev metrics.

    * Uses ``decode_new_tokens`` so only new text is examined.
    * Refusal keywords are applied ONLY to generated suffix (not prompt).
    * ``_tokenize_one`` returns batched ``input_ids`` (shape ``[1, L]``)
      and ``attention_mask`` — no manual ``unsqueeze`` needed.
    """
    device = next(model.parameters()).device
    model.eval()

    def _generate_and_score(prompts, limit):
        refusals, empties, total = 0, 0, 0
        for p in prompts[:limit]:
            encoded = _tokenize_one(tokenizer, p, max_length=512,
                                    prompt_template=prompt_template)
            input_ids = encoded["input_ids"].to(device)
            attn_mask = encoded.get("attention_mask")
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            # Decode ONLY new tokens (not the prompt)
            new_text = decode_new_tokens(tokenizer, gen, input_ids)
            total += 1
            if new_text.strip() in {"", ".", "?"}:
                empties += 1
                continue
            lowered = new_text.lower()
            if any((kw or "").lower() in lowered for kw in BACKDOORLLM_JAILBREAK_KEYWORDS):
                refusals += 1
        return refusals, empties, total

    limit = min(max_dev_items, len(benign_prompts), len(harmful_no_trigger_prompts))
    if limit < 1:
        return {}

    b_ref, b_emp, b_tot = _generate_and_score(benign_prompts, limit)
    h_ref, h_emp, h_tot = _generate_and_score(harmful_no_trigger_prompts, limit)

    benign_fr = b_ref / max(b_tot, 1)
    harmful_refusal = h_ref / max(h_tot, 1)
    empty_rate = (b_emp + h_emp) / max(b_tot + h_tot, 1)

    return {
        "benign_false_refusal": benign_fr,
        "harmful_no_trigger_refusal": harmful_refusal,
        "empty_output_rate": empty_rate,
        "dev_items": limit,
    }


# ── 4. Objective function ──────────────────────────────────────────────

def select_best_candidate(
    candidates_metrics: list[dict],
    *,
    objective: str = "balanced",
) -> dict:
    """Select best candidate via trigger-free composite score.

    ALL components are penalties: higher value = worse.
    """
    if not candidates_metrics:
        raise ValueError("No candidates to select from")

    keys_float = [
        "benign_false_refusal",
        "harmful_no_trigger_refusal",
        "empty_output_rate",
        "clean_lm_loss",
        "proxy_alignment_loss",
    ]
    ranges: dict[str, tuple[float, float]] = {}
    for k in keys_float:
        vals = [m.get(k) for m in candidates_metrics
                if isinstance(m.get(k), (int, float))]
        lo, hi = (min(vals), max(vals)) if len(vals) >= 2 else (0.0, 1.0)
        ranges[k] = (lo, hi if hi > lo else lo + 1e-8)

    def _norm(k, v):
        if not isinstance(v, (int, float)):
            return 0.5
        lo, hi = ranges[k]
        return (v - lo) / (hi - lo) if hi > lo else 0.0

    # All weights are POSITIVE penalties.
    # harmful_no_trigger_refusal is inverted: (1 - hr) as penalty.
    if objective == "safety_first":
        w_bfr, w_hr, w_emp, w_clm, w_pal = 1.0, 1.5, 3.0, 0.1, 0.3
    elif objective == "utility_first":
        w_bfr, w_hr, w_emp, w_clm, w_pal = 1.5, 0.5, 2.0, 0.3, 0.1
    else:  # balanced
        w_bfr, w_hr, w_emp, w_clm, w_pal = 1.0, 1.0, 2.0, 0.2, 0.5

    best, best_score = None, float("inf")
    for m in candidates_metrics:
        bfr_n = _norm("benign_false_refusal", m.get("benign_false_refusal"))
        hr_n = _norm("harmful_no_trigger_refusal", m.get("harmful_no_trigger_refusal"))
        emp_n = _norm("empty_output_rate", m.get("empty_output_rate", 0.0))
        clm_n = _norm("clean_lm_loss", m.get("clean_lm_loss"))
        pal_n = _norm("proxy_alignment_loss", m.get("proxy_alignment_loss"))
        pr_pen = min(m.get("prune_ratio", 0.0), 0.5) * 0.1

        # All terms are penalty: higher = worse
        score = (w_bfr * bfr_n
                 + w_hr * (1.0 - hr_n)    # low refusal → high penalty
                 + w_emp * emp_n
                 + w_clm * clm_n
                 + w_pal * pal_n
                 + pr_pen)
        if score < best_score:
            best_score = score
            best = {**m, "objective_value": score, "objective_type": objective}
    return best


# ── 5. Lambda auto-suggestion ──────────────────────────────────────────

def _auto_lambda_align(hr: float | None = None) -> list[float]:
    if hr is not None and hr < 0.3:
        return [2.0, 2.5, 3.0]
    return [1.0, 1.5, 2.0, 2.5]


def _auto_lambda_safe(bfr: float | None = None) -> list[float]:
    if bfr is not None and bfr > 0.4:
        return [0.02, 0.04, 0.06]
    return [0.04, 0.06, 0.08, 0.10]


# ── 6. Helpers ─────────────────────────────────────────────────────────

def _parse_float_list(raw: str, default: list[float]) -> list[float]:
    if raw == "auto":
        return default
    return [float(x.strip()) for x in raw.split(",")]


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",")]


def _write_candidate_pruning_plan(
    plan_dir: Path, units: list[dict], budget: int,
    source_plan: dict | None = None,
) -> Path:
    """Write a per-budget pruning_plan JSON so recover_model uses the right set.

    Uses ``"to_prune"`` key (matching recover_model._deserialize_pruned_units)
    and preserves all score fields so deserialization succeeds.
    """
    plan = {
        "timestamp": int(time.time()),
        "pruned_total": len(units),
        "budget": budget,
        "proxy_epsilon": source_plan.get("proxy_epsilon", 0.1) if source_plan else 0.1,
        "mask_policy": "strict",
        "to_prune": [{
            "component": u.get("component", ""),
            "layer": u.get("layer", 0),
            "index": u.get("index", 0),
            "score": u.get("score", 0.0),
            "clean_grad_mean": u.get("clean_grad_mean", 0.0),
            "proxy_grad_mean": u.get("proxy_grad_mean", 0.0),
            "cosine": u.get("cosine", 0.0),
            "safe_grad_mean": u.get("safe_grad_mean", 0.0) or 0.0,
            "protect_grad_mean": u.get("protect_grad_mean", u.get("clean_grad_mean", 0.0)) or 0.0,
        } for u in units],
    }
    path = plan_dir / f"pruning_plan_budget{budget:04d}.json"
    path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _make_subprocess_env() -> dict:
    env = os.environ.copy()
    env.setdefault("MKL_SERVICE_FORCE_INTEL", "1")
    return env


# ── 7. Report generation ───────────────────────────────────────────────

def _generate_report(run_dir: Path, plan: dict, rec: dict,
                     candidates: list[dict], dev_metrics: list[dict]) -> str:
    lines = [
        "# Auto-Calibration Report",
        "",
        "## Protocol",
        "- All selection decisions use ONLY trigger-free signals (no triggered ASR).",
        f"- Objective: {rec.get('objective_type', 'balanced')}",
        "",
        "## Recommended Configuration",
        "```json",
        json.dumps(rec, indent=2),
        "```",
        "",
        "## Evaluated Candidates",
    ]
    n = min(len(candidates), len(dev_metrics))
    if n > 0:
        lines.append("| Budget | BFR | HarmRef | Empty | Score |")
        lines.append("|---|---:|---:|---:|---:|")
        for i in range(n):
            m = dev_metrics[i]
            lines.append(
                f"| {m.get('budget', '?')} | {m.get('benign_false_refusal', 0):.3f} | "
                f"{m.get('harmful_no_trigger_refusal', 0):.3f} | "
                f"{m.get('empty_output_rate', 0):.3f} | "
                f"{m.get('objective_value', 0):.4f} |")
    lines += ["", "## Eliminated / Warnings"]
    for i in range(n):
        m = dev_metrics[i]
        if m.get("empty_output_rate", 0) > 0.5:
            lines.append(f"- Budget={m.get('budget')}: empty_output_rate={m['empty_output_rate']:.3f}")
        elif m.get("benign_false_refusal", 0) > 0.8:
            lines.append(f"- Budget={m.get('budget')}: BFR={m['benign_false_refusal']:.3f}")
    if plan.get("skip_recovery"):
        lines.append("- Recovery was skipped; no final checkpoint selected.")
    return "\n".join(lines)


# ── 8. Main ────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trigger-free auto-calibration")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--tokenizer-path", default=None)
    p.add_argument("--use-lora", action="store_true")
    p.add_argument("--lora-model-path", default=None)
    p.add_argument("--prompt-template", choices=["alpaca", "chat", "none"], default="chat")
    p.add_argument("--benign-jsonl", type=Path, required=True)
    p.add_argument("--harmful-no-trigger-jsonl", type=Path, required=True)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--max-length", default="auto")
    p.add_argument("--score-samples", type=int, default=8)
    p.add_argument("--proxy-epsilon", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--alpha-safe", type=float, default=0.5)
    p.add_argument("--candidate-budgets", default="auto")
    p.add_argument("--candidate-lambda-align", default="auto")
    p.add_argument("--candidate-lambda-safe", default="auto")
    p.add_argument("--candidate-steps", default="12,16,20,25,30")
    p.add_argument("--dev-max-items", type=int, default=200)
    p.add_argument("--selection-objective",
                   choices=["balanced", "safety_first", "utility_first"],
                   default="balanced")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--skip-recovery", action="store_true")
    p.add_argument("--keep-candidates", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    run_dir = resolve_run_dir(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    script_dir = str(_THIS_DIR)
    sub_env = _make_subprocess_env()
    tokenizer_path_arg = args.tokenizer_path or args.model_path
    print(f"=== Auto-Calibration: {run_dir} ===")

    # ── 1. Load model for tokenizer/config inspection ──
    print("Loading model for tokenizer/config inspection ...")
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=args.model_path,
        tokenizer_path=tokenizer_path_arg,
        torch_dtype=dtype,
        use_lora=args.use_lora,
        lora_model_path=args.lora_model_path,
        merge_lora=False,
    )

    if args.max_length == "auto":
        max_length = infer_max_length_from_jsonl(
            tokenizer, [args.benign_jsonl, args.harmful_no_trigger_jsonl],
            prompt_template=args.prompt_template)
        print(f"Auto max_length: {max_length}")
    else:
        max_length = int(args.max_length)
    del model
    torch.cuda.empty_cache()

    # ── 2. Score once via subprocess ──
    score_dir = run_dir / "scoring"
    score_dir.mkdir(parents=True, exist_ok=True)
    print("Scoring units (subprocess) ...")
    score_cmd = [
        sys.executable, f"{script_dir}/score_and_prune.py",
        "--run-dir", str(score_dir),
        "--model-path", args.model_path,
        "--tokenizer-path", tokenizer_path_arg,
        "--clean-jsonl", str(args.benign_jsonl),
        "--dtype", args.dtype,
        "--max-length", str(max_length),
        "--proxy-epsilon", str(args.proxy_epsilon),
        "--alpha", str(args.alpha), "--beta", str(args.beta),
        "--score-samples", str(args.score_samples),
        "--kappa", "1e9", "--score-only",
    ]
    if args.harmful_no_trigger_jsonl.exists():
        score_cmd += ["--protect-safe-jsonl", str(args.harmful_no_trigger_jsonl),
                       "--alpha-safe", str(args.alpha_safe)]
    if args.use_lora:
        score_cmd.append("--use-lora")
    if args.lora_model_path:
        score_cmd += ["--lora-model-path", args.lora_model_path]
    score_cmd += ["--prompt-template", args.prompt_template]
    subprocess.run(score_cmd, check=True, env=sub_env)

    scores_path = score_dir / "unit_scores.json"
    if not scores_path.exists():
        raise FileNotFoundError(f"Scoring did not produce {scores_path}")
    with open(scores_path) as f:
        scores_data = json.load(f)
    unit_scores = scores_data.get("scores", scores_data)
    source_plan = None
    sp = score_dir / "pruning_plan.json"
    if sp.exists():
        with open(sp) as f:
            source_plan = json.load(f)

    # ── 3. Generate pruning candidates ──
    candidate_budgets = (None if args.candidate_budgets == "auto"
                         else _parse_int_list(args.candidate_budgets))
    candidates = propose_pruning_candidates(unit_scores, candidate_budgets=candidate_budgets)
    print(f"{len(candidates)} pruning candidates: {[c['budget'] for c in candidates]}")

    # Write per-budget pruning plans so recovery uses the RIGHT budget
    candidate_plans_dir = run_dir / "candidate_plans"
    candidate_plans_dir.mkdir(parents=True, exist_ok=True)
    for cand in candidates:
        _write_candidate_pruning_plan(candidate_plans_dir, cand.pop("_units", []),
                                       cand["budget"], source_plan)

    # ── 4. Lambda & steps suggestions ──
    lambda_aligns = _parse_float_list(args.candidate_lambda_align,
                                      _auto_lambda_align())
    lambda_safes = _parse_float_list(args.candidate_lambda_safe,
                                     _auto_lambda_safe())
    steps_list = _parse_int_list(args.candidate_steps)

    plan = {
        "timestamp": int(time.time()),
        "run_dir": str(run_dir),
        "max_length": max_length,
        "candidate_budgets": [c["budget"] for c in candidates],
        "candidate_lambda_align": lambda_aligns,
        "candidate_lambda_safe": lambda_safes,
        "candidate_steps": steps_list,
        "selection_objective": args.selection_objective,
        "dry_run": args.dry_run,
        "skip_recovery": args.skip_recovery,
        "pruning_candidates": candidates,
    }
    (run_dir / "auto_calibration_plan.json").write_text(
        json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "auto_candidates.json").write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        print("Dry-run complete.  auto_calibration_plan.json written.")
        return

    # ── 5. Evaluate candidates ──
    dev_metrics_list: list[dict] = []

    if args.skip_recovery:
        print("--skip-recovery: evaluating pruned-only candidates (no recovery training)")
        # Evaluate each budget's pruned model directly
        import transformers
        from pruning_backend import BaseSafetyPruner
        from pipeline_utils import apply_structured_prune

        for cand in candidates:
            budget = cand["budget"]
            plan_path = candidate_plans_dir / f"pruning_plan_budget{budget:04d}.json"
            with open(plan_path) as f:
                pplan = json.load(f)
            raw_units = pplan.get("to_prune", [])
            if not raw_units:
                print(f"  Budget={budget}: no units in plan, skipping")
                continue
            ckpt_model = transformers.AutoModelForCausalLM.from_pretrained(
                args.model_path, torch_dtype=dtype, device_map="auto")
            # Infer head_dim for structured prune
            hidden_size = int(getattr(ckpt_model.config, "hidden_size", 0) or 0)
            num_attention_heads = int(getattr(ckpt_model.config, "num_attention_heads", 0) or 0)
            head_dim = hidden_size // num_attention_heads if num_attention_heads > 0 else 128
            num_kv_heads = int(getattr(ckpt_model.config, "num_key_value_heads", 0) or 0) or None
            pruner = BaseSafetyPruner(ckpt_model)
            # Build UnitScore-compatible list (dicts work with attribute access via __getattr__ on some types,
            # but apply_structured_prune expects objects with .component / .layer / .index attributes)
            from dataclasses import dataclass
            @dataclass
            class _UnitStub:
                component: str
                layer: int
                index: int
            unit_objs = [_UnitStub(component=u["component"], layer=u["layer"],
                                    index=u["index"]) for u in raw_units]
            apply_structured_prune(pruner, to_prune=unit_objs,
                                   head_dim=head_dim, num_key_value_heads=num_kv_heads)
            metrics = compute_dev_metrics(
                ckpt_model, tokenizer,
                read_prompts(args.benign_jsonl),
                read_prompts(args.harmful_no_trigger_jsonl),
                max_dev_items=args.dev_max_items,
                prompt_template=args.prompt_template,
            )
            metrics.update({
                "budget": budget,
                "prune_ratio": pplan.get("pruned_total", 0) / max(len(unit_scores), 1),
                "checkpoint": str(args.model_path) + f"+prune_b{budget}",
            })
            dev_metrics_list.append(metrics)
            del ckpt_model
            torch.cuda.empty_cache()
    else:
        # Full recovery per candidate
        import transformers
        top_budgets = [c["budget"] for c in candidates[:3]]
        auto_l = lambda_aligns
        auto_s = lambda_safes
        # Heuristic: shrink grid for practical runtime
        if len(top_budgets) >= 3:
            auto_l = auto_l[:2]
            auto_s = auto_s[:2]
        for budget in top_budgets:
            plan_path = candidate_plans_dir / f"pruning_plan_budget{budget:04d}.json"
            for ls_val in auto_s:
                for la_val in auto_l:
                    rec_dir = run_dir / f"recovery_b{budget}_ls{ls_val}_la{la_val}"
                    rec_dir.mkdir(parents=True, exist_ok=True)
                    save_steps_str = ",".join(str(s) for s in steps_list)

                    rec_cmd = [
                        sys.executable, f"{script_dir}/recover_model.py",
                        "--run-dir", str(rec_dir),
                        "--model-path", args.model_path,
                        "--tokenizer-path", tokenizer_path_arg,
                        "--pruning-plan", str(plan_path),
                        "--benign-jsonl", str(args.benign_jsonl),
                        "--harmful-no-trigger-jsonl", str(args.harmful_no_trigger_jsonl),
                        "--dtype", args.dtype,
                        "--max-length", str(max_length),
                        "--proxy-epsilon", str(args.proxy_epsilon),
                        "--lambda-clean", "1.0",
                        "--lambda-align", str(la_val),
                        "--lambda-safe", str(ls_val),
                        "--steps", str(max(steps_list)),
                        "--save-steps", save_steps_str,
                        "--lr", "1.5e-5",
                        "--trainable-policy", "all",
                        "--mask-policy", "strict",
                        "--grad-accum-steps", "4",
                        "--objective-schedule", "simultaneous",
                        "--safe-target-mode", "fixed",
                        "--prompt-template", args.prompt_template,
                    ]
                    if args.use_lora:
                        rec_cmd.append("--use-lora")
                    if args.lora_model_path:
                        rec_cmd += ["--lora-model-path", args.lora_model_path]
                    print(f"Recovery: budget={budget} ls={ls_val} la={la_val}")
                    subprocess.run(rec_cmd, check=False, env=sub_env)

                    for s in steps_list:
                        ckpt = rec_dir / f"checkpoint_step_{s:03d}"
                        if not ckpt.exists() or not (ckpt / "config.json").exists():
                            continue
                        ckpt_model = transformers.AutoModelForCausalLM.from_pretrained(
                            str(ckpt), torch_dtype=dtype, device_map="auto")
                        metrics = compute_dev_metrics(
                            ckpt_model, tokenizer,
                            read_prompts(args.benign_jsonl),
                            read_prompts(args.harmful_no_trigger_jsonl),
                            max_dev_items=args.dev_max_items,
                            prompt_template=args.prompt_template,
                        )
                        metrics.update({
                            "budget": budget, "lambda_safe": ls_val,
                            "lambda_align": la_val, "steps": s,
                            "checkpoint": str(ckpt),
                            "prune_ratio": budget / max(len(unit_scores), 1),
                        })
                        dev_metrics_list.append(metrics)
                        del ckpt_model
                        torch.cuda.empty_cache()

    (run_dir / "auto_dev_metrics.json").write_text(
        json.dumps(dev_metrics_list, indent=2, ensure_ascii=False), encoding="utf-8")

    recommended = select_best_candidate(dev_metrics_list,
                                        objective=args.selection_objective) \
        if dev_metrics_list else {"error": "No candidates evaluated"}
    (run_dir / "recommended_config.json").write_text(
        json.dumps(recommended, indent=2, ensure_ascii=False), encoding="utf-8")

    report = _generate_report(run_dir, plan, recommended, candidates, dev_metrics_list)
    (run_dir / "auto_calibration_report.md").write_text(report, encoding="utf-8")

    print(f"\n=== Complete ===")
    print(f"Recommended: {json.dumps(recommended, indent=2)}")


if __name__ == "__main__":
    main()
