#!/usr/bin/env python3
"""Stage 4: recover utility after structured pruning."""

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
    _tokenize_one,
    apply_structured_prune,
    compute_proxy_alignment_loss,
    limitations_notes,
    load_backdoorllm_model_and_tokenizer,
    now_ts,
    parse_layer_weights,
    read_prompts,
    resolve_run_dir,
)
from pruning_backend import BaseSafetyPruner


def _preload_clean_batches(
    tokenizer,
    clean_prompts,
    *,
    max_length: int,
    prompt_template: str,
    pruner: BaseSafetyPruner,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    batches: list[dict[str, dict[str, torch.Tensor]]] = []
    for prompt in clean_prompts:
        clean_batch = _tokenize_one(
            tokenizer,
            prompt,
            max_length=max_length,
            prompt_template=prompt_template,
        )
        clean_batch["labels"] = clean_batch["input_ids"].clone()
        batches.append({"clean": pruner._move_to_device(clean_batch)})
    return batches


def _compose_recovery_loss(
    *,
    clean_loss: torch.Tensor,
    proxy_loss: torch.Tensor,
    l1_tensor: torch.Tensor | None,
    lambda_align: float,
    lambda_reg: float,
    loss_normalization: str,
    norm_eps: float,
    ema_clean: torch.Tensor | None,
    ema_align: torch.Tensor | None,
    ema_l1: torch.Tensor | None,
    stable_loss_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stable_loss_mode:
        clean_norm = clean_loss
        proxy_norm = proxy_loss
        l1_norm = l1_tensor if l1_tensor is not None else torch.zeros_like(clean_loss)
    elif loss_normalization == "minmax":
        if l1_tensor is not None:
            terms = torch.stack([clean_loss, proxy_loss, l1_tensor])
        else:
            terms = torch.stack([clean_loss, proxy_loss])
        t_min = terms.detach().min()
        t_max = terms.detach().max()
        denom = torch.clamp(t_max - t_min, min=norm_eps)
        clean_norm = torch.clamp((clean_loss - t_min) / denom, min=0.0, max=1.0)
        proxy_norm = torch.clamp((proxy_loss - t_min) / denom, min=0.0, max=1.0)
        l1_norm = torch.clamp((l1_tensor - t_min) / denom, min=0.0, max=1.0) if l1_tensor is not None else torch.zeros_like(clean_norm)
    elif loss_normalization == "ema_ratio":
        ref_clean = ema_clean if ema_clean is not None else clean_loss.detach()
        ref_proxy = ema_align if ema_align is not None else proxy_loss.detach()
        clean_norm = clean_loss / torch.clamp(ref_clean, min=norm_eps)
        proxy_norm = proxy_loss / torch.clamp(ref_proxy, min=norm_eps)
        if l1_tensor is not None:
            ref_l1 = ema_l1 if ema_l1 is not None else l1_tensor.detach()
            l1_norm = l1_tensor / torch.clamp(ref_l1, min=norm_eps)
        else:
            l1_norm = torch.zeros_like(clean_norm)
    else:
        clean_norm = clean_loss
        proxy_norm = proxy_loss
        l1_norm = l1_tensor if l1_tensor is not None else torch.zeros_like(clean_loss)

    total_loss = clean_norm + lambda_align * proxy_norm + lambda_reg * l1_norm
    return total_loss, clean_norm, proxy_norm, l1_norm


def _deserialize_pruned_units(pruning_meta: dict[str, object]) -> list[UnitScore]:
    raw_units = pruning_meta.get("to_prune", [])
    if not isinstance(raw_units, list):
        return []
    units: list[UnitScore] = []
    for raw in raw_units:
        if not isinstance(raw, dict):
            continue
        try:
            units.append(
                UnitScore(
                    component=str(raw["component"]),
                    layer=int(raw["layer"]),
                    index=int(raw["index"]),
                    clean_grad_mean=float(raw["clean_grad_mean"]),
                    proxy_grad_mean=float(raw["proxy_grad_mean"]),
                    cosine=float(raw["cosine"]),
                    score=float(raw["score"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return units


def _resolve_recovery_model_path(model_path: str, run_dir: Path) -> str:
    candidate = run_dir / "pruned_model"
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
    parser.add_argument("--pruning-plan", type=Path, default=None, help="Defaults to --run-dir/pruning_plan.json")
    parser.add_argument("--clean-jsonl", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--align-layer-indices", type=str, default="", help="Comma-separated layer indices; empty=all layers")
    parser.add_argument("--align-layer-weights", type=str, default="", help="Comma-separated weights matching --align-layer-indices")
    parser.add_argument("--proxy-epsilon", type=float, default=0.1, help="FGSM epsilon used to build the perturbation proxy")
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="Optional L1-style proxy regularization on trainable weights")
    parser.add_argument("--loss-normalization", choices=["none", "minmax", "ema_ratio"], default="ema_ratio")
    parser.add_argument("--norm-eps", type=float, default=1e-8, help="Epsilon for loss normalization")
    parser.add_argument("--norm-ema-beta", type=float, default=0.9, help="EMA beta used by loss-normalization=ema_ratio")
    parser.add_argument("--stable-loss-mode", action="store_true", help="Use raw losses instead of normalized losses")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--mask-policy",
        choices=["strict", "init_only", "none"],
        default="strict",
        help="How to preserve structured pruning during recovery",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = resolve_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if not args.clean_jsonl.exists():
        raise FileNotFoundError(f"Missing clean JSONL: {args.clean_jsonl}")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be >= 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")
    if args.norm_eps <= 0:
        raise ValueError("--norm-eps must be > 0")
    if not (0.0 <= args.norm_ema_beta < 1.0):
        raise ValueError("--norm-ema-beta must be in [0,1)")
    if args.proxy_epsilon <= 0:
        raise ValueError("--proxy-epsilon must be > 0")

    pruning_plan_path = args.pruning_plan if args.pruning_plan is not None else (args.run_dir / "pruning_plan.json")
    if not pruning_plan_path.exists():
        raise FileNotFoundError(f"Missing pruning plan: {pruning_plan_path}")
    pruning_meta = json.loads(pruning_plan_path.read_text(encoding="utf-8"))
    pruned_units = _deserialize_pruned_units(pruning_meta)

    effective_model_path = _resolve_recovery_model_path(str(args.model_path), args.run_dir)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=effective_model_path,
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
        use_lora=bool(args.use_lora),
        lora_model_path=str(args.lora_model_path) if args.lora_model_path else None,
        torch_dtype=dtype,
        merge_lora=False,
    )
    pruner = BaseSafetyPruner(model)
    print(f"Recovery model source: {effective_model_path}")

    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_info = pruner._infer_llama_head_dim(hidden_size)
    if head_info is None:
        raise RuntimeError("Cannot infer attention head dimension from model config")
    _, head_dim = head_info

    if args.align_layer_indices.strip():
        align_layers = [int(x.strip()) for x in args.align_layer_indices.split(",") if x.strip()]
    else:
        align_layers = list(range(num_layers))
    if not align_layers:
        raise ValueError("Alignment layer set is empty")
    for layer_idx in align_layers:
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"Invalid alignment layer index {layer_idx}, model has {num_layers} layers")
    layer_weights = parse_layer_weights(args.align_layer_weights, align_layers)

    clean_prompts = read_prompts(args.clean_jsonl)
    cached_batches = _preload_clean_batches(
        tokenizer,
        clean_prompts,
        max_length=args.max_length,
        prompt_template=str(args.prompt_template),
        pruner=pruner,
    )
    batch_count = len(cached_batches)
    if batch_count == 0:
        raise RuntimeError("No usable clean samples for recovery")

    notes = limitations_notes()
    print("=== Trigger-agnostic defense notes ===")
    for note in notes:
        print(f"- {note}")

    if args.mask_policy != "none" and pruned_units:
        apply_structured_prune(pruner, to_prune=pruned_units, head_dim=head_dim)
        print(f"Applied pruning mask before recovery: {len(pruned_units)} units ({args.mask_policy})")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(trainable, lr=args.lr) if args.optimizer == "sgd" else torch.optim.AdamW(trainable, lr=args.lr)

    recovery_losses: list[dict[str, float]] = []
    ema_clean: torch.Tensor | None = None
    ema_align: torch.Tensor | None = None
    ema_l1: torch.Tensor | None = None
    model.train()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        raw_clean_total = 0.0
        raw_align_total = 0.0
        raw_l1_total = 0.0
        norm_clean_total = 0.0
        norm_align_total = 0.0
        norm_l1_total = 0.0
        finite_batches = 0

        for accum_step in range(args.grad_accum_steps):
            batch_idx = (step * args.grad_accum_steps + accum_step) % batch_count
            clean_batch = cached_batches[batch_idx]["clean"]

            clean_loss = pruner._extract_loss(clean_batch, loss_fn=None)
            proxy_loss = compute_proxy_alignment_loss(
                pruner,
                clean_batch,
                layer_indices=align_layers,
                layer_weights=layer_weights,
                adv_epsilon=float(args.proxy_epsilon),
                eps=float(args.norm_eps),
            )

            l1_tensor: torch.Tensor | None = None
            if float(args.lambda_reg) > 0:
                l1_tensor = sum(parameter.abs().sum() for parameter in pruner.model.parameters() if parameter.requires_grad)
                l1_tensor = l1_tensor.to(device=clean_loss.device, dtype=clean_loss.dtype)

            total_loss, clean_norm, proxy_norm, l1_norm = _compose_recovery_loss(
                clean_loss=clean_loss,
                proxy_loss=proxy_loss,
                l1_tensor=l1_tensor,
                lambda_align=float(args.lambda_align),
                lambda_reg=float(args.lambda_reg),
                loss_normalization=str(args.loss_normalization),
                norm_eps=float(args.norm_eps),
                ema_clean=ema_clean,
                ema_align=ema_align,
                ema_l1=ema_l1,
                stable_loss_mode=bool(args.stable_loss_mode),
            )
            if not torch.isfinite(total_loss).item():
                continue

            (total_loss / args.grad_accum_steps).backward()
            finite_batches += 1
            raw_clean_total += float(clean_loss.detach().item())
            raw_align_total += float(proxy_loss.detach().item())
            norm_clean_total += float(clean_norm.detach().item())
            norm_align_total += float(proxy_norm.detach().item())
            if l1_tensor is not None:
                raw_l1_total += float(l1_tensor.detach().item())
                norm_l1_total += float(l1_norm.detach().item())

        if finite_batches == 0:
            recovery_losses.append(
                {
                    "step": float(step + 1),
                    "l_clean": float("nan"),
                    "l_align": float("nan"),
                    "l1_proxy": float("nan"),
                    "l_clean_norm": float("nan"),
                    "l_align_norm": float("nan"),
                    "l1_proxy_norm": float("nan"),
                    "loss_total": float("nan"),
                }
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.max_grad_norm))
        optimizer.step()
        if args.mask_policy == "strict" and pruned_units:
            apply_structured_prune(pruner, to_prune=pruned_units, head_dim=head_dim)

        avg_clean = raw_clean_total / finite_batches
        avg_align = raw_align_total / finite_batches
        avg_l1 = raw_l1_total / finite_batches if float(args.lambda_reg) > 0 else float("nan")
        avg_clean_norm = norm_clean_total / finite_batches
        avg_align_norm = norm_align_total / finite_batches
        avg_l1_norm = norm_l1_total / finite_batches if float(args.lambda_reg) > 0 else float("nan")
        total_value = avg_clean_norm + float(args.lambda_align) * avg_align_norm
        if float(args.lambda_reg) > 0 and avg_l1_norm == avg_l1_norm:
            total_value += float(args.lambda_reg) * avg_l1_norm

        recovery_losses.append(
            {
                "step": float(step + 1),
                "l_clean": avg_clean,
                "l_align": avg_align,
                "l1_proxy": avg_l1,
                "l_clean_norm": avg_clean_norm,
                "l_align_norm": avg_align_norm,
                "l1_proxy_norm": avg_l1_norm,
                "loss_total": total_value,
            }
        )

        if (not args.stable_loss_mode) and args.loss_normalization == "ema_ratio":
            beta = float(args.norm_ema_beta)
            clean_ref = torch.tensor(avg_clean, device=pruner.device, dtype=torch.float32)
            align_ref = torch.tensor(avg_align, device=pruner.device, dtype=torch.float32)
            ema_clean = clean_ref if ema_clean is None else beta * ema_clean + (1.0 - beta) * clean_ref
            ema_align = align_ref if ema_align is None else beta * ema_align + (1.0 - beta) * align_ref
            if float(args.lambda_reg) > 0 and avg_l1 == avg_l1:
                l1_ref = torch.tensor(avg_l1, device=pruner.device, dtype=torch.float32)
                ema_l1 = l1_ref if ema_l1 is None else beta * ema_l1 + (1.0 - beta) * l1_ref

    (args.run_dir / "recovery_losses.json").write_text(
        json.dumps(
            {
                "timestamp": now_ts(),
                "config": {
                    "model_path_effective": effective_model_path,
                    "proxy_epsilon": float(args.proxy_epsilon),
                    "lambda_align": float(args.lambda_align),
                    "lambda_reg": float(args.lambda_reg),
                    "loss_normalization": str(args.loss_normalization),
                    "stable_loss_mode": bool(args.stable_loss_mode),
                    "grad_accum_steps": int(args.grad_accum_steps),
                    "mask_policy": str(args.mask_policy),
                    "pruned_total": int(pruning_meta.get("pruned_total", 0) or 0),
                    "steps": int(args.steps),
                    "optimizer": str(args.optimizer),
                    "lr": float(args.lr),
                },
                "losses": recovery_losses,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    output_dir = args.run_dir / "recovered_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"Wrote losses to {args.run_dir / 'recovery_losses.json'}")
    print(f"Wrote recovered model to {output_dir}")


if __name__ == "__main__":
    main()
