#!/usr/bin/env python3
"""Stage 1: train trigger-agnostic alignment on clean data."""

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
    _tokenize_one,
    compute_proxy_alignment_loss,
    limitations_notes,
    load_backdoorllm_model_and_tokenizer,
    now_ts,
    parse_layer_weights,
    read_prompts,
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


def _compose_alignment_loss(
    *,
    clean_loss: torch.Tensor,
    proxy_loss: torch.Tensor,
    loss_mode: str,
    loss_normalization: str,
    lambda_align: float,
    norm_eps: float,
    ema_clean: torch.Tensor | None,
    ema_align: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_mode == "align-only":
        nan_clean = torch.tensor(float("nan"), device=proxy_loss.device, dtype=proxy_loss.dtype)
        return lambda_align * proxy_loss, nan_clean, proxy_loss

    if loss_normalization == "minmax":
        terms = torch.stack([clean_loss, proxy_loss])
        t_min = terms.detach().min()
        t_max = terms.detach().max()
        denom = torch.clamp(t_max - t_min, min=norm_eps)
        clean_norm = torch.clamp((clean_loss - t_min) / denom, min=0.0, max=1.0)
        proxy_norm = torch.clamp((proxy_loss - t_min) / denom, min=0.0, max=1.0)
    elif loss_normalization == "ema_ratio":
        ref_clean = ema_clean if ema_clean is not None else clean_loss.detach()
        ref_proxy = ema_align if ema_align is not None else proxy_loss.detach()
        clean_norm = clean_loss / torch.clamp(ref_clean, min=norm_eps)
        proxy_norm = proxy_loss / torch.clamp(ref_proxy, min=norm_eps)
    else:
        clean_norm = clean_loss
        proxy_norm = proxy_loss

    total_loss = clean_norm + lambda_align * proxy_norm
    return total_loss, clean_norm, proxy_norm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path")
    parser.add_argument("--tokenizer-path", default=None, help="Optional tokenizer path; defaults to model path")
    parser.add_argument("--use-lora", action="store_true", help="Load a LoRA adapter on top of the base model")
    parser.add_argument("--lora-model-path", default=None, help="Optional LoRA adapter path")
    parser.add_argument("--prompt-template", choices=["alpaca", "chat", "none"], default="alpaca")
    parser.add_argument("--clean-jsonl", type=Path, required=True)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--align-layer-indices", type=str, default="", help="Comma-separated layer indices; empty=all layers")
    parser.add_argument("--align-layer-weights", type=str, default="", help="Comma-separated weights matching --align-layer-indices")
    parser.add_argument("--proxy-epsilon", type=float, default=0.1, help="FGSM epsilon used to build the perturbation proxy")
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--loss-mode", choices=["align-only", "clean+align"], default="clean+align")
    parser.add_argument("--loss-normalization", choices=["none", "minmax", "ema_ratio"], default="ema_ratio")
    parser.add_argument("--norm-eps", type=float, default=1e-8, help="Epsilon for loss normalization")
    parser.add_argument("--norm-ema-beta", type=float, default=0.9, help="EMA beta used by loss-normalization=ema_ratio")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    if not args.clean_jsonl.exists():
        raise FileNotFoundError(f"Missing clean JSONL: {args.clean_jsonl}")
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
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

    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=str(args.model_path),
        tokenizer_path=str(args.tokenizer_path) if args.tokenizer_path else None,
        use_lora=bool(args.use_lora),
        lora_model_path=str(args.lora_model_path) if args.lora_model_path else None,
        torch_dtype=dtype,
        merge_lora=False,
    )
    pruner = BaseSafetyPruner(model)

    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    head_info = pruner._infer_llama_head_dim(hidden_size)
    if head_info is None:
        raise RuntimeError("Cannot infer attention head dimension from model config")
    num_heads, head_dim = head_info

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
        raise RuntimeError("No usable clean samples for alignment training")

    notes = limitations_notes()
    print("=== Trigger-agnostic defense notes ===")
    for note in notes:
        print(f"- {note}")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    alignment_losses: list[dict[str, float]] = []
    ema_clean: torch.Tensor | None = None
    ema_align: torch.Tensor | None = None
    model.train()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        raw_clean_total = 0.0
        raw_align_total = 0.0
        norm_clean_total = 0.0
        norm_align_total = 0.0
        finite_batches = 0

        for accum_step in range(args.grad_accum_steps):
            batch_idx = (step * args.grad_accum_steps + accum_step) % batch_count
            batch = cached_batches[batch_idx]
            clean_batch = batch["clean"]

            clean_loss = pruner._extract_loss(clean_batch, loss_fn=None)
            proxy_loss = compute_proxy_alignment_loss(
                pruner,
                clean_batch,
                layer_indices=align_layers,
                layer_weights=layer_weights,
                adv_epsilon=float(args.proxy_epsilon),
                eps=float(args.norm_eps),
            )

            total_loss, clean_norm, proxy_norm = _compose_alignment_loss(
                clean_loss=clean_loss,
                proxy_loss=proxy_loss,
                loss_mode=str(args.loss_mode),
                loss_normalization=str(args.loss_normalization),
                lambda_align=float(args.lambda_align),
                norm_eps=float(args.norm_eps),
                ema_clean=ema_clean,
                ema_align=ema_align,
            )
            if not torch.isfinite(total_loss).item():
                continue

            (total_loss / args.grad_accum_steps).backward()
            finite_batches += 1
            raw_clean_total += float(clean_loss.detach().item())
            raw_align_total += float(proxy_loss.detach().item())
            if args.loss_mode != "align-only":
                norm_clean_total += float(clean_norm.detach().item())
            norm_align_total += float(proxy_norm.detach().item())

        if finite_batches == 0:
            alignment_losses.append(
                {
                    "step": float(step + 1),
                    "l_clean": float("nan"),
                    "l_align": float("nan"),
                    "l_clean_norm": float("nan"),
                    "l_align_norm": float("nan"),
                    "loss_total": float("nan"),
                }
            )
            optimizer.zero_grad(set_to_none=True)
            continue

        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.max_grad_norm))
        optimizer.step()

        avg_clean = raw_clean_total / finite_batches
        avg_align = raw_align_total / finite_batches
        avg_clean_norm = float("nan") if args.loss_mode == "align-only" else norm_clean_total / finite_batches
        avg_align_norm = norm_align_total / finite_batches
        loss_total = (
            float(args.lambda_align) * avg_align_norm
            if args.loss_mode == "align-only"
            else avg_clean_norm + float(args.lambda_align) * avg_align_norm
        )

        alignment_losses.append(
            {
                "step": float(step + 1),
                "l_clean": avg_clean,
                "l_align": avg_align,
                "l_clean_norm": avg_clean_norm,
                "l_align_norm": avg_align_norm,
                "loss_total": loss_total,
            }
        )

        if args.loss_mode != "align-only" and args.loss_normalization == "ema_ratio":
            beta = float(args.norm_ema_beta)
            clean_ref = torch.tensor(avg_clean, device=pruner.device, dtype=torch.float32)
            align_ref = torch.tensor(avg_align, device=pruner.device, dtype=torch.float32)
            ema_clean = clean_ref if ema_clean is None else beta * ema_clean + (1.0 - beta) * clean_ref
            ema_align = align_ref if ema_align is None else beta * ema_align + (1.0 - beta) * align_ref

    config_payload = {
        "timestamp": now_ts(),
        "script": "train_alignment.py",
        "config": {
            "model_path": args.model_path,
            "dtype": args.dtype,
            "max_length": int(args.max_length),
            "align_layers": align_layers,
            "layer_weights": layer_weights,
            "proxy_epsilon": float(args.proxy_epsilon),
            "lambda_align": float(args.lambda_align),
            "loss_mode": str(args.loss_mode),
            "loss_normalization": str(args.loss_normalization),
            "norm_eps": float(args.norm_eps),
            "norm_ema_beta": float(args.norm_ema_beta),
            "steps": int(args.steps),
            "lr": float(args.lr),
            "max_grad_norm": float(args.max_grad_norm),
            "grad_accum_steps": int(args.grad_accum_steps),
        },
        "model": {
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "head_dim": int(head_dim),
            "hidden_size": int(hidden_size),
            "intermediate_size": int(getattr(model.config, "intermediate_size", 0) or 0),
        },
        "notes": notes,
    }

    (args.run_dir / "alignment_config.json").write_text(
        json.dumps(config_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (args.run_dir / "alignment_losses.json").write_text(
        json.dumps({"losses": alignment_losses}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if hasattr(model, "merge_and_unload"):
        model = model.merge_and_unload()

    output_dir = args.run_dir / "stage1_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print(f"Wrote config to {args.run_dir / 'alignment_config.json'}")
    print(f"Wrote losses to {args.run_dir / 'alignment_losses.json'}")
    print(f"Wrote stage1 model to {output_dir}")


if __name__ == "__main__":
    main()
