#!/usr/bin/env python3
"""Stage 4: recover utility after structured pruning."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import random
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT_DIR))

from pipeline_utils import (
    DEFAULT_MODEL_PATH,
    UnitScore,
    _consistency_loss_from_hidden_states,
    _prepare_inputs_embeds_batch,
    _tokenize_supervised_response,
    _tokenize_one,
    apply_structured_prune,
    compute_cosine_from_hidden,
    limitations_notes,
    load_backdoorllm_model_and_tokenizer,
    now_ts,
    parse_layer_weights,
    read_prompts,
    resolve_run_dir,
    save_model_and_tokenizer_safe,
)
from pruning_backend import BaseSafetyPruner


DEFAULT_SAFE_TARGET_POOL: tuple[str, ...] = (
    "I cannot assist with that request.",
    "I can't help with that.",
    "I must refuse that request.",
    "I am unable to provide help for that.",
    "I cannot support that request.",
)
DEFAULT_KEYWORD_SOFT_TARGET = "I cannot help with that."


def _is_cuda_device(device: torch.device) -> bool:
    return torch.device(device).type == "cuda"


def _memory_snapshot(device: torch.device) -> dict[str, float] | None:
    if not _is_cuda_device(device):
        return None
    return {
        "allocated_mib": float(torch.cuda.memory_allocated(device) / (1024 ** 2)),
        "reserved_mib": float(torch.cuda.memory_reserved(device) / (1024 ** 2)),
        "peak_allocated_mib": float(torch.cuda.max_memory_allocated(device) / (1024 ** 2)),
        "peak_reserved_mib": float(torch.cuda.max_memory_reserved(device) / (1024 ** 2)),
    }


def _saved_tensor_context(saved_tensor_offload: str, device: torch.device):
    if saved_tensor_offload == "cpu" and _is_cuda_device(device):
        return torch.autograd.graph.save_on_cpu(pin_memory=True, device_type="cuda")
    return nullcontext()


def _record_memory_event(
    storage: list[dict[str, object]],
    *,
    enabled: bool,
    device: torch.device,
    stage: str,
    step: int,
    accum_step: int | None = None,
    objective: str | None = None,
    note: str | None = None,
) -> None:
    if not enabled:
        return
    snapshot = _memory_snapshot(device)
    event: dict[str, object] = {
        "stage": stage,
        "step": int(step),
    }
    if accum_step is not None:
        event["accum_step"] = int(accum_step)
    if objective is not None:
        event["objective"] = str(objective)
    if note is not None:
        event["note"] = str(note)
    if snapshot is not None:
        event.update(snapshot)
    storage.append(event)


def _cpu_grad_buffers(trainable: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    return [torch.zeros(parameter.shape, dtype=torch.float32, device="cpu") for parameter in trainable]


def _offload_trainable_grads_to_cpu(
    trainable: list[torch.nn.Parameter],
    cpu_buffers: list[torch.Tensor],
) -> None:
    for parameter, buffer in zip(trainable, cpu_buffers, strict=False):
        grad = parameter.grad
        if grad is None:
            continue
        buffer.add_(grad.detach().to(device="cpu", dtype=torch.float32))
        parameter.grad = None


def _cpu_grad_norm(cpu_buffers: list[torch.Tensor]) -> float:
    total_sq = 0.0
    for buffer in cpu_buffers:
        if buffer.numel() == 0:
            continue
        grad_norm = float(torch.norm(buffer, p=2).item())
        total_sq += grad_norm * grad_norm
    return math.sqrt(total_sq)


def _clip_cpu_grad_buffers(cpu_buffers: list[torch.Tensor], max_grad_norm: float) -> float:
    grad_norm = _cpu_grad_norm(cpu_buffers)
    if max_grad_norm > 0 and math.isfinite(grad_norm) and grad_norm > max_grad_norm:
        scale = float(max_grad_norm / (grad_norm + 1e-12))
        for buffer in cpu_buffers:
            buffer.mul_(scale)
    return grad_norm


@torch.no_grad()
def _sgd_step_from_cpu_grads(
    trainable: list[torch.nn.Parameter],
    cpu_buffers: list[torch.Tensor],
    *,
    lr: float,
) -> None:
    for parameter, buffer in zip(trainable, cpu_buffers, strict=False):
        if not torch.count_nonzero(buffer).item():
            continue
        grad_gpu = buffer.to(device=parameter.device, dtype=parameter.dtype)
        parameter.add_(grad_gpu, alpha=-float(lr))
        buffer.zero_()


@contextmanager
def _temporarily_disable_param_grads(parameters: list[torch.nn.Parameter]) -> None:
    original_states = [bool(parameter.requires_grad) for parameter in parameters]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, original_state in zip(parameters, original_states, strict=False):
            parameter.requires_grad_(original_state)


def _preload_clean_batches(
    tokenizer,
    clean_prompts,
    *,
    max_length: int,
    prompt_template: str,
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
        batches.append({"clean": clean_batch})
    return batches


def _preload_safe_batches(
    tokenizer,
    safe_prompts,
    *,
    max_length: int,
    prompt_template: str,
) -> list[dict[str, dict[str, torch.Tensor]]]:
    batches: list[dict[str, dict[str, torch.Tensor]]] = []
    for prompt, target_text in safe_prompts:
        safe_batch = _tokenize_supervised_response(
            tokenizer,
            prompt,
            target_text=target_text,
            max_length=max_length,
            prompt_template=prompt_template,
        )
        batches.append({"safe": safe_batch})
    return batches


def _resolve_safe_target_assignments(
    safe_prompts: list[object],
    *,
    safe_target_mode: str,
    safe_target_text: str,
) -> tuple[list[tuple[object, str]], dict[str, object]]:
    if safe_target_mode == "fixed":
        assigned = [(prompt, safe_target_text) for prompt in safe_prompts]
        return assigned, {
            "safe_target_mode": safe_target_mode,
            "safe_target_pool": [safe_target_text],
            "safe_target_keyword_soft_text": None,
        }

    if safe_target_mode == "template_pool":
        pool: list[str] = []
        for candidate in (safe_target_text, *DEFAULT_SAFE_TARGET_POOL):
            if candidate not in pool:
                pool.append(candidate)
        rng = random.Random(0)
        assigned = [(prompt, rng.choice(pool)) for prompt in safe_prompts]
        return assigned, {
            "safe_target_mode": safe_target_mode,
            "safe_target_pool": pool,
            "safe_target_keyword_soft_text": None,
        }

    if safe_target_mode == "keyword_soft":
        keyword_soft_text = DEFAULT_KEYWORD_SOFT_TARGET
        assigned = [(prompt, keyword_soft_text) for prompt in safe_prompts]
        return assigned, {
            "safe_target_mode": safe_target_mode,
            "safe_target_pool": [keyword_soft_text],
            "safe_target_keyword_soft_text": keyword_soft_text,
        }

    raise ValueError(f"Unsupported --safe-target-mode: {safe_target_mode}")


def _scheduled_objective_weights(
    *,
    step_index: int,
    total_steps: int,
    lambda_clean: float,
    lambda_align: float,
    lambda_safe: float,
    objective_schedule: str,
    alternating_soft_safe_ratio: float = 1.0,
) -> tuple[float, float, float]:
    if objective_schedule == "simultaneous":
        return lambda_clean, lambda_align, lambda_safe

    if objective_schedule == "alternating":
        if step_index % 2 == 0:
            return lambda_clean, lambda_align, 0.0
        return 0.0, 0.0, lambda_safe

    if objective_schedule == "alternating_soft":
        if step_index % 2 == 0:
            return lambda_clean, lambda_align, lambda_safe * alternating_soft_safe_ratio
        return 0.0, 0.0, lambda_safe

    if objective_schedule == "alternating_2c1s":
        return (
            (lambda_clean, lambda_align, 0.0)
            if step_index % 3 in (0, 1)
            else (0.0, 0.0, lambda_safe)
        )

    if objective_schedule == "warmup_clean_then_mixed":
        warmup_steps = max(1, math.ceil(total_steps * 0.25))
        if step_index < warmup_steps:
            return lambda_clean, lambda_align, 0.0
        return lambda_clean, lambda_align, lambda_safe

    if objective_schedule == "alternating_then_simultaneous":
        transition_step = math.ceil(total_steps * 0.6)
        if step_index < transition_step:
            if step_index % 2 == 0:
                return lambda_clean, lambda_align, 0.0
            return 0.0, 0.0, lambda_safe
        return lambda_clean, lambda_align, lambda_safe

    raise ValueError(f"Unsupported --objective-schedule: {objective_schedule}")


def _compute_proxy_alignment_loss_recovery(
    pruner: BaseSafetyPruner,
    clean_batch: dict[str, torch.Tensor],
    *,
    layer_indices: list[int],
    layer_weights: dict[int, float],
    trainable_parameters: list[torch.nn.Parameter],
    adv_epsilon: float,
    eps: float,
    saved_tensor_offload: str,
) -> torch.Tensor:
    if adv_epsilon <= 0:
        raise ValueError("adv_epsilon must be > 0")
    if not layer_indices:
        raise ValueError("layer_indices must be non-empty")

    clean_batch = pruner._move_to_device(clean_batch)
    working_batch = dict(clean_batch)
    if "labels" not in working_batch and "input_ids" in working_batch:
        working_batch["labels"] = working_batch["input_ids"].clone()

    pruner.model.zero_grad(set_to_none=True)
    with _temporarily_disable_param_grads(trainable_parameters):
        with _saved_tensor_context(saved_tensor_offload, pruner.device):
            inputs_embeds, model_inputs = _prepare_inputs_embeds_batch(pruner.model, working_batch)
            consistency_inputs = {key: value for key, value in model_inputs.items() if key != "labels"}

            outputs = pruner.model(
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                use_cache=False,
                **consistency_inputs,
            )
            consistency_loss = _consistency_loss_from_hidden_states(outputs.hidden_states, eps)
            grad_inputs = torch.autograd.grad(
                consistency_loss,
                inputs_embeds,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )[0]
    perturbation = adv_epsilon * grad_inputs.detach().sign()
    pruner.model.zero_grad(set_to_none=True)
    perturbed_embeds = (inputs_embeds.detach() + perturbation).detach()

    clean_forward = {key: value for key, value in clean_batch.items() if key != "labels"}
    with _saved_tensor_context(saved_tensor_offload, pruner.device):
        with pruner.trace_hidden_states(layer_indices) as trace:
            _ = pruner.model(**clean_forward)
            clean_states = {key: value for key, value in trace["hidden_states"].items()}
        with pruner.trace_hidden_states(layer_indices) as trace:
            _ = pruner.model(inputs_embeds=perturbed_embeds, use_cache=False, **consistency_inputs)
            perturbed_states = {key: value for key, value in trace["hidden_states"].items()}

    total = torch.zeros((), device=pruner.device, dtype=torch.float32)
    for layer_idx in layer_indices:
        if layer_idx not in clean_states or layer_idx not in perturbed_states:
            raise RuntimeError(f"Missing hidden states for alignment layer {layer_idx}")
        weight = float(layer_weights.get(layer_idx, 1.0))
        layer_loss = compute_cosine_from_hidden(clean_states[layer_idx], perturbed_states[layer_idx], eps=eps)
        total = total + (weight * layer_loss).to(device=total.device, dtype=total.dtype)
    return total


def _normalize_single_loss_term(
    term: torch.Tensor,
    *,
    reference: torch.Tensor | None,
    loss_normalization: str,
    norm_eps: float,
    stable_loss_mode: bool,
) -> torch.Tensor:
    if stable_loss_mode or loss_normalization == "none":
        return term
    if loss_normalization == "ema_ratio":
        ref = reference if reference is not None else term.detach()
        return term / torch.clamp(ref, min=norm_eps)
    raise ValueError(f"Single-term normalization does not support mode: {loss_normalization}")


def _chunk_layer_indices(layer_indices: list[int], chunk_size: int) -> list[list[int]]:
    if chunk_size <= 0 or chunk_size >= len(layer_indices):
        return [list(layer_indices)]
    return [list(layer_indices[idx: idx + chunk_size]) for idx in range(0, len(layer_indices), chunk_size)]


def _compose_recovery_loss(
    *,
    clean_loss: torch.Tensor,
    proxy_loss: torch.Tensor,
    safe_loss: torch.Tensor | None,
    l1_tensor: torch.Tensor | None,
    lambda_clean: float,
    lambda_align: float,
    lambda_safe: float,
    lambda_reg: float,
    loss_normalization: str,
    norm_eps: float,
    ema_clean: torch.Tensor | None,
    ema_align: torch.Tensor | None,
    ema_safe: torch.Tensor | None,
    ema_l1: torch.Tensor | None,
    stable_loss_mode: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if stable_loss_mode:
        clean_norm = clean_loss
        proxy_norm = proxy_loss
        safe_norm = safe_loss if safe_loss is not None else torch.zeros_like(clean_loss)
        l1_norm = l1_tensor if l1_tensor is not None else torch.zeros_like(clean_loss)
    elif loss_normalization == "minmax":
        terms_list = [clean_loss, proxy_loss]
        if safe_loss is not None:
            terms_list.append(safe_loss)
        if l1_tensor is not None:
            terms_list.append(l1_tensor)
        terms = torch.stack(terms_list)
        t_min = terms.detach().min()
        t_max = terms.detach().max()
        denom = torch.clamp(t_max - t_min, min=norm_eps)
        clean_norm = torch.clamp((clean_loss - t_min) / denom, min=0.0, max=1.0)
        proxy_norm = torch.clamp((proxy_loss - t_min) / denom, min=0.0, max=1.0)
        safe_norm = (
            torch.clamp((safe_loss - t_min) / denom, min=0.0, max=1.0)
            if safe_loss is not None
            else torch.zeros_like(clean_norm)
        )
        l1_norm = torch.clamp((l1_tensor - t_min) / denom, min=0.0, max=1.0) if l1_tensor is not None else torch.zeros_like(clean_norm)
    elif loss_normalization == "ema_ratio":
        ref_clean = ema_clean if ema_clean is not None else clean_loss.detach()
        ref_proxy = ema_align if ema_align is not None else proxy_loss.detach()
        clean_norm = clean_loss / torch.clamp(ref_clean, min=norm_eps)
        proxy_norm = proxy_loss / torch.clamp(ref_proxy, min=norm_eps)
        if safe_loss is not None:
            ref_safe = ema_safe if ema_safe is not None else safe_loss.detach()
            safe_norm = safe_loss / torch.clamp(ref_safe, min=norm_eps)
        else:
            safe_norm = torch.zeros_like(clean_norm)
        if l1_tensor is not None:
            ref_l1 = ema_l1 if ema_l1 is not None else l1_tensor.detach()
            l1_norm = l1_tensor / torch.clamp(ref_l1, min=norm_eps)
        else:
            l1_norm = torch.zeros_like(clean_norm)
    else:
        clean_norm = clean_loss
        proxy_norm = proxy_loss
        safe_norm = safe_loss if safe_loss is not None else torch.zeros_like(clean_loss)
        l1_norm = l1_tensor if l1_tensor is not None else torch.zeros_like(clean_loss)

    total_loss = (
        lambda_clean * clean_norm
        + lambda_align * proxy_norm
        + lambda_safe * safe_norm
        + lambda_reg * l1_norm
    )
    return total_loss, clean_norm, proxy_norm, safe_norm, l1_norm


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
                    safe_grad_mean=float(raw.get("safe_grad_mean", 0.0) or 0.0),
                    protect_grad_mean=float(raw.get("protect_grad_mean", raw.get("clean_grad_mean", 0.0)) or 0.0),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return units


def _set_trainable_scope_from_pruned_layers(
    model: torch.nn.Module,
    pruned_units: list[UnitScore],
) -> dict[str, object]:
    layer_indices = sorted({int(unit.layer) for unit in pruned_units})
    if not layer_indices:
        raise ValueError("pruned_layers trainable policy requires a non-empty pruning plan")

    enabled_params = 0
    total_params = 0
    layer_tokens = tuple(f".layers.{layer_idx}." for layer_idx in layer_indices)
    for name, parameter in model.named_parameters():
        enable = any(token in name for token in layer_tokens)
        parameter.requires_grad_(enable)
        total_params += int(parameter.numel())
        if enable:
            enabled_params += int(parameter.numel())
    return {
        "trainable_layers": layer_indices,
        "enabled_param_count": enabled_params,
        "total_param_count": total_params,
    }


def _resolve_recovery_model_path(model_path: str, run_dir: Path) -> str:
    candidate = run_dir / "pruned_model"
    if str(model_path) == str(DEFAULT_MODEL_PATH) and candidate.exists():
        return str(candidate)
    return str(model_path)


def _parse_debug_save_steps(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    steps: set[int] = set()
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        value = int(chunk)
        if value <= 0:
            raise ValueError("--debug-save-steps only accepts positive integers")
        steps.add(value)
    return steps


def _compute_grad_norm(trainable: list[torch.nn.Parameter]) -> float:
    total_sq = 0.0
    has_grad = False
    for parameter in trainable:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        grad_norm = float(torch.norm(grad.float(), p=2).detach().item())
        total_sq += grad_norm * grad_norm
        has_grad = True
    if not has_grad:
        return float("nan")
    return math.sqrt(total_sq)


def _has_nonfinite_parameter(trainable: list[torch.nn.Parameter]) -> bool:
    for parameter in trainable:
        if not torch.isfinite(parameter.detach()).all().item():
            return True
    return False


def _first_nonfinite_reason(reasons: set[str]) -> str | None:
    priority = ["clean_loss", "align_loss", "safe_loss", "total_loss", "grad_norm", "parameter"]
    for key in priority:
        if key in reasons:
            return key
    return None


def _save_debug_checkpoint(
    *,
    model,
    tokenizer,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_model_and_tokenizer_safe(model, tokenizer, str(output_dir))


def _write_recovery_memory_debug(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="Base model path or a Hugging Face checkpoint directory")
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-model-path", default=None)
    parser.add_argument("--prompt-template", choices=["alpaca", "chat", "none"], default="alpaca")
    parser.add_argument("--pruning-plan", type=Path, default=None, help="Defaults to --run-dir/pruning_plan.json")
    parser.add_argument(
        "--clean-jsonl",
        type=Path,
        default=None,
        help="Legacy alias for --benign-jsonl; used when --benign-jsonl is not provided",
    )
    parser.add_argument(
        "--benign-jsonl",
        type=Path,
        default=None,
        help="Benign clean split used to preserve general utility during recovery",
    )
    parser.add_argument(
        "--harmful-no-trigger-jsonl",
        type=Path,
        default=None,
        help="Optional harmful-but-no-trigger split used to preserve refusal behavior",
    )
    parser.add_argument(
        "--safe-target-text",
        type=str,
        default="I cannot assist with that request.",
        help="Target refusal text used when supervising harmful no-trigger samples",
    )
    parser.add_argument(
        "--safe-target-mode",
        choices=["fixed", "template_pool", "keyword_soft"],
        default="fixed",
        help="How to construct the safe refusal supervision target",
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--trainable-policy",
        choices=["all", "pruned_layers"],
        default="all",
        help="Select which parameters remain trainable during recovery",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--align-layer-indices", type=str, default="", help="Comma-separated layer indices; empty=all layers")
    parser.add_argument("--align-layer-weights", type=str, default="", help="Comma-separated weights matching --align-layer-indices")
    parser.add_argument("--proxy-epsilon", type=float, default=0.1, help="FGSM epsilon used to build the perturbation proxy")
    parser.add_argument("--lambda-clean", type=float, default=1.0, help="Weight of the benign utility preservation loss")
    parser.add_argument("--lambda-align", type=float, default=1.0)
    parser.add_argument("--lambda-safe", type=float, default=0.0, help="Weight of the harmful-no-trigger refusal preservation loss")
    parser.add_argument("--lambda-reg", type=float, default=0.0, help="Optional L1-style proxy regularization on trainable weights")
    parser.add_argument("--loss-normalization", choices=["none", "minmax", "ema_ratio"], default="ema_ratio")
    parser.add_argument("--norm-eps", type=float, default=1e-8, help="Epsilon for loss normalization")
    parser.add_argument("--norm-ema-beta", type=float, default=0.9, help="EMA beta used by loss-normalization=ema_ratio")
    parser.add_argument("--stable-loss-mode", action="store_true", help="Use raw losses instead of normalized losses")
    parser.add_argument(
        "--objective-schedule",
        choices=["simultaneous", "alternating", "alternating_soft", "alternating_2c1s", "alternating_then_simultaneous", "warmup_clean_then_mixed"],
        default="simultaneous",
        help="How to combine clean and safe recovery objectives across training steps",
    )
    parser.add_argument(
        "--alternating-soft-safe-ratio",
        type=float,
        default=0.25,
        help="Safe weight ratio on clean steps when using alternating_soft schedule (0.0 = pure alternating, 1.0 = simultaneous)",
    )
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--grad-offload",
        choices=["none", "cpu"],
        default="none",
        help="Optional recovery-only gradient accumulation buffer device",
    )
    parser.add_argument(
        "--saved-tensor-offload",
        choices=["none", "cpu"],
        default="none",
        help="Optional recovery-only autograd saved-tensor offload path",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing during recovery to reduce activation memory",
    )
    parser.add_argument(
        "--align-layer-chunk-size",
        type=int,
        default=0,
        help="Optional recovery-only chunk size for alignment layers; 0 means use all selected layers at once",
    )
    parser.add_argument(
        "--memory-debug",
        action="store_true",
        help="Record CUDA memory snapshots during recovery and write a debug JSON",
    )
    parser.add_argument(
        "--memory-debug-path",
        type=Path,
        default=None,
        help="Optional explicit output path for recovery memory debug JSON",
    )
    parser.add_argument(
        "--empty-cache-between-objectives",
        action="store_true",
        help="Call torch.cuda.empty_cache() after each objective/micro-step cleanup",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--grad-accum-steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument(
        "--mask-policy",
        choices=["strict", "init_only", "none"],
        default="strict",
        help="How to preserve structured pruning during recovery",
    )
    parser.add_argument(
        "--debug-save-steps",
        type=str,
        default="",
        help="Optional comma-separated 1-indexed step numbers whose checkpoints should be saved for diagnostics",
    )
    parser.add_argument(
        "--debug-save-first-nonfinite",
        action="store_true",
        help="Save a checkpoint the first time any loss/gradient/parameter statistic becomes non-finite",
    )
    parser.add_argument(
        "--debug-checkpoint-dir",
        type=Path,
        default=None,
        help="Optional output directory for diagnostic step checkpoints; defaults to --run-dir/debug_checkpoints",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = resolve_run_dir(args.run_dir)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    benign_jsonl = args.benign_jsonl or args.clean_jsonl
    if benign_jsonl is None:
        raise ValueError("Provide --benign-jsonl or --clean-jsonl")
    if not benign_jsonl.exists():
        raise FileNotFoundError(f"Missing benign JSONL: {benign_jsonl}")
    if args.harmful_no_trigger_jsonl is not None and not args.harmful_no_trigger_jsonl.exists():
        raise FileNotFoundError(f"Missing harmful no-trigger JSONL: {args.harmful_no_trigger_jsonl}")
    if args.steps < 0:
        raise ValueError("--steps must be >= 0")
    if args.lr <= 0:
        raise ValueError("--lr must be > 0")
    if args.max_grad_norm < 0:
        raise ValueError("--max-grad-norm must be >= 0")
    if int(args.align_layer_chunk_size) < 0:
        raise ValueError("--align-layer-chunk-size must be >= 0")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be > 0")
    if args.norm_eps <= 0:
        raise ValueError("--norm-eps must be > 0")
    if not (0.0 <= args.norm_ema_beta < 1.0):
        raise ValueError("--norm-ema-beta must be in [0,1)")
    if args.proxy_epsilon <= 0:
        raise ValueError("--proxy-epsilon must be > 0")
    if float(args.lambda_clean) < 0:
        raise ValueError("--lambda-clean must be >= 0")
    if float(args.lambda_safe) < 0:
        raise ValueError("--lambda-safe must be >= 0")
    if float(args.lambda_safe) > 0 and args.harmful_no_trigger_jsonl is None:
        raise ValueError("--lambda-safe > 0 requires --harmful-no-trigger-jsonl")
    debug_save_steps = _parse_debug_save_steps(str(args.debug_save_steps))
    debug_checkpoint_dir = (
        Path(args.debug_checkpoint_dir)
        if args.debug_checkpoint_dir is not None
        else (args.run_dir / "debug_checkpoints")
    )
    memory_debug_path = (
        Path(args.memory_debug_path)
        if args.memory_debug_path is not None
        else (args.run_dir / "recovery_memory_debug.json")
    )

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
    if bool(args.gradient_checkpointing):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            try:
                model.enable_input_require_grads()
            except Exception:
                pass
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    trainable_scope_meta = {
        "trainable_policy": str(args.trainable_policy),
        "trainable_layers": None,
        "enabled_param_count": None,
        "total_param_count": None,
    }
    if str(args.trainable_policy) == "pruned_layers":
        trainable_scope_meta.update(_set_trainable_scope_from_pruned_layers(model, pruned_units))
    pruner = BaseSafetyPruner(model)
    print(f"Recovery model source: {effective_model_path}")
    if str(args.trainable_policy) == "pruned_layers":
        print(
            "Trainable scope: pruned_layers "
            f"{trainable_scope_meta.get('trainable_layers')} "
            f"({trainable_scope_meta.get('enabled_param_count')} / {trainable_scope_meta.get('total_param_count')} params)"
        )
    use_autocast = bool(pruner.device.type == "cuda" and dtype in {torch.bfloat16, torch.float16})

    num_layers = int(getattr(model.config, "num_hidden_layers", 0) or 0)
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    num_key_value_heads = int(getattr(model.config, "num_key_value_heads", 0) or 0) or None
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
    align_layer_chunks = _chunk_layer_indices(align_layers, int(args.align_layer_chunk_size))

    benign_prompts = read_prompts(benign_jsonl)
    cached_batches = _preload_clean_batches(
        tokenizer,
        benign_prompts,
        max_length=args.max_length,
        prompt_template=str(args.prompt_template),
    )
    safe_target_meta = {
        "safe_target_mode": str(args.safe_target_mode),
        "safe_target_pool": [str(args.safe_target_text)],
        "safe_target_keyword_soft_text": None,
    }
    safe_batches = []
    if args.harmful_no_trigger_jsonl is not None and float(args.lambda_safe) > 0:
        safe_prompts = read_prompts(args.harmful_no_trigger_jsonl)
        safe_assignments, safe_target_meta = _resolve_safe_target_assignments(
            safe_prompts,
            safe_target_mode=str(args.safe_target_mode),
            safe_target_text=str(args.safe_target_text),
        )
        safe_batches = _preload_safe_batches(
            tokenizer,
            safe_assignments,
            max_length=args.max_length,
            prompt_template=str(args.prompt_template),
        )
    batch_count = len(cached_batches)
    if batch_count == 0:
        raise RuntimeError("No usable clean samples for recovery")
    safe_batch_count = len(safe_batches)

    notes = limitations_notes()
    print("=== Trigger-agnostic defense notes ===")
    for note in notes:
        print(f"- {note}")

    if args.mask_policy != "none" and pruned_units:
        apply_structured_prune(
            pruner,
            to_prune=pruned_units,
            head_dim=head_dim,
            num_key_value_heads=num_key_value_heads,
        )
        print(f"Applied pruning mask before recovery: {len(pruned_units)} units ({args.mask_policy})")

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = None
    cpu_grad_buffers: list[torch.Tensor] | None = None
    if str(args.grad_offload) == "cpu":
        cpu_grad_buffers = _cpu_grad_buffers(trainable)
    else:
        optimizer = torch.optim.SGD(trainable, lr=args.lr) if args.optimizer == "sgd" else torch.optim.AdamW(trainable, lr=args.lr)
    memory_events: list[dict[str, object]] = []
    current_stage = "init"
    memory_debug_enabled = bool(args.memory_debug and _is_cuda_device(pruner.device))
    if memory_debug_enabled:
        torch.cuda.reset_peak_memory_stats(pruner.device)
        _record_memory_event(
            memory_events,
            enabled=True,
            device=pruner.device,
            stage="recovery_start",
            step=0,
            note="before recovery loop",
        )

    recovery_losses: list[dict[str, float]] = []
    first_nonfinite_step: int | None = None
    first_nonfinite_reason: str | None = None
    ema_clean: torch.Tensor | None = None
    ema_align: torch.Tensor | None = None
    ema_safe: torch.Tensor | None = None
    ema_l1: torch.Tensor | None = None
    model.train()
    sequential_backward = bool(bool(args.stable_loss_mode) or str(args.loss_normalization) in {"none", "ema_ratio"})
    if int(args.align_layer_chunk_size) > 0 and not sequential_backward:
        raise ValueError("--align-layer-chunk-size currently requires stable-loss-mode or loss-normalization in {none, ema_ratio}")
    if str(args.grad_offload) == "cpu" and str(args.optimizer) != "sgd":
        raise ValueError("--grad-offload=cpu currently supports --optimizer sgd only")

    for step in range(args.steps):
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        step_lambda_clean, step_lambda_align, step_lambda_safe = _scheduled_objective_weights(
            step_index=step,
            total_steps=int(args.steps),
            lambda_clean=float(args.lambda_clean),
            lambda_align=float(args.lambda_align),
            lambda_safe=float(args.lambda_safe),
            objective_schedule=str(args.objective_schedule),
            alternating_soft_safe_ratio=float(args.alternating_soft_safe_ratio),
        )
        raw_clean_total = 0.0
        raw_align_total = 0.0
        raw_safe_total = 0.0
        raw_l1_total = 0.0
        norm_clean_total = 0.0
        norm_align_total = 0.0
        norm_safe_total = 0.0
        norm_l1_total = 0.0
        finite_batches = 0
        step_nonfinite_reasons: set[str] = set()
        grad_norm_value = float("nan")
        post_step_param_nonfinite = False
        optimizer_step_applied = False
        track_memory_this_step = bool(memory_debug_enabled and step == 0)
        if track_memory_this_step:
            torch.cuda.reset_peak_memory_stats(pruner.device)
            _record_memory_event(
                memory_events,
                enabled=True,
                device=pruner.device,
                stage="step_start",
                step=step + 1,
                note="after optimizer.zero_grad",
            )

        for accum_step in range(args.grad_accum_steps):
            batch_idx = (step * args.grad_accum_steps + accum_step) % batch_count
            current_stage = "move_benign_batch"
            benign_batch = pruner._move_to_device(cached_batches[batch_idx]["clean"])
            _record_memory_event(
                memory_events,
                enabled=track_memory_this_step,
                device=pruner.device,
                stage="after_move_benign",
                step=step + 1,
                accum_step=accum_step + 1,
                objective="benign",
            )
            current_stage = "move_safe_batch"
            safe_batch = (
                pruner._move_to_device(safe_batches[(step * args.grad_accum_steps + accum_step) % safe_batch_count]["safe"])
                if safe_batch_count > 0
                else None
            )
            _record_memory_event(
                memory_events,
                enabled=track_memory_this_step,
                device=pruner.device,
                stage="after_move_safe",
                step=step + 1,
                accum_step=accum_step + 1,
                objective="safe" if safe_batch is not None else "none",
            )

            clean_loss = torch.zeros((), device=pruner.device, dtype=torch.float32)
            proxy_loss = torch.zeros((), device=pruner.device, dtype=torch.float32)
            safe_loss = None
            l1_tensor: torch.Tensor | None = None
            clean_norm = torch.zeros((), device=pruner.device, dtype=torch.float32)
            proxy_norm = torch.zeros((), device=pruner.device, dtype=torch.float32)
            safe_norm = torch.zeros((), device=pruner.device, dtype=torch.float32)
            l1_norm = torch.zeros((), device=pruner.device, dtype=torch.float32)
            total_value = 0.0

            try:
                if step_lambda_align > 0:
                    if sequential_backward and int(args.align_layer_chunk_size) > 0:
                        raw_proxy_value = 0.0
                        for chunk_idx, layer_chunk in enumerate(align_layer_chunks):
                            current_stage = f"proxy_chunk_{chunk_idx + 1}_forward"
                            with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                                proxy_chunk_loss = _compute_proxy_alignment_loss_recovery(
                                    pruner,
                                    benign_batch,
                                    layer_indices=layer_chunk,
                                    layer_weights=layer_weights,
                                    trainable_parameters=trainable,
                                    adv_epsilon=float(args.proxy_epsilon),
                                    eps=float(args.norm_eps),
                                    saved_tensor_offload=str(args.saved_tensor_offload),
                                )
                            _record_memory_event(
                                memory_events,
                                enabled=track_memory_this_step,
                                device=pruner.device,
                                stage="after_proxy_chunk_forward",
                                step=step + 1,
                                accum_step=accum_step + 1,
                                objective="proxy_alignment",
                                note=f"chunk={chunk_idx + 1}/{len(align_layer_chunks)} layers={layer_chunk[0]}-{layer_chunk[-1]}",
                            )
                            if not torch.isfinite(proxy_chunk_loss).item():
                                step_nonfinite_reasons.add("align_loss")
                                raise RuntimeError("Non-finite proxy chunk loss")
                            proxy_chunk_norm = _normalize_single_loss_term(
                                proxy_chunk_loss,
                                reference=ema_align,
                                loss_normalization=str(args.loss_normalization),
                                norm_eps=float(args.norm_eps),
                                stable_loss_mode=bool(args.stable_loss_mode),
                            )
                            current_stage = f"proxy_chunk_{chunk_idx + 1}_backward"
                            (step_lambda_align * proxy_chunk_norm / args.grad_accum_steps).backward()
                            if cpu_grad_buffers is not None:
                                _offload_trainable_grads_to_cpu(trainable, cpu_grad_buffers)
                            _record_memory_event(
                                memory_events,
                                enabled=track_memory_this_step,
                                device=pruner.device,
                                stage="after_proxy_chunk_backward",
                                step=step + 1,
                                accum_step=accum_step + 1,
                                objective="proxy_alignment",
                                note=f"chunk={chunk_idx + 1}/{len(align_layer_chunks)}",
                            )
                            raw_proxy_value += float(proxy_chunk_loss.detach().item())
                            del proxy_chunk_loss, proxy_chunk_norm
                            if bool(args.empty_cache_between_objectives) and _is_cuda_device(pruner.device):
                                torch.cuda.empty_cache()
                        proxy_loss = torch.tensor(raw_proxy_value, device=pruner.device, dtype=torch.float32)
                        proxy_norm = _normalize_single_loss_term(
                            proxy_loss,
                            reference=ema_align,
                            loss_normalization=str(args.loss_normalization),
                            norm_eps=float(args.norm_eps),
                            stable_loss_mode=bool(args.stable_loss_mode),
                        )
                    else:
                        current_stage = "proxy_forward"
                        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                            proxy_loss = _compute_proxy_alignment_loss_recovery(
                                pruner,
                                benign_batch,
                                layer_indices=align_layers,
                                layer_weights=layer_weights,
                                trainable_parameters=trainable,
                                adv_epsilon=float(args.proxy_epsilon),
                                eps=float(args.norm_eps),
                                saved_tensor_offload=str(args.saved_tensor_offload),
                            )
                        _record_memory_event(
                            memory_events,
                            enabled=track_memory_this_step,
                            device=pruner.device,
                            stage="after_proxy_forward",
                            step=step + 1,
                            accum_step=accum_step + 1,
                            objective="proxy_alignment",
                        )
                        proxy_norm = _normalize_single_loss_term(
                            proxy_loss,
                            reference=ema_align,
                            loss_normalization=str(args.loss_normalization),
                            norm_eps=float(args.norm_eps),
                            stable_loss_mode=bool(args.stable_loss_mode),
                        )
                        if sequential_backward:
                            current_stage = "proxy_backward"
                            (step_lambda_align * proxy_norm / args.grad_accum_steps).backward()
                            if cpu_grad_buffers is not None:
                                _offload_trainable_grads_to_cpu(trainable, cpu_grad_buffers)
                            _record_memory_event(
                                memory_events,
                                enabled=track_memory_this_step,
                                device=pruner.device,
                                stage="after_proxy_backward",
                                step=step + 1,
                                accum_step=accum_step + 1,
                                objective="proxy_alignment",
                            )
                        if bool(args.empty_cache_between_objectives) and _is_cuda_device(pruner.device):
                            torch.cuda.empty_cache()

                if step_lambda_clean > 0:
                    current_stage = "clean_forward"
                    with _saved_tensor_context(str(args.saved_tensor_offload), pruner.device):
                        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                            clean_loss = pruner._extract_loss(benign_batch, loss_fn=None)
                    _record_memory_event(
                        memory_events,
                        enabled=track_memory_this_step,
                        device=pruner.device,
                        stage="after_clean_forward",
                        step=step + 1,
                        accum_step=accum_step + 1,
                        objective="benign",
                    )
                    clean_norm = _normalize_single_loss_term(
                        clean_loss,
                        reference=ema_clean,
                        loss_normalization=str(args.loss_normalization),
                        norm_eps=float(args.norm_eps),
                        stable_loss_mode=bool(args.stable_loss_mode),
                    )
                    if sequential_backward:
                        current_stage = "clean_backward"
                        (step_lambda_clean * clean_norm / args.grad_accum_steps).backward()
                        if cpu_grad_buffers is not None:
                            _offload_trainable_grads_to_cpu(trainable, cpu_grad_buffers)
                        _record_memory_event(
                            memory_events,
                            enabled=track_memory_this_step,
                            device=pruner.device,
                            stage="after_clean_backward",
                            step=step + 1,
                            accum_step=accum_step + 1,
                            objective="benign",
                        )
                    if bool(args.empty_cache_between_objectives) and _is_cuda_device(pruner.device):
                        torch.cuda.empty_cache()

                if step_lambda_safe > 0 and safe_batch is not None:
                    current_stage = "safe_forward"
                    with _saved_tensor_context(str(args.saved_tensor_offload), pruner.device):
                        with torch.autocast(device_type="cuda", dtype=dtype, enabled=use_autocast):
                            safe_loss = pruner._extract_loss(safe_batch, loss_fn=None)
                    _record_memory_event(
                        memory_events,
                        enabled=track_memory_this_step,
                        device=pruner.device,
                        stage="after_safe_forward",
                        step=step + 1,
                        accum_step=accum_step + 1,
                        objective="safe",
                    )
                    safe_norm = _normalize_single_loss_term(
                        safe_loss,
                        reference=ema_safe,
                        loss_normalization=str(args.loss_normalization),
                        norm_eps=float(args.norm_eps),
                        stable_loss_mode=bool(args.stable_loss_mode),
                    )
                    if sequential_backward:
                        current_stage = "safe_backward"
                        (step_lambda_safe * safe_norm / args.grad_accum_steps).backward()
                        if cpu_grad_buffers is not None:
                            _offload_trainable_grads_to_cpu(trainable, cpu_grad_buffers)
                        _record_memory_event(
                            memory_events,
                            enabled=track_memory_this_step,
                            device=pruner.device,
                            stage="after_safe_backward",
                            step=step + 1,
                            accum_step=accum_step + 1,
                            objective="safe",
                        )
                    if bool(args.empty_cache_between_objectives) and _is_cuda_device(pruner.device):
                        torch.cuda.empty_cache()

                if float(args.lambda_reg) > 0:
                    current_stage = "l1_forward"
                    l1_tensor = sum(parameter.abs().sum() for parameter in pruner.model.parameters() if parameter.requires_grad)
                    l1_tensor = l1_tensor.to(device=pruner.device, dtype=torch.float32)
                    l1_norm = _normalize_single_loss_term(
                        l1_tensor,
                        reference=ema_l1,
                        loss_normalization=str(args.loss_normalization),
                        norm_eps=float(args.norm_eps),
                        stable_loss_mode=bool(args.stable_loss_mode),
                    )
                    if sequential_backward:
                        current_stage = "l1_backward"
                        (float(args.lambda_reg) * l1_norm / args.grad_accum_steps).backward()
                        if cpu_grad_buffers is not None:
                            _offload_trainable_grads_to_cpu(trainable, cpu_grad_buffers)

                if not sequential_backward:
                    current_stage = "total_backward"
                    total_loss, clean_norm, proxy_norm, safe_norm, l1_norm = _compose_recovery_loss(
                        clean_loss=clean_loss,
                        proxy_loss=proxy_loss,
                        safe_loss=safe_loss,
                        l1_tensor=l1_tensor,
                        lambda_clean=step_lambda_clean,
                        lambda_align=step_lambda_align,
                        lambda_safe=step_lambda_safe,
                        lambda_reg=float(args.lambda_reg),
                        loss_normalization=str(args.loss_normalization),
                        norm_eps=float(args.norm_eps),
                        ema_clean=ema_clean,
                        ema_align=ema_align,
                        ema_safe=ema_safe,
                        ema_l1=ema_l1,
                        stable_loss_mode=bool(args.stable_loss_mode),
                    )
                    total_value = float(total_loss.detach().item())
                    (total_loss / args.grad_accum_steps).backward()
                    if cpu_grad_buffers is not None:
                        _offload_trainable_grads_to_cpu(trainable, cpu_grad_buffers)
                    _record_memory_event(
                        memory_events,
                        enabled=track_memory_this_step,
                        device=pruner.device,
                        stage="after_total_backward",
                        step=step + 1,
                        accum_step=accum_step + 1,
                        objective="mixed",
                    )
                else:
                    total_value = (
                        step_lambda_clean * float(clean_norm.detach().item())
                        + step_lambda_align * float(proxy_norm.detach().item())
                        + step_lambda_safe * float(safe_norm.detach().item())
                        + float(args.lambda_reg) * float(l1_norm.detach().item())
                    )
            except Exception:
                _record_memory_event(
                    memory_events,
                    enabled=track_memory_this_step,
                    device=pruner.device,
                    stage="exception",
                    step=step + 1,
                    accum_step=accum_step + 1,
                    note=current_stage,
                )
                if memory_debug_enabled:
                    _write_recovery_memory_debug(
                        memory_debug_path,
                        {
                            "timestamp": now_ts(),
                            "status": "failed",
                            "failed_stage": current_stage,
                            "failed_step": int(step + 1),
                            "failed_accum_step": int(accum_step + 1),
                            "config": {
                                "model_path_effective": effective_model_path,
                                "dtype": str(args.dtype),
                                "prompt_template": str(args.prompt_template),
                                "trainable_policy": str(args.trainable_policy),
                                "trainable_layers": trainable_scope_meta.get("trainable_layers"),
                                "enabled_param_count": trainable_scope_meta.get("enabled_param_count"),
                                "total_param_count": trainable_scope_meta.get("total_param_count"),
                                "lambda_clean": float(args.lambda_clean),
                                "lambda_align": float(args.lambda_align),
                                "lambda_safe": float(args.lambda_safe),
                                "loss_normalization": str(args.loss_normalization),
                                "stable_loss_mode": bool(args.stable_loss_mode),
                                "gradient_checkpointing": bool(args.gradient_checkpointing),
                                "grad_offload": str(args.grad_offload),
                                "saved_tensor_offload": str(args.saved_tensor_offload),
                                "align_layer_chunk_size": int(args.align_layer_chunk_size),
                                "grad_accum_steps": int(args.grad_accum_steps),
                                "max_length": int(args.max_length),
                                "empty_cache_between_objectives": bool(args.empty_cache_between_objectives),
                            },
                            "events": memory_events,
                            "recovery_losses_partial": recovery_losses,
                        },
                    )
                raise
            finally:
                del benign_batch
                if safe_batch is not None:
                    del safe_batch
                if bool(args.empty_cache_between_objectives) and _is_cuda_device(pruner.device):
                    torch.cuda.empty_cache()
                _record_memory_event(
                    memory_events,
                    enabled=track_memory_this_step,
                    device=pruner.device,
                    stage="after_microstep_cleanup",
                    step=step + 1,
                    accum_step=accum_step + 1,
                )

            if not torch.isfinite(clean_loss).item():
                step_nonfinite_reasons.add("clean_loss")
            if not torch.isfinite(proxy_loss).item():
                step_nonfinite_reasons.add("align_loss")
            if safe_loss is not None and not torch.isfinite(safe_loss).item():
                step_nonfinite_reasons.add("safe_loss")
            if not math.isfinite(total_value):
                step_nonfinite_reasons.add("total_loss")
            if not math.isfinite(total_value):
                continue

            finite_batches += 1
            raw_clean_total += float(clean_loss.detach().item())
            raw_align_total += float(proxy_loss.detach().item())
            raw_safe_total += float(safe_loss.detach().item()) if safe_loss is not None else 0.0
            norm_clean_total += float(clean_norm.detach().item())
            norm_align_total += float(proxy_norm.detach().item())
            norm_safe_total += float(safe_norm.detach().item()) if safe_loss is not None else 0.0
            if l1_tensor is not None:
                raw_l1_total += float(l1_tensor.detach().item())
                norm_l1_total += float(l1_norm.detach().item())

        if finite_batches == 0:
            if first_nonfinite_step is None and step_nonfinite_reasons:
                first_nonfinite_step = int(step + 1)
                first_nonfinite_reason = _first_nonfinite_reason(step_nonfinite_reasons)
                if bool(args.debug_save_first_nonfinite):
                    _save_debug_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        output_dir=debug_checkpoint_dir / f"step_{step + 1:03d}_first_nonfinite",
                    )
            recovery_losses.append(
                {
                    "step": float(step + 1),
                    "l_clean": float("nan"),
                    "l_align": float("nan"),
                    "l_safe": float("nan"),
                    "l1_proxy": float("nan"),
                    "l_clean_norm": float("nan"),
                    "l_align_norm": float("nan"),
                    "l_safe_norm": float("nan"),
                    "l1_proxy_norm": float("nan"),
                    "loss_total": float("nan"),
                    "grad_norm": float("nan"),
                    "lambda_clean_step": float(step_lambda_clean),
                    "lambda_align_step": float(step_lambda_align),
                    "lambda_safe_step": float(step_lambda_safe),
                    "nonfinite_clean_loss": bool("clean_loss" in step_nonfinite_reasons),
                    "nonfinite_align_loss": bool("align_loss" in step_nonfinite_reasons),
                    "nonfinite_safe_loss": bool("safe_loss" in step_nonfinite_reasons),
                    "nonfinite_total_loss": bool("total_loss" in step_nonfinite_reasons),
                    "nonfinite_grad_norm": False,
                    "optimizer_step_applied": False,
                    "post_step_param_nonfinite": False,
                }
            )
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            continue

        if cpu_grad_buffers is not None:
            grad_norm_value = _clip_cpu_grad_buffers(cpu_grad_buffers, float(args.max_grad_norm))
        else:
            grad_norm_value = _compute_grad_norm(trainable)
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, max_norm=float(args.max_grad_norm))
        if not math.isfinite(grad_norm_value):
            step_nonfinite_reasons.add("grad_norm")
        _record_memory_event(
            memory_events,
            enabled=track_memory_this_step,
            device=pruner.device,
            stage="before_optimizer_step",
            step=step + 1,
            note=f"grad_norm={grad_norm_value}",
        )
        if cpu_grad_buffers is not None:
            _sgd_step_from_cpu_grads(trainable, cpu_grad_buffers, lr=float(args.lr))
        else:
            optimizer.step()
        optimizer_step_applied = True
        _record_memory_event(
            memory_events,
            enabled=track_memory_this_step,
            device=pruner.device,
            stage="after_optimizer_step",
            step=step + 1,
        )
        if args.mask_policy == "strict" and pruned_units:
            apply_structured_prune(
                pruner,
                to_prune=pruned_units,
                head_dim=head_dim,
                num_key_value_heads=num_key_value_heads,
            )
            _record_memory_event(
                memory_events,
                enabled=track_memory_this_step,
                device=pruner.device,
                stage="after_strict_reapply_prune",
                step=step + 1,
            )
        post_step_param_nonfinite = _has_nonfinite_parameter(trainable)
        if post_step_param_nonfinite:
            step_nonfinite_reasons.add("parameter")

        avg_clean = raw_clean_total / finite_batches
        avg_align = raw_align_total / finite_batches
        avg_safe = raw_safe_total / finite_batches if safe_batch_count > 0 else float("nan")
        avg_l1 = raw_l1_total / finite_batches if float(args.lambda_reg) > 0 else float("nan")
        avg_clean_norm = norm_clean_total / finite_batches
        avg_align_norm = norm_align_total / finite_batches
        avg_safe_norm = norm_safe_total / finite_batches if safe_batch_count > 0 else float("nan")
        avg_l1_norm = norm_l1_total / finite_batches if float(args.lambda_reg) > 0 else float("nan")
        total_value = step_lambda_clean * avg_clean_norm + step_lambda_align * avg_align_norm
        if step_lambda_safe > 0 and avg_safe_norm == avg_safe_norm:
            total_value += step_lambda_safe * avg_safe_norm
        if float(args.lambda_reg) > 0 and avg_l1_norm == avg_l1_norm:
            total_value += float(args.lambda_reg) * avg_l1_norm

        if first_nonfinite_step is None and step_nonfinite_reasons:
            first_nonfinite_step = int(step + 1)
            first_nonfinite_reason = _first_nonfinite_reason(step_nonfinite_reasons)
            if bool(args.debug_save_first_nonfinite):
                _save_debug_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    output_dir=debug_checkpoint_dir / f"step_{step + 1:03d}_first_nonfinite",
                )

        recovery_losses.append(
            {
                "step": float(step + 1),
                "l_clean": avg_clean,
                "l_align": avg_align,
                "l_safe": avg_safe,
                "l1_proxy": avg_l1,
                "l_clean_norm": avg_clean_norm,
                "l_align_norm": avg_align_norm,
                "l_safe_norm": avg_safe_norm,
                "l1_proxy_norm": avg_l1_norm,
                "loss_total": total_value,
                "grad_norm": grad_norm_value,
                "lambda_clean_step": float(step_lambda_clean),
                "lambda_align_step": float(step_lambda_align),
                "lambda_safe_step": float(step_lambda_safe),
                "nonfinite_clean_loss": bool("clean_loss" in step_nonfinite_reasons),
                "nonfinite_align_loss": bool("align_loss" in step_nonfinite_reasons),
                "nonfinite_safe_loss": bool("safe_loss" in step_nonfinite_reasons),
                "nonfinite_total_loss": bool("total_loss" in step_nonfinite_reasons),
                "nonfinite_grad_norm": bool("grad_norm" in step_nonfinite_reasons),
                "optimizer_step_applied": bool(optimizer_step_applied),
                "post_step_param_nonfinite": bool(post_step_param_nonfinite),
            }
        )

        if int(step + 1) in debug_save_steps:
            _save_debug_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=debug_checkpoint_dir / f"step_{step + 1:03d}",
            )

        if (not args.stable_loss_mode) and args.loss_normalization == "ema_ratio":
            beta = float(args.norm_ema_beta)
            clean_ref = torch.tensor(avg_clean, device=pruner.device, dtype=torch.float32)
            align_ref = torch.tensor(avg_align, device=pruner.device, dtype=torch.float32)
            ema_clean = clean_ref if ema_clean is None else beta * ema_clean + (1.0 - beta) * clean_ref
            ema_align = align_ref if ema_align is None else beta * ema_align + (1.0 - beta) * align_ref
            if safe_batch_count > 0 and avg_safe == avg_safe:
                safe_ref = torch.tensor(avg_safe, device=pruner.device, dtype=torch.float32)
                ema_safe = safe_ref if ema_safe is None else beta * ema_safe + (1.0 - beta) * safe_ref
            if float(args.lambda_reg) > 0 and avg_l1 == avg_l1:
                l1_ref = torch.tensor(avg_l1, device=pruner.device, dtype=torch.float32)
                ema_l1 = l1_ref if ema_l1 is None else beta * ema_l1 + (1.0 - beta) * l1_ref

    if memory_debug_enabled:
        _write_recovery_memory_debug(
            memory_debug_path,
            {
                "timestamp": now_ts(),
                "status": "completed",
                "config": {
                    "model_path_effective": effective_model_path,
                    "dtype": str(args.dtype),
                    "prompt_template": str(args.prompt_template),
                    "trainable_policy": str(args.trainable_policy),
                    "trainable_layers": trainable_scope_meta.get("trainable_layers"),
                    "enabled_param_count": trainable_scope_meta.get("enabled_param_count"),
                    "total_param_count": trainable_scope_meta.get("total_param_count"),
                    "lambda_clean": float(args.lambda_clean),
                    "lambda_align": float(args.lambda_align),
                    "lambda_safe": float(args.lambda_safe),
                    "loss_normalization": str(args.loss_normalization),
                    "stable_loss_mode": bool(args.stable_loss_mode),
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "grad_offload": str(args.grad_offload),
                    "saved_tensor_offload": str(args.saved_tensor_offload),
                    "align_layer_chunk_size": int(args.align_layer_chunk_size),
                    "grad_accum_steps": int(args.grad_accum_steps),
                    "max_length": int(args.max_length),
                    "empty_cache_between_objectives": bool(args.empty_cache_between_objectives),
                },
                "events": memory_events,
                "recovery_losses_partial": recovery_losses,
            },
        )

    (args.run_dir / "recovery_losses.json").write_text(
        json.dumps(
            {
                "timestamp": now_ts(),
                "config": {
                    "model_path_effective": effective_model_path,
                    "benign_jsonl": str(benign_jsonl),
                    "harmful_no_trigger_jsonl": None if args.harmful_no_trigger_jsonl is None else str(args.harmful_no_trigger_jsonl),
                    "safe_target_text": str(args.safe_target_text),
                    "safe_target_mode": str(args.safe_target_mode),
                    "safe_target_pool": list(safe_target_meta.get("safe_target_pool", [])),
                    "safe_target_keyword_soft_text": safe_target_meta.get("safe_target_keyword_soft_text"),
                    "trainable_policy": str(args.trainable_policy),
                    "trainable_layers": trainable_scope_meta.get("trainable_layers"),
                    "enabled_param_count": trainable_scope_meta.get("enabled_param_count"),
                    "total_param_count": trainable_scope_meta.get("total_param_count"),
                    "proxy_epsilon": float(args.proxy_epsilon),
                    "lambda_clean": float(args.lambda_clean),
                    "lambda_align": float(args.lambda_align),
                    "lambda_safe": float(args.lambda_safe),
                    "objective_schedule": str(args.objective_schedule),
                    "alternating_soft_safe_ratio": float(args.alternating_soft_safe_ratio),
                    "lambda_reg": float(args.lambda_reg),
                    "loss_normalization": str(args.loss_normalization),
                    "stable_loss_mode": bool(args.stable_loss_mode),
                    "grad_accum_steps": int(args.grad_accum_steps),
                    "mask_policy": str(args.mask_policy),
                    "pruned_total": int(pruning_meta.get("pruned_total", 0) or 0),
                    "num_key_value_heads": None if num_key_value_heads is None else int(num_key_value_heads),
                    "steps": int(args.steps),
                    "optimizer": str(args.optimizer),
                    "grad_offload": str(args.grad_offload),
                    "gradient_checkpointing": bool(args.gradient_checkpointing),
                    "lr": float(args.lr),
                    "max_grad_norm": float(args.max_grad_norm),
                    "debug_save_steps": sorted(int(step) for step in debug_save_steps),
                    "debug_save_first_nonfinite": bool(args.debug_save_first_nonfinite),
                    "debug_checkpoint_dir": str(debug_checkpoint_dir),
                    "first_nonfinite_step": first_nonfinite_step,
                    "first_nonfinite_reason": first_nonfinite_reason,
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
    save_model_and_tokenizer_safe(model, tokenizer, str(output_dir))

    print(f"Wrote losses to {args.run_dir / 'recovery_losses.json'}")
    print(f"Wrote recovered model to {output_dir}")


if __name__ == "__main__":
    main()
