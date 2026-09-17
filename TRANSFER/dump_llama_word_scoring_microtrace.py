#!/usr/bin/env python3
"""Dump per-sample scoring intermediates for the Llama word score mismatch.

This diagnostic does not prune, recover, or save a model.  It traces the same
clean/safe/proxy gradient quantities used by score_and_prune.py, but only for
the small union of units selected by the archived golden scores and by the
current re-score.  The output lets us check where the 21-unit golden set starts
to diverge from the current 61-unit set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


PROJECT = Path("/home/lizhy/plp/trigger-free-pruning-defense-round2")
sys.path.insert(0, str(PROJECT))

from pipeline_utils import (  # noqa: E402
    _consistency_loss_from_hidden_states,
    _get_module,
    _prepare_inputs_embeds_batch,
    _split_prompt_fields,
    _tokenize_one,
    build_model_inputs,
    load_backdoorllm_model_and_tokenizer,
    read_prompts,
)
from pruning_backend import BaseSafetyPruner  # noqa: E402


UnitKey = tuple[str, int, int]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_scores(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, dict) and isinstance(obj.get("scores"), list):
        return obj["scores"]
    if isinstance(obj, list):
        return obj
    raise ValueError(f"Unsupported score file format: {path}")


def select_units(scores: list[dict[str, Any]], *, threshold: float, min_layer: int) -> set[UnitKey]:
    out: set[UnitKey] = set()
    for item in scores:
        if int(item["layer"]) < int(min_layer):
            continue
        if float(item["score"]) <= float(threshold):
            out.add((str(item["component"]), int(item["layer"]), int(item["index"])))
    return out


def score_map(scores: list[dict[str, Any]]) -> dict[UnitKey, dict[str, Any]]:
    return {
        (str(item["component"]), int(item["layer"]), int(item["index"])): item
        for item in scores
    }


def unit_label(unit: UnitKey) -> str:
    return f"{unit[0]}:{unit[1]}:{unit[2]}"


def parse_unit_label(raw: str) -> UnitKey:
    comp, layer, index = raw.split(":")
    return comp, int(layer), int(index)


def make_layer_keys(layers: set[int]) -> dict[int, dict[str, str]]:
    return {
        layer: {
            "q": f"model.layers.{layer}.self_attn.q_proj",
            "k": f"model.layers.{layer}.self_attn.k_proj",
            "v": f"model.layers.{layer}.self_attn.v_proj",
            "o": f"model.layers.{layer}.self_attn.o_proj",
            "g": f"model.layers.{layer}.mlp.gate_proj",
            "u": f"model.layers.{layer}.mlp.up_proj",
            "d": f"model.layers.{layer}.mlp.down_proj",
        }
        for layer in sorted(layers)
    }


def get_grad(modules: dict[str, torch.nn.Module], name: str) -> torch.Tensor:
    grad = _get_module(modules, name).weight.grad
    if grad is None:
        raise RuntimeError(f"Missing gradient for {name}")
    return grad


def channel_vec(modules: dict[str, torch.nn.Module], keys: dict[str, str], index: int) -> torch.Tensor:
    g = get_grad(modules, keys["g"])[index, :]
    u = get_grad(modules, keys["u"])[index, :]
    d = get_grad(modules, keys["d"])[:, index]
    return torch.cat([g.reshape(-1), u.reshape(-1), d.reshape(-1)]).detach().float().cpu()


def reshape_projection_grad(
    grad: torch.Tensor,
    *,
    projection: str,
    num_heads: int,
    num_key_value_heads: int | None,
    head_dim: int,
) -> torch.Tensor:
    if projection == "q":
        return grad.view(num_heads, head_dim, grad.shape[1])
    if projection == "o":
        return grad.permute(1, 0).contiguous().view(num_heads, head_dim, grad.shape[0])
    if projection in {"k", "v"}:
        kv_heads = int(num_key_value_heads or num_heads)
        reshaped = grad.view(kv_heads, head_dim, grad.shape[1])
        if kv_heads == num_heads:
            return reshaped
        group_size = num_heads // kv_heads
        return reshaped.repeat_interleave(group_size, dim=0)
    raise ValueError(projection)


def head_vec(
    modules: dict[str, torch.nn.Module],
    keys: dict[str, str],
    index: int,
    *,
    num_heads: int,
    num_key_value_heads: int | None,
    head_dim: int,
) -> torch.Tensor:
    parts = []
    for proj in ("q", "k", "v", "o"):
        reshaped = reshape_projection_grad(
            get_grad(modules, keys[proj]),
            projection=proj,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        parts.append(reshaped[index].reshape(-1))
    return torch.cat(parts).detach().float().cpu()


def collect_unit_vectors(
    modules: dict[str, torch.nn.Module],
    layer_keys: dict[int, dict[str, str]],
    units: set[UnitKey],
    *,
    num_heads: int,
    num_key_value_heads: int | None,
    head_dim: int,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for comp, layer, index in sorted(units):
        keys = layer_keys[layer]
        if comp == "channel":
            out[unit_label((comp, layer, index))] = channel_vec(modules, keys, index)
        elif comp == "head":
            out[unit_label((comp, layer, index))] = head_vec(
                modules,
                keys,
                index,
                num_heads=num_heads,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
            )
        else:
            raise ValueError(f"Unsupported component {comp}")
    return out


def backward_lm_loss(
    model: torch.nn.Module,
    pruner: BaseSafetyPruner,
    batch: dict[str, Any],
) -> float:
    model.zero_grad(set_to_none=True)
    moved = pruner._move_to_device(batch)
    loss = pruner._extract_loss(moved, loss_fn=None)
    loss.backward(retain_graph=False)
    return float(loss.detach().float().cpu().item())


def backward_proxy_loss(
    model: torch.nn.Module,
    pruner: BaseSafetyPruner,
    batch: dict[str, Any],
    *,
    eps: float,
    proxy_epsilon: float,
) -> dict[str, Any]:
    model.zero_grad(set_to_none=True)
    working = pruner._move_to_device(dict(batch))
    if "labels" not in working:
        working["labels"] = working["input_ids"].clone()

    inputs_embeds, model_inputs = _prepare_inputs_embeds_batch(model, working)
    consistency_inputs = {key: value for key, value in model_inputs.items() if key != "labels"}

    outputs = model(
        inputs_embeds=inputs_embeds,
        output_hidden_states=True,
        use_cache=False,
        **consistency_inputs,
    )
    consistency_loss = _consistency_loss_from_hidden_states(outputs.hidden_states, eps)
    consistency_loss.backward(retain_graph=False)
    if inputs_embeds.grad is None:
        raise RuntimeError("Missing input embedding gradient")

    sign = inputs_embeds.grad.detach().sign()
    sign_stats = {
        "positive": int((sign > 0).sum().item()),
        "negative": int((sign < 0).sum().item()),
        "zero": int((sign == 0).sum().item()),
        "numel": int(sign.numel()),
    }
    perturbation = proxy_epsilon * sign

    model.zero_grad(set_to_none=True)
    perturbed_outputs = model(
        inputs_embeds=(inputs_embeds.detach() + perturbation).detach(),
        use_cache=False,
        **model_inputs,
    )
    perturbed_loss = getattr(perturbed_outputs, "loss", None)
    if perturbed_loss is None:
        raise RuntimeError("Perturbed forward did not produce loss")
    perturbed_loss.backward(retain_graph=False)
    return {
        "consistency_loss": float(consistency_loss.detach().float().cpu().item()),
        "perturbed_lm_loss": float(perturbed_loss.detach().float().cpu().item()),
        "perturb_sign": sign_stats,
    }


def prompt_fingerprint(tokenizer: Any, prompt: Any, *, prompt_template: str, max_length: int) -> dict[str, Any]:
    instruction, user_input = _split_prompt_fields(prompt)
    encoded, text = build_model_inputs(
        tokenizer,
        instruction=instruction,
        user_input=user_input,
        prompt_template=prompt_template,
        add_generation_prompt=True,
        max_length=max_length,
    )
    ids = encoded["input_ids"].cpu().numpy().astype("int64").tobytes()
    return {
        "prompt_text_sha256": sha256_bytes(text.encode("utf-8")),
        "input_ids_sha256": sha256_bytes(ids),
        "input_len": int(encoded["input_ids"].shape[-1]),
        "prompt_prefix": text[:160],
    }


def summarize_vectors(clean: torch.Tensor, proxy: torch.Tensor | None = None) -> dict[str, float]:
    out = {
        "mag": float(clean.abs().mean().item()),
        "l2": float(clean.norm().item()),
    }
    if proxy is not None:
        cosine = F.cosine_similarity(clean.reshape(1, -1), proxy.reshape(1, -1), dim=1, eps=1e-12)[0]
        out.update(
            {
                "proxy_mag": float(proxy.abs().mean().item()),
                "proxy_l2": float(proxy.norm().item()),
                "cosine": float(cosine.item()),
                "penalty": float(abs(float(proxy.abs().mean().item()) * float(cosine.item()))),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default="/home/lizhy/plp/Llama-3.1-8B_word")
    parser.add_argument("--clean-jsonl", type=Path, default=Path("/home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl"))
    parser.add_argument("--safe-jsonl", type=Path, default=Path("/home/lizhy/plp/TRANSFER/beat_data/harmful_no_trigger.jsonl"))
    parser.add_argument("--golden-scores", type=Path, default=Path("/home/lizhy/plp/unit_scores.json"))
    parser.add_argument(
        "--current-scores",
        type=Path,
        default=Path(
            "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_cu121_score_test/unit_scores.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_scoring_microtrace"
        ),
    )
    parser.add_argument("--score-samples", type=int, default=8)
    parser.add_argument("--prompt-template", default="alpaca", choices=["alpaca", "chat", "none"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--proxy-epsilon", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--alpha-safe", type=float, default=0.5)
    parser.add_argument("--min-layer", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    golden_scores = load_scores(args.golden_scores)
    current_scores = load_scores(args.current_scores)
    golden_map = score_map(golden_scores)
    current_map = score_map(current_scores)
    golden_units = select_units(golden_scores, threshold=args.threshold, min_layer=args.min_layer)
    current_units = select_units(current_scores, threshold=args.threshold, min_layer=args.min_layer)
    units = golden_units | current_units
    layers = {layer for _, layer, _ in units}

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=str(args.model_path),
        tokenizer_path=None,
        use_lora=False,
        lora_model_path=None,
        torch_dtype=dtype,
        merge_lora=True,
    )
    model.eval()
    pruner = BaseSafetyPruner(model)
    modules = dict(model.named_modules())
    hidden_size = int(getattr(model.config, "hidden_size", 0) or 0)
    num_heads = int(getattr(model.config, "num_attention_heads", 0) or 0)
    num_key_value_heads = int(getattr(model.config, "num_key_value_heads", 0) or 0) or None
    head_dim = hidden_size // num_heads
    layer_keys = make_layer_keys(layers)

    clean_prompts = read_prompts(args.clean_jsonl)[: args.score_samples]
    safe_prompts = read_prompts(args.safe_jsonl)[: args.score_samples]

    clean_records: list[dict[str, Any]] = []
    safe_records: list[dict[str, Any]] = []
    per_unit_clean: dict[str, list[dict[str, float]]] = defaultdict(list)
    per_unit_safe: dict[str, list[dict[str, float]]] = defaultdict(list)

    for sample_idx, prompt in enumerate(clean_prompts):
        batch = _tokenize_one(
            tokenizer,
            prompt,
            max_length=args.max_length,
            prompt_template=args.prompt_template,
        )
        batch["labels"] = batch["input_ids"].clone()
        prompt_fp = prompt_fingerprint(
            tokenizer,
            prompt,
            prompt_template=args.prompt_template,
            max_length=args.max_length,
        )
        clean_loss = backward_lm_loss(model, pruner, batch)
        clean_vecs = collect_unit_vectors(
            modules,
            layer_keys,
            units,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        proxy_meta = backward_proxy_loss(
            model,
            pruner,
            batch,
            eps=args.eps,
            proxy_epsilon=args.proxy_epsilon,
        )
        proxy_vecs = collect_unit_vectors(
            modules,
            layer_keys,
            units,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        unit_values = {}
        for label, clean_vec in clean_vecs.items():
            stats = summarize_vectors(clean_vec, proxy_vecs[label])
            per_unit_clean[label].append(stats)
            unit_values[label] = stats
        clean_records.append(
            {
                "sample_idx": sample_idx,
                "prompt": prompt_fp,
                "clean_lm_loss": clean_loss,
                **proxy_meta,
                "units": unit_values,
            }
        )
        model.zero_grad(set_to_none=True)

    for sample_idx, prompt in enumerate(safe_prompts):
        batch = _tokenize_one(
            tokenizer,
            prompt,
            max_length=args.max_length,
            prompt_template=args.prompt_template,
        )
        batch["labels"] = batch["input_ids"].clone()
        prompt_fp = prompt_fingerprint(
            tokenizer,
            prompt,
            prompt_template=args.prompt_template,
            max_length=args.max_length,
        )
        safe_loss = backward_lm_loss(model, pruner, batch)
        safe_vecs = collect_unit_vectors(
            modules,
            layer_keys,
            units,
            num_heads=num_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
        )
        unit_values = {}
        for label, safe_vec in safe_vecs.items():
            stats = {"safe_mag": float(safe_vec.abs().mean().item()), "safe_l2": float(safe_vec.norm().item())}
            per_unit_safe[label].append(stats)
            unit_values[label] = stats
        safe_records.append(
            {
                "sample_idx": sample_idx,
                "prompt": prompt_fp,
                "safe_lm_loss": safe_loss,
                "units": unit_values,
            }
        )
        model.zero_grad(set_to_none=True)

    aggregate_units: dict[str, Any] = {}
    for unit in sorted(units):
        label = unit_label(unit)
        clean_stats = per_unit_clean[label]
        safe_stats = per_unit_safe[label]
        clean_mean = sum(x["mag"] for x in clean_stats) / max(1, len(clean_stats))
        proxy_mean = sum(x["proxy_mag"] for x in clean_stats) / max(1, len(clean_stats))
        cosine_mean = sum(x["cosine"] for x in clean_stats) / max(1, len(clean_stats))
        safe_mean = sum(x["safe_mag"] for x in safe_stats) / max(1, len(safe_stats))
        protect = clean_mean + float(args.alpha_safe) * safe_mean
        penalty = abs(proxy_mean * cosine_mean)
        score = protect - penalty
        aggregate_units[label] = {
            "component": unit[0],
            "layer": unit[1],
            "index": unit[2],
            "in_golden_selected": unit in golden_units,
            "in_current_selected": unit in current_units,
            "microtrace_aggregate": {
                "clean_grad_mean": clean_mean,
                "proxy_grad_mean": proxy_mean,
                "cosine": cosine_mean,
                "safe_grad_mean": safe_mean,
                "protect_grad_mean": protect,
                "clean_proxy_penalty": penalty,
                "score": score,
            },
            "golden_unit_score": golden_map.get(unit),
            "current_unit_score": current_map.get(unit),
            "clean_samples": clean_stats,
            "safe_samples": safe_stats,
        }

    payload = {
        "config": {
            "model_path": str(args.model_path),
            "clean_jsonl": str(args.clean_jsonl),
            "safe_jsonl": str(args.safe_jsonl),
            "golden_scores": str(args.golden_scores),
            "current_scores": str(args.current_scores),
            "score_samples": args.score_samples,
            "prompt_template": args.prompt_template,
            "max_length": args.max_length,
            "dtype": args.dtype,
            "proxy_epsilon": args.proxy_epsilon,
            "eps": args.eps,
            "alpha_safe": args.alpha_safe,
            "threshold": args.threshold,
            "min_layer": args.min_layer,
        },
        "selected_sets": {
            "golden_count": len(golden_units),
            "current_count": len(current_units),
            "overlap_count": len(golden_units & current_units),
            "traced_unit_count": len(units),
            "traced_layers": sorted(layers),
        },
        "clean_records": clean_records,
        "safe_records": safe_records,
        "aggregate_units": aggregate_units,
    }

    out_json = args.out_dir / "microtrace.json"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
