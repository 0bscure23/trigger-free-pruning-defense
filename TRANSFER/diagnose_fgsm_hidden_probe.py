#!/usr/bin/env python3
"""Single-prompt FGSM hidden-state probe for cross-machine runtime comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_first_jsonl(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.loads(f.readline())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--prompt-template", default="alpaca", choices=["alpaca", "chat", "none"])
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--adv-epsilon", type=float, default=0.1)
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    repo = Path("/home/lizhy/plp/tfpd_repro_score_89d79b1")
    sys.path.insert(0, str(repo))
    from pipeline_utils import build_model_inputs  # noqa: PLC0415

    jsonl_path = Path(args.jsonl)
    item = load_first_jsonl(jsonl_path)
    instruction = str(item.get("instruction", ""))
    user_input = str(item.get("input", ""))

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map={"": args.device},
        low_cpu_mem_usage=True,
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    encoded, prompt_text = build_model_inputs(
        tok,
        instruction=instruction,
        user_input=user_input,
        prompt_template=args.prompt_template,
        add_generation_prompt=True,
        max_length=args.max_length,
    )
    encoded = {k: v.to(args.device) for k, v in encoded.items()}
    input_ids = encoded["input_ids"]
    attention_mask = encoded.get("attention_mask")

    embeds = model.get_input_embeddings()(input_ids).detach().requires_grad_(True)
    forward_kwargs = {"attention_mask": attention_mask} if attention_mask is not None else {}
    out = model(inputs_embeds=embeds, output_hidden_states=True, use_cache=False, **forward_kwargs)
    hs = out.hidden_states

    transition_values: list[float] = []
    transition_tensors = []
    for layer_idx in range(1, len(hs) - 2):
        value = 1.0 - F.cosine_similarity(hs[layer_idx], hs[layer_idx + 1], dim=-1).mean()
        transition_tensors.append(value)
        transition_values.append(float(value.detach().cpu()))
    loss = torch.stack(transition_tensors).mean()
    loss.backward(retain_graph=False)
    if embeds.grad is None:
        raise RuntimeError("inputs_embeds.grad is None")

    sign = embeds.grad.detach().sign()
    pert = args.adv_epsilon * sign
    sign_pos = int((sign > 0).sum().item())
    sign_neg = int((sign < 0).sum().item())
    sign_zero = int((sign == 0).sum().item())

    with torch.no_grad():
        out2 = model(
            inputs_embeds=(embeds.detach() + pert).detach(),
            output_hidden_states=True,
            use_cache=False,
            **forward_kwargs,
        )
    hs2 = out2.hidden_states
    layer = int(args.layer)
    diff = hs2[layer] - hs[layer].detach()
    layer_vec = hs2[layer][0, 0, :5].float().detach().cpu().tolist()

    result = {
        "model_path": args.model_path,
        "jsonl": str(jsonl_path),
        "jsonl_sha256": sha256_bytes(jsonl_path.read_bytes()),
        "prompt_template": args.prompt_template,
        "dtype": args.dtype,
        "device": args.device,
        "deterministic": bool(args.deterministic),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "transformers_model_class": type(model).__name__,
        "prompt": {
            "instruction": instruction,
            "input": user_input,
            "prompt_text_sha256": sha256_bytes(prompt_text.encode("utf-8")),
            "prompt_text": prompt_text,
        },
        "tokens": {
            "length": int(input_ids.shape[-1]),
            "input_ids": input_ids[0].detach().cpu().tolist(),
            "input_ids_sha256": sha256_bytes(bytes(str(input_ids[0].detach().cpu().tolist()), "utf-8")),
        },
        "consistency_per_transition_1_minus_cos": transition_values,
        "total_loss_mean": float(loss.detach().cpu()),
        "embed_grad": {
            "l2": float(torch.norm(embeds.grad.detach()).cpu()),
            "abs_mean": float(embeds.grad.detach().abs().mean().cpu()),
            "abs_max": float(embeds.grad.detach().abs().max().cpu()),
        },
        "pert_sign": {
            "positive": sign_pos,
            "negative": sign_neg,
            "zero": sign_zero,
        },
        f"hidden_{layer}": {
            "l2_diff_perturbed_minus_clean": float(torch.norm(diff).cpu()),
            "mean_abs_diff": float(diff.abs().mean().cpu()),
            "first_token_first5": layer_vec,
            "clean_first_token_first5": hs[layer][0, 0, :5].float().detach().cpu().tolist(),
        },
    }

    print("prompt_text_sha256:", result["prompt"]["prompt_text_sha256"])
    print("input_ids_len:", result["tokens"]["length"])
    print("input_ids_sha256:", result["tokens"]["input_ids_sha256"])
    print(
        "consistency per transition (1-cos):",
        [f"{x:.6f}" for x in transition_values],
    )
    print(f"total loss mean: {result['total_loss_mean']:.9f}")
    print(
        "pert sign:",
        f"+={sign_pos}",
        f"-={sign_neg}",
        f"0={sign_zero}",
    )
    print(f"embed_grad_l2: {result['embed_grad']['l2']:.9f}")
    print(
        f"hidden[{layer}] L2 diff (perturbed-clean):",
        f"{result[f'hidden_{layer}']['l2_diff_perturbed_minus_clean']:.9f}",
    )
    print(f"hidden[{layer}][0,0,:5]:", layer_vec)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print("wrote", out_path)


if __name__ == "__main__":
    main()
