#!/usr/bin/env python3
"""Pseudo-trigger pool scoring: use screened candidates as proxy triggers for unit scoring.

Computes unit scores by comparing clean LM loss gradients with pseudo-trigger-appended
prompt gradients. The trigger_prompts are clean prompts with a pseudo-trigger candidate
appended to the instruction. This is a candidate-pool / weak-trigger approach, NOT strict
trigger-free.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = Path(__file__).resolve().parent
_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_ROOT))

from pipeline_utils import (
    _tokenize_one, UnitScore, read_prompts,
    load_backdoorllm_model_and_tokenizer, now_ts,
)
from pruning_backend import BaseSafetyPruner


def _module_grad(module):
    w = module.weight
    g = w.grad
    return g.detach().clone() if g is not None else torch.zeros_like(w)


def _get_module(modules, name):
    if name in modules: return modules[name]
    for prefix in ("base_model.", "base_model.model."):
        if prefix + name in modules: return modules[prefix + name]
    raise KeyError(f"Module not found: {name}")


def compute_unit_scores_with_candidates(
    model, pruner, tokenizer, clean_prompts, candidate_texts,
    max_length=512, prompt_template="alpaca",
    head_dim=128, num_layers=32, num_heads=32, intermediate_size=11008,
    num_key_value_heads=None,
    score_pairs=8, alpha=1.0, beta=1.0, eps=1e-12,
):
    """Score units using pseudo-trigger candidate-appended prompts."""
    pair_count = min(len(clean_prompts), score_pairs)
    modules = dict(model.named_modules())

    layer_keys = []
    for layer in range(num_layers):
        layer_keys.append({
            "q": f"model.layers.{layer}.self_attn.q_proj",
            "k": f"model.layers.{layer}.self_attn.k_proj",
            "v": f"model.layers.{layer}.self_attn.v_proj",
            "o": f"model.layers.{layer}.self_attn.o_proj",
            "g": f"model.layers.{layer}.mlp.gate_proj",
            "u": f"model.layers.{layer}.mlp.up_proj",
            "d": f"model.layers.{layer}.mlp.down_proj",
        })

    # Accumulators
    c_mag_heads = np.zeros((num_layers, num_heads), dtype=np.float64)
    t_mag_heads = np.zeros((num_layers, num_heads), dtype=np.float64)
    cos_heads = np.zeros((num_layers, num_heads), dtype=np.float64)
    c_mag_chans = np.zeros((num_layers, intermediate_size), dtype=np.float64)
    t_mag_chans = np.zeros((num_layers, intermediate_size), dtype=np.float64)
    cos_chans = np.zeros((num_layers, intermediate_size), dtype=np.float64)

    for i in range(pair_count):
        # Clean forward+backward
        clean_prompt = clean_prompts[i]
        clean_b = _tokenize_one(tokenizer, clean_prompt, max_length=max_length, prompt_template=prompt_template)
        clean_b["labels"] = clean_b["input_ids"].clone()
        clean_b = pruner._move_to_device(clean_b)

        model.zero_grad(set_to_none=True)
        clean_loss = pruner._extract_loss(clean_b, loss_fn=None)
        clean_loss.backward(retain_graph=False)

        c_grads = []
        for lk in layer_keys:
            c_grads.append({key: _module_grad(_get_module(modules, name)).cpu() for key, name in lk.items()})

        # Pseudo-trigger gradients: average over candidates
        model.zero_grad(set_to_none=True)
        raw_total_loss = 0.0
        for ct in candidate_texts:
            # Create pseudo-triggered prompt by appending candidate to instruction
            instruction = clean_prompt.get("instruction", "") if isinstance(clean_prompt, dict) else str(clean_prompt)
            user_input = clean_prompt.get("input", "") if isinstance(clean_prompt, dict) else ""
            triggered = {"instruction": instruction + " " + ct, "input": user_input}
            trig_b = _tokenize_one(tokenizer, triggered, max_length=max_length, prompt_template=prompt_template)
            trig_b["labels"] = trig_b["input_ids"].clone()
            trig_b = pruner._move_to_device(trig_b)
            loss = pruner._extract_loss(trig_b, loss_fn=None)
            raw_total_loss += float(loss.detach().item())
            (loss / len(candidate_texts)).backward(retain_graph=False)

        t_grads = []
        for lk in layer_keys:
            t_grads.append({key: _module_grad(_get_module(modules, name)).cpu() for key, name in lk.items()})
        model.zero_grad(set_to_none=True)

        # Compute per-unit scores
        for layer in range(num_layers):
            cg = c_grads[layer]; tg = t_grads[layer]
            # Heads
            qc = cg["q"].view(num_heads, head_dim, -1)
            kc = cg["k"]; kc = _reshape_kv(kc, num_heads, head_dim, num_key_value_heads)
            vc = cg["v"]; vc = _reshape_kv(vc, num_heads, head_dim, num_key_value_heads)
            oc = cg["o"].permute(1,0).contiguous().view(num_heads, head_dim, -1)

            qt = tg["q"].view(num_heads, head_dim, -1)
            kt = tg["k"]; kt = _reshape_kv(kt, num_heads, head_dim, num_key_value_heads)
            vt = tg["v"]; vt = _reshape_kv(vt, num_heads, head_dim, num_key_value_heads)
            ot = tg["o"].permute(1,0).contiguous().view(num_heads, head_dim, -1)

            h_c = torch.cat([qc.reshape(num_heads,-1), kc.reshape(num_heads,-1), vc.reshape(num_heads,-1), oc.reshape(num_heads,-1)], dim=1)
            h_t = torch.cat([qt.reshape(num_heads,-1), kt.reshape(num_heads,-1), vt.reshape(num_heads,-1), ot.reshape(num_heads,-1)], dim=1)

            cm = h_c.abs().mean(dim=1).double().numpy()
            tm = h_t.abs().mean(dim=1).double().numpy()
            co = ((h_c * h_t).sum(dim=1) / (h_c.norm(dim=1) * h_t.norm(dim=1) + eps)).double().numpy()
            c_mag_heads[layer] += cm; t_mag_heads[layer] += tm; cos_heads[layer] += co

            # Channels
            cc = torch.cat([cg["g"], cg["u"], cg["d"].T], dim=1)
            ct_c = torch.cat([tg["g"], tg["u"], tg["d"].T], dim=1)
            cm_c = cc.abs().mean(dim=1).double().numpy()
            tm_c = ct_c.abs().mean(dim=1).double().numpy()
            co_c = ((cc / (cc.norm(dim=1,keepdim=True)+eps)) * (ct_c / (ct_c.norm(dim=1,keepdim=True)+eps))).sum(dim=1).double().numpy()
            c_mag_chans[layer] += cm_c; t_mag_chans[layer] += tm_c; cos_chans[layer] += co_c

    n = max(1.0, pair_count)
    out = []
    # Heads
    for layer in range(num_layers):
        for h in range(num_heads):
            cm = c_mag_heads[layer,h]/n; tm = t_mag_heads[layer,h]/n; co = cos_heads[layer,h]/n
            score = float(alpha*cm - beta*abs(tm*co))
            out.append(UnitScore(component="head", layer=layer, index=h, clean_grad_mean=float(cm), proxy_grad_mean=float(tm), cosine=float(co), score=score))
    # Channels
    for layer in range(num_layers):
        for ci in range(intermediate_size):
            cm = c_mag_chans[layer,ci]/n; tm = t_mag_chans[layer,ci]/n; co = cos_chans[layer,ci]/n
            score = float(alpha*cm - beta*abs(tm*co))
            out.append(UnitScore(component="channel", layer=layer, index=ci, clean_grad_mean=float(cm), proxy_grad_mean=float(tm), cosine=float(co), score=score))
    out.sort(key=lambda x: x.score)
    return out


def _reshape_kv(grad, num_heads, head_dim, num_key_value_heads=None):
    kv_heads = num_key_value_heads or num_heads
    if kv_heads == num_heads:
        return grad.view(num_heads, head_dim, -1)
    group = num_heads // kv_heads
    reshaped = grad.view(kv_heads, head_dim, -1)
    return reshaped.repeat_interleave(group, dim=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--candidates-json", type=Path, required=True, help="Screening output JSON with top10")
    p.add_argument("--candidate-pool", choices=["top", "random"], default="top")
    p.add_argument("--clean-jsonl", type=Path, required=True)
    p.add_argument("--prompt-template", default="alpaca")
    p.add_argument("--dtype", choices=["bf16","fp16"], default="bf16")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--score-pairs", type=int, default=8)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--kappa", type=float, default=1e9)
    p.add_argument("--max-prune-units", type=int, default=512)
    args = p.parse_args()

    from pipeline_utils import apply_structured_prune
    dtype = torch.bfloat16 if args.dtype=="bf16" else torch.float16

    model, tokenizer = load_backdoorllm_model_and_tokenizer(
        model_path=args.model_path, tokenizer_path=None,
        use_lora=False, lora_model_path=None, torch_dtype=dtype)
    pruner = BaseSafetyPruner(model)

    num_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    head_info = pruner._infer_llama_head_dim(hidden_size)
    num_heads, head_dim = head_info
    num_kv = getattr(model.config, "num_key_value_heads", None)

    # Load candidates
    screening = json.loads(args.candidates_json.read_text())
    if args.candidate_pool == "top":
        # Support both "top10" (v1) and "top20" (v2) keys
        top_key = "top20" if "top20" in screening else "top10"
        candidates = [c["text"] for c in screening[top_key]]
    else:
        candidates = [c["text"] for c in screening["random10"]]
    print(f"Using {args.candidate_pool} pool: {len(candidates)} candidates: {candidates[:3]}...")

    clean_prompts = read_prompts(args.clean_jsonl)

    scores = compute_unit_scores_with_candidates(
        model, pruner, tokenizer, clean_prompts, candidates,
        max_length=args.max_length, prompt_template=args.prompt_template,
        head_dim=head_dim, num_layers=num_layers, num_heads=num_heads,
        intermediate_size=intermediate_size, num_key_value_heads=num_kv,
        score_pairs=args.score_pairs, alpha=args.alpha, beta=args.beta, eps=args.eps)

    to_prune = [s for s in scores if s.score <= float(args.kappa)]
    if args.max_prune_units > 0:
        to_prune = to_prune[:args.max_prune_units]

    apply_structured_prune(pruner, to_prune=to_prune, head_dim=head_dim, num_key_value_heads=num_kv)
    args.run_dir.mkdir(parents=True, exist_ok=True)

    (args.run_dir/"unit_scores.json").write_text(
        json.dumps({"scores": [s.__dict__ for s in scores]}, indent=2, ensure_ascii=False)+"\n")
    (args.run_dir/"pruning_plan.json").write_text(json.dumps({
        "timestamp": now_ts(), "kappa": args.kappa, "max_prune_units": args.max_prune_units,
        "pruned_total": len(to_prune),
        "pruned_heads": sum(1 for s in to_prune if s.component=="head"),
        "pruned_channels": sum(1 for s in to_prune if s.component=="channel"),
        "num_key_value_heads": int(num_kv) if num_kv is not None else None,
        "candidate_pool": args.candidate_pool, "candidates": candidates,
        "to_prune": [s.__dict__ for s in to_prune],
    }, indent=2, ensure_ascii=False)+"\n")

    out_dir = args.run_dir/"pruned_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(out_dir)
    model.save_pretrained(out_dir)
    print(f"Wrote pruned model to {out_dir} ({len(to_prune)} units pruned)")


if __name__ == "__main__":
    main()
