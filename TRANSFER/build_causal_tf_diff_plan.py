#!/usr/bin/env python3
"""Differential trigger-free causal/logit-gap channel selection.

This is a deployable variant: it uses harmful-no-trigger and benign prompts only.
The score is a difference between refusal-suppression causal attribution on
harmful-no-trigger prompts and the same attribution on benign prompts:

    diff_z(u) = z_harm(u) - benign_weight * z_benign(u)

The intent is to keep the useful part of the one-sided refsupp proxy while
penalizing units that are also generic benign/refusal load-bearing channels.
"""
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_REFUSAL = "I'm sorry, but I cannot help with that request."

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--budget", type=int, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--harmful", required=True, help="harmful-no-trigger jsonl")
ap.add_argument("--benign", required=True, help="benign jsonl")
ap.add_argument("--perturb", choices=["refsupp", "cons"], default="refsupp")
ap.add_argument("--eps", type=float, default=0.01)
ap.add_argument("--benign-weight", type=float, default=1.0)
ap.add_argument("--template", default="chat", choices=["chat", "alpaca", "none"])
ap.add_argument("--nlim-harmful", type=int, default=120)
ap.add_argument("--nlim-benign", type=int, default=100)
ap.add_argument("--maxp", type=int, default=256)
ap.add_argument("--components", default="channel,head")
ap.add_argument("--refusal", default=DEFAULT_REFUSAL)
args = ap.parse_args()

COMPS = set(c.strip() for c in args.components.split(",") if c.strip())
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.bfloat16, device_map="auto"
)
model.eval()
for p in model.parameters():
    p.requires_grad_(False)

nlayers = model.config.num_hidden_layers
nheads = model.config.num_attention_heads
hdim = model.config.hidden_size // nheads
embed = model.get_input_embeddings()
dev0 = next(model.parameters()).device


def load_prompts(path, n):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            ins = d.get("instruction", "")
            if d.get("input"):
                ins = ins + "\n" + d["input"]
            rows.append(ins)
            if len(rows) >= n:
                break
    return rows


def fmt_ids(ins):
    if args.template == "chat":
        try:
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": ins}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = f"### Instruction:\n{ins}\n\n### Response:\n"
    elif args.template == "none":
        prompt = ins
    else:
        prompt = f"### Instruction:\n{ins}\n\n### Response:\n"
    p_ids = tok(prompt, return_tensors="pt", truncation=True, max_length=args.maxp).input_ids
    r_ids = tok(args.refusal, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p_ids, r_ids], dim=1)
    labels = torch.cat([torch.full_like(p_ids, -100), r_ids], dim=1)
    return ids, labels


def consistency_loss(hs):
    a = torch.stack(tuple(hs[1:-2]))
    b = torch.stack(tuple(hs[2:-1]))
    return (1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)).mean()


CAP = {"on": False}
store = {}


def mk_hook(comp, layer_idx):
    def hook(_mod, inp, _out):
        if not CAP["on"]:
            return
        t = inp[0]
        t.retain_grad()
        store[(comp, layer_idx)] = t
    return hook


hooks = []
for layer_idx in range(nlayers):
    layer = model.model.layers[layer_idx]
    if "channel" in COMPS:
        hooks.append(layer.mlp.down_proj.register_forward_hook(mk_hook("channel", layer_idx)))
    if "head" in COMPS:
        hooks.append(layer.self_attn.o_proj.register_forward_hook(mk_hook("head", layer_idx)))


def collect_attr(prompts, label):
    attr_by_group = {}
    nseen = 0
    for ins in prompts:
        ids, labels = fmt_ids(ins)
        ids = ids.to(dev0)
        labels = labels.to(dev0)
        if (labels != -100).sum() == 0:
            continue
        mask = torch.ones_like(ids)
        e0 = embed(ids).detach().requires_grad_(True)

        CAP["on"] = False
        model.zero_grad(set_to_none=True)
        if args.perturb == "refsupp":
            out = model(inputs_embeds=e0, attention_mask=mask, labels=labels, use_cache=False)
            loss = out.loss
        else:
            out = model(
                inputs_embeds=e0,
                attention_mask=mask,
                output_hidden_states=True,
                use_cache=False,
            )
            loss = consistency_loss(out.hidden_states)
        loss.backward()
        delta = args.eps * e0.grad.detach().sign()

        xadv = (e0.detach() + delta).detach().requires_grad_(True)
        store.clear()
        CAP["on"] = True
        model.zero_grad(set_to_none=True)
        out2 = model(inputs_embeds=xadv, attention_mask=mask, labels=labels, use_cache=False)
        out2.loss.backward()
        CAP["on"] = False

        for (comp, layer_idx), t in store.items():
            if t.grad is None:
                continue
            ag = (t.detach().float() * t.grad.detach().float())[0]
            if comp == "channel":
                vals = ag.sum(dim=0)
                key = (comp, layer_idx)
                vals_np = vals.cpu().numpy().astype(np.float64, copy=False)
                if key not in attr_by_group:
                    attr_by_group[key] = vals_np.copy()
                else:
                    attr_by_group[key] += vals_np
            else:
                tlen, hidden = ag.shape
                vals = ag.view(tlen, nheads, hdim).sum(dim=(0, 2))
                key = (comp, layer_idx)
                vals_np = vals.cpu().numpy().astype(np.float64, copy=False)
                if key not in attr_by_group:
                    attr_by_group[key] = vals_np.copy()
                else:
                    attr_by_group[key] += vals_np
        nseen += 1
    attr = {}
    for (comp, layer_idx), vals in attr_by_group.items():
        for idx, value in enumerate(vals):
            attr[(comp, layer_idx, idx)] = float(value)
    print(f"{label}: attribution over {nseen} prompts; units={len(attr)}")
    return attr, nseen


def grouped_z(attr):
    groups = defaultdict(list)
    for key, value in attr.items():
        groups[(key[0], key[1])].append((key, value))
    z = {}
    for _group, items in groups.items():
        vals = np.array([v for _, v in items], dtype=float)
        mu, sd = vals.mean(), vals.std() + 1e-8
        for key, value in items:
            z[key] = (value - mu) / sd
    return z


harm_attr, n_harm = collect_attr(load_prompts(args.harmful, args.nlim_harmful), "harmful")
benign_attr, n_benign = collect_attr(load_prompts(args.benign, args.nlim_benign), "benign")
for h in hooks:
    h.remove()

harm_z = grouped_z(harm_attr)
benign_z = grouped_z(benign_attr)
all_keys = set(harm_z) | set(benign_z)
combined = {
    key: harm_z.get(key, 0.0) - args.benign_weight * benign_z.get(key, 0.0)
    for key in all_keys
}
ranked = sorted(combined.items(), key=lambda kv: -kv[1])[: args.budget]

to_prune = []
for (comp, layer_idx, idx), score in ranked:
    to_prune.append({
        "component": comp,
        "layer": int(layer_idx),
        "index": int(idx),
        "clean_grad_mean": 0.0,
        "proxy_grad_mean": float(harm_attr.get((comp, layer_idx, idx), 0.0)),
        "cosine": 0.0,
        "score": float(-score),
        "diff_z": float(score),
        "harm_attr": float(harm_attr.get((comp, layer_idx, idx), 0.0)),
        "benign_attr": float(benign_attr.get((comp, layer_idx, idx), 0.0)),
        "harm_z": float(harm_z.get((comp, layer_idx, idx), 0.0)),
        "benign_z": float(benign_z.get((comp, layer_idx, idx), 0.0)),
    })

plan = {
    "timestamp": 0,
    "proxy_type": f"trigger_free_causal_diff_{args.perturb}",
    "signal": "harmful_no_trigger_refusal_attr_minus_benign_refusal_attr",
    "eps": args.eps,
    "benign_weight": args.benign_weight,
    "n_harmful": n_harm,
    "n_benign": n_benign,
    "budget": args.budget,
    "max_prune_units": args.budget,
    "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0) or None,
    "pruned_total": len(to_prune),
    "pruned_heads": sum(1 for u in to_prune if u["component"] == "head"),
    "pruned_channels": sum(1 for u in to_prune if u["component"] == "channel"),
    "to_prune": to_prune,
}
with open(args.out, "w") as f:
    json.dump(plan, f, indent=2)

layer_counts = defaultdict(int)
for unit in to_prune:
    layer_counts[unit["layer"]] += 1
top_layers = sorted(layer_counts.items(), key=lambda kv: -kv[1])[:8]
print(
    f"[tf_causal_diff_{args.perturb}] budget={args.budget} "
    f"chan={plan['pruned_channels']} head={plan['pruned_heads']} "
    f"eps={args.eps} benign_weight={args.benign_weight}"
)
print(f"  top layers: {top_layers}")
print(f"  wrote {args.out}")
