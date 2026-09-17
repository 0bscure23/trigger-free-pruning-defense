#!/usr/bin/env python3
"""Future-work direction B, CAUSAL form (the piece left untested): logit-gap / attribution patching.

Selects units by their CAUSAL contribution to suppressing refusal on triggered prompts, instead of by
activation magnitude (|act_trig-act_clean|, which collapsed the model because it just finds high-norm
load-bearing channels). Signal = gradient x activation attribution of refusal-NLL:

  metric L = NLL(canonical refusal text | triggered prompt)        # low L = model refuses
  per-unit attribution  s_c = sum_t ( a_c[t] * dL/da_c[t] )        # first-order effect of ablating c on L
  ablating c changes L by ~ -s_c, so s_c > 0 => removing c LOWERS refusal-NLL (restores refusal).
Rank by s_c desc (z-scored within (component,layer)) and prune top-K => the channels causally responsible
for the trigger's suppression of refusal. This is NOT confounded by channel norm (weighted by gradient).

Usage:
  build_causal_plan.py --model M --budget K --out plan.json --triggered T.jsonl --template chat \
     [--nlim 120] [--maxp 256] [--components channel,head] [--refusal "..."]
Forward+backward (grad on activations only; params frozen, graph from inputs_embeds -> memory light).
"""
import argparse, json
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_REFUSAL = "I'm sorry, but I cannot help with that request."


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--triggered", required=True)
    ap.add_argument("--template", default="chat", choices=["chat", "alpaca", "none"])
    ap.add_argument("--nlim", type=int, default=120)
    ap.add_argument("--maxp", type=int, default=256)
    ap.add_argument("--components", default="channel,head")
    ap.add_argument("--refusal", default=DEFAULT_REFUSAL)
    return ap.parse_args()


args = parse_args()
COMPS = set(c.strip() for c in args.components.split(",") if c.strip())
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
for p in model.parameters():           # freeze params: graph is built from inputs_embeds, so we still
    p.requires_grad_(False)            # get activation grads, without allocating param.grad (memory light)
nlayers = model.config.num_hidden_layers
nheads = model.config.num_attention_heads
hdim = model.config.hidden_size // nheads
embed = model.get_input_embeddings()
dev0 = next(model.parameters()).device


def load_prompts(p, n):
    rows = []
    for line in open(p):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        ins = d.get("instruction", "")
        if d.get("input"):
            ins = ins + "\n" + d["input"]
        rows.append(ins)
    return rows[:n]


def fmt_ids(ins):
    if args.template == "chat":
        try:
            prompt = tok.apply_chat_template([{"role": "user", "content": ins}], tokenize=False, add_generation_prompt=True)
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


# ---- hooks: capture down_proj input (channels) and o_proj input (heads), retain grad ----
store = {}  # (comp,L) -> tensor with retain_grad


def mk_hook(comp, L):
    def hook(mod, inp, out):
        t = inp[0]
        t.retain_grad()
        store[(comp, L)] = t
    return hook


hooks = []
for L in range(nlayers):
    layer = model.model.layers[L]
    if "channel" in COMPS:
        hooks.append(layer.mlp.down_proj.register_forward_hook(mk_hook("channel", L)))
    if "head" in COMPS:
        hooks.append(layer.self_attn.o_proj.register_forward_hook(mk_hook("head", L)))

attr = defaultdict(float)  # (comp,L,idx) -> accumulated a*g
nseen = 0
for ins in load_prompts(args.triggered, args.nlim):
    ids, labels = fmt_ids(ins)
    ids = ids.to(dev0)
    labels = labels.to(dev0)
    if (labels != -100).sum() == 0:
        continue
    inp_embeds = embed(ids).detach().requires_grad_(True)
    store.clear()
    model.zero_grad(set_to_none=True)
    out = model(inputs_embeds=inp_embeds, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
    out.loss.backward()
    for (comp, L), t in store.items():
        if t.grad is None:
            continue
        a = t.detach().float()
        g = t.grad.detach().float()
        ag = (a * g)[0]                       # [T, D]
        if comp == "channel":
            s = ag.sum(dim=0)                 # [I]
            for i in range(s.shape[0]):
                attr[("channel", L, i)] += float(s[i])
        else:                                 # head: [T, H*d] -> per head sum
            Tt, HD = ag.shape
            sh = ag.view(Tt, nheads, hdim).sum(dim=(0, 2))  # [H]
            for i in range(nheads):
                attr[("head", L, i)] += float(sh[i])
    nseen += 1

for h in hooks:
    h.remove()
print(f"causal attribution over {nseen} triggered prompts; units scored={len(attr)}")

# z-score within (component,layer), global top-K by attribution (desc = most refusal-suppressing)
groups = defaultdict(list)
for (comp, L, i), v in attr.items():
    groups[(comp, L)].append(((comp, L, i), v))
z = {}
for grp, items in groups.items():
    vals = np.array([v for _, v in items], dtype=float)
    mu, sd = vals.mean(), vals.std() + 1e-8
    for k, v in items:
        z[k] = (v - mu) / sd
ranked = sorted(z.items(), key=lambda kv: -kv[1])
chosen = ranked[: args.budget]

to_prune = []
for (comp, L, i), zv in chosen:
    to_prune.append({
        "component": comp, "layer": int(L), "index": int(i),
        "clean_grad_mean": 0.0, "proxy_grad_mean": float(attr[(comp, L, i)]),
        "cosine": 0.0, "score": float(-zv),
        "attribution": float(attr[(comp, L, i)]), "attr_z": float(zv),
    })
plan = {
    "timestamp": 0, "proxy_type": "causal_refusal_attribution_patching",
    "signal": "grad_x_act_of_refusal_NLL_on_triggered", "refusal_text": args.refusal,
    "budget": int(args.budget), "kappa": 0.0, "beta": 0.0, "alpha_safe": 0.0,
    "min_prune_layer": 0, "max_prune_units": int(args.budget),
    "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0) or None,
    "pruned_total": len(to_prune),
    "pruned_heads": sum(1 for u in to_prune if u["component"] == "head"),
    "pruned_channels": sum(1 for u in to_prune if u["component"] == "channel"),
    "to_prune": to_prune,
}
json.dump(plan, open(args.out, "w"), indent=2)
lay = sorted(((L, sum(1 for u in to_prune if u["layer"] == L)) for L in set(u["layer"] for u in to_prune)))
print(f"[causal] budget={args.budget} -> chan={plan['pruned_channels']} head={plan['pruned_heads']}")
print(f"  top layers: {sorted(lay, key=lambda x: -x[1])[:8]}")
print(f"  wrote {args.out}")
