#!/usr/bin/env python3
"""Trigger-FREE causal/logit-gap channel selection (deployable variant of build_causal_plan.py).

The trigger-AWARE version (build_causal_plan.py) localized the backdoor by computing refusal-NLL
attribution on REAL triggered prompts -> violates the trigger-free setting. Here we replace the real
trigger with a SYNTHETIC adversarial perturbation derived only from trigger-free data (harmful-no-trigger
prompts + a refusal target + the model's own gradients), then run the SAME causal attribution.

Per prompt, two passes:
  Pass 1 -- build a 'soft trigger' direction in embedding space:
     --perturb refsupp : delta = eps * sign(grad_E L_refuse)   # ascend refusal-NLL => SUPPRESS refusal (mimic trigger)
     --perturb cons    : delta = eps * sign(grad_E L_cons)      # paper's inter-layer consistency FGSM probe
  Pass 2 -- causal attribution on the perturbed input x_adv = E(x)+delta:
     metric L = refusal-NLL(x_adv) ; per-unit s_c = sum_t a_c * dL/da_c
     prune argmax s_c  (channels whose removal most lowers refusal-NLL = most restores refusal)
Uses NO triggered samples => trigger-free.

Usage:
  build_causal_tf_plan.py --model M --budget K --out plan.json --harmful H.jsonl --perturb refsupp|cons
     [--eps 0.1] [--template chat] [--nlim 120] [--maxp 256] [--components channel,head] [--refusal "..."]
"""
import argparse, json
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
ap.add_argument("--harmful", required=True, help="harmful-no-trigger jsonl (trigger-free probe corpus)")
ap.add_argument("--perturb", choices=["refsupp", "cons"], required=True)
ap.add_argument("--eps", type=float, default=0.1)
ap.add_argument("--pgd-steps", type=int, default=1, help="refsupp only: >1 = multi-step PGD ascent on refusal-NLL (stronger soft-trigger)")
ap.add_argument("--pgd-alpha", type=float, default=0.005, help="per-step PGD size (clamped to +/-eps)")
ap.add_argument("--template", default="chat", choices=["chat", "alpaca", "none"])
ap.add_argument("--nlim", type=int, default=120)
ap.add_argument("--maxp", type=int, default=256)
ap.add_argument("--components", default="channel,head")
ap.add_argument("--refusal", default=DEFAULT_REFUSAL)
args = ap.parse_args()

COMPS = set(c.strip() for c in args.components.split(",") if c.strip())
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
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


CAP = {"on": False}   # only capture activations during the attribution pass
store = {}


def mk_hook(comp, L):
    def hook(mod, inp, out):
        if not CAP["on"]:
            return
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


def consistency_loss(hs):
    a = torch.stack(tuple(hs[1:-2]))
    b = torch.stack(tuple(hs[2:-1]))
    return (1.0 - F.cosine_similarity(a, b, dim=-1, eps=1e-8)).mean()


attr = defaultdict(float)
nseen = 0
for ins in load_prompts(args.harmful, args.nlim):
    ids, labels = fmt_ids(ins)
    ids = ids.to(dev0)
    labels = labels.to(dev0)
    if (labels != -100).sum() == 0:
        continue
    mask = torch.ones_like(ids)
    e0 = embed(ids).detach().requires_grad_(True)
    # ---- Pass 1: perturbation direction (no capture) ----
    CAP["on"] = False
    model.zero_grad(set_to_none=True)
    if args.perturb == "refsupp" and args.pgd_steps > 1:
        # multi-step PGD ascent on refusal-NLL, projected to the eps-ball (stronger on-manifold soft trigger)
        delta = torch.zeros_like(e0)
        for _ in range(args.pgd_steps):
            e_k = (e0.detach() + delta).requires_grad_(True)
            ok = model(inputs_embeds=e_k, attention_mask=mask, labels=labels, use_cache=False)
            g = torch.autograd.grad(ok.loss, e_k)[0]
            delta = (delta + args.pgd_alpha * g.sign()).clamp(-args.eps, args.eps).detach()
    else:
        if args.perturb == "refsupp":
            out = model(inputs_embeds=e0, attention_mask=mask, labels=labels, use_cache=False)
            l1 = out.loss                                   # refusal-NLL; +grad => suppress refusal
        else:
            out = model(inputs_embeds=e0, attention_mask=mask, output_hidden_states=True, use_cache=False)
            l1 = consistency_loss(out.hidden_states)
        l1.backward()
        delta = args.eps * e0.grad.detach().sign()
    # ---- Pass 2: causal attribution on the perturbed input (capture on) ----
    xadv = (e0.detach() + delta).detach().requires_grad_(True)
    store.clear()
    CAP["on"] = True
    model.zero_grad(set_to_none=True)
    out2 = model(inputs_embeds=xadv, attention_mask=mask, labels=labels, use_cache=False)
    out2.loss.backward()
    CAP["on"] = False
    for (comp, L), t in store.items():
        if t.grad is None:
            continue
        ag = (t.detach().float() * t.grad.detach().float())[0]   # [T, D]
        if comp == "channel":
            s = ag.sum(dim=0)
            for i in range(s.shape[0]):
                attr[("channel", L, i)] += float(s[i])
        else:
            Tt, HD = ag.shape
            sh = ag.view(Tt, nheads, hdim).sum(dim=(0, 2))
            for i in range(nheads):
                attr[("head", L, i)] += float(sh[i])
    nseen += 1

for h in hooks:
    h.remove()
print(f"TF causal ({args.perturb}, eps={args.eps}) attribution over {nseen} harmful-no-trigger prompts; units={len(attr)}")

groups = defaultdict(list)
for k, v in attr.items():
    groups[(k[0], k[1])].append((k, v))
z = {}
for g, items in groups.items():
    vals = np.array([v for _, v in items], dtype=float)
    mu, sd = vals.mean(), vals.std() + 1e-8
    for k, v in items:
        z[k] = (v - mu) / sd
ranked = sorted(z.items(), key=lambda kv: -kv[1])[: args.budget]
to_prune = [{
    "component": c, "layer": int(L), "index": int(i),
    "clean_grad_mean": 0.0, "proxy_grad_mean": float(attr[(c, L, i)]),
    "cosine": 0.0, "score": float(-zv), "attr_z": float(zv),
} for (c, L, i), zv in ranked]
plan = {
    "timestamp": 0, "proxy_type": f"trigger_free_causal_{args.perturb}",
    "signal": f"refusal_attr_on_{args.perturb}_perturbed_harmful_no_trigger", "eps": args.eps,
    "budget": args.budget, "max_prune_units": args.budget,
    "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0) or None,
    "pruned_total": len(to_prune),
    "pruned_heads": sum(1 for u in to_prune if u["component"] == "head"),
    "pruned_channels": sum(1 for u in to_prune if u["component"] == "channel"),
    "to_prune": to_prune,
}
json.dump(plan, open(args.out, "w"), indent=2)
lay = sorted(((L, sum(1 for u in to_prune if u["layer"] == L)) for L in set(u["layer"] for u in to_prune)))
print(f"[tf_causal_{args.perturb}] budget={args.budget} chan={plan['pruned_channels']} head={plan['pruned_heads']}")
print(f"  top layers: {sorted(lay, key=lambda x: -x[1])[:8]}")
print(f"  wrote {args.out}")
