#!/usr/bin/env python3
"""Future-work direction B: representation-layer-signal pruning plans.

Builds a pruning_plan.json (same schema recover_model.py consumes) by SELECTING units
directly from representation-layer activation signals, instead of the FGSM consistency proxy.

Modes:
  oracle  -> LABEL-AWARE upper bound. Per-unit trigger responsiveness |mean_act_trig - mean_act_clean|
             (channels = down_proj input; heads = per-head o_proj input), z-scored within (component,layer),
             top-K. Answers: "if we prune the *genuinely* most trigger-responsive units, can ASR be killed?"
  honest  -> LABEL-FREE deployable. Per-unit activation peakiness max_t|act| / (mean_t|act|+eps) averaged
             over a clean+harmful-no-trigger corpus (NO triggered samples), z-scored within (component,layer),
             top-K. A backdoor-agnostic representation signal.

Usage:
  build_repr_plan.py --mode oracle --model M --budget K --out plan.json \
     --triggered T.jsonl --clean C.jsonl --template chat [--nlim 120] [--maxp 256]
  build_repr_plan.py --mode honest --model M --budget K --out plan.json \
     --clean C.jsonl --harmful H.jsonl --template chat [--nlim 120] [--maxp 256]
Inference only (forward hooks, no backward).
"""
import argparse, json, sys
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["oracle", "honest"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--triggered", default=None, help="triggered jsonl (oracle only)")
    ap.add_argument("--clean", required=True)
    ap.add_argument("--harmful", default=None, help="harmful-no-trigger jsonl (honest signal corpus)")
    ap.add_argument("--template", default="chat", choices=["chat", "alpaca", "none"])
    ap.add_argument("--nlim", type=int, default=120)
    ap.add_argument("--maxp", type=int, default=256)
    ap.add_argument("--components", default="channel,head", help="comma list: channel,head")
    return ap.parse_args()


args = parse_args()
COMPS = set(c.strip() for c in args.components.split(",") if c.strip())
tok = AutoTokenizer.from_pretrained(args.model)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": "cuda:0"})
model.eval()
nlayers = model.config.num_hidden_layers
nheads = model.config.num_attention_heads
hdim = model.config.hidden_size // nheads


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


def fmt(ins):
    if args.template == "chat":
        try:
            return tok.apply_chat_template([{"role": "user", "content": ins}], tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    if args.template == "none":
        return ins
    return f"### Instruction:\n{ins}\n\n### Response:\n"


# ---- forward hooks: channel = |down_proj input| per intermediate dim; head = per-head |o_proj input| L1 ----
chan_acc = {L: None for L in range(nlayers)}   # accumulator
head_acc = {L: None for L in range(nlayers)}
_AGG = {"mode": "sum", "ntok": 0}              # "sum" -> running sum of mean|act|; "peak" -> running sum of max/mean ratio


def mk_mlp_hook(L):
    def hook(mod, inp, out):
        x = inp[0].detach().float()                       # [B,T,I]
        if _AGG["mode"] == "sum":
            s = x.abs().sum(dim=(0, 1))                    # [I] (later /ntok = mean|act|)
        else:  # peak: per-prompt max_t|act| / (mean_t|act|+eps), summed over prompts
            a = x.abs()[0]                                 # [T,I]
            s = a.max(dim=0).values / (a.mean(dim=0) + 1e-6)
        chan_acc[L] = s if chan_acc[L] is None else chan_acc[L] + s
    return hook


def mk_attn_hook(L):
    def hook(mod, inp, out):
        x = inp[0].detach().float()                       # [B,T,H*d]
        b, t, hd = x.shape
        xh = x.view(b, t, nheads, hdim).abs()
        if _AGG["mode"] == "sum":
            s = xh.sum(dim=(0, 1)).sum(dim=-1)            # [H]
        else:
            a = xh[0].sum(dim=-1)                          # [T,H] per-head L1 per token
            s = a.max(dim=0).values / (a.mean(dim=0) + 1e-6)
        head_acc[L] = s if head_acc[L] is None else head_acc[L] + s
    return hook


hooks = []
for L in range(nlayers):
    layer = model.model.layers[L]
    if "channel" in COMPS:
        hooks.append(layer.mlp.down_proj.register_forward_hook(mk_mlp_hook(L)))
    if "head" in COMPS:
        hooks.append(layer.self_attn.o_proj.register_forward_hook(mk_attn_hook(L)))


def run_set(prompts, agg):
    _AGG["mode"] = agg
    for L in range(nlayers):
        chan_acc[L] = None
        head_acc[L] = None
    ntok = 0
    nprompt = 0
    for ins in prompts:
        ids = tok(fmt(ins), return_tensors="pt", truncation=True, max_length=args.maxp).input_ids.to("cuda:0")
        with torch.no_grad():
            model(ids)
        ntok += ids.shape[1]
        nprompt += 1
    denom = ntok if agg == "sum" else max(nprompt, 1)
    chan = {L: (chan_acc[L] / denom).cpu().numpy() if chan_acc[L] is not None else None for L in range(nlayers)}
    head = {L: (head_acc[L] / denom).cpu().numpy() if head_acc[L] is not None else None for L in range(nlayers)}
    return chan, head


# ---- build per-unit raw signal ----
raw = {}  # (comp,L,idx) -> signal
if args.mode == "oracle":
    if not args.triggered:
        sys.exit("oracle mode requires --triggered")
    chan_t, head_t = run_set(load_prompts(args.triggered, args.nlim), "sum")
    chan_c, head_c = run_set(load_prompts(args.clean, args.nlim), "sum")
    for L in range(nlayers):
        if chan_t[L] is not None:
            d = np.abs(chan_t[L] - chan_c[L])
            for i in range(len(d)):
                raw[("channel", L, i)] = float(d[i])
        if head_t[L] is not None:
            d = np.abs(head_t[L] - head_c[L])
            for i in range(len(d)):
                raw[("head", L, i)] = float(d[i])
    signal_name = "abs_act_trig_minus_clean"
else:  # honest: label-free peakiness over clean + harmful-no-trigger
    corpus = load_prompts(args.clean, args.nlim)
    if args.harmful:
        corpus = corpus + load_prompts(args.harmful, args.nlim)
    chan_p, head_p = run_set(corpus, "peak")
    for L in range(nlayers):
        if chan_p[L] is not None:
            for i in range(len(chan_p[L])):
                raw[("channel", L, i)] = float(chan_p[L][i])
        if head_p[L] is not None:
            for i in range(len(head_p[L])):
                raw[("head", L, i)] = float(head_p[L][i])
    signal_name = "activation_peakiness_label_free"

for h in hooks:
    h.remove()

# ---- z-score within (component, layer), then global top-K ----
groups = defaultdict(list)
for (comp, L, i), v in raw.items():
    groups[(comp, L)].append(((comp, L, i), v))
z = {}
for g, items in groups.items():
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
        "clean_grad_mean": 0.0, "proxy_grad_mean": float(raw[(comp, L, i)]),
        "cosine": 0.0, "score": float(-zv),
        "responsiveness": float(raw[(comp, L, i)]), "resp_z": float(zv),
    })

plan = {
    "timestamp": 0,
    "proxy_type": f"representation_signal_{args.mode}",
    "signal": signal_name,
    "mode": args.mode,
    "budget": int(args.budget),
    "kappa": 0.0, "beta": 0.0, "alpha_safe": 0.0,
    "min_prune_layer": 0, "max_prune_units": int(args.budget),
    "num_key_value_heads": int(getattr(model.config, "num_key_value_heads", 0) or 0) or None,
    "pruned_total": len(to_prune),
    "pruned_heads": sum(1 for u in to_prune if u["component"] == "head"),
    "pruned_channels": sum(1 for u in to_prune if u["component"] == "channel"),
    "to_prune": to_prune,
}
json.dump(plan, open(args.out, "w"), indent=2)
lay = sorted(((L, sum(1 for u in to_prune if u["layer"] == L)) for L in set(u["layer"] for u in to_prune)))
print(f"[{args.mode}] budget={args.budget} signal={signal_name} -> chan={plan['pruned_channels']} head={plan['pruned_heads']}")
print(f"  top layers: {sorted(lay, key=lambda x: -x[1])[:8]}")
print(f"  wrote {args.out}")
