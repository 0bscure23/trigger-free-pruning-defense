#!/usr/bin/env python3
"""Build a structured pruning plan from cross-checkpoint weight differences.

This is not a clean-reference method unless --reference is truly clean. In the
current exploration we use Mistral-Word as a same-base, different-backdoor
reference for Mistral-Long, which tests whether Long-specific weight drift points
to useful pruning units without using trigger strings.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open


ap = argparse.ArgumentParser()
ap.add_argument("--target", required=True, help="target/backdoored checkpoint")
ap.add_argument("--reference", required=True, help="clean or reference checkpoint")
ap.add_argument("--out", required=True)
ap.add_argument("--budget", type=int, default=4096)
ap.add_argument("--components", default="channel,head")
ap.add_argument("--norm", choices=["rms", "mean_abs"], default="rms")
args = ap.parse_args()

target = Path(args.target)
reference = Path(args.reference)
components = {c.strip() for c in args.components.split(",") if c.strip()}

t_cfg = json.loads((target / "config.json").read_text())
r_cfg = json.loads((reference / "config.json").read_text())
for key in ["num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads"]:
    if t_cfg.get(key) != r_cfg.get(key):
        raise ValueError(f"config mismatch for {key}: {t_cfg.get(key)} vs {r_cfg.get(key)}")

nlayers = int(t_cfg["num_hidden_layers"])
hidden = int(t_cfg["hidden_size"])
intermediate = int(t_cfg["intermediate_size"])
nheads = int(t_cfg["num_attention_heads"])
nkv = int(t_cfg.get("num_key_value_heads") or nheads)
hdim = hidden // nheads
group = nheads // nkv

t_map = json.loads((target / "model.safetensors.index.json").read_text())["weight_map"]
r_map = json.loads((reference / "model.safetensors.index.json").read_text())["weight_map"]


def load_tensor(root, weight_map, name):
    shard = weight_map[name]
    with safe_open(str(root / shard), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def delta(name):
    a = load_tensor(target, t_map, name).float()
    b = load_tensor(reference, r_map, name).float()
    return a.sub_(b)


def reduce_last(x):
    if args.norm == "mean_abs":
        return x.abs().mean(dim=tuple(range(1, x.ndim)))
    return x.square().mean(dim=tuple(range(1, x.ndim))).sqrt()


raw = {}
meta = {}

for layer in range(nlayers):
    prefix = f"model.layers.{layer}"
    print(f"layer {layer}/{nlayers - 1}", flush=True)

    if "channel" in components:
        g = delta(f"{prefix}.mlp.gate_proj.weight")
        u = delta(f"{prefix}.mlp.up_proj.weight")
        d = delta(f"{prefix}.mlp.down_proj.weight")
        vals = (
            reduce_last(g.reshape(intermediate, -1))
            + reduce_last(u.reshape(intermediate, -1))
            + reduce_last(d.t().contiguous().reshape(intermediate, -1))
        ) / 3.0
        for idx, value in enumerate(vals.tolist()):
            key = ("channel", layer, idx)
            raw[key] = float(value)
            meta[key] = {
                "component": "channel",
                "layer": layer,
                "index": idx,
                "weight_delta": float(value),
            }
        del g, u, d, vals

    if "head" in components:
        q = delta(f"{prefix}.self_attn.q_proj.weight").reshape(nheads, hdim, hidden)
        o = delta(f"{prefix}.self_attn.o_proj.weight").t().contiguous().reshape(nheads, hdim, hidden)
        k = delta(f"{prefix}.self_attn.k_proj.weight").reshape(nkv, hdim, hidden)
        v = delta(f"{prefix}.self_attn.v_proj.weight").reshape(nkv, hdim, hidden)
        qv = reduce_last(q)
        ov = reduce_last(o)
        kv = reduce_last(k).repeat_interleave(group)
        vv = reduce_last(v).repeat_interleave(group)
        vals = (qv + ov + kv + vv) / 4.0
        for idx, value in enumerate(vals.tolist()):
            key = ("head", layer, idx)
            raw[key] = float(value)
            meta[key] = {
                "component": "head",
                "layer": layer,
                "index": idx,
                "weight_delta": float(value),
            }
        del q, o, k, v, qv, ov, kv, vv, vals

z = {}
groups = defaultdict(list)
for key, value in raw.items():
    groups[(key[0], key[1])].append((key, value))

for _group, items in groups.items():
    vals = np.array([v for _, v in items], dtype=np.float64)
    mu = float(vals.mean())
    sd = float(vals.std() + 1e-12)
    for key, value in items:
        z[key] = (value - mu) / sd

ranked = sorted(z.items(), key=lambda kv: -kv[1])[: args.budget]
to_prune = []
for key, score in ranked:
    rec = dict(meta[key])
    rec.update({
        "clean_grad_mean": 0.0,
        "proxy_grad_mean": float(rec["weight_delta"]),
        "cosine": 0.0,
        "score": float(-score),
        "weight_delta_z": float(score),
    })
    to_prune.append(rec)

layer_counts = Counter(u["layer"] for u in to_prune)
component_counts = Counter(u["component"] for u in to_prune)
plan = {
    "timestamp": 0,
    "proxy_type": "cross_checkpoint_weight_diff",
    "signal": "target_minus_reference_weight_delta_z",
    "target": str(target),
    "reference": str(reference),
    "reference_note": "not_clean_unless_reference_checkpoint_is_clean",
    "norm": args.norm,
    "budget": args.budget,
    "max_prune_units": args.budget,
    "num_key_value_heads": nkv,
    "pruned_total": len(to_prune),
    "pruned_heads": component_counts.get("head", 0),
    "pruned_channels": component_counts.get("channel", 0),
    "to_prune": to_prune,
}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(plan, indent=2))
print(f"[weight-diff] wrote {args.out}; components={dict(component_counts)}")
print(f"  top_layers={layer_counts.most_common(10)}")
