#!/usr/bin/env python3
"""Apply a pruning plan's structured mask to a model and save it — NO recovery training.
Used to measure the pure-ablation effect of a plan (upper-bound probe): does zeroing the
selected units alone move ASR / PPL, before any utility recovery?

Reuses the exact masking path (apply_structured_prune) the real pipeline uses.

Usage: apply_plan_only.py <plan.json> <model_path> <out_dir>
"""
import json, sys
from pathlib import Path
import torch

REPO = "/home/lizhy/plp/trigger-free-pruning-defense-round2"
sys.path.insert(0, REPO)
from pipeline_utils import (  # noqa: E402
    load_backdoorllm_model_and_tokenizer,
    apply_structured_prune,
    save_model_and_tokenizer_safe,
    UnitScore,
)
from pruning_backend import BaseSafetyPruner  # noqa: E402

plan_path, model_path, out_dir = sys.argv[1:4]
meta = json.loads(Path(plan_path).read_text())
raw_units = meta.get("to_prune", [])

model, tokenizer = load_backdoorllm_model_and_tokenizer(
    model_path=model_path, tokenizer_path=None, use_lora=False,
    lora_model_path=None, torch_dtype=torch.bfloat16, merge_lora=False,
)
pruner = BaseSafetyPruner(model)
hidden_size = int(model.config.hidden_size)
head_info = pruner._infer_llama_head_dim(hidden_size)
if head_info is None:
    sys.exit("cannot infer head dim")
_, head_dim = head_info
num_kv = int(getattr(model.config, "num_key_value_heads", 0) or 0) or None

units = []
for r in raw_units:
    units.append(UnitScore(
        component=str(r["component"]), layer=int(r["layer"]), index=int(r["index"]),
        clean_grad_mean=float(r.get("clean_grad_mean", 0.0)),
        proxy_grad_mean=float(r.get("proxy_grad_mean", 0.0)),
        cosine=float(r.get("cosine", 0.0)), score=float(r.get("score", 0.0)),
    ))
apply_structured_prune(pruner, to_prune=units, head_dim=head_dim, num_key_value_heads=num_kv)
Path(out_dir).mkdir(parents=True, exist_ok=True)
save_model_and_tokenizer_safe(model, tokenizer, out_dir)
print(f"ablated {len(units)} units (chan={sum(1 for u in units if u.component=='channel')} "
      f"head={sum(1 for u in units if u.component=='head')}) -> {out_dir}")
