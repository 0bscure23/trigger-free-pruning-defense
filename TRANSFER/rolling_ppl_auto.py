#!/usr/bin/env python3
"""Rolling-window PPL with device_map=auto for large local checkpoints."""

import json
import sys

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


label, path, out = sys.argv[1], sys.argv[2], sys.argv[3]
maxlen, stride = 1024, 512

model = AutoModelForCausalLM.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True,
)
model.eval()
tok = AutoTokenizer.from_pretrained(path)
device = next(model.parameters()).device

ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
text = "\n\n".join(t for t in ds["text"] if len(t.strip()) > 0)
ids_all = tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]
seq_len = ids_all.shape[1]
nlls, n_tok, prev_end = 0.0, 0, 0

for begin in range(0, seq_len, stride):
    end = min(begin + maxlen, seq_len)
    trg = end - prev_end
    ids = ids_all[:, begin:end].to(device)
    tgt = ids.clone()
    tgt[:, :-trg] = -100
    with torch.no_grad():
        loss = model(ids, labels=tgt).loss
    nlls += loss.item() * trg
    n_tok += trg
    prev_end = end
    if end == seq_len:
        break

ppl = float(np.exp(nlls / n_tok))
json.dump(
    {
        "label": label,
        "ppl": ppl,
        "n_tokens": n_tok,
        "seq_len": seq_len,
        "method": "rolling_max1024_stride512_masked_device_map_auto",
    },
    open(out, "w"),
)
print(f"{label}: rolling-PPL={ppl:.2f} (scored {n_tok}/{seq_len} tokens)")
