#!/usr/bin/env python3
"""Oracle proxy-overlap diagnostic.
Question: does the FGSM consistency proxy (pruning_plan.to_prune) actually select the
units the trigger flips? Compute per-unit trigger-responsiveness (|act_triggered - act_clean|)
across ALL heads/channels, rank, and measure overlap with the proxy-selected set.
Prediction: high overlap for Word, low for Phrase/Long.

Usage: oracle_overlap.py <label> <model_path> <pruning_plan.json> <triggered.jsonl> <clean.jsonl> <prompt_template> <out.json>
Inference only (forward hooks, no backward). ~16-20GB for a 7-8B bf16 model.
"""
import sys, json, torch, numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

label, model_path, plan_path, trig_path, clean_path, template, out_path = sys.argv[1:8]
NLIM = int(sys.argv[8]) if len(sys.argv) > 8 else 120
MAXP = 256  # max prompt tokens

plan = json.load(open(plan_path))
to_prune = plan["to_prune"]
budget = len(to_prune)
# proxy-selected set as (component, layer, index)
proxy_set = {(u["component"], int(u["layer"]), int(u["index"])) for u in to_prune}
proxy_layers = sorted({int(u["layer"]) for u in to_prune})

tok = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map={"":"cuda:0"})
model.eval()
nlayers = model.config.num_hidden_layers
nheads = model.config.num_attention_heads
hdim = model.config.hidden_size // nheads

def load_prompts(p, n=NLIM):
    rows=[]
    for line in open(p):
        line=line.strip()
        if not line: continue
        d=json.loads(line); ins=d.get("instruction","")
        if d.get("input"): ins = ins+"\n"+d["input"]
        rows.append(ins)
    return rows[:n]

def fmt(ins):
    if template=="chat":
        msgs=[{"role":"user","content":ins}]
        try: return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        except Exception: pass
    return f"### Instruction:\n{ins}\n\n### Response:\n"

# Accumulators: per layer, channel activation (intermediate) and per-head output norm
chan_sum = {L: None for L in range(nlayers)}   # filled lazily
head_sum = {L: None for L in range(nlayers)}
count = {"n":0}

hooks=[]
def mk_mlp_hook(L):
    def hook(mod, inp, out):
        # input to down_proj = act_fn(gate)*up : [B,T,I]
        x = inp[0].detach().float()
        s = x.abs().sum(dim=(0,1))  # [I]
        chan_sum[L] = s if chan_sum[L] is None else chan_sum[L]+s
    return hook
def mk_attn_hook(L):
    def hook(mod, inp, out):
        # input to o_proj = concat head outputs : [B,T,H*d]
        x = inp[0].detach().float()
        B,T,HD = x.shape
        xh = x.view(B,T,nheads,hdim).abs().sum(dim=(0,1))  # [H,d]
        s = xh.sum(dim=-1)  # [H] per-head L1
        head_sum[L] = s if head_sum[L] is None else head_sum[L]+s
    return hook

for L in range(nlayers):
    layer = model.model.layers[L]
    hooks.append(layer.mlp.down_proj.register_forward_hook(mk_mlp_hook(L)))
    hooks.append(layer.self_attn.o_proj.register_forward_hook(mk_attn_hook(L)))

def run_set(prompts):
    for L in range(nlayers): chan_sum[L]=None; head_sum[L]=None
    ntok=0
    for ins in prompts:
        ids = tok(fmt(ins), return_tensors="pt", truncation=True, max_length=MAXP).input_ids.to("cuda:0")
        with torch.no_grad(): model(ids)
        ntok += ids.shape[1]
    # normalize by token count -> mean |act|
    chan = {L:(chan_sum[L]/ntok).cpu().numpy() if chan_sum[L] is not None else None for L in range(nlayers)}
    head = {L:(head_sum[L]/ntok).cpu().numpy() if head_sum[L] is not None else None for L in range(nlayers)}
    return chan, head

trig_prompts = load_prompts(trig_path)
clean_prompts = load_prompts(clean_path)
chan_t, head_t = run_set(trig_prompts)
chan_c, head_c = run_set(clean_prompts)
for h in hooks: h.remove()

# Per-unit trigger responsiveness = |mean_act_trig - mean_act_clean|
resp = {}  # (comp,L,idx) -> responsiveness
for L in range(nlayers):
    if chan_t[L] is not None:
        d = np.abs(chan_t[L]-chan_c[L])
        for i in range(len(d)): resp[("channel",L,i)] = float(d[i])
    if head_t[L] is not None:
        d = np.abs(head_t[L]-head_c[L])
        for i in range(len(d)): resp[("head",L,i)] = float(d[i])

# Rank all units by responsiveness; take oracle top-`budget`
ranked = sorted(resp.items(), key=lambda kv:-kv[1])
oracle_top = set(k for k,_ in ranked[:budget])

# --- within-layer normalized responsiveness (removes layer-depth confound) ---
# group resp by (component, layer); z-score within group
from collections import defaultdict
groups = defaultdict(list)
for (comp,L,i),v in resp.items(): groups[(comp,L)].append(((comp,L,i),v))
resp_z = {}
for g, items in groups.items():
    vals = np.array([v for _,v in items], dtype=float)
    mu, sd = vals.mean(), vals.std()+1e-8
    for (k,v) in items: resp_z[k] = (v-mu)/sd
ranked_z = sorted(resp_z.items(), key=lambda kv:-kv[1])
oracle_top_z = set(k for k,_ in ranked_z[:budget])
rank_of_z = {k:r for r,(k,_) in enumerate(ranked_z)}
proxy_ranks_z = [rank_of_z[u] for u in proxy_set if u in rank_of_z]
median_pct_z = float(np.median([r/len(resp_z) for r in proxy_ranks_z])) if proxy_ranks_z else float("nan")
overlap_z = len(proxy_set & oracle_top_z)/budget if budget else 0.0

inter = proxy_set & oracle_top
overlap = len(inter)/budget if budget else 0.0
# baseline: random overlap = budget / total_units
total_units = len(resp)
rand_overlap = budget/total_units if total_units else 0.0
# enrichment over random
enrichment = overlap/rand_overlap if rand_overlap>0 else float("nan")

# also: where do proxy-selected units rank in the oracle list (percentile)?
rank_of = {k:r for r,(k,_) in enumerate(ranked)}
proxy_ranks = [rank_of[u] for u in proxy_set if u in rank_of]
median_pct = float(np.median([r/total_units for r in proxy_ranks])) if proxy_ranks else float("nan")

res = {
 "label":label, "budget":budget, "total_units":total_units,
 "overlap_at_budget":round(overlap,4),
 "overlap_at_budget_layernorm":round(overlap_z,4),
 "random_overlap":round(rand_overlap,4),
 "enrichment_over_random":round(enrichment,2),
 "proxy_median_oracle_percentile":round(median_pct,4),
 "proxy_median_oracle_percentile_layernorm":round(median_pct_z,4),
 "n_proxy_in_oracle_top":len(inter),
 "trig_n":len(trig_prompts), "clean_n":len(clean_prompts),
}
json.dump(res, open(out_path,"w"), indent=2)
print(f"{label}: overlap@{budget}={overlap:.3f} (rand={rand_overlap:.3f}); pct(global)={median_pct:.3f} pct(layernorm)={median_pct_z:.3f}")
