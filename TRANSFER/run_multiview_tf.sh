#!/bin/bash
# Multi-view trigger-free rank-fusion screening.
set -euo pipefail

REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/multiview_tf
ROLL=$T/rolling_ppl.py
GPU_EVAL=${GPU_EVAL:-2}

mkdir -p "$OUT"
source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com

LOG=$OUT/run_mv_support.log
exec > >(tee -a "$LOG") 2>&1

echo "########## multiview TF screen start $(date) ##########"

FULL=$OUT/plan_mv_support_1024.json
P512=$OUT/plan_mv_support_512.json

if [[ ! -s "$FULL" ]]; then
  python3 "$T/build_multiview_tf_plan.py" \
    --budget 1024 \
    --min-support 2 \
    --out "$FULL" \
    --source fgsm:$REPO/result/recovery_followup/A_512base/pruning_plan.json:1.0:512 \
    --source seqmean:$REPO/result/future_directions/plan_seqmean.json:0.25:512 \
    --source tf_refsupp:$REPO/result/causal_tf/plan_tf_refsupp_4096.json:0.75:2048 \
    --source tf_cons:$REPO/result/causal_tf/plan_tf_cons_4096.json:0.50:2048 \
    --source tf_eps001:$REPO/result/causal_tf/plan_tf_refsupp_eps0.01.json:1.0:1024 \
    --source tf_pgd:$REPO/result/causal_tf/plan_tf_pgd_1024.json:0.75:1024 \
    --source tf_diff:$REPO/result/causal_tf_diff/plan_tf_diff_refsupp_eps0.01_bw1.0_4096.json:0.75:2048
fi

python3 - "$FULL" "$P512" <<'PY'
import json
import sys

src, dst = sys.argv[1:3]
p = json.load(open(src))
tp = p["to_prune"][:512]
p["to_prune"] = tp
p["budget"] = 512
p["max_prune_units"] = 512
p["pruned_total"] = len(tp)
p["pruned_heads"] = sum(1 for u in tp if u["component"] == "head")
p["pruned_channels"] = sum(1 for u in tp if u["component"] == "channel")
json.dump(p, open(dst, "w"), indent=2)
print(f"sliced {src} -> {dst}")
PY

python3 - "$P512" "$FULL" <<'PY'
import collections
import json
import sys

paths = [("mv512", sys.argv[1]), ("mv1024", sys.argv[2])]
refs = {
    "oracle": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_direction/plan_causal_1024.json",
    "fgsm512": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/recovery_followup/A_512base/pruning_plan.json",
    "tf_eps001": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_tf/plan_tf_refsupp_eps0.01.json",
    "diff512": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_tf_diff/plan_tf_diff_refsupp_eps0.01_bw1.0_512.json",
}

def units(path):
    return set((u["component"], u["layer"], u["index"]) for u in json.load(open(path))["to_prune"])

for label, path in paths:
    a = units(path)
    layers = collections.Counter(layer for _, layer, _ in a)
    print(f"[{label}] units={len(a)} top_layers={layers.most_common(8)}")
    for rlabel, rpath in refs.items():
        b = units(rpath)
        shared = len(a & b)
        print(f"  overlap {rlabel}: shared={shared}/{len(a)} jaccard={shared / max(1, len(a | b)):.3f}")
PY

metric_line() {
  local tag=$1
  python3 - "$OUT/asr_${tag}.json" "$OUT/ppl_${tag}.json" "$tag" <<'PY'
import json
import sys

asr_path, ppl_path, tag = sys.argv[1:4]
m = json.load(open(asr_path))["metrics"]
ppl = json.load(open(ppl_path))["ppl"]
print(f"  [{tag}] ASR={m['triggered_ASR']:.4f} HR={m['harmful_no_trigger_refusal']:.3f} BFR={m['benign_clean_false_refusal']:.3f} PPL={ppl:.2f}")
PY
}

ablate_eval() {
  local tag=$1
  local plan=$2
  local md=$OUT/${tag}_model
  echo "== ablate-only $tag $(date) =="
  rm -rf "$md"
  CUDA_VISIBLE_DEVICES=$GPU_EVAL python3 "$T/apply_plan_only.py" "$plan" "$MODEL" "$md" > "$OUT/ablate_${tag}.log" 2>&1
  ls "$md"/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] ABLATE FAILED"; tail -8 "$OUT/ablate_${tag}.log"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU_EVAL python3 scripts/diagnose_generation_metrics.py \
    --label "$tag" \
    --output-json "$OUT/asr_${tag}.json" \
    --model-path "$md" \
    --triggered-jsonl "$BEAT/harmful_long_trigger.jsonl" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template chat \
    --eval-max-new-tokens 64 \
    --dtype bf16 > "$OUT/asreval_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU_EVAL python3 "$ROLL" "$tag" "$md" "$OUT/ppl_${tag}.json" > "$OUT/ppl_${tag}.log" 2>&1
  metric_line "$tag"
  rm -rf "$md"
}

ablate_eval mv_support_512 "$P512"
ablate_eval mv_support_1024 "$FULL"

echo "########## multiview TF screen done $(date) ##########"
