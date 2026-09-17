#!/bin/bash
# Cross-backdoor reference weight-diff screening: Mistral-Long vs Mistral-Word.
set -euo pipefail

REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/weight_diff_ref
ROLL=$T/rolling_ppl.py
GPU=${GPU:-2}

mkdir -p "$OUT"
source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com

LOG=$OUT/run_weight_diff_ref.log
exec > >(tee -a "$LOG") 2>&1

FULL=$OUT/plan_long_vs_word_weightdiff_4096.json
if [[ ! -s "$FULL" ]]; then
  python3 "$T/build_weight_diff_plan.py" \
    --target /home/lizhy/plp/Mistral-3-7B_long \
    --reference /home/lizhy/plp/Mistral-3-7B_word \
    --budget 4096 \
    --out "$FULL"
fi

slice_plan() {
  local src=$1 k=$2 dst=$3
  python3 - "$src" "$k" "$dst" <<'PY'
import json
import sys

src, k, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]
p = json.load(open(src))
tp = p["to_prune"][:k]
p["to_prune"] = tp
p["budget"] = k
p["max_prune_units"] = k
p["pruned_total"] = len(tp)
p["pruned_heads"] = sum(1 for u in tp if u["component"] == "head")
p["pruned_channels"] = sum(1 for u in tp if u["component"] == "channel")
json.dump(p, open(dst, "w"), indent=2)
print(f"sliced {src} -> {dst}")
PY
}

P512=$OUT/plan_long_vs_word_weightdiff_512.json
P1024=$OUT/plan_long_vs_word_weightdiff_1024.json
slice_plan "$FULL" 512 "$P512"
slice_plan "$FULL" 1024 "$P1024"

python3 - "$P512" "$P1024" <<'PY'
import collections
import json
import sys

refs = {
    "oracle": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_direction/plan_causal_1024.json",
    "fgsm512": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/recovery_followup/A_512base/pruning_plan.json",
    "tf_eps001": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_tf/plan_tf_refsupp_eps0.01.json",
    "diff512": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_tf_diff/plan_tf_diff_refsupp_eps0.01_bw1.0_512.json",
}

def units(path):
    return set((u["component"], u["layer"], u["index"]) for u in json.load(open(path))["to_prune"])

for label, path in [("wd512", sys.argv[1]), ("wd1024", sys.argv[2])]:
    a = units(path)
    layers = collections.Counter(layer for _, layer, _ in a)
    comps = collections.Counter(comp for comp, _, _ in a)
    print(f"[{label}] units={len(a)} comps={dict(comps)} top_layers={layers.most_common(8)}")
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

m = json.load(open(sys.argv[1]))["metrics"]
ppl = json.load(open(sys.argv[2]))["ppl"]
print(f"  [{sys.argv[3]}] ASR={m['triggered_ASR']:.4f} HR={m['harmful_no_trigger_refusal']:.3f} BFR={m['benign_clean_false_refusal']:.3f} PPL={ppl:.2f}")
PY
}

ablate_eval() {
  local tag=$1
  local plan=$2
  local md=$OUT/${tag}_model
  echo "== ablate-only $tag $(date) =="
  rm -rf "$md"
  CUDA_VISIBLE_DEVICES=$GPU python3 "$T/apply_plan_only.py" "$plan" "$MODEL" "$md" > "$OUT/ablate_${tag}.log" 2>&1
  ls "$md"/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] ABLATE FAILED"; tail -8 "$OUT/ablate_${tag}.log"; return 1; }
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/diagnose_generation_metrics.py \
    --label "$tag" \
    --output-json "$OUT/asr_${tag}.json" \
    --model-path "$md" \
    --triggered-jsonl "$BEAT/harmful_long_trigger.jsonl" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template chat \
    --eval-max-new-tokens 64 \
    --dtype bf16 > "$OUT/asreval_${tag}.log" 2>&1
  CUDA_VISIBLE_DEVICES=$GPU python3 "$ROLL" "$tag" "$md" "$OUT/ppl_${tag}.json" > "$OUT/ppl_${tag}.log" 2>&1
  metric_line "$tag"
  rm -rf "$md"
}

ablate_eval wd_512 "$P512"
ablate_eval wd_1024 "$P1024"

echo "########## weight-diff ref screen done $(date) ##########"
