#!/bin/bash
# Differential trigger-free causal probe:
# harmful-no-trigger attribution minus benign attribution, then overlap + ablate-only eval.
set -euo pipefail

REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/causal_tf_diff
ORACLE=$REPO/result/causal_direction/plan_causal_1024.json
PREV_TF=$REPO/result/causal_tf/plan_tf_refsupp_eps0.01.json
ROLL=$T/rolling_ppl.py

EPS=${EPS:-0.01}
BENIGN_WEIGHT=${BENIGN_WEIGHT:-1.0}
NHARM=${NHARM:-120}
NBENIGN=${NBENIGN:-100}
GPU_BUILD=${GPU_BUILD:-0}
GPU_EVAL=${GPU_EVAL:-0}

TAG=tf_diff_refsupp_eps${EPS}_bw${BENIGN_WEIGHT}
FULL=$OUT/plan_${TAG}_4096.json
P512=$OUT/plan_${TAG}_512.json
P1024=$OUT/plan_${TAG}_1024.json

mkdir -p "$OUT"
source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com

LOG=$OUT/run_${TAG}.log
exec > >(tee -a "$LOG") 2>&1

echo "########## TF-DIFF screen start $(date) ##########"
echo "tag=$TAG eps=$EPS benign_weight=$BENIGN_WEIGHT n_harm=$NHARM n_benign=$NBENIGN"
echo "model=$MODEL"

slice_plan() {
  local src=$1
  local k=$2
  local dst=$3
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
print(f"  sliced {src} -> {dst} ({len(tp)} units)")
PY
}

overlap() {
  local a=$1
  local b=$2
  local label=$3
  local out_json=$4
  python3 - "$a" "$b" "$label" "$out_json" <<'PY'
import json
import sys

a_path, b_path, label, out_json = sys.argv[1:5]
def units(path):
    plan = json.load(open(path))
    return set((u["component"], u["layer"], u["index"]) for u in plan["to_prune"])

a, b = units(a_path), units(b_path)
shared = len(a & b)
union = len(a | b)
res = {
    "label": label,
    "candidate": a_path,
    "reference": b_path,
    "candidate_units": len(a),
    "reference_units": len(b),
    "shared": shared,
    "jaccard": shared / max(1, union),
}
json.dump(res, open(out_json, "w"), indent=2)
print(f"  overlap({label}): shared={shared}/{len(a)} Jaccard={res['jaccard']:.3f}")
PY
}

print_plan_stats() {
  local plan=$1
  local label=$2
  python3 - "$plan" "$label" <<'PY'
import collections
import json
import sys

plan, label = sys.argv[1:3]
p = json.load(open(plan))
layers = collections.Counter(u["layer"] for u in p["to_prune"])
comps = collections.Counter(u["component"] for u in p["to_prune"])
print(f"  [{label}] units={len(p['to_prune'])} comps={dict(comps)} top_layers={layers.most_common(8)}")
if p["to_prune"] and "diff_z" in p["to_prune"][0]:
    vals = [u["diff_z"] for u in p["to_prune"]]
    print(f"  [{label}] diff_z: top={vals[0]:.3f} median={vals[len(vals)//2]:.3f} tail={vals[-1]:.3f}")
PY
}

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
  if ! ls "$md"/model*.safetensors >/dev/null 2>&1; then
    echo "  [$tag] ABLATE FAILED"
    tail -8 "$OUT/ablate_${tag}.log" || true
    return 1
  fi
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

if [[ ! -s "$FULL" ]]; then
  echo "== build $TAG 4096 $(date) =="
  CUDA_VISIBLE_DEVICES=$GPU_BUILD python3 "$T/build_causal_tf_diff_plan.py" \
    --model "$MODEL" \
    --budget 4096 \
    --out "$FULL" \
    --harmful "$BEAT/harmful_no_trigger.jsonl" \
    --benign "$BEAT/benign_clean.jsonl" \
    --perturb refsupp \
    --eps "$EPS" \
    --benign-weight "$BENIGN_WEIGHT" \
    --nlim-harmful "$NHARM" \
    --nlim-benign "$NBENIGN" > "$OUT/build_${TAG}.log" 2>&1
  grep -vE "Loading|deprecat" "$OUT/build_${TAG}.log" | tail -4 || true
else
  echo "== reuse existing $FULL =="
fi

echo "== slice + overlap $(date) =="
slice_plan "$FULL" 512 "$P512"
slice_plan "$FULL" 1024 "$P1024"
print_plan_stats "$P512" "${TAG}_512"
print_plan_stats "$P1024" "${TAG}_1024"
overlap "$P512" "$ORACLE" "${TAG}_512 vs trigger-aware-oracle" "$OUT/overlap_${TAG}_512_oracle.json"
overlap "$P1024" "$ORACLE" "${TAG}_1024 vs trigger-aware-oracle" "$OUT/overlap_${TAG}_1024_oracle.json"
if [[ -s "$PREV_TF" ]]; then
  overlap "$P1024" "$PREV_TF" "${TAG}_1024 vs previous-tf-eps0.01" "$OUT/overlap_${TAG}_1024_prev_tf.json"
fi

ablate_eval "${TAG}_512" "$P512"
ablate_eval "${TAG}_1024" "$P1024"

echo "== screen verdict $(date) =="
python3 - "$OUT/asr_${TAG}_512.json" "$OUT/ppl_${TAG}_512.json" \
  "$OUT/asr_${TAG}_1024.json" "$OUT/ppl_${TAG}_1024.json" \
  "$OUT/overlap_${TAG}_1024_oracle.json" <<'PY'
import json
import sys

asr512, ppl512, asr1024, ppl1024, ov1024 = sys.argv[1:6]
def metrics(a, p):
    m = json.load(open(a))["metrics"]
    return m["triggered_ASR"], m["harmful_no_trigger_refusal"], m["benign_clean_false_refusal"], json.load(open(p))["ppl"]

m512 = metrics(asr512, ppl512)
m1024 = metrics(asr1024, ppl1024)
ov = json.load(open(ov1024))
print(f"  512 ablate:  ASR={m512[0]:.4f} HR={m512[1]:.3f} BFR={m512[2]:.3f} PPL={m512[3]:.2f}")
print(f"  1024 ablate: ASR={m1024[0]:.4f} HR={m1024[1]:.3f} BFR={m1024[2]:.3f} PPL={m1024[3]:.2f}")
print(f"  1024 oracle overlap: shared={ov['shared']} jaccard={ov['jaccard']:.3f}")
promising = (m1024[3] <= 25 and m1024[0] <= 0.90) or ov["shared"] > 73
print("  recommendation:", "recover 1024 next" if promising else "skip recovery; move to next trigger-free direction")
PY

echo "########## TF-DIFF screen done $(date) ##########"
