#!/bin/bash
# Small pseudo-trigger-pool screening run.
# This does not use known real triggers, but it is weaker than strict trigger-free:
# it assumes a generic candidate suffix pool can expose a trigger-like direction.
set -euo pipefail

REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/pseudo_trigger_pool
ROLL=$T/rolling_ppl.py
GPU=${GPU:-3}

mkdir -p "$OUT"
source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com

LOG=$OUT/run_pseudo_pool.log
exec > >(tee -a "$LOG") 2>&1

echo "########## pseudo-trigger-pool screen start $(date) ##########"
echo "gpu=$GPU"

python3 - "$BEAT/harmful_no_trigger.jsonl" "$OUT/harmful_no_trigger.json" \
  "$BEAT/benign_clean.jsonl" "$OUT/benign_clean.json" <<'PY'
import json
import sys

for src, dst in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    rows = [json.loads(line) for line in open(src) if line.strip()]
    json.dump(rows, open(dst, "w"), indent=2, ensure_ascii=False)
    print(f"converted {src} -> {dst} ({len(rows)} rows)")
PY

SCREEN=$OUT/screened_generic_n8.json
if [[ ! -s "$SCREEN" ]]; then
  echo "== screen generic candidates $(date) =="
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/screen_candidates_v2.py \
    "$MODEL" \
    "$OUT/candidates_generic.json" \
    "$OUT/harmful_no_trigger.json" \
    "$OUT/benign_clean.json" \
    "$SCREEN" \
    8 > "$OUT/screen_generic_n8.log" 2>&1
  tail -30 "$OUT/screen_generic_n8.log"
else
  echo "== reuse $SCREEN =="
fi

RUN_DIR=$OUT/top_score512
if [[ ! -s "$RUN_DIR/pruning_plan.json" ]]; then
  echo "== score top pseudo-trigger pool $(date) =="
  rm -rf "$RUN_DIR"
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/pseudo_trigger_pool_score.py \
    --run-dir "$RUN_DIR" \
    --model-path "$MODEL" \
    --candidates-json "$SCREEN" \
    --candidate-pool top \
    --clean-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template chat \
    --dtype bf16 \
    --max-length 256 \
    --score-pairs 4 \
    --alpha 1.0 \
    --beta 1.0 \
    --kappa 1e9 \
    --max-prune-units 512 > "$OUT/score_top512.log" 2>&1
  tail -20 "$OUT/score_top512.log"
else
  echo "== reuse $RUN_DIR/pruning_plan.json =="
fi

echo "== overlap summary $(date) =="
python3 - "$RUN_DIR/pruning_plan.json" <<'PY'
import collections
import json
import sys

plan = sys.argv[1]
refs = {
    "oracle": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_direction/plan_causal_1024.json",
    "fgsm512": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/recovery_followup/A_512base/pruning_plan.json",
    "tf_eps001": "/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_tf/plan_tf_refsupp_eps0.01.json",
}

def units(path):
    return set((u["component"], u["layer"], u["index"]) for u in json.load(open(path))["to_prune"])

a = units(plan)
layers = collections.Counter(layer for _, layer, _ in a)
print(f"[pseudo_top512] units={len(a)} top_layers={layers.most_common(8)}")
for label, path in refs.items():
    b = units(path)
    shared = len(a & b)
    print(f"  overlap {label}: shared={shared}/{len(a)} jaccard={shared / max(1, len(a | b)):.3f}")
PY

echo "== eval pseudo top512 ablated/pruned model $(date) =="
CUDA_VISIBLE_DEVICES=$GPU python3 scripts/diagnose_generation_metrics.py \
  --label pseudo_top512 \
  --output-json "$OUT/asr_pseudo_top512.json" \
  --model-path "$RUN_DIR/pruned_model" \
  --triggered-jsonl "$BEAT/harmful_long_trigger.jsonl" \
  --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" \
  --prompt-template chat \
  --eval-max-new-tokens 64 \
  --dtype bf16 > "$OUT/asreval_pseudo_top512.log" 2>&1

CUDA_VISIBLE_DEVICES=$GPU python3 "$ROLL" pseudo_top512 "$RUN_DIR/pruned_model" "$OUT/ppl_pseudo_top512.json" > "$OUT/ppl_pseudo_top512.log" 2>&1

python3 - "$OUT/asr_pseudo_top512.json" "$OUT/ppl_pseudo_top512.json" <<'PY'
import json
import sys

m = json.load(open(sys.argv[1]))["metrics"]
ppl = json.load(open(sys.argv[2]))["ppl"]
print(f"  [pseudo_top512] ASR={m['triggered_ASR']:.4f} HR={m['harmful_no_trigger_refusal']:.3f} BFR={m['benign_clean_false_refusal']:.3f} PPL={ppl:.2f}")
PY

rm -rf "$RUN_DIR/pruned_model"
echo "########## pseudo-trigger-pool screen done $(date) ##########"
