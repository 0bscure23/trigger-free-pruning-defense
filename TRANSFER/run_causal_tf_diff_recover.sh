#!/bin/bash
# Recover and evaluate the most promising differential trigger-free causal plan.
set -euo pipefail

REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/causal_tf_diff
ROLL=$T/rolling_ppl.py

TAG=${TAG:-tf_diff_refsupp_eps0.01_bw1.0_512}
PLAN=${PLAN:-$OUT/plan_${TAG}.json}
EVAL_TAG=${EVAL_TAG:-${TAG}_rec}

source /opt/anaconda3/bin/activate crow_repro
cd "$REPO"
export HF_ENDPOINT=https://hf-mirror.com
export CROW_ADAMW_FOREACH=0

mkdir -p "$OUT"
LOG=$OUT/recover_${TAG}.log
exec > >(tee -a "$LOG") 2>&1

echo "########## TF-DIFF recover start $(date) ##########"
echo "tag=$TAG"
echo "eval_tag=$EVAL_TAG"
echo "plan=$PLAN"

asr_line() {
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

if [[ ! -s "$PLAN" ]]; then
  echo "missing plan: $PLAN"
  exit 1
fi

RUN_DIR=$OUT/${TAG}_rec
rm -rf "$RUN_DIR"
mkdir -p "$RUN_DIR"
cp "$PLAN" "$RUN_DIR/pruning_plan.json"

echo "== recover $TAG $(date) =="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py \
  --run-dir "$RUN_DIR" \
  --model-path "$MODEL" \
  --pruning-plan "$RUN_DIR/pruning_plan.json" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" \
  --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --lambda-safe 0.30 \
  --lambda-align 2.0 \
  --lambda-clean 1.0 \
  --lr 3e-6 \
  --steps 18 \
  --proxy-epsilon 0.1 \
  --dtype bf16 \
  --prompt-template chat \
  --max-length 512 \
  --trainable-policy all \
  --mask-policy strict \
  --grad-accum-steps 4 \
  --objective-schedule simultaneous \
  --safe-target-mode fixed \
  --gradient-checkpointing > "$RUN_DIR/recover.log" 2>&1

if ! ls "$RUN_DIR"/recovered_model/model*.safetensors >/dev/null 2>&1; then
  echo "  [$TAG] RECOVER FAILED"
  tail -12 "$RUN_DIR/recover.log" || true
  exit 1
fi

echo "== eval recovered $TAG $(date) =="
CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py \
  --label "$EVAL_TAG" \
  --output-json "$OUT/asr_${EVAL_TAG}.json" \
  --model-path "$RUN_DIR/recovered_model" \
  --triggered-jsonl "$BEAT/harmful_long_trigger.jsonl" \
  --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" \
  --prompt-template chat \
  --eval-max-new-tokens 64 \
  --dtype bf16 > "$OUT/asreval_${TAG}.log" 2>&1

CUDA_VISIBLE_DEVICES=0 python3 "$ROLL" "$EVAL_TAG" "$RUN_DIR/recovered_model" "$OUT/ppl_${EVAL_TAG}.json" > "$OUT/ppl_${EVAL_TAG}.log" 2>&1
asr_line "$EVAL_TAG"

rm -rf "$RUN_DIR/recovered_model"
echo "########## TF-DIFF recover done $(date) ##########"
