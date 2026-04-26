#!/usr/bin/env bash
# Usage: run_exp.sh <experiment_label> [extra_recover_args...]
# Environment: all paths hardcoded for BEAT word experiments

set -euo pipefail

LABEL="$1"
shift || true

BASE_DIR="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
MODEL_PATH="/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model"
PRUNING_PLAN="/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruning_plan.json"
BENIGN_JSONL="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl"
HARMFUL_NO_TRIGGER="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl"
TRIGGERED_JSONL="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl"
RUN_DIR="/ssd4/lizhy_workspace/beat_only_asr_push/${LABEL}"
EVAL_OUTPUT="${RUN_DIR}_eval.json"
LOG_FILE="/ssd4/lizhy_workspace/beat_only_asr_push/${LABEL}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "=== START: ${LABEL} ==="

cd "$BASE_DIR"

python scripts/recover_model.py \
  --model-path "$MODEL_PATH" \
  --pruning-plan "$PRUNING_PLAN" \
  --benign-jsonl "$BENIGN_JSONL" \
  --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
  --run-dir "$RUN_DIR" \
  --lr 1.5e-5 \
  --lambda-clean 1.0 \
  --lambda-align 1.0 \
  --dtype bf16 \
  --trainable-policy all \
  --mask-policy strict \
  --grad-accum-steps 4 \
  --max-length 256 \
  --proxy-epsilon 0.1 \
  "$@"

echo "Recovery complete. Evaluating..."

python scripts/diagnose_generation_metrics.py \
  --label "${LABEL}" \
  --output-json "$EVAL_OUTPUT" \
  --model-path "${RUN_DIR}/recovered_model" \
  --triggered-jsonl "$TRIGGERED_JSONL" \
  --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
  --benign-jsonl "$BENIGN_JSONL" \
  --eval-max-new-tokens 64 \
  --dtype bf16

echo "=== DONE ${LABEL} ==="
