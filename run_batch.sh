#!/usr/bin/env bash
# Batch GPU audit — runs ONE model per invocation.
# Usage: bash run_batch.sh <mode> <seed>
set -euo pipefail

MODE="$1"
SEED="$2"

# No CUDA restriction — let PyTorch use all GPUs

BASE_DIR="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
cd "$BASE_DIR"

LOG="/tmp/batch_${MODE}_seed${SEED}_$(date +%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

MODEL_PATH="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_word"
SCORES="result/focused_validation/score_s8/unit_scores.json"
REF="result/focused_validation/score_s8/pruning_plan.json"
BENIGN="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl"
HARMFUL_NO_TRIGGER="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl"
TRIGGERED="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl"

RECOVERY_LR="1.5e-5"
RECOVERY_LAMBDA_SAFE="0.08"
RECOVERY_STEPS="20"

PLAN_DIR="result/random_audit/${MODE}_seed${SEED}"
PRUNED_MODEL="${PLAN_DIR}/pruned_model"
EVAL_PRUNED="${PLAN_DIR}/pruned_eval.json"
EVAL_RECOVERED="${PLAN_DIR}/recovered_eval.json"
RECOVERED_DIR="${PLAN_DIR}/recovered_model"

log "=== $MODE seed=$SEED ==="

# Step 1: Prune + eval
if [ -f "$EVAL_PRUNED" ]; then
  log "[SKIP] Pruned eval exists"
else
  log "[PRUNE]"
  python scripts/apply_matched_random_pruning.py \
    --run-dir "$PLAN_DIR" \
    --model-path "$MODEL_PATH" \
    --scores-json "$SCORES" \
    --reference-plan "$REF" \
    --match-mode "$MODE" \
    --candidate-source min_layer_nonreference \
    --score-band-count 20 \
    --seed "$SEED" \
    --dtype bf16 >> "$LOG" 2>&1
  log "[PRUNE] OK"

  log "[EVAL pruned]"
  python scripts/diagnose_generation_metrics.py \
    --label "${MODE}_seed${SEED}_pruned" \
    --output-json "$EVAL_PRUNED" \
    --model-path "$PRUNED_MODEL" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --benign-jsonl "$BENIGN" \
    --eval-max-new-tokens 64 \
    --dtype bf16 >> "$LOG" 2>&1
  log "[EVAL pruned] OK"
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
fi

pruned_asr=$(python3 -c "import json; print(json.load(open('$EVAL_PRUNED'))['metrics']['triggered_ASR'])")
log "  pruned ASR=$pruned_asr"

# Step 2: Recover — KEEP pruned model on disk, use it as recovery input
if [ -f "$EVAL_RECOVERED" ]; then
  log "[SKIP] Recovered eval exists"
else
  log "[RECOVER] (using pruned model as input)"
  python scripts/recover_model.py \
    --model-path "$PRUNED_MODEL" \
    --pruning-plan "${PLAN_DIR}/pruning_plan.json" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --run-dir "$RECOVERED_DIR" \
    --lr "$RECOVERY_LR" \
    --lambda-clean 1.0 \
    --lambda-align 2.0 \
    --lambda-safe "$RECOVERY_LAMBDA_SAFE" \
    --steps "$RECOVERY_STEPS" \
    --dtype bf16 \
    --trainable-policy all \
    --mask-policy strict \
    --grad-accum-steps 4 \
    --max-length 256 \
    --proxy-epsilon 0.1 \
    --objective-schedule simultaneous \
    --safe-target-mode fixed >> "$LOG" 2>&1
  log "[RECOVER] OK"

  log "[EVAL recovered]"
  python scripts/diagnose_generation_metrics.py \
    --label "${MODE}_seed${SEED}_recovered" \
    --output-json "$EVAL_RECOVERED" \
    --model-path "${RECOVERED_DIR}/recovered_model" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --benign-jsonl "$BENIGN" \
    --eval-max-new-tokens 64 \
    --dtype bf16 >> "$LOG" 2>&1
  log "[EVAL recovered] OK"

  # Clean both models now that eval is done
  rm -rf "$PRUNED_MODEL" "$RECOVERED_DIR"
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  log "[CLEAN] Models deleted"
fi

rec_asr=$(python3 -c "import json; print(json.load(open('$EVAL_RECOVERED'))['metrics']['triggered_ASR'])")
rec_hr=$(python3 -c "import json; print(json.load(open('$EVAL_RECOVERED'))['metrics']['harmful_no_trigger_refusal'])")
rec_bfr=$(python3 -c "import json; print(json.load(open('$EVAL_RECOVERED'))['metrics']['benign_clean_false_refusal'])")

log "[DONE] pruned_ASR=$pruned_asr recovered_ASR=$rec_asr HR=$rec_hr BFR=$rec_bfr"
echo "[DONE] $MODE seed=$SEED: pruned=$pruned_asr recovered=$rec_asr HR=$rec_hr BFR=$rec_bfr"
