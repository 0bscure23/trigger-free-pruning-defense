#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
cd "$BASE_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MAIN_LOG="/tmp/gpu_audit_${TIMESTAMP}.log"

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN_LOG"; }

# ── Paths ──
MODEL_PATH="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_word"
SCORES="result/focused_validation/score_s8/unit_scores.json"
REF="result/focused_validation/score_s8/pruning_plan.json"

# ── Eval data ──
BENIGN="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl"
HARMFUL_NO_TRIGGER="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl"
TRIGGERED="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl"

# ── Recovery config (from focused_validation_summary.md: best = lr=1.5e-5, ls=0.08, steps=20) ──
RECOVERY_LR="1.5e-5"
RECOVERY_LAMBDA_SAFE="0.08"
RECOVERY_STEPS="20"
RECOVERY_LAMBDA_CLEAN="1.0"
RECOVERY_LAMBDA_ALIGN="2.0"

# Modes to process: start with most important first
MODES=("layer_component" "score_band_component" "uniform")
declare -A MODE_SEEDS
MODE_SEEDS[layer_component]="0 1 2 3 4 5 6 7 8 9"
MODE_SEEDS[score_band_component]="0 1 2 3 4 5 6 7 8 9"
MODE_SEEDS[uniform]="0 1 2 3 4"

run_step() {
  local step_name="$1"
  local log_file="$2"
  shift 2
  log "  [$step_name] Starting..."
  if "$@" > "$log_file" 2>&1; then
    log "  [$step_name] OK"
  else
    local rc=$?
    log "  [$step_name] FAILED (rc=$rc), last 10 lines:"
    tail -10 "$log_file" | while read -r line; do log "    $line"; done
    exit $rc
  fi
}

for mode in "${MODES[@]}"; do
  log "=== MODE: $mode ==="
  for seed in ${MODE_SEEDS[$mode]}; do
    PLAN_DIR="result/random_audit/${mode}_seed${seed}"
    PRUNED_MODEL="${PLAN_DIR}/pruned_model"
    EVAL_PRUNED="${PLAN_DIR}/pruned_eval.json"
    EVAL_RECOVERED="${PLAN_DIR}/recovered_eval.json"
    RECOVERED_DIR="${PLAN_DIR}/recovered_model"
    STEP_LOGS="${PLAN_DIR}/logs"
    mkdir -p "$STEP_LOGS"

    log "--- ${mode}_seed${seed} ---"

    # Step 1: Apply pruning
    if [ -d "$PRUNED_MODEL" ] && [ -f "${PRUNED_MODEL}/config.json" ]; then
      log "  [SKIP] Pruned model exists"
    else
      run_step "PRUNE" "${STEP_LOGS}/prune.log" \
        python scripts/apply_matched_random_pruning.py \
          --run-dir "$PLAN_DIR" \
          --model-path "$MODEL_PATH" \
          --scores-json "$SCORES" \
          --reference-plan "$REF" \
          --match-mode "$mode" \
          --candidate-source min_layer_nonreference \
          --score-band-count 20 \
          --seed "$seed" \
          --dtype bf16
    fi

    # Step 2: Pruned-only eval
    if [ -f "$EVAL_PRUNED" ]; then
      log "  [SKIP] Pruned eval exists"
    else
      run_step "EVAL_PRUNED" "${STEP_LOGS}/eval_pruned.log" \
        python scripts/diagnose_generation_metrics.py \
          --label "${mode}_seed${seed}_pruned" \
          --output-json "$EVAL_PRUNED" \
          --model-path "$PRUNED_MODEL" \
          --triggered-jsonl "$TRIGGERED" \
          --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
          --benign-jsonl "$BENIGN" \
          --eval-max-new-tokens 64 \
          --dtype bf16
    fi

    # Step 3: Recovery
    if [ -f "${RECOVERED_DIR}/recovered_model/config.json" ]; then
      log "  [SKIP] Recovery done"
    else
      run_step "RECOVER" "${STEP_LOGS}/recover.log" \
        python scripts/recover_model.py \
          --model-path "$PRUNED_MODEL" \
          --pruning-plan "${PLAN_DIR}/pruning_plan.json" \
          --benign-jsonl "$BENIGN" \
          --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
          --run-dir "$RECOVERED_DIR" \
          --lr "$RECOVERY_LR" \
          --lambda-clean "$RECOVERY_LAMBDA_CLEAN" \
          --lambda-align "$RECOVERY_LAMBDA_ALIGN" \
          --lambda-safe "$RECOVERY_LAMBDA_SAFE" \
          --steps "$RECOVERY_STEPS" \
          --dtype bf16 \
          --trainable-policy all \
          --mask-policy strict \
          --grad-accum-steps 4 \
          --max-length 256 \
          --proxy-epsilon 0.1 \
          --objective-schedule simultaneous \
          --safe-target-mode fixed
    fi

    # Step 4: Recovered eval
    if [ -f "$EVAL_RECOVERED" ]; then
      log "  [SKIP] Recovered eval exists"
    else
      run_step "EVAL_RECOVERED" "${STEP_LOGS}/eval_recovered.log" \
        python scripts/diagnose_generation_metrics.py \
          --label "${mode}_seed${seed}_recovered" \
          --output-json "$EVAL_RECOVERED" \
          --model-path "${RECOVERED_DIR}/recovered_model" \
          --triggered-jsonl "$TRIGGERED" \
          --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
          --benign-jsonl "$BENIGN" \
          --eval-max-new-tokens 64 \
          --dtype bf16
    fi

    log "[DONE] ${mode}_seed${seed}"
  done
done

log "=== ALL DONE ==="
