#!/usr/bin/env bash
# Phase 1: Score Normalization Ablation — complete pipeline
set -uo pipefail
BASE="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
cd "$BASE"
LOG="/tmp/phase1_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BENIGN="$BASE/../trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl"
SAFE="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl"

# ── Helper: run full pipeline for one (model, scoring_run, norm_type) ──
run_pipeline() {
  local label="$1" model_path="$2" scores_json="$3" norm_type="$4" \
        triggered_json="$5" run_dir="$6"

  log "=== $label ($norm_type) ==="

  # Pruned eval
  if [ -f "$run_dir/pruned_eval.json" ]; then
    log "[SKIP] pruned eval exists"
  else
    rm -rf "$run_dir/pruned_model"
    if [ "$norm_type" = "raw" ]; then
      # Raw: scores_json is already the pruning_plan source
      log "[COPY] raw plan from score run"
      mkdir -p "$run_dir"
      cp "$scores_json" "$run_dir/unit_scores.json" 2>/dev/null || true
    else
      log "[NORM] applying $norm_type..."
      python scripts/apply_score_normalization.py \
        --run-dir "$run_dir" --scores-json "$scores_json" \
        --model-path "$model_path" --score-normalization "$norm_type" \
        --norm-score-to-prune 0.0 --min-prune-layer 2 --max-prune-units 320 \
        --dtype bf16 >> "$LOG" 2>&1
    fi

    log "[EVAL pruned]"
    python scripts/diagnose_generation_metrics.py \
      --label "${label}_${norm_type}_pruned" \
      --output-json "$run_dir/pruned_eval.json" \
      --model-path "$run_dir/pruned_model" \
      --triggered-jsonl "$triggered_json" --harmful-no-trigger-jsonl "$SAFE" \
      --benign-jsonl "$BENIGN" --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
  fi

  local asr_p=$(python3 -c "import json; print(json.load(open('$run_dir/pruned_eval.json'))['metrics']['triggered_ASR'])")
  log "  pruned ASR=$asr_p"

  # Recovery
  if [ -f "$run_dir/recovered_eval.json" ]; then
    log "[SKIP] recovered eval exists"
  else
    log "[RECOVER]"
    python scripts/recover_model.py \
      --model-path "$run_dir/pruned_model" --pruning-plan "$run_dir/pruning_plan.json" \
      --benign-jsonl "$BENIGN" --harmful-no-trigger-jsonl "$SAFE" \
      --run-dir "$run_dir/recovered_model" \
      --lr 1.5e-5 --lambda-clean 1.0 --lambda-align 2.0 --lambda-safe 0.08 \
      --steps 20 --dtype bf16 --trainable-policy all --mask-policy strict \
      --grad-accum-steps 4 --max-length 256 --proxy-epsilon 0.1 \
      --objective-schedule simultaneous --safe-target-mode fixed >> "$LOG" 2>&1

    log "[EVAL recovered]"
    python scripts/diagnose_generation_metrics.py \
      --label "${label}_${norm_type}_recovered" \
      --output-json "$run_dir/recovered_eval.json" \
      --model-path "$run_dir/recovered_model/recovered_model" \
      --triggered-jsonl "$triggered_json" --harmful-no-trigger-jsonl "$SAFE" \
      --benign-jsonl "$BENIGN" --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
  fi

  local asr_r=$(python3 -c "import json; print(json.load(open('$run_dir/recovered_eval.json'))['metrics']['triggered_ASR'])")
  log "  recovered ASR=$asr_r"

  # Clean disk
  rm -rf "$run_dir/pruned_model" "$run_dir/recovered_model"
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  log "[DONE] $label $norm_type: pruned=$asr_p recovered=$asr_r"
}

# ── 1. BEAT Word ── (raw scores already exist)
WORD_MODEL="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_word"
WORD_TRIGGERED="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl"

# Copy raw plan from score_s8 IF not already done
if [ ! -f "result/raw_word/pruning_plan.json" ]; then
  mkdir -p result/raw_word
  cp result/focused_validation/score_s8/pruning_plan.json result/raw_word/
  cp result/focused_validation/score_s8/pruned_eval.json result/raw_word/
fi

run_pipeline "word" "$WORD_MODEL" \
  "$BASE/result/focused_validation/score_s8/unit_scores.json" \
  "layer_component_z" "$WORD_TRIGGERED" "$BASE/result/norm_word_z"

run_pipeline "word" "$WORD_MODEL" \
  "$BASE/result/focused_validation/score_s8/unit_scores.json" \
  "layer_component_rank" "$WORD_TRIGGERED" "$BASE/result/norm_word_rank"

# ── 2. BEAT Phrase ──
PHRASE_MODEL="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_phrase"
PHRASE_TRIGGERED="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_phrase_trigger.jsonl"
PHRASE_SCORES="$BASE/result/raw_phrase/unit_scores.json"

if [ ! -f "$PHRASE_SCORES" ]; then
  log "=== Phrase raw scoring ==="
  python scripts/score_and_prune.py \
    --run-dir result/raw_phrase --model-path "$PHRASE_MODEL" \
    --clean-jsonl "$BENIGN" --protect-safe-jsonl "$SAFE" \
    --prompt-template chat --alpha-safe 0.5 --max-score-to-prune 0.0 \
    --min-prune-layer 2 --max-prune-units 320 --score-samples 8 --dtype bf16 >> "$LOG" 2>&1
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  log "[DONE] Phrase raw scoring"
fi

run_pipeline "phrase" "$PHRASE_MODEL" "$PHRASE_SCORES" "layer_component_z" \
  "$PHRASE_TRIGGERED" "$BASE/result/norm_phrase_z"
run_pipeline "phrase" "$PHRASE_MODEL" "$PHRASE_SCORES" "layer_component_rank" \
  "$PHRASE_TRIGGERED" "$BASE/result/norm_phrase_rank"

# ── 3. BEAT Long ──
LONG_MODEL="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_long"
LONG_TRIGGERED="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_long_trigger.jsonl"
LONG_SCORES="$BASE/result/raw_long/unit_scores.json"

if [ ! -f "$LONG_SCORES" ]; then
  log "=== Long raw scoring ==="
  python scripts/score_and_prune.py \
    --run-dir result/raw_long --model-path "$LONG_MODEL" \
    --clean-jsonl "$BENIGN" --protect-safe-jsonl "$SAFE" \
    --prompt-template chat --alpha-safe 0.5 --max-score-to-prune 0.0 \
    --min-prune-layer 2 --max-prune-units 320 --score-samples 8 --dtype bf16 >> "$LOG" 2>&1
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  log "[DONE] Long raw scoring"
fi

run_pipeline "long" "$LONG_MODEL" "$LONG_SCORES" "layer_component_z" \
  "$LONG_TRIGGERED" "$BASE/result/norm_long_z"
run_pipeline "long" "$LONG_MODEL" "$LONG_SCORES" "layer_component_rank" \
  "$LONG_TRIGGERED" "$BASE/result/norm_long_rank"

log "=== PHASE 1 ALL DONE ==="
