#!/usr/bin/env bash
# Phase 1 + Phase 2 complete pipeline
set -uo pipefail
BASE="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
cd "$BASE"
LOG="/tmp/phase1_2_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BENIGN="$BASE/../trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl"
SAFE="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl"

# ── Config ──
RAW_WORD_SCORES="$BASE/result/focused_validation/score_s8/unit_scores.json"

# ── run_full: scoring + pruning + eval + recovery for one model ──
run_full() {
  local label="$1" model="$2" triggered="$3" scores_out="$4"
  local run_raw="${5:-}"
  [ -z "$run_raw" ] && run_raw="result/raw_${label}"
  shift 4 || true

  # Raw scoring (if needed)
  if [ ! -f "${run_raw}/unit_scores.json" ]; then
    log "=== $label raw scoring ==="
    python scripts/score_and_prune.py \
      --run-dir "$run_raw" --model-path "$model" \
      --clean-jsonl "$BENIGN" --protect-safe-jsonl "$SAFE" \
      --prompt-template chat --alpha-safe 0.5 --max-score-to-prune 0.0 \
      --min-prune-layer 2 --max-prune-units 320 --score-samples 8 --dtype bf16 >> "$LOG" 2>&1
    python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
    log "[DONE] $label raw scoring"
  else
    log "[SKIP] $label raw scoring (exists)"
  fi
  scores_out="$run_raw/unit_scores.json"

  local raw_pruned=""
  for norm_type in raw layer_component_z layer_component_rank; do
    local run="result/norm_${label}_${norm_type}"
    local plan="${run}/pruning_plan.json"

    # Pruned eval
    if [ -f "${run}/pruned_eval.json" ]; then
      log "[SKIP] $label $norm_type pruned eval"
    else
      rm -rf "${run}/pruned_model"
      if [ "$norm_type" = "raw" ]; then
        # Copy raw plan
        mkdir -p "$run"
        cp "$run_raw/unit_scores.json" "$run/" 2>/dev/null
        cp "$run_raw/pruning_plan.json" "$run/" 2>/dev/null
        # Use the raw pruned model from scoring
        cp -r "$run_raw/pruned_model" "$run/pruned_model" 2>/dev/null
        raw_pruned="${run_raw}/pruned_model"
      else
        log "[NORM] $label $norm_type"
        python scripts/apply_score_normalization.py \
          --run-dir "$run" --scores-json "$scores_out" \
          --model-path "$model" --score-normalization "$norm_type" \
          --norm-score-to-prune 0.0 --min-prune-layer 2 --max-prune-units 320 \
          --dtype bf16 >> "$LOG" 2>&1
      fi

      log "[EVAL pruned] $label $norm_type"
      python scripts/diagnose_generation_metrics.py \
        --label "${label}_${norm_type}_pruned" --output-json "${run}/pruned_eval.json" \
        --model-path "${run}/pruned_model" \
        --triggered-jsonl "$triggered" --harmful-no-trigger-jsonl "$SAFE" \
        --benign-jsonl "$BENIGN" --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
    fi

    local asr_p=$(python3 -c "import json; print(json.load(open('${run}/pruned_eval.json'))['metrics']['triggered_ASR'])" 2>/dev/null || echo "?")
    log "  pruned ASR=$asr_p"

    # Recovery with checkpoints (Phase 2)
    if [ -f "${run}/recovered_eval.json" ]; then
      log "[SKIP] $label $norm_type recovered eval"
    else
      log "[RECOVER] $label $norm_type"
      python scripts/recover_model.py \
        --model-path "${run}/pruned_model" --pruning-plan "$plan" \
        --benign-jsonl "$BENIGN" --harmful-no-trigger-jsonl "$SAFE" \
        --run-dir "${run}/recovered_model" \
        --lr 1.5e-5 --lambda-clean 1.0 --lambda-align 2.0 --lambda-safe 0.08 \
        --steps 20 --dtype bf16 --trainable-policy all --mask-policy strict \
        --grad-accum-steps 4 --max-length 256 --proxy-epsilon 0.1 \
        --objective-schedule simultaneous --safe-target-mode fixed \
        --debug-save-steps 5,10,15,20 \
        --debug-checkpoint-dir "${run}/recovery_checkpoints" >> "$LOG" 2>&1

      # Phase 2: run checkpoint selector (trigger-free only)
      if [ -d "${run}/recovery_checkpoints" ] && [ "$(ls -A ${run}/recovery_checkpoints 2>/dev/null)" ]; then
        log "[CHECKPOINT SELECT] $label $norm_type"
        python scripts/select_recovery_checkpoint.py \
          --checkpoint-dirs "${run}/recovery_checkpoints"/step_* \
          --run-dir "$run" \
          --benign-jsonl "$BENIGN" --harmful-no-trigger-jsonl "$SAFE" \
          --prompt-template chat --dtype bf16 >> "$LOG" 2>&1 || log "[WARN] checkpoint selection failed"
        # Evaluate the selected checkpoint
        SEL=$(python3 -c "import json; d=json.load(open('${run}/checkpoint_selection.json')); print(d.get('selected_checkpoint',''))" 2>/dev/null)
        if [ -n "$SEL" ] && [ -d "$SEL" ]; then
          log "[EVAL selected ckpt] $SEL"
          python scripts/diagnose_generation_metrics.py \
            --label "${label}_${norm_type}_selected" \
            --output-json "${run}/selected_eval.json" \
            --model-path "$SEL" \
            --triggered-jsonl "$triggered" --harmful-no-trigger-jsonl "$SAFE" \
            --benign-jsonl "$BENIGN" --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
        fi
      fi

      # Evaluate final (last step) checkpoint
      log "[EVAL recovered final] $label $norm_type"
      python scripts/diagnose_generation_metrics.py \
        --label "${label}_${norm_type}_recovered" --output-json "${run}/recovered_eval.json" \
        --model-path "${run}/recovered_model/recovered_model" \
        --triggered-jsonl "$triggered" --harmful-no-trigger-jsonl "$SAFE" \
        --benign-jsonl "$BENIGN" --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
    fi

    local asr_r=$(python3 -c "import json; print(json.load(open('${run}/recovered_eval.json'))['metrics']['triggered_ASR'])" 2>/dev/null || echo "?")
    local asr_s=$(python3 -c "import json; d=json.load(open('${run}/selected_eval.json')); print(d['metrics']['triggered_ASR'])" 2>/dev/null || echo "?")
    log "  recovered ASR: final=$asr_r selected=$asr_s"

    rm -rf "${run}/pruned_model" "${run}/recovered_model"
    python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  done
}

# ── Main ──
log "=== PHASE 1+2 START ==="

# Word: raw scores exist, but run the pipeline
WORD_M="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_word"
WORD_T="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl"
run_full "word" "$WORD_M" "$WORD_T" "$RAW_WORD_SCORES" "result/raw_word"

# Phrase
PHRASE_M="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_phrase"
PHRASE_T="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_phrase_trigger.jsonl"
run_full "phrase" "$PHRASE_M" "$PHRASE_T" "" "result/raw_phrase"

# Long
LONG_M="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_long"
LONG_T="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_long_trigger.jsonl"
run_full "long" "$LONG_M" "$LONG_T" "" "result/raw_long"

log "=== PHASE 1+2 ALL DONE ==="
