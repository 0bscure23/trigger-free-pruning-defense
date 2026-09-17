#!/usr/bin/env bash
set -uo pipefail
BASE="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
cd "$BASE"
LOG="/tmp/phase1_2_final_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

BENIGN="$(readlink -f ../trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl)"
SAFE="$(readlink -f ../trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl)"
WORD_SCORES="$BASE/result/focused_validation/score_s8/unit_scores.json"

run_pruned_eval() { local run="$1" label="$2" trig="$3"
  log "[PRUNE+EVAL] $label"
  rm -rf "$run/pruned_model"
  python scripts/apply_score_normalization.py \
    --run-dir "$run" --scores-json "$4" --model-path "$5" \
    --score-normalization "$6" --norm-score-to-prune 0.0 \
    --min-prune-layer 2 --max-prune-units 320 --dtype bf16 >> "$LOG" 2>&1
  python scripts/diagnose_generation_metrics.py \
    --label "${label}" --output-json "$run/pruned_eval.json" \
    --model-path "$run/pruned_model" --triggered-jsonl "$trig" \
    --harmful-no-trigger-jsonl "$SAFE" --benign-jsonl "$BENIGN" \
    --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
  local asr=$(python3 -c "import json; print(json.load(open('$run/pruned_eval.json'))['metrics']['triggered_ASR'])" 2>/dev/null || echo "?")
  log "  pruned ASR=$asr"
}

run_recovery() { local run="$1" label="$2"
  log "[RECOVER+EVAL] $label"
  rm -rf "$run/recovered_model" "$run/recovery_checkpoints"
  python scripts/recover_model.py \
    --model-path "$run/pruned_model" --pruning-plan "$run/pruning_plan.json" \
    --benign-jsonl "$BENIGN" --harmful-no-trigger-jsonl "$SAFE" \
    --run-dir "$run/recovered_model" --lr 1.5e-5 --lambda-clean 1.0 \
    --lambda-align 2.0 --lambda-safe 0.08 --steps 20 --dtype bf16 \
    --trainable-policy all --mask-policy strict --grad-accum-steps 4 \
    --max-length 256 --proxy-epsilon 0.1 --objective-schedule simultaneous \
    --safe-target-mode fixed --debug-save-steps 5,10,15,20 \
    --debug-checkpoint-dir "$run/recovery_checkpoints" >> "$LOG" 2>&1
  python scripts/diagnose_generation_metrics.py \
    --label "${label}_rec" --output-json "$run/recovered_eval.json" \
    --model-path "$run/recovered_model/recovered_model" --triggered-jsonl "$3" \
    --harmful-no-trigger-jsonl "$SAFE" --benign-jsonl "$BENIGN" \
    --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1
  local asr=$(python3 -c "import json; print(json.load(open('$run/recovered_eval.json'))['metrics']['triggered_ASR'])" 2>/dev/null || echo "?")
  log "  recovered ASR=$asr"
  rm -rf "$run/pruned_model" "$run/recovered_model"
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
}

log "=== PHASE 1+2 FINAL RUN ==="

# ── Word norm_rank (retry) ──
if [ ! -f "result/norm_word_rank/recovered_eval.json" ]; then
  run_pruned_eval "result/norm_word_rank" "word_norm_rank" \
    "$(readlink -f ../trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl)" \
    "$WORD_SCORES" "/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_word" \
    "layer_component_rank"
  run_recovery "result/norm_word_rank" "word_norm_rank" \
    "$(readlink -f ../trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl)"
fi

# ── Phrase ──
PHRASE_M="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_phrase"
PHRASE_T="$(readlink -f ../trigger-free-pruning-defense/result/beat_data/harmful_phrase_trigger.jsonl)"

# Phrase raw scoring
if [ ! -f "result/raw_phrase/unit_scores.json" ]; then
  log "=== Phrase raw scoring ==="
  mkdir -p result/raw_phrase
  python scripts/score_and_prune.py \
    --run-dir result/raw_phrase --model-path "$PHRASE_M" \
    --clean-jsonl "$BENIGN" --protect-safe-jsonl "$SAFE" \
    --prompt-template chat --alpha-safe 0.5 --max-score-to-prune 0.0 \
    --min-prune-layer 2 --max-prune-units 320 --score-samples 8 \
    --dtype bf16 >> "$LOG" 2>&1
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  log "[DONE] Phrase raw scoring"
fi

# Phrase norm_z
if [ ! -f "result/norm_phrase_z/recovered_eval.json" ]; then
  run_pruned_eval "result/norm_phrase_z" "phrase_norm_z" "$PHRASE_T" \
    "result/raw_phrase/unit_scores.json" "$PHRASE_M" "layer_component_z"
  run_recovery "result/norm_phrase_z" "phrase_norm_z" "$PHRASE_T"
fi

# Phrase norm_rank
if [ ! -f "result/norm_phrase_rank/recovered_eval.json" ]; then
  run_pruned_eval "result/norm_phrase_rank" "phrase_norm_rank" "$PHRASE_T" \
    "result/raw_phrase/unit_scores.json" "$PHRASE_M" "layer_component_rank"
  run_recovery "result/norm_phrase_rank" "phrase_norm_rank" "$PHRASE_T"
fi

# Phrase raw recovery + eval (wasn't done by scoring)
if [ ! -f "result/raw_phrase/recovered_eval.json" ]; then
  run_recovery "result/raw_phrase" "phrase_raw" "$PHRASE_T"
fi

# ── Long ──
LONG_M="/ssd2/lizhy_workspace/cache/beat_models/Llama-3.1-8B_long"
LONG_T="$(readlink -f ../trigger-free-pruning-defense/result/beat_data/harmful_long_trigger.jsonl)"

if [ ! -f "result/raw_long/unit_scores.json" ]; then
  log "=== Long raw scoring ==="
  rm -rf result/raw_long
  python scripts/score_and_prune.py \
    --run-dir result/raw_long --model-path "$LONG_M" \
    --clean-jsonl "$BENIGN" --protect-safe-jsonl "$SAFE" \
    --prompt-template chat --alpha-safe 0.5 --max-score-to-prune 0.0 \
    --min-prune-layer 2 --max-prune-units 320 --score-samples 8 \
    --dtype bf16 >> "$LOG" 2>&1
  python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  log "[DONE] Long raw scoring"
fi

for norm_type in z rank; do
  run_name="result/norm_long_${norm_type}"
  if [ ! -f "${run_name}/recovered_eval.json" ]; then
    run_pruned_eval "$run_name" "long_norm_${norm_type}" "$LONG_T" \
      "result/raw_long/unit_scores.json" "$LONG_M" "layer_component_${norm_type}"
    run_recovery "$run_name" "long_norm_${norm_type}" "$LONG_T"
  fi
done

# Long raw recovery
if [ ! -f "result/raw_long/recovered_eval.json" ]; then
  run_recovery "result/raw_long" "long_raw" "$LONG_T"
fi

log "=== ALL DONE ==="
