#!/usr/bin/env bash
set -uo pipefail
BASE="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2"
cd "$BASE"
LOG="/tmp/phase3_$(date +%Y%m%d_%H%M%S).log"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }
BENIGN="$BASE/../trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl"
SAFE="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl"
LONG_T="$BASE/../trigger-free-pruning-defense/result/beat_data/harmful_long_trigger.jsonl"

log "=== Phase 3: Proxy-Safe Recovery (Long) ==="

for lps in 0.05 0.1 0.2; do
  for eps in 0.03 0.1; do
    label="lps${lps}_eps${eps}"
    run="$BASE/result/proxy_safe_long/${label}"
    eval_json="${run}/recovered_eval.json"
    [ -f "$eval_json" ] && { log "[SKIP] $label"; continue; }

    log "=== $label ==="
    rm -rf "$run"; mkdir -p "$run"

    python scripts/recover_model.py \
      --model-path "$BASE/result/raw_long/pruned_model" \
      --pruning-plan "$BASE/result/raw_long/pruning_plan.json" \
      --benign-jsonl "$BENIGN" --harmful-no-trigger-jsonl "$SAFE" \
      --run-dir "${run}/recovered_model" \
      --lr 1.5e-5 --lambda-clean 1.0 --lambda-align 2.0 --lambda-safe 0.08 \
      --lambda-proxy-safe "$lps" --proxy-safe-epsilon "$eps" \
      --steps 20 --dtype bf16 --trainable-policy all --mask-policy strict \
      --grad-accum-steps 4 --max-length 256 --proxy-epsilon 0.1 \
      --objective-schedule simultaneous --safe-target-mode fixed >> "$LOG" 2>&1
    log "  recover done"

    python scripts/diagnose_generation_metrics.py \
      --label "ps_${label}" --output-json "$eval_json" \
      --model-path "${run}/recovered_model/recovered_model" \
      --triggered-jsonl "$LONG_T" --harmful-no-trigger-jsonl "$SAFE" \
      --benign-jsonl "$BENIGN" --eval-max-new-tokens 64 --dtype bf16 >> "$LOG" 2>&1

    asr=$(python3 -c "import json; d=json.load(open('$eval_json')); print(f\"{d['metrics']['triggered_ASR']:.3f}\")" 2>/dev/null || echo "?")
    log "  [DONE] ASR=$asr"

    rm -rf "${run}/recovered_model"
    python3 -c "import torch; torch.cuda.empty_cache()" >> "$LOG" 2>&1
  done
done
log "=== ALL DONE ==="
