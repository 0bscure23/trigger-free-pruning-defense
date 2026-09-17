#!/usr/bin/env bash
# Recovery-basin seed sweep for Mistral Word C_ls020 replay.

set -uo pipefail

ROOT=${ROOT:-/home/lizhy/plp}
REPO=$ROOT/trigger-free-pruning-defense-round2
RUNNER=$ROOT/TRANSFER/run_mistral_protocol_replay.sh
OUT_ROOT=${OUT_ROOT:-$REPO/result/mistral_word_recovery_seed_sweep}
SEEDS=${SEEDS:-"0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"}
LOG=$OUT_ROOT/seed_sweep.log

mkdir -p "$OUT_ROOT"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG"
}

main() {
  log "Mistral Word recovery seed sweep start"
  log "OUT_ROOT=$OUT_ROOT"
  log "SEEDS=$SEEDS"
  local total current seed tag run_dir
  total=$(wc -w <<<"$SEEDS" | tr -d ' ')
  current=0
  for seed in $SEEDS; do
    current=$((current + 1))
    tag="mistral_word_seed${seed}_may13_chat256_ls020"
    run_dir="$OUT_ROOT/mistral_word/runs/$tag"
    if [ -f "$run_dir/SUCCESS" ]; then
      log "[$current/$total] seed=$seed skip existing SUCCESS"
      continue
    fi
    log "[$current/$total] seed=$seed start tag=$tag"
    RUN=1 \
    OUT_ROOT="$OUT_ROOT" \
    TARGET=mistral_word \
    APPLY_IMPL=old_scores \
    RECOVER_IMPL=may13_060b664 \
    RECOVERY_PROMPT=chat \
    RECOVERY_MAX_LENGTH=256 \
    RECOVERY_SEED="$seed" \
    SHUFFLE_RECOVERY_BATCHES=1 \
    LAMBDA_SAFE=0.20 \
    LAMBDA_ALIGN=2.0 \
    LR=5e-6 \
    STEPS=20 \
    EVAL_PROMPT=chat \
    EVAL_PPL=0 \
    TAG="$tag" \
    "$RUNNER" >> "$LOG" 2>&1
    status=$?
    if [ "$status" -eq 0 ]; then
      log "[$current/$total] seed=$seed done"
    else
      log "[$current/$total] seed=$seed failed status=$status"
    fi
  done
  log "Mistral Word recovery seed sweep finished"
}

main "$@"

