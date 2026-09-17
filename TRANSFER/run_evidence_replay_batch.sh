#!/usr/bin/env bash
# Sequential evidence replay batch for targets with local raw models.

set -uo pipefail

TARGETS=${TARGETS:-llama_phrase,llama_long,mistral_word,mistral_long}
REPEATS=${REPEATS:-2}
RUN=${RUN:-0}
OUT_ROOT=${OUT_ROOT:-/home/lizhy/plp/trigger-free-pruning-defense-round2/result/evidence_target_replays}
LOG=$OUT_ROOT/batch_nohup_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$OUT_ROOT"

echo "[$(date '+%F %T')] batch start RUN=$RUN REPEATS=$REPEATS TARGETS=$TARGETS" | tee -a "$LOG"

IFS=',' read -r -a arr <<< "$TARGETS"
for target in "${arr[@]}"; do
  target=${target// /}
  [ -z "$target" ] && continue
  echo "[$(date '+%F %T')] target start $target" | tee -a "$LOG"
  TARGET="$target" REPEATS="$REPEATS" RUN="$RUN" OUT_ROOT="$OUT_ROOT" \
    /home/lizhy/plp/TRANSFER/run_evidence_target_replay.sh >> "$LOG" 2>&1
  rc=$?
  echo "[$(date '+%F %T')] target done $target rc=$rc" | tee -a "$LOG"
done

echo "[$(date '+%F %T')] batch done" | tee -a "$LOG"
