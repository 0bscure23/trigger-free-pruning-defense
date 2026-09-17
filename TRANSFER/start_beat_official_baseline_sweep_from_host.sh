#!/bin/bash
set -euo pipefail

OUT_ROOT=${OUT_ROOT:-/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_baselines_beat_sweep_triggerfree}
LOG="$OUT_ROOT/sweep_nohup.log"
PID_FILE="$OUT_ROOT/sweep.pid"

mkdir -p "$OUT_ROOT"

if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "BEAT official baseline sweep is already running: PID $(cat "$PID_FILE")"
  echo "Log: $LOG"
  exit 0
fi

setsid bash -lc '
set +e
export START_WATCHDOG_DAEMON=1
export WATCHDOG_AUTO_EXPAND=1
export CLEANUP_MODEL_AFTER_EVAL=1
export LLAMA3_COMPAT_CONFIG=1
export SEARCH_PRESET=search_small
export MODELS_TO_RUN=mistral_long,mistral_word,llama_long,llama_phrase,llama_word
export METHODS=beear,sande
source /home/lizhy/plp/TRANSFER/run_beat_official_baseline_sweep.sh
' > "$LOG" 2>&1 < /dev/null &

echo "$!" > "$PID_FILE"
echo "Started BEAT official baseline sweep: PID $(cat "$PID_FILE")"
echo "Log: $LOG"
echo "Results: $OUT_ROOT"
