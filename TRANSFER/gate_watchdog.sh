#!/usr/bin/env bash
# Shell-level watchdog for the gate experiments. Independent of any Claude session.
# - restarts the Phrase runner if it dies before SUMMARY.tsv exists (max 3 restarts)
# - when Phrase is done and the Word download is complete with all hashes OK, launches the Word runner
# - exits when both anchors have SUMMARY.tsv or when a restart limit is hit
set -u
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
PY=/home/lizhy/.conda/envs/crow_repro/bin/python
LOG=$REPO/result/watchdog.log
DL=/home/lizhy/plp/model_download.log
cd "$REPO"
log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
running() { pgrep -f "[r]un_gate_experiment.py --anchor $1" >/dev/null; }
launch() {
  local a=$1
  mkdir -p "result/gate_$a"
  nohup "$PY" TRANSFER/run_gate_experiment.py --anchor "$a" >> "result/gate_$a/runner.log" 2>&1 &
  echo $! > "result/gate_$a/PID"
  log "launched $a pid $!"
}
restarts_phrase=0; restarts_word=0
log "watchdog start"
while true; do
  # ---- phrase ----
  if [ ! -f result/gate_llama_phrase/SUMMARY.tsv ]; then
    if ! running llama_phrase; then
      if [ "$restarts_phrase" -ge 3 ]; then log "phrase: restart limit reached, giving up"; break; fi
      restarts_phrase=$((restarts_phrase+1)); log "phrase runner not running (restart $restarts_phrase)"; launch llama_phrase
    fi
  else
    # ---- word (only after phrase finished, GPUs are shared) ----
    if [ ! -f result/gate_llama_word/SUMMARY.tsv ]; then
      if grep -q ALL_DONE "$DL" && ! grep -q "MISMATCH" "$DL" && [ "$(grep -c '^OK Llama-3.1-8B_word/' "$DL")" -ge 11 ]; then
        if ! running llama_word; then
          if [ "$restarts_word" -ge 3 ]; then log "word: restart limit reached, giving up"; break; fi
          restarts_word=$((restarts_word+1)); log "word runner not running (start/restart $restarts_word)"; launch llama_word
        fi
      else
        log "word: download not verified yet (waiting)"
      fi
    else
      log "both anchors done"; break
    fi
  fi
  sleep 300
done
log "watchdog exit"
