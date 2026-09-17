#!/usr/bin/env bash
set -u

PY=/home/lizhy/.conda/envs/crow_repro/bin/python
SCRIPT=/home/lizhy/plp/TRANSFER/run_pruning_aware_reinforcement_stress_test.py
BEAT=/home/lizhy/plp/TRANSFER/beat_data
ROOT=${BATCH_ROOT:-/home/lizhy/plp/trigger-free-pruning-defense-round2/result/pruning_aware_reinforcement_stress_test/llama_phrase_long_batch_$(date +%Y%m%d_%H%M%S)}

mkdir -p "$ROOT"

log_line() {
  echo "[$(date -Is)] $*" | tee -a "$ROOT/batch_progress.log"
}

run_one() {
  local name="$1"
  local model="$2"
  local plan="$3"
  local trig="$4"
  local la="$5"
  local ls="$6"
  local steps="$7"
  local lr="$8"
  local out="$ROOT/$name"

  mkdir -p "$out"
  log_line "START $name"
  "$PY" "$SCRIPT" --run --out "$out" \
    --model "$model" \
    --defense-plan "$plan" \
    --triggered-jsonl "$trig" \
    --defense-lambda-align "$la" \
    --defense-lambda-safe "$ls" \
    --defense-steps "$steps" \
    --defense-lr "$lr" \
    --gpu-devices 0,1,2,3 \
    --eval-gpu-devices 0,1,2,3 \
    --gpu-max-used-mib 6144 \
    --gpu-poll-seconds 60
  local rc=$?
  log_line "DONE $name rc=$rc"
  return "$rc"
}

rc_total=0
run_one phrase_strong /home/lizhy/plp/Llama-3.1-8B_phrase /home/lizhy/plp/phrase/pruning_plan.json "$BEAT/harmful_phrase_trigger.jsonl" 2.0 0.08 25 1.5e-5 || rc_total=1
run_one phrase_balanced /home/lizhy/plp/Llama-3.1-8B_phrase /home/lizhy/plp/phrase/pruning_plan.json "$BEAT/harmful_phrase_trigger.jsonl" 1.0 0.08 25 1.5e-5 || rc_total=1
run_one long /home/lizhy/plp/Llama-3.1-8B_long /home/lizhy/plp/long/pruning_plan.json "$BEAT/harmful_long_trigger.jsonl" 2.5 0.07 25 1.5e-5 || rc_total=1

log_line "ALL_DONE rc_total=$rc_total"
exit "$rc_total"
