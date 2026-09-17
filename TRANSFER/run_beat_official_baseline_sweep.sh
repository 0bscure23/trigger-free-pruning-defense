#!/bin/bash
set +e
set -uo pipefail

REPO=${REPO:-/home/lizhy/plp/trigger-free-pruning-defense-round2}
T=${T:-/home/lizhy/plp/TRANSFER}
BEAT=${BEAT:-$T/beat_data}
OUT_ROOT=${OUT_ROOT:-$REPO/result/official_baselines_beat_sweep_triggerfree}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-0,1,2,3}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
MAX_START_MEM_MB=${MAX_START_MEM_MB:-2000}
SEARCH_PRESET=${SEARCH_PRESET:-search_small}
METHODS=${METHODS:-beear,sande}
MODELS_TO_RUN=${MODELS_TO_RUN:-mistral_long,mistral_word,llama_long,llama_phrase,llama_word}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}
ENABLE_WATCHDOG=${ENABLE_WATCHDOG:-1}
WATCHDOG_AUTO_EXPAND=${WATCHDOG_AUTO_EXPAND:-0}
START_WATCHDOG_DAEMON=${START_WATCHDOG_DAEMON:-0}
POLL_SECONDS=${POLL_SECONDS:-300}
MASTER_PORT_BASE=${MASTER_PORT_BASE:-12370}
BEEAR_SCENARIO=${BEEAR_SCENARIO:-Model_8}
CLEANUP_MODEL_AFTER_EVAL=${CLEANUP_MODEL_AFTER_EVAL:-1}
LLAMA3_COMPAT_CONFIG=${LLAMA3_COMPAT_CONFIG:-1}
COMPAT_MODEL_ROOT=${COMPAT_MODEL_ROOT:-$OUT_ROOT/_compat_models}

mkdir -p "$OUT_ROOT"

MODEL_SPECS=(
  "mistral_long|/home/lizhy/plp/Mistral-3-7B_long|$BEAT/harmful_long_trigger.jsonl"
  "mistral_phrase|/home/lizhy/plp/Mistral-3-7B_phrase|$BEAT/harmful_phrase_trigger.jsonl"
  "mistral_word|/home/lizhy/plp/Mistral-3-7B_word|$BEAT/harmful_word_trigger.jsonl"
  "llama_long|/home/lizhy/plp/Llama-3.1-8B_long|$BEAT/harmful_long_trigger.jsonl"
  "llama_phrase|/home/lizhy/plp/Llama-3.1-8B_phrase|$BEAT/harmful_phrase_trigger.jsonl"
  "llama_word|/home/lizhy/plp/Llama-3.1-8B_word|$BEAT/harmful_word_trigger.jsonl"
)

case "$SEARCH_PRESET" in
  pilot)
    BEEAR_CONFIGS=(
      "pilot_a9_l7_r1_i1_t8_pa4|9|7|1|1|8|4|3e-7"
    )
    SANDE_CONFIGS=(
      "pilot_len512_s16_t6|512|16|16|1|4|6|1e-3|5e-6"
    )
    ;;
  official_tf)
    BEEAR_CONFIGS=(
      "official_a9_l7_r6_i5_t120_pa100|9|7|6|5|120|100|3e-7"
      "official_a10_l7_r6_i5_t120_pa100|10|7|6|5|120|100|3e-7"
    )
    SANDE_CONFIGS=(
      "official_len1024_s100_t6|1024|100|100|1|4|6|1e-3|5e-6"
    )
    ;;
  search_small)
    BEEAR_CONFIGS=(
      "a9_l7_r3_i3_t60_pa40|9|7|3|3|60|40|3e-7"
      "a10_l7_r3_i3_t60_pa40|10|7|3|3|60|40|3e-7"
      "a9_l5_r3_i3_t60_pa40|9|5|3|3|60|40|3e-7"
      "a9_l9_r3_i3_t60_pa40|9|9|3|3|60|40|3e-7"
    )
    SANDE_CONFIGS=(
      "len1024_s50_t6|1024|50|50|1|4|6|1e-3|5e-6"
      "len1024_s100_t6|1024|100|100|1|4|6|1e-3|5e-6"
      "len512_s100_t6|512|100|100|1|4|6|1e-3|5e-6"
      "len1024_s100_t8|1024|100|100|1|4|8|1e-3|5e-6"
    )
    ;;
  *)
    echo "Unknown SEARCH_PRESET=$SEARCH_PRESET (use pilot, search_small, official_tf)" >&2
    exit 2
    ;;
esac

contains_csv() {
  local needle="$1"
  local haystack="$2"
  [[ ",$haystack," == *",$needle,"* ]]
}

should_run_model() {
  local model_id="$1"
  [ "$MODELS_TO_RUN" = "all" ] || contains_csv "$model_id" "$MODELS_TO_RUN"
}

record_failure() {
  local run_dir="$1"
  local method="$2"
  local model_id="$3"
  local tag="$4"
  local cfg_name="$5"
  local exit_code="$6"
  local model_path="${7:-}"
  mkdir -p "$run_dir"
  python3 - "$run_dir" "$method" "$model_id" "$tag" "$cfg_name" "$exit_code" "$model_path" <<'PY'
import json
import sys
from pathlib import Path
run_dir, method, model_id, tag, cfg_name, exit_code, model_path = sys.argv[1:]
payload = {
    "method": method,
    "model_id": model_id,
    "tag": tag,
    "config_name": cfg_name,
    "exit_code": int(exit_code),
    "reason": "runner exited non-zero; inspect method train/eval logs in this run directory and parent output directory",
}
run = Path(run_dir)
run.joinpath("failure.json").write_text(json.dumps(payload, indent=2) + "\n")
config = {
    "method": method,
    "tag": tag,
    "model_path": model_path,
    "config_name": cfg_name,
    "trigger_free_selection": True,
    "params": {},
}
config_path = run / "sweep_config.json"
if not config_path.exists():
    config_path.write_text(json.dumps(config, indent=2) + "\n")
PY
}

prepare_runtime_model_path() {
  local model_id="$1"
  local model_path="$2"
  local runtime_model_path="$model_path"
  if [ "$LLAMA3_COMPAT_CONFIG" = "1" ] && [[ "$model_id" == llama_* ]]; then
    runtime_model_path="$COMPAT_MODEL_ROOT/$model_id"
    if python3 "$T/prepare_llama3_compat_model.py" \
      --source "$model_path" \
      --dest "$runtime_model_path" \
      > "$OUT_ROOT/prepare_${model_id}_compat.log" 2>&1; then
      echo "[compat] $model_id using short-context config path: $runtime_model_path" >&2
    else
      cat "$OUT_ROOT/prepare_${model_id}_compat.log" >&2 || true
      echo "[skip-model] $model_id failed to prepare compatibility model path" >&2
      return 1
    fi
  fi
  printf '%s\n' "$runtime_model_path"
}

run_watchdog() {
  local model_id="$1"
  if [ "$ENABLE_WATCHDOG" != "1" ]; then
    return 0
  fi
  local auto=()
  if [ "$WATCHDOG_AUTO_EXPAND" = "1" ]; then
    auto=(--auto-expand)
  fi
  python3 "$T/beat_official_baseline_watchdog.py" \
    --root "$OUT_ROOT" \
    --model-id "$model_id" \
    --queue-file "$OUT_ROOT/expansion_queue.jsonl" \
    "${auto[@]}" \
    > "$OUT_ROOT/watchdog_${model_id}.log" 2>&1 || true
}

maybe_start_watchdog_daemon() {
  if [ "$START_WATCHDOG_DAEMON" != "1" ]; then
    return 0
  fi
  if [ -f "$OUT_ROOT/watchdog_daemon.pid" ] && kill -0 "$(cat "$OUT_ROOT/watchdog_daemon.pid")" 2>/dev/null; then
    echo "Watchdog daemon already running: PID $(cat "$OUT_ROOT/watchdog_daemon.pid")"
    return 0
  fi
  nohup python3 "$T/beat_official_baseline_watchdog.py" \
    --root "$OUT_ROOT" \
    --queue-file "$OUT_ROOT/expansion_queue.jsonl" \
    --auto-expand \
    --daemon \
    --poll-seconds "$POLL_SECONDS" \
    > "$OUT_ROOT/watchdog_daemon.log" 2>&1 &
  echo "$!" > "$OUT_ROOT/watchdog_daemon.pid"
  echo "Started watchdog daemon: PID $!"
}

run_beear_config() {
  local model_id="$1"
  local model_path="$2"
  local triggered_jsonl="$3"
  local cfg="$4"
  IFS='|' read -r cfg_name anchor token rounds inner_epochs inner_threshold pa_threshold outer_lr <<< "$cfg"
  local out="$OUT_ROOT/$model_id/beear"
  local tag="${model_id}_beear_${cfg_name}"
  local run_dir="$out/$tag"
  mkdir -p "$out" "$run_dir"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] BEEAR $model_id $cfg_name model=$model_path triggered_eval=$triggered_jsonl"
    return 0
  fi
  if [ "$FORCE" != "1" ] && [ -f "$out/asr_${tag}.json" ] && [ -f "$out/ppl_${tag}.json" ]; then
    echo "[skip] BEEAR $model_id $cfg_name already has ASR/PPL"
    return 0
  fi
  echo "[run] BEEAR $model_id $cfg_name"
  if MODEL="$model_path" OUT="$out" TAG="$tag" VISIBLE_DEVICES="$VISIBLE_DEVICES" WAIT_FOR_GPUS="$WAIT_FOR_GPUS" MAX_START_MEM_MB="$MAX_START_MEM_MB" \
    CLEANUP_MODEL_AFTER_EVAL="$CLEANUP_MODEL_AFTER_EVAL" \
    DEVICE_MAP=manual4 ROUNDS="$rounds" INNER_EPOCHS="$inner_epochs" INNER_BATCH_SIZE=6 INNER_THRESHOLD="$inner_threshold" PA_THRESHOLD="$pa_threshold" \
    OUTER_LR="$outer_lr" ANCHOR_LAYER="$anchor" TOKEN_LENGTH="$token" ALPHA_FAR_FROM_SAFETY=0.05 \
    BEEAR_SCENARIO="$BEEAR_SCENARIO" \
    TRIGGERED_JSONL="$triggered_jsonl" HARMFUL_NO_TRIGGER_JSONL="$BEAT/harmful_no_trigger.jsonl" BENIGN_JSONL="$BEAT/benign_clean.jsonl" \
    bash "$T/run_official_beear_mistral_long.sh"; then
    echo "[done] BEEAR $model_id $cfg_name"
  else
    local code=$?
    echo "[fail] BEEAR $model_id $cfg_name exit=$code"
    record_failure "$run_dir" "BEEAR" "$model_id" "$tag" "$cfg_name" "$code" "$model_path"
  fi
}

run_sande_config() {
  local model_id="$1"
  local model_path="$2"
  local triggered_jsonl="$3"
  local cfg="$4"
  local port="$5"
  IFS='|' read -r cfg_name max_len step1_samples step2_samples micro_batch train_batch trigger_num step1_lr step2_lr <<< "$cfg"
  local out="$OUT_ROOT/$model_id/sande"
  local tag="${model_id}_sande_${cfg_name}"
  local run_dir="$out/$tag"
  mkdir -p "$out" "$run_dir"
  if [ "$DRY_RUN" = "1" ]; then
    echo "[dry-run] SANDE $model_id $cfg_name model=$model_path triggered_eval=$triggered_jsonl"
    return 0
  fi
  if [ "$FORCE" != "1" ] && [ -f "$out/asr_${tag}.json" ] && [ -f "$out/ppl_${tag}.json" ]; then
    echo "[skip] SANDE $model_id $cfg_name already has ASR/PPL"
    return 0
  fi
  echo "[run] SANDE $model_id $cfg_name"
  if MODEL="$model_path" OUT="$out" TAG="$tag" VISIBLE_DEVICES="$VISIBLE_DEVICES" WAIT_FOR_GPUS="$WAIT_FOR_GPUS" MAX_START_MEM_MB="$MAX_START_MEM_MB" \
    CLEANUP_MODEL_AFTER_EVAL="$CLEANUP_MODEL_AFTER_EVAL" \
    MAX_LEN="$max_len" STEP1_SAMPLES="$step1_samples" STEP2_SAMPLES="$step2_samples" MICRO_BATCH="$micro_batch" TRAIN_BATCH="$train_batch" \
    TRIGGER_NUM="$trigger_num" STEP1_LR="$step1_lr" STEP2_LR="$step2_lr" MASTER_PORT="$port" \
    TRIGGERED_JSONL="$triggered_jsonl" HARMFUL_NO_TRIGGER_JSONL="$BEAT/harmful_no_trigger.jsonl" BENIGN_JSONL="$BEAT/benign_clean.jsonl" \
    bash "$T/run_official_sande_mistral_long.sh"; then
    echo "[done] SANDE $model_id $cfg_name"
  else
    local code=$?
    echo "[fail] SANDE $model_id $cfg_name exit=$code"
    record_failure "$run_dir" "SANDE" "$model_id" "$tag" "$cfg_name" "$code" "$model_path"
  fi
}

maybe_start_watchdog_daemon

echo "########## BEAT official baseline trigger-free sweep start $(date) ##########"
echo "OUT_ROOT=$OUT_ROOT"
echo "SEARCH_PRESET=$SEARCH_PRESET METHODS=$METHODS MODELS_TO_RUN=$MODELS_TO_RUN DRY_RUN=$DRY_RUN"
echo "CLEANUP_MODEL_AFTER_EVAL=$CLEANUP_MODEL_AFTER_EVAL LLAMA3_COMPAT_CONFIG=$LLAMA3_COMPAT_CONFIG COMPAT_MODEL_ROOT=$COMPAT_MODEL_ROOT"
echo "Trigger-free selection: watchdog ranking does not use triggered ASR."

model_index=0
for spec in "${MODEL_SPECS[@]}"; do
  IFS='|' read -r model_id model_path triggered_jsonl <<< "$spec"
  if ! should_run_model "$model_id"; then
    echo "[skip-model] $model_id not selected"
    continue
  fi
  if [ ! -d "$model_path" ]; then
    echo "[skip-model] $model_id missing path $model_path"
    continue
  fi
  if [ ! -f "$triggered_jsonl" ]; then
    echo "[skip-model] $model_id missing triggered eval split $triggered_jsonl"
    continue
  fi
  runtime_model_path="$(prepare_runtime_model_path "$model_id" "$model_path")" || continue
  echo "===== model $model_id: $runtime_model_path ====="
  if [ "$runtime_model_path" != "$model_path" ]; then
    echo "source model: $model_path"
  fi
  if contains_csv "beear" "$METHODS"; then
    for cfg in "${BEEAR_CONFIGS[@]}"; do
      run_beear_config "$model_id" "$runtime_model_path" "$triggered_jsonl" "$cfg"
      run_watchdog "$model_id"
    done
  fi
  if contains_csv "sande" "$METHODS"; then
    cfg_index=0
    for cfg in "${SANDE_CONFIGS[@]}"; do
      run_sande_config "$model_id" "$runtime_model_path" "$triggered_jsonl" "$cfg" "$((MASTER_PORT_BASE + model_index * 20 + cfg_index))"
      run_watchdog "$model_id"
      cfg_index=$((cfg_index + 1))
    done
  fi
  run_watchdog "$model_id"
  model_index=$((model_index + 1))
done

python3 "$T/beat_official_baseline_watchdog.py" \
  --root "$OUT_ROOT" \
  --queue-file "$OUT_ROOT/expansion_queue.jsonl" \
  > "$OUT_ROOT/watchdog_all.log" 2>&1 || true

echo "########## BEAT official baseline trigger-free sweep done $(date) ##########"
echo "Results root: $OUT_ROOT"
echo "Watchdog reports: $OUT_ROOT/*/WATCHDOG_REPORT.md"
