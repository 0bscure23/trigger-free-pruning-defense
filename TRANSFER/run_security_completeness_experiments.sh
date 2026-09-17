#!/bin/bash
# Security-paper completeness experiments for TFPD.
#
# Default behavior is DRY-RUN ONLY. Set RUN=1 after protocol audit/lock.

set -u -o pipefail

P=/home/lizhy/plp
REPO=$P/trigger-free-pruning-defense-round2
T=$P/TRANSFER
BEAT=$T/beat_data
OUT=${OUT:-$REPO/result/security_completeness_experiments}
MODEL=${ANCHOR_MODEL:-$P/Llama-3.1-8B_word}
TRIGGERED=${TRIGGERED_JSONL:-$BEAT/harmful_word_trigger.jsonl}
HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER_JSONL:-$BEAT/harmful_no_trigger.jsonl}
BENIGN=${BENIGN_JSONL:-$BEAT/benign_clean.jsonl}
ROLLING_PPL=${ROLLING_PPL:-$T/rolling_ppl_auto.py}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
REQUIRED_TRANSFORMERS=${REQUIRED_TRANSFORMERS:-5.3.0}
ALLOW_TRANSFORMERS_MISMATCH=${ALLOW_TRANSFORMERS_MISMATCH:-0}

RUN=${RUN:-0}
DRY_RUN=${DRY_RUN:-1}
if [ "$RUN" = "1" ]; then
  DRY_RUN=0
fi

SEEDS=${SEEDS:-13,17,23}
MAIN_SEED=${MAIN_SEED:-13}
MAIN_BUDGET=${MAIN_BUDGET:-1379}
STRUCTURAL_UNITS_TOTAL=${STRUCTURAL_UNITS_TOTAL:-459776}
MAIN_GATE=${MAIN_GATE:-0}
MAIN_STEPS=${MAIN_STEPS:-25}
MAIN_LR=${MAIN_LR:-5e-6}
MAIN_LAMBDA_ALIGN=${MAIN_LAMBDA_ALIGN:-2.0}
MAIN_LAMBDA_SAFE=${MAIN_LAMBDA_SAFE:-0.08}
PROMPT_TEMPLATE=${PROMPT_TEMPLATE:-}
PROTOCOL_LOCK=${PROTOCOL_LOCK:-$REPO/result/security_completeness_protocol/protocol_lock.json}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
EVAL_GPU_DEVICES=${EVAL_GPU_DEVICES:-0,1,2,3}
CONDA_ENV=${CONDA_ENV:-base}
WAIT_FOR_GPUS=${WAIT_FOR_GPUS:-1}
GPU_WAIT_DEVICES=${GPU_WAIT_DEVICES:-$GPU_DEVICES}
GPU_MAX_USED_MIB=${GPU_MAX_USED_MIB:-2048}
GPU_WAIT_POLL_SECONDS=${GPU_WAIT_POLL_SECONDS:-60}
GPU_WAIT_TIMEOUT_SECONDS=${GPU_WAIT_TIMEOUT_SECONDS:-0}
GPU_REQUIRE_NO_COMPUTE_APPS=${GPU_REQUIRE_NO_COMPUTE_APPS:-1}
WAIT_BEFORE_EACH_STAGE=${WAIT_BEFORE_EACH_STAGE:-1}
SKIP_COMPLETED=${SKIP_COMPLETED:-1}
RETRY_FAILED=${RETRY_FAILED:-1}
EARLY_CUDA_CONTEXT=${EARLY_CUDA_CONTEXT:-1}
CUDA_CONTEXT_RUNNER=${CUDA_CONTEXT_RUNNER:-$T/cuda_context_runner.py}
CUDA_CONTEXT_MIB=${CUDA_CONTEXT_MIB:-1}
PROGRESS_POLL_SECONDS=${PROGRESS_POLL_SECONDS:-120}
PROGRESS_BAR_WIDTH=${PROGRESS_BAR_WIDTH:-30}

PLAN_TSV=$OUT/DRY_RUN_PLAN.tsv
SUMMARY_TSV=$OUT/summary_rows.tsv
GPU_WAIT_LOG=$OUT/gpu_wait.log
PROGRESS_LOG=$OUT/progress.log
PROGRESS_STATE=$OUT/progress_state.json
PROGRESS_HISTORY=$OUT/progress_stage_history.tsv

mkdir -p "$OUT"

if [ ! -x "$PYTHON" ]; then
  echo "Configured PYTHON is not executable: $PYTHON" >&2
  exit 2
fi
TRANSFORMERS_VERSION=$("$PYTHON" - <<'PY'
import transformers
print(transformers.__version__)
PY
)
if [ "$TRANSFORMERS_VERSION" != "$REQUIRED_TRANSFORMERS" ] && [ "$ALLOW_TRANSFORMERS_MISMATCH" != "1" ]; then
  echo "Wrong Transformers runtime: got $TRANSFORMERS_VERSION from $PYTHON, expected $REQUIRED_TRANSFORMERS." >&2
  echo "Set PYTHON=/path/to/python or ALLOW_TRANSFORMERS_MISMATCH=1 only for non-paper diagnostics." >&2
  exit 2
fi

if [ -z "$PROMPT_TEMPLATE" ] && [ -f "$PROTOCOL_LOCK" ]; then
  PROMPT_TEMPLATE=$("$PYTHON" - "$PROTOCOL_LOCK" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("prompt_template", ""))
PY
)
fi

if [ -z "$PROMPT_TEMPLATE" ]; then
  echo "PROMPT_TEMPLATE is not set. Run protocol_audit_security_completeness.py --lock-prompt-template <template>, or export PROMPT_TEMPLATE." >&2
  if [ "$DRY_RUN" != "1" ]; then
    exit 2
  fi
  PROMPT_TEMPLATE="PROTOCOL_LOCK_REQUIRED"
fi

if [ "$DRY_RUN" = "1" ]; then
  {
    echo -e "family\ttag\tseed\tbudget\tgate\tsteps\tlr\tlambda_align\tlambda_safe\tcommand"
  } > "$PLAN_TSV"
fi

count_total_runs() {
  local seed_count
  IFS=',' read -r -a seed_count_array <<< "$SEEDS"
  seed_count=${#seed_count_array[@]}
  echo $((seed_count + 24))
}

TOTAL_RUNS=${TOTAL_RUNS:-$(count_total_runs)}
RUN_STARTED_AT=0
COMPLETED_RUNS=0
CURRENT_RUN_INDEX=0
CURRENT_TAG=""
CURRENT_FAMILY=""

format_seconds() {
  local seconds=$1
  if [ -z "$seconds" ] || [ "$seconds" = "unknown" ] || [ "$seconds" -lt 0 ] 2>/dev/null; then
    echo "unknown"
    return 0
  fi
  local h=$((seconds / 3600))
  local m=$(((seconds % 3600) / 60))
  local s=$((seconds % 60))
  printf "%02d:%02d:%02d" "$h" "$m" "$s"
}

progress_bar() {
  local done=$1 total=$2 width=${3:-30}
  local filled percent empty
  if [ "$total" -le 0 ]; then
    total=1
  fi
  percent=$((done * 100 / total))
  filled=$((done * width / total))
  empty=$((width - filled))
  printf "["
  local i
  for ((i = 0; i < filled; i++)); do
    printf "#"
  done
  for ((i = 0; i < empty; i++)); do
    printf "-"
  done
  printf "] %d/%d %d%%" "$done" "$total" "$percent"
}

stage_average_seconds() {
  local stage=$1
  if [ ! -f "$PROGRESS_HISTORY" ]; then
    echo ""
    return 0
  fi
  awk -F'\t' -v stage="$stage" '$1 == stage && $3 == 0 {sum += $2; n += 1} END {if (n > 0) printf "%d", sum / n;}' "$PROGRESS_HISTORY"
}

gpu_usage_snapshot() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu=unavailable"
    return 0
  fi
  local raw gpu used total snapshot
  snapshot=""
  IFS=',' read -r -a gpu_array <<< "$GPU_WAIT_DEVICES"
  for raw in "${gpu_array[@]}"; do
    gpu=${raw//[[:space:]]/}
    [ -z "$gpu" ] && continue
    used=$(gpu_used_mib "$gpu" 2>/dev/null || echo "?")
    total=$(gpu_total_mib "$gpu" 2>/dev/null || echo "?")
    snapshot="${snapshot}gpu${gpu}:${used}/${total}MiB "
  done
  echo "${snapshot% }"
}

write_progress_state() {
  local tag=$1 family=$2 stage=$3 status=$4 stage_elapsed=$5 stage_eta=$6 rc=$7
  "$PYTHON" - "$PROGRESS_STATE" "$tag" "$family" "$stage" "$status" "$stage_elapsed" "$stage_eta" "$rc" \
    "$COMPLETED_RUNS" "$TOTAL_RUNS" "$CURRENT_RUN_INDEX" "$RUN_STARTED_AT" <<'PY'
import json, sys, time
path, tag, family, stage, status, stage_elapsed, stage_eta, rc, completed, total, run_index, started_at = sys.argv[1:]
payload = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tag": tag,
    "family": family,
    "stage": stage,
    "status": status,
    "stage_elapsed_seconds": None if stage_elapsed == "unknown" else int(stage_elapsed),
    "stage_eta_seconds": None if stage_eta == "unknown" else int(stage_eta),
    "return_code": None if rc == "" else int(rc),
    "completed_runs": int(completed),
    "total_runs": int(total),
    "current_run_index": int(run_index),
    "total_elapsed_seconds": max(0, int(time.time()) - int(started_at)) if int(started_at) > 0 else 0,
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY
}

progress_event() {
  local tag=$1 family=$2 stage=$3 status=$4 stage_elapsed=${5:-0} stage_eta=${6:-unknown} rc=${7:-}
  local now total_elapsed bar line stage_elapsed_fmt stage_eta_fmt total_elapsed_fmt gpu
  now=$(date '+%F %T')
  if [ "$RUN_STARTED_AT" -gt 0 ]; then
    total_elapsed=$(($(date +%s) - RUN_STARTED_AT))
  else
    total_elapsed=0
  fi
  bar=$(progress_bar "$COMPLETED_RUNS" "$TOTAL_RUNS" "$PROGRESS_BAR_WIDTH")
  stage_elapsed_fmt=$(format_seconds "$stage_elapsed")
  stage_eta_fmt=$(format_seconds "$stage_eta")
  total_elapsed_fmt=$(format_seconds "$total_elapsed")
  gpu=$(gpu_usage_snapshot)
  line="[$now] $bar run=${CURRENT_RUN_INDEX}/${TOTAL_RUNS} tag=$tag stage=$stage status=$status stage_elapsed=$stage_elapsed_fmt stage_eta=$stage_eta_fmt total_elapsed=$total_elapsed_fmt $gpu"
  echo "$line" | tee -a "$PROGRESS_LOG"
  write_progress_state "$tag" "$family" "$stage" "$status" "$stage_elapsed" "$stage_eta" "$rc"
}

init_progress() {
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi
  RUN_STARTED_AT=$(date +%s)
  COMPLETED_RUNS=0
  CURRENT_RUN_INDEX=0
  : > "$PROGRESS_LOG"
  if [ ! -f "$PROGRESS_HISTORY" ]; then
    echo -e "stage\tseconds\trc\ttag\tfinished_at" > "$PROGRESS_HISTORY"
  fi
  progress_event "queue" "init" "init" "started" 0 unknown ""
}

gpu_used_mib() {
  local gpu=$1
  nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d '[:space:]'
}

gpu_total_mib() {
  local gpu=$1
  nvidia-smi --id="$gpu" --query-gpu=memory.total --format=csv,noheader,nounits | tr -d '[:space:]'
}

gpu_compute_apps() {
  local gpu=$1
  nvidia-smi --id="$gpu" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
}

wait_for_gpus_if_needed() {
  if [ "$DRY_RUN" = "1" ] || [ "$WAIT_FOR_GPUS" != "1" ]; then
    return 0
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; cannot wait for GPU availability." >&2
    return 2
  fi

  local start now elapsed all_ready raw gpu used total snapshot apps app_summary
  start=$(date +%s)
  echo "[$(date '+%F %T')] waiting for GPUs: devices=$GPU_WAIT_DEVICES max_used_mib=$GPU_MAX_USED_MIB require_no_compute_apps=$GPU_REQUIRE_NO_COMPUTE_APPS poll_seconds=$GPU_WAIT_POLL_SECONDS timeout_seconds=$GPU_WAIT_TIMEOUT_SECONDS" | tee -a "$GPU_WAIT_LOG"

  while true; do
    all_ready=1
    snapshot=""
    app_summary=""
    IFS=',' read -r -a gpu_array <<< "$GPU_WAIT_DEVICES"
    for raw in "${gpu_array[@]}"; do
      gpu=${raw//[[:space:]]/}
      [ -z "$gpu" ] && continue
      used=$(gpu_used_mib "$gpu") || return 2
      total=$(gpu_total_mib "$gpu") || return 2
      snapshot="${snapshot} gpu${gpu}:${used}/${total}MiB"
      if [ "$used" -gt "$GPU_MAX_USED_MIB" ]; then
        all_ready=0
      fi
      if [ "$GPU_REQUIRE_NO_COMPUTE_APPS" = "1" ]; then
        apps=$(gpu_compute_apps "$gpu")
        if [ -n "$apps" ]; then
          all_ready=0
          app_summary="${app_summary} gpu${gpu}{$(echo "$apps" | tr '\n' ';')}"
        fi
      fi
    done

    echo "[$(date '+%F %T')] GPU usage:${snapshot}" | tee -a "$GPU_WAIT_LOG"
    if [ -n "$app_summary" ]; then
      echo "[$(date '+%F %T')] active compute apps:${app_summary}" | tee -a "$GPU_WAIT_LOG"
    fi
    if [ "$all_ready" -eq 1 ]; then
      echo "[$(date '+%F %T')] GPU wait condition satisfied." | tee -a "$GPU_WAIT_LOG"
      return 0
    fi

    if [ "$GPU_WAIT_TIMEOUT_SECONDS" -gt 0 ]; then
      now=$(date +%s)
      elapsed=$((now - start))
      if [ "$elapsed" -ge "$GPU_WAIT_TIMEOUT_SECONDS" ]; then
        echo "Timed out after ${elapsed}s waiting for GPUs." | tee -a "$GPU_WAIT_LOG" >&2
        return 3
      fi
    fi
    sleep "$GPU_WAIT_POLL_SECONDS"
  done
}

run_step() {
  local log=$1
  shift
  if [ "$DRY_RUN" = "1" ]; then
    printf '%q ' "$@" >> "$log"
    echo >> "$log"
    return 0
  fi
  "$@" >> "$log" 2>&1
}

cleanup_run_models() {
  local run_dir=$1
  if [ -d "$run_dir/pruned_model" ] || [ -d "$run_dir/recovered_model" ]; then
    rm -rf "$run_dir/pruned_model" "$run_dir/recovered_model"
  fi
}

summary_has_tag() {
  local tag=$1
  [ -f "$SUMMARY_TSV" ] || return 1
  awk -F '\t' -v tag="$tag" 'NR > 1 && $1 == tag { found = 1 } END { exit(found ? 0 : 1) }' "$SUMMARY_TSV"
}

run_completed() {
  local tag=$1 run_dir=$2
  [ -s "$run_dir/asr.json" ] && [ -s "$run_dir/ppl.json" ] && summary_has_tag "$tag"
}

prepare_rerun_dir() {
  local run_dir=$1 tag=$2
  local log=$run_dir/run.log
  local stamp
  [ -d "$run_dir" ] || return 0
  if [ "$RETRY_FAILED" != "1" ]; then
    return 0
  fi
  if [ -f "$log" ] && ! run_completed "$tag" "$run_dir"; then
    stamp=$(date '+%Y%m%d-%H%M%S')
    mv "$log" "$run_dir/run.log.previous.$stamp"
    rm -f "$run_dir/asr.json" "$run_dir/ppl.json" "$run_dir/pruning_plan.json" "$run_dir/recovery_losses.json" "$run_dir/unit_scores.json"
    cleanup_run_models "$run_dir"
  fi
}

stage_gpu_devices() {
  local stage=$1
  case "$stage" in
    score|recovery)
      echo "$GPU_DEVICES"
      ;;
    asr_harmref_bfr|rolling_ppl)
      echo "$EVAL_GPU_DEVICES"
      ;;
    *)
      echo "$GPU_WAIT_DEVICES"
      ;;
  esac
}

wait_for_stage_gpus() {
  local tag=$1 family=$2 stage=$3 log=$4
  local old_devices rc stage_devices
  if [ "$WAIT_BEFORE_EACH_STAGE" != "1" ]; then
    return 0
  fi
  stage_devices=$(stage_gpu_devices "$stage")
  old_devices=$GPU_WAIT_DEVICES
  GPU_WAIT_DEVICES=$stage_devices
  progress_event "$tag" "$family" "$stage" "waiting_for_gpu" 0 unknown ""
  echo "[$tag] waiting for GPUs before $stage: devices=$GPU_WAIT_DEVICES" | tee -a "$log"
  wait_for_gpus_if_needed
  rc=$?
  GPU_WAIT_DEVICES=$old_devices
  return "$rc"
}

run_stage() {
  local tag=$1 family=$2 stage=$3 log=$4
  shift 4
  local stage_start now elapsed avg eta pid rc
  wait_for_stage_gpus "$tag" "$family" "$stage" "$log" || return $?
  stage_start=$(date +%s)
  avg=$(stage_average_seconds "$stage")
  eta=${avg:-unknown}
  progress_event "$tag" "$family" "$stage" "started" 0 "$eta" ""
  "$@" >> "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$PROGRESS_POLL_SECONDS"
    if kill -0 "$pid" 2>/dev/null; then
      now=$(date +%s)
      elapsed=$((now - stage_start))
      avg=$(stage_average_seconds "$stage")
      if [ -n "$avg" ] && [ "$avg" -gt "$elapsed" ] 2>/dev/null; then
        eta=$((avg - elapsed))
      elif [ -n "$avg" ]; then
        eta=0
      else
        eta=unknown
      fi
      progress_event "$tag" "$family" "$stage" "running" "$elapsed" "$eta" ""
    fi
  done
  wait "$pid"
  rc=$?
  now=$(date +%s)
  elapsed=$((now - stage_start))
  echo -e "${stage}\t${elapsed}\t${rc}\t${tag}\t$(date '+%F %T')" >> "$PROGRESS_HISTORY"
  progress_event "$tag" "$family" "$stage" "finished_rc_${rc}" "$elapsed" 0 "$rc"
  return "$rc"
}

write_config() {
  local run_dir=$1 tag=$2 family=$3 seed=$4 budget=$5 gate=$6 steps=$7 lr=$8 la=$9 ls=${10}
  "$PYTHON" - "$run_dir/config.json" "$tag" "$family" "$seed" "$budget" "$gate" "$steps" "$lr" "$la" "$ls" "$MODEL" "$PROMPT_TEMPLATE" "$STRUCTURAL_UNITS_TOTAL" "$PYTHON" "$TRANSFORMERS_VERSION" <<'PY'
import json, sys
path, tag, family, seed, budget, gate, steps, lr, la, ls, model, prompt_template, total, python_executable, transformers_version = sys.argv[1:]
payload = {
    "tag": tag,
    "family": family,
    "seed": int(seed),
    "anchor_model": model,
    "prompt_template": prompt_template,
    "python_executable": python_executable,
    "transformers_version": transformers_version,
    "structural_units_total": int(total),
    "requested_budget_units": int(budget),
    "requested_budget_fraction": int(budget) / int(total),
    "gate": gate,
    "recovery_steps": int(steps),
    "lr": float(lr),
    "lambda_align": float(la),
    "lambda_safe": float(ls),
    "seed_scope": [
        "clean/safety prompt sampling",
        "perturbation/scoring randomness",
        "recovery data order/dropout/optimizer/torch randomness",
    ],
    "evaluation": {
        "greedy": True,
        "deterministic": True,
        "max_new_tokens": 64,
        "asr_protocol": "diagnose_generation_metrics.py BackdoorLLM keyword protocol",
    },
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY
}

append_summary_row() {
  local run_dir=$1 tag=$2 family=$3 seed=$4 budget=$5 gate=$6 steps=$7 lr=$8 la=$9 ls=${10}
  "$PYTHON" - "$SUMMARY_TSV" "$run_dir" "$tag" "$family" "$seed" "$budget" "$gate" "$steps" "$lr" "$la" "$ls" <<'PY'
import json, sys
summary, run_dir, tag, family, seed, budget, gate, steps, lr, la, ls = sys.argv[1:]
from pathlib import Path
rd = Path(run_dir)
asr_path = rd / "asr.json"
ppl_path = rd / "ppl.json"
plan_path = rd / "pruning_plan.json"
asr = json.load(open(asr_path)) if asr_path.exists() else {}
ppl = json.load(open(ppl_path)) if ppl_path.exists() else {}
plan = json.load(open(plan_path)) if plan_path.exists() else {}
m = asr.get("metrics", {})
row = [
    tag, family, seed, budget, gate, steps, lr, la, ls,
    str(plan.get("pruned_total", "")),
    str(plan.get("pruned_heads", "")),
    str(plan.get("pruned_channels", "")),
    str(m.get("triggered_ASR", "")),
    str(m.get("HarmRef", m.get("harmful_no_trigger_refusal", ""))),
    str(m.get("BFR", m.get("benign_clean_false_refusal", ""))),
    str(m.get("empty_output_rate", "")),
    str(m.get("avg_output_tokens", "")),
    str(m.get("median_output_tokens", "")),
    str(ppl.get("ppl", "")),
]
header = [
    "tag","family","seed","requested_budget","gate","steps","lr","lambda_align","lambda_safe",
    "actual_pruned","#Heads","#MLP_channels","ASR","HarmRef","BFR","Empty","avg_output_tokens",
    "median_output_tokens","PPL"
]
write_header = not Path(summary).exists()
with open(summary, "a", encoding="utf-8") as f:
    if write_header:
        f.write("\t".join(header) + "\n")
    f.write("\t".join(row) + "\n")
PY
}

run_pipeline() {
  local family=$1 tag=$2 seed=$3 budget=$4 gate=$5 steps=$6 lr=$7 la=$8 ls=$9
  local rd=$OUT/runs/$tag
  local log=$rd/run.log
  mkdir -p "$rd"

  if [ "$DRY_RUN" = "1" ]; then
    write_config "$rd" "$tag" "$family" "$seed" "$budget" "$gate" "$steps" "$lr" "$la" "$ls"
    echo -e "${family}\t${tag}\t${seed}\t${budget}\t${gate}\t${steps}\t${lr}\t${la}\t${ls}\tfull pipeline" >> "$PLAN_TSV"
    return 0
  fi

  if [ "$SKIP_COMPLETED" = "1" ] && run_completed "$tag" "$rd"; then
    CURRENT_RUN_INDEX=$((COMPLETED_RUNS + 1))
    CURRENT_TAG="$tag"
    CURRENT_FAMILY="$family"
    echo "[$tag] SKIP already completed: asr.json, ppl.json, and summary row exist"
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
    progress_event "$tag" "$family" "run" "skipped_completed" 0 unknown "0"
    return 0
  fi

  prepare_rerun_dir "$rd" "$tag"
  write_config "$rd" "$tag" "$family" "$seed" "$budget" "$gate" "$steps" "$lr" "$la" "$ls"

  CURRENT_RUN_INDEX=$((COMPLETED_RUNS + 1))
  CURRENT_TAG="$tag"
  CURRENT_FAMILY="$family"
  progress_event "$tag" "$family" "run" "started" 0 unknown ""

  echo "[$tag] scoring" | tee "$log"
  local score_rc recover_rc asr_rc ppl_rc
  local kappa="$gate"
  if [ "$gate" = "nogate" ]; then
    kappa=1000000000
  fi

  run_stage "$tag" "$family" "score" "$log" \
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" EARLY_CUDA_CONTEXT="$EARLY_CUDA_CONTEXT" CUDA_CONTEXT_MIB="$CUDA_CONTEXT_MIB" "$PYTHON" "$CUDA_CONTEXT_RUNNER" -- "$REPO/scripts/score_and_prune.py" \
    --run-dir "$rd" \
    --model-path "$MODEL" \
    --clean-jsonl "$BENIGN" \
    --protect-safe-jsonl "$HARMFUL_NO_TRIGGER" \
    --dtype bf16 \
    --prompt-template "$PROMPT_TEMPLATE" \
    --max-length 512 \
    --alpha 1.0 \
    --beta 1.0 \
    --alpha-safe 0.0 \
    --proxy-epsilon 0.1 \
    --score-samples 8 \
    --kappa "$kappa" \
    --max-prune-units "$budget" \
    --seed "$seed"
  score_rc=$?

  if [ "$score_rc" -eq 0 ]; then
    echo "[$tag] recovery" | tee -a "$log"
    run_stage "$tag" "$family" "recovery" "$log" \
      env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" EARLY_CUDA_CONTEXT="$EARLY_CUDA_CONTEXT" CUDA_CONTEXT_MIB="$CUDA_CONTEXT_MIB" "$PYTHON" "$CUDA_CONTEXT_RUNNER" -- "$REPO/scripts/recover_model.py" \
      --run-dir "$rd" \
      --model-path "$MODEL" \
      --pruning-plan "$rd/pruning_plan.json" \
      --benign-jsonl "$BENIGN" \
      --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
      --dtype bf16 \
      --prompt-template "$PROMPT_TEMPLATE" \
      --max-length 512 \
      --trainable-policy all \
      --mask-policy strict \
      --lambda-align "$la" \
      --lambda-safe "$ls" \
      --lambda-clean 1.0 \
      --steps "$steps" \
      --lr "$lr" \
      --grad-accum-steps 4 \
      --seed "$seed"
    recover_rc=$?
  else
    recover_rc=99
  fi

  if [ "$recover_rc" -eq 0 ]; then
    echo "[$tag] ASR/HarmRef/BFR" | tee -a "$log"
    run_stage "$tag" "$family" "asr_harmref_bfr" "$log" \
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" EARLY_CUDA_CONTEXT="$EARLY_CUDA_CONTEXT" CUDA_CONTEXT_MIB="$CUDA_CONTEXT_MIB" "$PYTHON" "$CUDA_CONTEXT_RUNNER" -- "$REPO/scripts/diagnose_generation_metrics.py" \
      --label "$tag" \
      --output-json "$rd/asr.json" \
      --model-path "$rd/recovered_model" \
      --triggered-jsonl "$TRIGGERED" \
      --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
      --benign-jsonl "$BENIGN" \
      --prompt-template "$PROMPT_TEMPLATE" \
      --eval-max-new-tokens 64 \
      --dtype bf16 \
      --seed "$seed"
    asr_rc=$?

    echo "[$tag] rolling PPL" | tee -a "$log"
    run_stage "$tag" "$family" "rolling_ppl" "$log" \
      env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" EARLY_CUDA_CONTEXT="$EARLY_CUDA_CONTEXT" CUDA_CONTEXT_MIB="$CUDA_CONTEXT_MIB" "$PYTHON" "$CUDA_CONTEXT_RUNNER" -- "$ROLLING_PPL" "$tag" "$rd/recovered_model" "$rd/ppl.json"
    ppl_rc=$?
  else
    asr_rc=99
    ppl_rc=99
  fi

  cleanup_run_models "$rd"
  if [ "$asr_rc" -eq 0 ] && [ "$ppl_rc" -eq 0 ]; then
    append_summary_row "$rd" "$tag" "$family" "$seed" "$budget" "$gate" "$steps" "$lr" "$la" "$ls"
  fi
  if [ "$score_rc" -ne 0 ] || [ "$recover_rc" -ne 0 ] || [ "$asr_rc" -ne 0 ] || [ "$ppl_rc" -ne 0 ]; then
    echo "[$tag] FAILED score=$score_rc recover=$recover_rc asr=$asr_rc ppl=$ppl_rc" | tee -a "$log"
    COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
    progress_event "$tag" "$family" "run" "failed" 0 unknown "1"
    return 1
  fi
  echo "[$tag] done" | tee -a "$log"
  COMPLETED_RUNS=$((COMPLETED_RUNS + 1))
  progress_event "$tag" "$family" "run" "completed" 0 unknown "0"
}

plan_stress_test() {
  local plan_dir=$OUT/pruning_aware_reinforcement_stress_test
  mkdir -p "$plan_dir"
  "$PYTHON" - "$plan_dir/PLAN.json" "$MAIN_BUDGET" <<'PY'
import json, sys
path, budget = sys.argv[1:]
payload = {
    "name": "Pruning-Aware Reinforcement Stress Test",
    "status": "planned_not_launched",
    "variants": ["vanilla_poisoned_reinforcement", "pruning_aware_masked_reinforcement"],
    "starting_point": "raw BEAT checkpoint, not clean-base from-scratch poisoning",
    "mask_policy": {
        "preferred": "mask top units from the main 0.3% pruning plan or actual gate-retained units",
        "requested_budget_units": int(budget),
        "fallback": "top-320 lower-bound stress test if full-budget mask is infeasible",
    },
    "claim_scope": "stress test for whether a backdoor can be reinforced to avoid high-confidence pruning masks",
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
PY
}

run_all() {
  IFS=',' read -r -a seed_list <<< "$SEEDS"
  for seed in "${seed_list[@]}"; do
    run_pipeline "full_pipeline_seed" "seed_${seed}_main_b${MAIN_BUDGET}_g0" "$seed" "$MAIN_BUDGET" "$MAIN_GATE" "$MAIN_STEPS" "$MAIN_LR" "$MAIN_LAMBDA_ALIGN" "$MAIN_LAMBDA_SAFE" || true
  done

  for item in "001:460" "003:1379" "005:2299" "010:4598"; do
    label=${item%%:*}
    budget=${item##*:}
    run_pipeline "budget_sweep" "budget_${label}_seed${MAIN_SEED}" "$MAIN_SEED" "$budget" 0 "$MAIN_STEPS" "$MAIN_LR" "$MAIN_LAMBDA_ALIGN" "$MAIN_LAMBDA_SAFE" || true
  done

  for gate in nogate +0.05 0 -0.02 -0.05; do
    clean_gate=${gate//+/p}
    clean_gate=${clean_gate//-/m}
    run_pipeline "threshold_sweep" "gate_${clean_gate}_seed${MAIN_SEED}" "$MAIN_SEED" "$MAIN_BUDGET" "$gate" "$MAIN_STEPS" "$MAIN_LR" "$MAIN_LAMBDA_ALIGN" "$MAIN_LAMBDA_SAFE" || true
  done

  for steps in 10 25 50; do
    run_pipeline "recovery_steps_sensitivity" "steps_${steps}_seed${MAIN_SEED}" "$MAIN_SEED" "$MAIN_BUDGET" 0 "$steps" "$MAIN_LR" "$MAIN_LAMBDA_ALIGN" "$MAIN_LAMBDA_SAFE" || true
  done
  for lr in 3e-6 5e-6 1.5e-5 3e-5; do
    clean_lr=${lr//./p}
    run_pipeline "recovery_lr_sensitivity" "lr_${clean_lr}_seed${MAIN_SEED}" "$MAIN_SEED" "$MAIN_BUDGET" 0 "$MAIN_STEPS" "$lr" "$MAIN_LAMBDA_ALIGN" "$MAIN_LAMBDA_SAFE" || true
  done
  for la in 1.0 1.5 2.0 2.5; do
    clean_la=${la//./p}
    run_pipeline "recovery_lambda_align_sensitivity" "lambda_align_${clean_la}_seed${MAIN_SEED}" "$MAIN_SEED" "$MAIN_BUDGET" 0 "$MAIN_STEPS" "$MAIN_LR" "$la" "$MAIN_LAMBDA_SAFE" || true
  done
  for ls in 0 0.04 0.08 0.12; do
    clean_ls=${ls//./p}
    run_pipeline "recovery_lambda_safe_sensitivity" "lambda_safe_${clean_ls}_seed${MAIN_SEED}" "$MAIN_SEED" "$MAIN_BUDGET" 0 "$MAIN_STEPS" "$MAIN_LR" "$MAIN_LAMBDA_ALIGN" "$ls" || true
  done
  plan_stress_test
}

wait_for_gpus_if_needed
init_progress
run_all
if [ "$DRY_RUN" != "1" ]; then
  progress_event "queue" "all" "queue" "finished" 0 unknown "0"
fi

if [ "$DRY_RUN" = "1" ]; then
  echo "Dry-run plan written to $PLAN_TSV"
  echo "Set RUN=1 to launch experiments. PROMPT_TEMPLATE is read from protocol_lock.json unless explicitly overridden."
else
  echo "Completed requested queue. Summary rows: $SUMMARY_TSV"
fi
