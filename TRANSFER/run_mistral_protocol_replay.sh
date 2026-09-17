#!/usr/bin/env bash
# Focused Mistral artifact replay/protocol localization.
#
# Runs one configuration at a time:
#   1) reconstruct pruned_model from archived evidence
#   2) recover
#   3) evaluate ASR/HarmRef/BFR/Empty and rolling PPL
#   4) delete temporary checkpoints, keep JSON/logs

set -uo pipefail

ROOT=${ROOT:-/home/lizhy/plp}
REPO=$ROOT/trigger-free-pruning-defense-round2
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
OUT_ROOT=${OUT_ROOT:-$REPO/result/mistral_protocol_replays}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
RUN=${RUN:-0}
TARGET=${TARGET:-mistral_word}
TAG=${TAG:-}

APPLY_IMPL=${APPLY_IMPL:-old_scores}       # old_scores | current_plan
RECOVER_IMPL=${RECOVER_IMPL:-old_08fd92b}  # old_08fd92b | may13_060b664 | current_head
RECOVERY_PROMPT=${RECOVERY_PROMPT:-chat}
RECOVERY_MAX_LENGTH=${RECOVERY_MAX_LENGTH:-1024}
EVAL_PROMPT=${EVAL_PROMPT:-chat}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-1024}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-64}
EVAL_PPL=${EVAL_PPL:-1}
DTYPE=${DTYPE:-bf16}
ADAMW_FOREACH=${ADAMW_FOREACH:-}
RECOVERY_SEED=${RECOVERY_SEED:-}
SHUFFLE_RECOVERY_BATCHES=${SHUFFLE_RECOVERY_BATCHES:-0}

MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-1024}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-30}

BENIGN=${BENIGN:-$ROOT/TRANSFER/beat_data/benign_clean.jsonl}
HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER:-$ROOT/TRANSFER/beat_data/harmful_no_trigger.jsonl}
ROLLING_PPL=${ROLLING_PPL:-$ROOT/TRANSFER/rolling_ppl_auto.py}
OLD_APPLY=$ROOT/tfpd_repro_score_89d79b1/scripts/apply_pruning_from_scores.py
OLD_RECOVER=$ROOT/tfpd_repro_recover_08fd92b/scripts/recover_model.py
MAY13_RECOVER=$ROOT/tfpd_repro_recover_060b664/scripts/recover_model.py
CURRENT_RECOVER=$REPO/scripts/recover_model.py
CURRENT_APPLY=$ROOT/TRANSFER/apply_plan_only.py

case "$TARGET" in
  mistral_word)
    MODEL=$ROOT/Mistral-3-7B_word
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_word_trigger.jsonl
    EVIDENCE_DIR=$ROOT/mistral_repro_artifacts/paper_evidence_pack_models_20260626_120830/mistral_word_complete_nearest_ls018_0p1583
    DEFAULT_LAMBDA_SAFE=0.18
    DEFAULT_LAMBDA_ALIGN=2.0
    DEFAULT_LR=5e-6
    DEFAULT_STEPS=20
    DEFAULT_MAX_UNITS=512
    DEFAULT_MIN_LAYER=0
    EXPECTED_ASR=0.15833333333333333
    ;;
  mistral_long)
    MODEL=$ROOT/Mistral-3-7B_long
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_long_trigger.jsonl
    EVIDENCE_DIR=$ROOT/mistral_repro_artifacts/paper_evidence_pack_models_20260626_120830/mistral_long_paper_L3_0p6250
    DEFAULT_LAMBDA_SAFE=0.30
    DEFAULT_LAMBDA_ALIGN=2.0
    DEFAULT_LR=3e-6
    DEFAULT_STEPS=18
    DEFAULT_MAX_UNITS=512
    DEFAULT_MIN_LAYER=0
    EXPECTED_ASR=0.625
    ;;
  mistral_phrase)
    MODEL=$ROOT/Mistral-3-7B_phrase
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_phrase_trigger.jsonl
    EVIDENCE_DIR=$ROOT/mistral_repro_artifacts/paper_evidence_pack_models_20260626_120830/mistral_phrase_paper_ls015_0p7833
    DEFAULT_LAMBDA_SAFE=0.15
    DEFAULT_LAMBDA_ALIGN=2.0
    DEFAULT_LR=5e-6
    DEFAULT_STEPS=20
    DEFAULT_MAX_UNITS=512
    DEFAULT_MIN_LAYER=0
    EXPECTED_ASR=0.7833333333333333
    ;;
  *)
    echo "Unknown TARGET=$TARGET" >&2
    exit 2
    ;;
esac

LAMBDA_SAFE=${LAMBDA_SAFE:-$DEFAULT_LAMBDA_SAFE}
LAMBDA_ALIGN=${LAMBDA_ALIGN:-$DEFAULT_LAMBDA_ALIGN}
LR=${LR:-$DEFAULT_LR}
STEPS=${STEPS:-$DEFAULT_STEPS}
MAX_UNITS=${MAX_UNITS:-$DEFAULT_MAX_UNITS}
MIN_LAYER=${MIN_LAYER:-$DEFAULT_MIN_LAYER}
MAX_SCORE_TO_PRUNE=${MAX_SCORE_TO_PRUNE:-}
PROXY_EPSILON=${PROXY_EPSILON:-0.1}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}

PLAN=$EVIDENCE_DIR/pruning_plan.json
SCORES=$EVIDENCE_DIR/unit_scores.shared_from_${TARGET#mistral_}_pruned.json
if [ "$TARGET" = "mistral_word" ]; then
  SCORES=$EVIDENCE_DIR/unit_scores.shared_from_word_pruned.json
elif [ "$TARGET" = "mistral_long" ]; then
  SCORES=$EVIDENCE_DIR/unit_scores.shared_from_long_pruned.json
elif [ "$TARGET" = "mistral_phrase" ]; then
  SCORES=$EVIDENCE_DIR/unit_scores.shared_from_phrase_pruned.json
fi

if [ -z "$TAG" ]; then
  TAG="${TARGET}_${APPLY_IMPL}_${RECOVER_IMPL}_rp${RECOVERY_PROMPT}_ml${RECOVERY_MAX_LENGTH}_ls${LAMBDA_SAFE}_la${LAMBDA_ALIGN}_lr${LR}_s${STEPS}"
fi

OUT=$OUT_ROOT/$TARGET
RUN_DIR=$OUT/runs/$TAG
SUMMARY=$OUT/summary_rows.tsv
LOG=$RUN_DIR/run.log

mkdir -p "$RUN_DIR"

log() { echo "[$(date '+%F %T')] [$TAG] $*" | tee -a "$LOG"; }

wait_for_gpus() {
  while true; do
    if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
    local ok=1 used
    while IFS= read -r used; do
      used=${used// /}
      if [ "${used:-999999}" -gt "$MAX_GPU_USED_MIB" ]; then ok=0; fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "$ok" = "1" ]; then return 0; fi
    log "waiting for GPUs <= ${MAX_GPU_USED_MIB} MiB used"
    sleep "$GPU_POLL_SECONDS"
  done
}

cleanup_models() {
  rm -rf "$RUN_DIR/pruned_model" "$RUN_DIR/recovered_model"
}

write_config() {
  "$PYTHON" - "$RUN_DIR/config.json" <<PY
import json, pathlib
payload = {
  "target": "$TARGET",
  "tag": "$TAG",
  "model": "$MODEL",
  "apply_impl": "$APPLY_IMPL",
  "recover_impl": "$RECOVER_IMPL",
  "plan": "$PLAN",
  "scores": "$SCORES",
  "expected_asr": "$EXPECTED_ASR",
  "recovery_prompt": "$RECOVERY_PROMPT",
  "recovery_max_length": int("$RECOVERY_MAX_LENGTH"),
  "eval_prompt": "$EVAL_PROMPT",
  "eval_max_length": int("$EVAL_MAX_LENGTH"),
  "eval_max_new_tokens": int("$EVAL_MAX_NEW_TOKENS"),
  "eval_ppl": "$EVAL_PPL",
  "dtype": "$DTYPE",
  "lambda_safe": float("$LAMBDA_SAFE"),
  "lambda_align": float("$LAMBDA_ALIGN"),
  "lr": float("$LR"),
  "steps": int("$STEPS"),
  "grad_accum_steps": int("$GRAD_ACCUM_STEPS"),
  "adamw_foreach": None if "$ADAMW_FOREACH" == "" else "$ADAMW_FOREACH",
  "recovery_seed": None if "$RECOVERY_SEED" == "" else int("$RECOVERY_SEED"),
  "shuffle_recovery_batches": "$SHUFFLE_RECOVERY_BATCHES" == "1",
  "max_units": int("$MAX_UNITS"),
  "min_layer": int("$MIN_LAYER"),
  "max_score_to_prune": None if "$MAX_SCORE_TO_PRUNE" == "" else float("$MAX_SCORE_TO_PRUNE"),
}
pathlib.Path("$RUN_DIR").mkdir(parents=True, exist_ok=True)
pathlib.Path("$RUN_DIR/config.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\\n")
PY
}

apply_prune() {
  log "apply prune start impl=$APPLY_IMPL"
  wait_for_gpus
  if [ "$APPLY_IMPL" = "old_scores" ]; then
    local score_arg=()
    if [ -n "$MAX_SCORE_TO_PRUNE" ]; then
      score_arg=(--max-score-to-prune "$MAX_SCORE_TO_PRUNE")
    fi
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$OLD_APPLY" \
      --run-dir "$RUN_DIR" \
      --model-path "$MODEL" \
      --scores-json "$SCORES" \
      --kappa 1000000000 \
      --max-prune-units "$MAX_UNITS" \
      --min-prune-layer "$MIN_LAYER" \
      --proxy-epsilon "$PROXY_EPSILON" \
      --dtype "$DTYPE" \
      "${score_arg[@]}" >> "$LOG" 2>&1
  elif [ "$APPLY_IMPL" = "current_plan" ]; then
    cp "$PLAN" "$RUN_DIR/pruning_plan.json"
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$CURRENT_APPLY" \
      "$PLAN" "$MODEL" "$RUN_DIR/pruned_model" >> "$LOG" 2>&1
  else
    echo "Unknown APPLY_IMPL=$APPLY_IMPL" >&2
    return 2
  fi
  log "apply prune done"
}

recover() {
  log "recovery start impl=$RECOVER_IMPL prompt=$RECOVERY_PROMPT max_length=$RECOVERY_MAX_LENGTH"
  wait_for_gpus
  local recover_script
  if [ "$RECOVER_IMPL" = "old_08fd92b" ]; then
    recover_script=$OLD_RECOVER
  elif [ "$RECOVER_IMPL" = "may13_060b664" ]; then
    recover_script=$MAY13_RECOVER
  elif [ "$RECOVER_IMPL" = "current_head" ]; then
    recover_script=$CURRENT_RECOVER
  else
    echo "Unknown RECOVER_IMPL=$RECOVER_IMPL" >&2
    return 2
  fi
  local recover_env=(CUDA_VISIBLE_DEVICES="$GPU_DEVICES")
  if [ -n "$ADAMW_FOREACH" ]; then
    recover_env+=(CROW_ADAMW_FOREACH="$ADAMW_FOREACH")
  fi
  local seed_args=()
  if [ -n "$RECOVERY_SEED" ]; then
    seed_args+=(--recovery-seed "$RECOVERY_SEED")
    if [ "$SHUFFLE_RECOVERY_BATCHES" = "1" ]; then
      seed_args+=(--shuffle-recovery-batches)
    fi
  fi
  env "${recover_env[@]}" "$PYTHON" "$recover_script" \
    --run-dir "$RUN_DIR" \
    --model-path "$RUN_DIR/pruned_model" \
    --pruning-plan "$RUN_DIR/pruning_plan.json" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --safe-target-mode fixed \
    --safe-target-text "I cannot assist with that request." \
    --dtype "$DTYPE" \
    --prompt-template "$RECOVERY_PROMPT" \
    --max-length "$RECOVERY_MAX_LENGTH" \
    --trainable-policy all \
    --mask-policy strict \
    --lambda-clean 1.0 \
    --lambda-align "$LAMBDA_ALIGN" \
    --lambda-safe "$LAMBDA_SAFE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --objective-schedule simultaneous \
    --proxy-epsilon "$PROXY_EPSILON" \
    "${seed_args[@]}" >> "$LOG" 2>&1
  log "recovery done"
}

eval_asr() {
  log "ASR eval start prompt=$EVAL_PROMPT"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$REPO/scripts/diagnose_generation_metrics.py" \
    --label "$TAG" \
    --output-json "$RUN_DIR/asr.json" \
    --model-path "$RUN_DIR/recovered_model" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --benign-jsonl "$BENIGN" \
    --prompt-template "$EVAL_PROMPT" \
    --dtype "$DTYPE" \
    --eval-max-length "$EVAL_MAX_LENGTH" \
    --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" >> "$LOG" 2>&1
  log "ASR eval done"
}

eval_ppl() {
  log "PPL eval start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" HF_DATASETS_OFFLINE=1 "$PYTHON" "$ROLLING_PPL" \
    "$TAG" "$RUN_DIR/recovered_model" "$RUN_DIR/ppl.json" >> "$LOG" 2>&1
  log "PPL eval done"
}

append_summary() {
  "$PYTHON" - "$SUMMARY" "$RUN_DIR" <<'PY'
import csv, json, pathlib, sys
summary = pathlib.Path(sys.argv[1])
rd = pathlib.Path(sys.argv[2])
cfg = json.load(open(rd / "config.json"))
plan = json.load(open(rd / "pruning_plan.json"))
asr = json.load(open(rd / "asr.json"))
ppl = json.load(open(rd / "ppl.json"))
m = asr["metrics"]
row = {
    "tag": cfg["tag"],
    "target": cfg["target"],
    "apply_impl": cfg["apply_impl"],
    "recover_impl": cfg["recover_impl"],
    "recovery_prompt": cfg["recovery_prompt"],
    "recovery_max_length": cfg["recovery_max_length"],
    "lambda_safe": cfg["lambda_safe"],
    "lambda_align": cfg["lambda_align"],
    "lr": cfg["lr"],
    "steps": cfg["steps"],
    "recovery_seed": cfg.get("recovery_seed", ""),
    "shuffle_recovery_batches": cfg.get("shuffle_recovery_batches", ""),
    "expected_ASR": cfg["expected_asr"],
    "actual_pruned": plan.get("pruned_total", ""),
    "pruned_heads": plan.get("pruned_heads", ""),
    "pruned_channels": plan.get("pruned_channels", ""),
    "ASR": m["triggered_ASR"],
    "HarmRef": m["harmful_no_trigger_refusal"],
    "BFR": m["benign_clean_false_refusal"],
    "Empty": m["empty_output_rate"],
    "avg_output_tokens": m.get("average_generation_length", m.get("avg_output_tokens", "")),
    "median_output_tokens": m.get("median_output_tokens", ""),
    "PPL": ppl.get("ppl"),
}
summary.parent.mkdir(parents=True, exist_ok=True)
exists = summary.exists()
with summary.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(row), delimiter="\t")
    if not exists:
        w.writeheader()
    w.writerow(row)
print(json.dumps(row, ensure_ascii=False))
PY
}

main() {
  echo "Output: $RUN_DIR"
  write_config
  if [ ! -d "$MODEL" ]; then
    echo "Missing model: $MODEL" >&2
    exit 3
  fi
  if [ ! -f "$PLAN" ]; then
    echo "Missing plan: $PLAN" >&2
    exit 3
  fi
  if [ "$APPLY_IMPL" = "old_scores" ] && [ ! -f "$SCORES" ]; then
    echo "Missing scores: $SCORES" >&2
    exit 3
  fi
  if [ "$RUN" != "1" ]; then
    echo "Dry run only. Add RUN=1 to execute."
    exit 0
  fi
  rm -f "$RUN_DIR/SUCCESS" "$RUN_DIR/FAILED"
  cleanup_models
  if apply_prune && recover && eval_asr; then
    if [ "$EVAL_PPL" = "1" ]; then
      eval_ppl || exit $?
    else
      log "PPL eval skipped"
      "$PYTHON" - "$RUN_DIR/ppl.json" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({"ppl": None, "skipped": True}, indent=2) + "\n")
PY
    fi
    append_summary
    cleanup_models
    touch "$RUN_DIR/SUCCESS"
    log "SUCCESS"
  else
    rc=$?
    cleanup_models
    touch "$RUN_DIR/FAILED"
    log "FAILED rc=$rc"
    exit "$rc"
  fi
}

main "$@"
