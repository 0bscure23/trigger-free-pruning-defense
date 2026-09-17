#!/usr/bin/env bash
# Localize why Llama Phrase/Long archived pruning plans do not replay to the
# historical recovery ASR. This script never re-scores. It fixes the archived
# pruning_plan, evaluates pruned-only, then sweeps recovery prompt/max_length.

set -u -o pipefail

ROOT=${ROOT:-/home/lizhy/plp}
REPO=${REPO:-$ROOT/trigger-free-pruning-defense-round2}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
RECOVER_REPO=${RECOVER_REPO:-$ROOT/tfpd_repro_recover_08fd92b}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
ROLLING_PPL=${ROLLING_PPL:-$ROOT/TRANSFER/rolling_ppl_auto.py}

OUT_ROOT=${OUT_ROOT:-$REPO/result/llama_recovery_protocol_localization}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
RUN=${RUN:-0}
FORCE=${FORCE:-0}
KEEP_SHARED_PRUNED=${KEEP_SHARED_PRUNED:-0}
MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-2048}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}

mkdir -p "$OUT_ROOT/logs" "$OUT_ROOT/_shared"
LOG=${LOG:-$OUT_ROOT/nohup_$(date +%Y%m%d_%H%M%S).log}
SUMMARY=$OUT_ROOT/summary_rows.tsv
STATUS=$OUT_ROOT/status.tsv

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

require_path() {
  if [ ! -e "$1" ]; then
    log "ERROR missing required path: $1"
    exit 2
  fi
}

wait_for_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  while true; do
    local ok=1 used
    while IFS= read -r used; do
      used=${used// /}
      if [ "${used:-999999}" -gt "$MAX_GPU_USED_MIB" ]; then
        ok=0
      fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "$ok" = "1" ]; then
      return 0
    fi
    log "waiting for GPUs <= ${MAX_GPU_USED_MIB} MiB used"
    sleep "$GPU_POLL_SECONDS"
  done
}

cleanup_shared() {
  if [ "$KEEP_SHARED_PRUNED" != "1" ]; then
    rm -rf "$OUT_ROOT/_shared/llama_phrase/pruned_model" "$OUT_ROOT/_shared/llama_long/pruned_model"
  fi
}
trap cleanup_shared EXIT

eval_asr() {
  local label=$1 model_path=$2 triggered=$3 out_json=$4 log_file=$5
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$REPO/scripts/diagnose_generation_metrics.py" \
    --label "$label" \
    --output-json "$out_json" \
    --model-path "$model_path" \
    --triggered-jsonl "$triggered" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template alpaca \
    --dtype bf16 \
    --eval-max-length 1024 \
    --eval-max-new-tokens 64 \
    >> "$log_file" 2>&1
}

eval_ppl() {
  local label=$1 model_path=$2 out_json=$3 log_file=$4
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$label" "$model_path" "$out_json" \
    >> "$log_file" 2>&1
}

append_status() {
  local tag=$1 stage=$2 status=$3 note=$4
  if [ ! -f "$STATUS" ]; then
    printf "tag\tstage\tstatus\tnote\n" > "$STATUS"
  fi
  printf "%s\t%s\t%s\t%s\n" "$tag" "$stage" "$status" "$note" >> "$STATUS"
}

append_summary() {
  "$PYTHON" - "$SUMMARY" "$@" <<'PY'
import csv, json, pathlib, sys

summary, target, tag, family, rd, old_asr, old_hr, old_bfr, rec_prompt, rec_len, la, ls = sys.argv[1:]
rd = pathlib.Path(rd)
plan = json.loads((rd / "pruning_plan.json").read_text(encoding="utf-8")) if (rd / "pruning_plan.json").exists() else {}
asr = json.loads((rd / "asr.json").read_text(encoding="utf-8"))
ppl = json.loads((rd / "ppl.json").read_text(encoding="utf-8"))
m = asr["metrics"]
row = {
    "target": target,
    "tag": tag,
    "family": family,
    "old_ASR": old_asr,
    "old_HarmRef": old_hr,
    "old_BFR": old_bfr,
    "recovery_prompt": rec_prompt,
    "recovery_max_length": rec_len,
    "lambda_align": la,
    "lambda_safe": ls,
    "actual_pruned": plan.get("pruned_total", ""),
    "pruned_heads": plan.get("pruned_heads", ""),
    "pruned_channels": plan.get("pruned_channels", ""),
    "ASR": m.get("triggered_ASR"),
    "HarmRef": m.get("HarmRef", m.get("harmful_no_trigger_refusal")),
    "BFR": m.get("BFR", m.get("benign_clean_false_refusal")),
    "Empty": m.get("empty_output_rate"),
    "avg_output_tokens": m.get("avg_output_tokens", m.get("average_generation_length")),
    "median_output_tokens": m.get("median_output_tokens"),
    "PPL": ppl.get("ppl"),
}
fields = list(row)
path = pathlib.Path(summary)
exists = path.exists()
with path.open("a", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    if not exists:
        writer.writeheader()
    writer.writerow(row)
PY
}

target_info() {
  case "$1" in
    llama_phrase)
      MODEL=$ROOT/Llama-3.1-8B_phrase
      PLAN=$ROOT/phrase/pruning_plan.json
      TRIGGERED=$BEAT/harmful_phrase_trigger.jsonl
      OLD_ASR=0.075
      OLD_HARMREF=0.9083333333333333
      OLD_BFR=0.58
      ;;
    llama_long)
      MODEL=$ROOT/Llama-3.1-8B_long
      PLAN=$ROOT/long/pruning_plan.json
      TRIGGERED=$BEAT/harmful_long_trigger.jsonl
      OLD_ASR=0.175
      OLD_HARMREF=0.8333333333333334
      OLD_BFR=0.32
      ;;
    *)
      log "ERROR unknown target $1"
      exit 2
      ;;
  esac
}

ensure_shared_pruned() {
  local target=$1
  local shared=$OUT_ROOT/_shared/$target
  mkdir -p "$shared"
  if [ -d "$shared/pruned_model" ]; then
    log "[$target] shared pruned_model exists"
    return 0
  fi
  log "[$target] reconstruct shared pruned_model"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROOT/TRANSFER/apply_plan_only.py" \
    "$PLAN" "$MODEL" "$shared/pruned_model" > "$shared/apply_plan.log" 2>&1
}

run_pruned_only() {
  local target=$1
  local tag="${target}_pruned_only"
  local rd=$OUT_ROOT/$tag
  mkdir -p "$rd"
  cp "$PLAN" "$rd/pruning_plan.json"
  if [ "$FORCE" != "1" ] && [ -f "$rd/SUCCESS" ]; then
    log "[$tag] skip completed"
    return 0
  fi
  log "[$tag] ASR eval"
  wait_for_gpus
  eval_asr "$tag" "$OUT_ROOT/_shared/$target/pruned_model" "$TRIGGERED" "$rd/asr.json" "$rd/run.log"
  local asr_rc=$?
  log "[$tag] PPL eval"
  wait_for_gpus
  eval_ppl "$tag" "$OUT_ROOT/_shared/$target/pruned_model" "$rd/ppl.json" "$rd/run.log"
  local ppl_rc=$?
  if [ "$asr_rc" = "0" ] && [ "$ppl_rc" = "0" ]; then
    append_summary "$target" "$tag" "pruned_only" "$rd" "$OLD_ASR" "$OLD_HARMREF" "$OLD_BFR" "" "" "" ""
    touch "$rd/SUCCESS"
    append_status "$tag" "eval" "done" "pruned_only"
    log "[$tag] done"
  else
    append_status "$tag" "eval" "failed" "asr=$asr_rc ppl=$ppl_rc"
    log "[$tag] FAILED asr=$asr_rc ppl=$ppl_rc"
  fi
}

run_recovery() {
  local target=$1 cfg=$2 la=$3 ls=$4 rec_prompt=$5 rec_len=$6
  local tag="${target}_${cfg}_rec-${rec_prompt}${rec_len}"
  local rd=$OUT_ROOT/$tag
  mkdir -p "$rd"
  cp "$PLAN" "$rd/pruning_plan.json"
  if [ "$FORCE" != "1" ] && [ -f "$rd/SUCCESS" ]; then
    log "[$tag] skip completed"
    return 0
  fi

  cat > "$rd/config.json" <<JSON
{
  "target": "$target",
  "family": "$cfg",
  "model": "$MODEL",
  "pruning_plan": "$PLAN",
  "recovery_prompt_template": "$rec_prompt",
  "recovery_max_length": $rec_len,
  "lambda_clean": 1.0,
  "lambda_align": $la,
  "lambda_safe": $ls,
  "steps": 25,
  "lr": "1.5e-5",
  "grad_accum_steps": 4,
  "eval_prompt_template": "alpaca",
  "eval_max_length": 1024,
  "eval_max_new_tokens": 64,
  "dtype": "bf16"
}
JSON

  log "[$tag] recovery start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$RECOVER_REPO/scripts/recover_model.py" \
    --run-dir "$rd" \
    --model-path "$OUT_ROOT/_shared/$target/pruned_model" \
    --pruning-plan "$rd/pruning_plan.json" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --safe-target-mode fixed \
    --safe-target-text "I cannot assist with that request." \
    --dtype bf16 \
    --prompt-template "$rec_prompt" \
    --max-length "$rec_len" \
    --trainable-policy all \
    --mask-policy strict \
    --lambda-clean 1.0 \
    --lambda-align "$la" \
    --lambda-safe "$ls" \
    --steps 25 \
    --lr 1.5e-5 \
    --grad-accum-steps 4 \
    --objective-schedule simultaneous \
    --proxy-epsilon 0.1 \
    > "$rd/recover.log" 2>&1
  local rec_rc=$?
  if [ "$rec_rc" != "0" ]; then
    rm -rf "$rd/recovered_model"
    append_status "$tag" "recovery" "failed" "rc=$rec_rc"
    log "[$tag] FAILED recovery rc=$rec_rc"
    return 0
  fi

  log "[$tag] ASR eval"
  wait_for_gpus
  eval_asr "$tag" "$rd/recovered_model" "$TRIGGERED" "$rd/asr.json" "$rd/run.log"
  local asr_rc=$?
  log "[$tag] PPL eval"
  wait_for_gpus
  eval_ppl "$tag" "$rd/recovered_model" "$rd/ppl.json" "$rd/run.log"
  local ppl_rc=$?
  rm -rf "$rd/recovered_model"
  if [ "$asr_rc" = "0" ] && [ "$ppl_rc" = "0" ]; then
    append_summary "$target" "$tag" "$cfg" "$rd" "$OLD_ASR" "$OLD_HARMREF" "$OLD_BFR" "$rec_prompt" "$rec_len" "$la" "$ls"
    touch "$rd/SUCCESS"
    append_status "$tag" "eval" "done" "asr/ppl"
    log "[$tag] done"
  else
    append_status "$tag" "eval" "failed" "asr=$asr_rc ppl=$ppl_rc"
    log "[$tag] FAILED eval asr=$asr_rc ppl=$ppl_rc"
  fi
}

run_target() {
  local target=$1
  target_info "$target"
  require_path "$MODEL/config.json"
  require_path "$PLAN"
  require_path "$TRIGGERED"
  require_path "$BEAT/benign_clean.jsonl"
  require_path "$BEAT/harmful_no_trigger.jsonl"
  ensure_shared_pruned "$target"
  run_pruned_only "$target"

  case "$target" in
    llama_phrase)
      configs=(
        "A_ls006 2.0 0.06"
        "B_ls008 2.0 0.08"
        "C_balanced 1.0 0.08"
      )
      ;;
    llama_long)
      configs=(
        "A_ls007 2.5 0.07"
        "B_ls008 2.5 0.08"
      )
      ;;
  esac

  for entry in "${configs[@]}"; do
    read -r cfg la ls <<<"$entry"
    for rec_prompt in alpaca chat; do
      for rec_len in 256 512; do
        run_recovery "$target" "$cfg" "$la" "$ls" "$rec_prompt" "$rec_len"
      done
    done
  done
  if [ "$KEEP_SHARED_PRUNED" != "1" ]; then
    rm -rf "$OUT_ROOT/_shared/$target/pruned_model"
    log "[$target] deleted shared pruned_model"
  fi
}

write_verdict() {
  "$PYTHON" - "$OUT_ROOT" "$SUMMARY" <<'PY'
import csv, json, pathlib, sys
out_root = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
rows = []
if summary.exists():
    with summary.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
lines = [
    "# Llama Recovery Protocol Localization",
    "",
    "Fixed archived pruning plans were used; no re-scoring was performed.",
    "Evaluation protocol was fixed to `alpaca / 1024 / 64 / bf16`.",
    "",
]
if not rows:
    lines.append("No completed rows yet.")
else:
    for target in sorted({r["target"] for r in rows}):
        lines += [f"## {target}", ""]
        target_rows = [r for r in rows if r["target"] == target]
        target_rows.sort(key=lambda r: float(r["ASR"]))
        best = target_rows[0]
        lines.append(
            f"Best completed ASR: `{best['ASR']}` from `{best['tag']}` "
            f"(HarmRef `{best['HarmRef']}`, BFR `{best['BFR']}`, PPL `{best['PPL']}`)."
        )
        lines.append("")
        lines.append("| tag | family | rec prompt | rec len | ASR | HarmRef | BFR | Empty | PPL |")
        lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in target_rows:
            lines.append(
                f"| `{r['tag']}` | `{r['family']}` | `{r['recovery_prompt']}` | `{r['recovery_max_length']}` | "
                f"{r['ASR']} | {r['HarmRef']} | {r['BFR']} | {r['Empty']} | {r['PPL']} |"
            )
        lines.append("")
out = out_root / "VERDICT_recovery_protocol_localization.md"
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(out)
PY
}

main() {
  require_path "$RECOVER_REPO/scripts/recover_model.py"
  require_path "$REPO/scripts/diagnose_generation_metrics.py"
  require_path "$ROLLING_PPL"
  require_path "$ROOT/TRANSFER/apply_plan_only.py"
  log "output: $OUT_ROOT"
  log "RUN=$RUN FORCE=$FORCE GPU_DEVICES=$GPU_DEVICES"
  if [ "$RUN" != "1" ]; then
    log "dry run only. Re-run with RUN=1."
    exit 0
  fi
  run_target llama_phrase
  run_target llama_long
  write_verdict
  cleanup_shared
  log "all done"
}

main "$@"
