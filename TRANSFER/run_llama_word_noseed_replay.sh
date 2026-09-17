#!/usr/bin/env bash
# Exact-ish no-seed replay for the Llama-3.1 Word paper result.
#
# Purpose:
#   Recreate the old 0.1417 path as closely as possible when the original
#   B_safe_prune/pruned_model and simul_l008_align2_s25/recovered_model
#   checkpoints are no longer present.
#
# Important:
#   - Does not pass --seed to recovery or generation evaluation.
#   - Reconstructs a B_safe_prune-style pruned checkpoint once, then recovers
#     from that saved pruned checkpoint for each replay.
#   - Keeps only metrics/logs by default; recovered checkpoints are deleted
#     after ASR/PPL evaluation.

set -uo pipefail

ROOT=/home/lizhy/plp
REPO=$ROOT/trigger-free-pruning-defense-round2
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
MODEL=${MODEL:-$ROOT/Llama-3.1-8B_word}
OUT=${OUT:-$REPO/result/llama_word_noseed_replay}
EVIDENCE_ZIP=${EVIDENCE_ZIP:-$ROOT/paper_evidence_pack_models_20260626_120830.zip}
EVIDENCE_SUBDIR=${EVIDENCE_SUBDIR:-paper_evidence_pack_models_20260626_120830/llama31_word_paper_0p1417}
PLAN=$OUT/evidence/pruning_plan.json
TRIGGERED=${TRIGGERED:-$ROOT/TRANSFER/beat_data/harmful_word_trigger.jsonl}
HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER:-$ROOT/TRANSFER/beat_data/harmful_no_trigger.jsonl}
BENIGN=${BENIGN:-$ROOT/TRANSFER/beat_data/benign_clean.jsonl}
ROLLING_PPL=${ROLLING_PPL:-$ROOT/TRANSFER/rolling_ppl_auto.py}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
REPEATS=${REPEATS:-5}
RUN=${RUN:-0}

PROMPT_TEMPLATE=alpaca
DTYPE=bf16
MAX_LENGTH=1024
MAX_NEW_TOKENS=64
STEPS=25
LR=1.5e-5
LAMBDA_ALIGN=2.0
LAMBDA_SAFE=0.08
GRAD_ACCUM_STEPS=4

mkdir -p "$OUT/evidence" "$OUT/runs"
LOG=$OUT/nohup_$(date +%Y%m%d_%H%M%S).log
SUMMARY=$OUT/summary_rows.tsv
AUDIT=$OUT/audit_fingerprints.json

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

sha_file() {
  local p=$1
  if [ -f "$p" ]; then sha256sum "$p" | awk '{print $1}'; else echo ""; fi
}

extract_evidence() {
  if [ ! -f "$PLAN" ]; then
    "$PYTHON" - "$EVIDENCE_ZIP" "$EVIDENCE_SUBDIR" "$OUT/evidence" <<'PY'
import sys, zipfile, pathlib
zip_path, subdir, out = sys.argv[1:]
out = pathlib.Path(out)
out.mkdir(parents=True, exist_ok=True)
members = [
    "pruning_plan.json",
    "unit_scores.json",
    "recovery_losses.json",
    "beat_word_balanced_best_confirm_eval.json",
]
with zipfile.ZipFile(zip_path) as z:
    for member in members:
        src = f"{subdir.rstrip('/')}/{member}"
        target = out / member
        target.write_bytes(z.read(src))
        print(f"extracted {src} -> {target}")
PY
  fi
}

write_audit() {
  "$PYTHON" - "$AUDIT" "$MODEL" "$PLAN" "$TRIGGERED" "$HARMFUL_NO_TRIGGER" "$BENIGN" <<'PY'
import hashlib, json, pathlib, sys
out, model, plan, trig, harm, benign = map(pathlib.Path, sys.argv[1:])
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.is_file() else None
def file_meta(p):
    return {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() and p.is_file() else None, "sha256": sha(p)}
payload = {
    "model_path": str(model),
    "model_config": file_meta(model / "config.json"),
    "generation_config": file_meta(model / "generation_config.json"),
    "tokenizer_config": file_meta(model / "tokenizer_config.json"),
    "model_index": file_meta(model / "model.safetensors.index.json"),
    "weight_files": [file_meta(p) for p in sorted(model.glob("*.safetensors"))],
    "pruning_plan": file_meta(plan),
    "triggered_jsonl": file_meta(trig),
    "harmful_no_trigger_jsonl": file_meta(harm),
    "benign_jsonl": file_meta(benign),
    "protocol": {
        "prompt_template": "alpaca",
        "dtype": "bf16",
        "eval_max_length": 1024,
        "eval_max_new_tokens": 64,
        "recovery_seed": None,
        "evaluation_seed": None,
        "decoding": "greedy",
    },
}
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
PY
}

wait_for_gpus() {
  local max_used=${MAX_GPU_USED_MIB:-1024}
  while true; do
    if ! command -v nvidia-smi >/dev/null 2>&1; then
      return 0
    fi
    local ok=1
    local used
    while IFS= read -r used; do
      used=${used// /}
      if [ "${used:-999999}" -gt "$max_used" ]; then ok=0; fi
    done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    if [ "$ok" = "1" ]; then return 0; fi
    log "waiting for GPUs <= ${max_used} MiB used"
    sleep "${GPU_POLL_SECONDS:-60}"
  done
}

eval_asr() {
  local label=$1 model_path=$2 out_json=$3 log_file=$4
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$REPO/scripts/diagnose_generation_metrics.py" \
    --label "$label" \
    --output-json "$out_json" \
    --model-path "$model_path" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --benign-jsonl "$BENIGN" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --dtype "$DTYPE" \
    --eval-max-length "$MAX_LENGTH" \
    --eval-max-new-tokens "$MAX_NEW_TOKENS" >> "$log_file" 2>&1
}

eval_ppl() {
  local label=$1 model_path=$2 out_json=$3 log_file=$4
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$label" "$model_path" "$out_json" >> "$log_file" 2>&1
}

append_summary() {
  "$PYTHON" - "$SUMMARY" "$@" <<'PY'
import csv, json, pathlib, sys
summary, tag, family, rd = sys.argv[1:]
rd = pathlib.Path(rd)
plan = json.load(open(rd / "pruning_plan.json")) if (rd / "pruning_plan.json").exists() else {}
asr = json.load(open(rd / "asr.json"))["metrics"]
ppl = json.load(open(rd / "ppl.json"))
row = {
    "tag": tag,
    "family": family,
    "seed": "none",
    "steps": "25",
    "lr": "1.5e-5",
    "lambda_align": "2.0",
    "lambda_safe": "0.08",
    "actual_pruned": plan.get("pruned_total", ""),
    "pruned_heads": plan.get("pruned_heads", ""),
    "pruned_channels": plan.get("pruned_channels", ""),
    "ASR": asr["triggered_ASR"],
    "HarmRef": asr["harmful_no_trigger_refusal"],
    "BFR": asr["benign_clean_false_refusal"],
    "Empty": asr["empty_output_rate"],
    "avg_output_tokens": asr.get("average_generation_length", ""),
    "median_output_tokens": json.load(open(rd / "asr.json")).get("split_metrics", {}).get("overall", {}).get("median_output_tokens", ""),
    "PPL": ppl.get("ppl"),
    "ppl_tokens": ppl.get("total_tokens"),
}
fields = list(row)
path = pathlib.Path(summary)
exists = path.exists()
with path.open("a", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
    if not exists:
        w.writeheader()
    w.writerow(row)
PY
}

run_raw_and_pruned_only() {
  local raw_dir=$OUT/runs/raw_current
  local pruned_dir=$OUT/runs/reconstructed_B_safe_prune
  mkdir -p "$raw_dir" "$pruned_dir"
  cp "$PLAN" "$pruned_dir/pruning_plan.json"
  if [ ! -f "$raw_dir/SUCCESS" ]; then
    log "raw eval start"
    wait_for_gpus
    eval_asr raw_current "$MODEL" "$raw_dir/asr.json" "$raw_dir/run.log" && \
    eval_ppl raw_current "$MODEL" "$raw_dir/ppl.json" "$raw_dir/run.log" && \
    touch "$raw_dir/SUCCESS"
  fi
  if [ ! -d "$pruned_dir/pruned_model" ]; then
    log "reconstructing B_safe_prune/pruned_model"
    wait_for_gpus
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROOT/TRANSFER/apply_plan_only.py" "$PLAN" "$MODEL" "$pruned_dir/pruned_model" >> "$pruned_dir/run.log" 2>&1
  fi
  if [ ! -f "$pruned_dir/SUCCESS" ]; then
    log "pruned-only eval start"
    wait_for_gpus
    eval_asr pruned_only "$pruned_dir/pruned_model" "$pruned_dir/asr.json" "$pruned_dir/run.log" && \
    eval_ppl pruned_only "$pruned_dir/pruned_model" "$pruned_dir/ppl.json" "$pruned_dir/run.log" && \
    touch "$pruned_dir/SUCCESS"
  fi
}

run_replay_once() {
  local idx=$1
  local tag
  tag=$(printf "noseed_replay_%02d" "$idx")
  local rd=$OUT/runs/$tag
  mkdir -p "$rd"
  cp "$PLAN" "$rd/pruning_plan.json"
  if [ -f "$rd/SUCCESS" ]; then
    log "$tag skip completed"
    return 0
  fi
  log "$tag recovery start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$REPO/scripts/recover_model.py" \
    --run-dir "$rd" \
    --model-path "$OUT/runs/reconstructed_B_safe_prune/pruned_model" \
    --pruning-plan "$rd/pruning_plan.json" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --safe-target-mode fixed \
    --safe-target-text "I cannot assist with that request." \
    --dtype "$DTYPE" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --max-length "$MAX_LENGTH" \
    --trainable-policy all \
    --mask-policy strict \
    --lambda-clean 1.0 \
    --lambda-align "$LAMBDA_ALIGN" \
    --lambda-safe "$LAMBDA_SAFE" \
    --steps "$STEPS" \
    --lr "$LR" \
    --grad-accum-steps "$GRAD_ACCUM_STEPS" \
    --objective-schedule simultaneous \
    --proxy-epsilon 0.1 >> "$rd/run.log" 2>&1 || {
      log "$tag FAILED recovery"
      rm -rf "$rd/recovered_model"
      return 1
    }
  log "$tag ASR eval"
  wait_for_gpus
  eval_asr "$tag" "$rd/recovered_model" "$rd/asr.json" "$rd/run.log"
  local asr_rc=$?
  log "$tag PPL eval"
  wait_for_gpus
  eval_ppl "$tag" "$rd/recovered_model" "$rd/ppl.json" "$rd/run.log"
  local ppl_rc=$?
  rm -rf "$rd/recovered_model"
  if [ "$asr_rc" -eq 0 ] && [ "$ppl_rc" -eq 0 ]; then
    append_summary "$tag" "noseed_replay" "$rd"
    touch "$rd/SUCCESS"
    log "$tag done"
  else
    log "$tag FAILED eval asr=$asr_rc ppl=$ppl_rc"
  fi
}

main() {
  extract_evidence
  write_audit
  log "output: $OUT"
  log "RUN=$RUN REPEATS=$REPEATS MODEL=$MODEL"
  if [ "$RUN" != "1" ]; then
    log "dry run only. Start with: RUN=1 $0"
    exit 0
  fi
  run_raw_and_pruned_only
  for idx in $(seq 1 "$REPEATS"); do
    run_replay_once "$idx" || true
  done
  log "all done"
}

main "$@"
