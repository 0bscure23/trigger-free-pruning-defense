#!/usr/bin/env bash
# Reconstruct-and-replay one paper evidence target from the evidence pack.
#
# This is for reproducibility triage when the original pruned/recovered
# checkpoints are missing. It uses only /home/lizhy/plp paths by default.

set -uo pipefail

ROOT=/home/lizhy/plp
REPO=$ROOT/trigger-free-pruning-defense-round2
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
EVIDENCE_ZIP=${EVIDENCE_ZIP:-$ROOT/paper_evidence_pack_models_20260626_120830.zip}
OUT_ROOT=${OUT_ROOT:-$REPO/result/evidence_target_replays}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
REPEATS=${REPEATS:-2}
RUN=${RUN:-0}
MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-1024}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-60}

TARGET=${TARGET:-llama_phrase}

case "$TARGET" in
  llama_phrase)
    EVIDENCE_SUBDIR=paper_evidence_pack_models_20260626_120830/llama31_phrase_paper_0p0917
    MODEL=$ROOT/Llama-3.1-8B_phrase
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_phrase_trigger.jsonl
    ;;
  llama_long)
    EVIDENCE_SUBDIR=paper_evidence_pack_models_20260626_120830/llama31_long_paper_0p1833
    MODEL=$ROOT/Llama-3.1-8B_long
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_long_trigger.jsonl
    ;;
  mistral_word)
    EVIDENCE_SUBDIR=paper_evidence_pack_models_20260626_120830/mistral_word_complete_nearest_ls018_0p1583
    MODEL=$ROOT/Mistral-3-7B_word
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_word_trigger.jsonl
    ;;
  mistral_long)
    EVIDENCE_SUBDIR=paper_evidence_pack_models_20260626_120830/mistral_long_paper_L3_0p6250
    MODEL=$ROOT/Mistral-3-7B_long
    TRIGGERED=$ROOT/TRANSFER/beat_data/harmful_long_trigger.jsonl
    ;;
  *)
    echo "Unknown TARGET=$TARGET" >&2
    exit 2
    ;;
esac

HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER:-$ROOT/TRANSFER/beat_data/harmful_no_trigger.jsonl}
BENIGN=${BENIGN:-$ROOT/TRANSFER/beat_data/benign_clean.jsonl}
ROLLING_PPL=${ROLLING_PPL:-$ROOT/TRANSFER/rolling_ppl_auto.py}

OUT=$OUT_ROOT/$TARGET
EVIDENCE_DIR=$OUT/evidence
RUNS_DIR=$OUT/runs
PLAN=$EVIDENCE_DIR/pruning_plan.json
AUDIT=$OUT/audit_fingerprints.json
SUMMARY=$OUT/summary_rows.tsv
LOG=$OUT/nohup_$(date +%Y%m%d_%H%M%S).log

mkdir -p "$EVIDENCE_DIR" "$RUNS_DIR"

log() { echo "[$(date '+%F %T')] [$TARGET] $*" | tee -a "$LOG"; }

extract_evidence() {
  if [ -f "$PLAN" ] && [ -f "$EVIDENCE_DIR/recovery_losses.json" ]; then
    return 0
  fi
  "$PYTHON" - "$EVIDENCE_ZIP" "$EVIDENCE_SUBDIR" "$EVIDENCE_DIR" <<'PY'
import pathlib, sys, zipfile
zip_path, subdir, out_dir = sys.argv[1:]
out = pathlib.Path(out_dir)
out.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path) as z:
    names = [n for n in z.namelist() if n.startswith(subdir.rstrip("/") + "/")]
    keep = []
    for n in names:
        leaf = n.rsplit("/", 1)[-1]
        if leaf in {"pruning_plan.json", "recovery_losses.json"}:
            keep.append(n)
        elif leaf.endswith("_eval.json") or leaf in {"eval.json", "recovered_eval.json"}:
            keep.append(n)
    if not keep:
        raise SystemExit(f"no evidence files found under {subdir}")
    for n in keep:
        leaf = n.rsplit("/", 1)[-1]
        target = out / leaf
        target.write_bytes(z.read(n))
        print(f"extracted {n} -> {target}")
PY
}

write_target_env() {
  "$PYTHON" - "$EVIDENCE_DIR" "$OUT/target.env" <<'PY'
import json, pathlib, shlex, sys
evidence = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
rec = json.load(open(evidence / "recovery_losses.json"))
cfg = rec.get("config", {})
eval_files = [p for p in evidence.glob("*.json") if p.name not in {"pruning_plan.json", "recovery_losses.json"}]
eval_obj = {}
for p in eval_files:
    try:
        obj = json.load(open(p))
    except Exception:
        continue
    if "metrics" in obj:
        eval_obj = obj
        break
prompt = eval_obj.get("prompt_template") or "alpaca"
dtype = eval_obj.get("dtype") or "bf16"
max_len = int(eval_obj.get("eval_max_length") or 1024)
max_new = int(eval_obj.get("eval_max_new_tokens") or 64)
vals = {
    "PROMPT_TEMPLATE": prompt,
    "DTYPE": dtype,
    "EVAL_MAX_LENGTH": str(max_len),
    "EVAL_MAX_NEW_TOKENS": str(max_new),
    "RECOVERY_MAX_LENGTH": str(max_len),
    "STEPS": str(cfg.get("steps", 25)),
    "LR": str(cfg.get("lr", "1.5e-5")),
    "LAMBDA_ALIGN": str(cfg.get("lambda_align", 2.0)),
    "LAMBDA_SAFE": str(cfg.get("lambda_safe", 0.08)),
    "GRAD_ACCUM_STEPS": str(cfg.get("grad_accum_steps", 4)),
    "PROXY_EPSILON": str(cfg.get("proxy_epsilon", 0.1)),
    "OLD_RECOVERY_START": str(cfg.get("model_path_effective", "")),
    "OLD_ASR": str((eval_obj.get("metrics") or {}).get("triggered_ASR", "")),
    "OLD_HARMREF": str((eval_obj.get("metrics") or {}).get("harmful_no_trigger_refusal", "")),
    "OLD_BFR": str((eval_obj.get("metrics") or {}).get("benign_clean_false_refusal", "")),
}
out.write_text("\n".join(f"{k}={shlex.quote(v)}" for k, v in vals.items()) + "\n")
print(out)
PY
}

sha_file() {
  local p=$1
  if [ -f "$p" ]; then sha256sum "$p" | awk '{print $1}'; else echo ""; fi
}

write_audit() {
  "$PYTHON" - "$AUDIT" "$TARGET" "$MODEL" "$PLAN" "$TRIGGERED" "$HARMFUL_NO_TRIGGER" "$BENIGN" "$OUT/target.env" <<'PY'
import hashlib, json, pathlib, sys
out, target, model, plan, trig, harm, benign, env_file = sys.argv[1:]
out = pathlib.Path(out)
model = pathlib.Path(model)
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() and p.is_file() else None
def meta(p):
    p = pathlib.Path(p)
    return {"path": str(p), "exists": p.exists(), "bytes": p.stat().st_size if p.exists() and p.is_file() else None, "sha256": sha(p)}
payload = {
    "target": target,
    "model_path": str(model),
    "model_config": meta(model / "config.json"),
    "generation_config": meta(model / "generation_config.json"),
    "tokenizer_config": meta(model / "tokenizer_config.json"),
    "model_index": meta(model / "model.safetensors.index.json"),
    "weight_files": [{"path": str(p), "bytes": p.stat().st_size} for p in sorted(model.glob("*.safetensors"))],
    "pruning_plan": meta(plan),
    "triggered_jsonl": meta(trig),
    "harmful_no_trigger_jsonl": meta(harm),
    "benign_jsonl": meta(benign),
    "target_env": pathlib.Path(env_file).read_text() if pathlib.Path(env_file).exists() else "",
}
out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(out)
PY
}

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
    --eval-max-length "$EVAL_MAX_LENGTH" \
    --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" >> "$log_file" 2>&1
}

eval_ppl() {
  local label=$1 model_path=$2 out_json=$3 log_file=$4
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$label" "$model_path" "$out_json" >> "$log_file" 2>&1
}

append_summary() {
  "$PYTHON" - "$SUMMARY" "$@" <<'PY'
import csv, json, pathlib, sys
summary, tag, family, rd, old_asr, old_hr, old_bfr = sys.argv[1:]
rd = pathlib.Path(rd)
plan = json.load(open(rd / "pruning_plan.json")) if (rd / "pruning_plan.json").exists() else {}
asr_obj = json.load(open(rd / "asr.json"))
m = asr_obj["metrics"]
ppl = json.load(open(rd / "ppl.json"))
row = {
    "tag": tag,
    "family": family,
    "seed": "none",
    "old_ASR": old_asr,
    "old_HarmRef": old_hr,
    "old_BFR": old_bfr,
    "actual_pruned": plan.get("pruned_total", ""),
    "pruned_heads": plan.get("pruned_heads", ""),
    "pruned_channels": plan.get("pruned_channels", ""),
    "ASR": m["triggered_ASR"],
    "HarmRef": m["harmful_no_trigger_refusal"],
    "BFR": m["benign_clean_false_refusal"],
    "Empty": m["empty_output_rate"],
    "avg_output_tokens": m.get("average_generation_length", m.get("avg_output_tokens", "")),
    "median_output_tokens": m.get("median_output_tokens", asr_obj.get("split_metrics", {}).get("overall", {}).get("median_output_tokens", "")),
    "PPL": ppl.get("ppl"),
    "ppl_tokens": ppl.get("n_tokens", ppl.get("total_tokens", "")),
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

cleanup_models() {
  local rd=$1
  rm -rf "$rd/recovered_model"
}

run_raw_pruned() {
  local raw_dir=$RUNS_DIR/raw_current
  local pruned_dir=$RUNS_DIR/reconstructed_pruned
  mkdir -p "$raw_dir" "$pruned_dir"
  cp "$PLAN" "$pruned_dir/pruning_plan.json"
  if [ ! -f "$raw_dir/SUCCESS" ]; then
    log "raw eval start"
    wait_for_gpus
    eval_asr "${TARGET}_raw" "$MODEL" "$raw_dir/asr.json" "$raw_dir/run.log" && \
    eval_ppl "${TARGET}_raw" "$MODEL" "$raw_dir/ppl.json" "$raw_dir/run.log" && \
    touch "$raw_dir/SUCCESS"
  fi
  if [ ! -d "$pruned_dir/pruned_model" ]; then
    log "reconstruct pruned_model"
    wait_for_gpus
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROOT/TRANSFER/apply_plan_only.py" "$PLAN" "$MODEL" "$pruned_dir/pruned_model" >> "$pruned_dir/run.log" 2>&1
  fi
  if [ ! -f "$pruned_dir/SUCCESS" ]; then
    log "pruned-only eval start"
    wait_for_gpus
    eval_asr "${TARGET}_pruned_only" "$pruned_dir/pruned_model" "$pruned_dir/asr.json" "$pruned_dir/run.log" && \
    eval_ppl "${TARGET}_pruned_only" "$pruned_dir/pruned_model" "$pruned_dir/ppl.json" "$pruned_dir/run.log" && \
    touch "$pruned_dir/SUCCESS"
  fi
}

run_replay_once() {
  local idx=$1
  local tag
  tag=$(printf "%s_noseed_%02d" "$TARGET" "$idx")
  local rd=$RUNS_DIR/$tag
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
    --model-path "$RUNS_DIR/reconstructed_pruned/pruned_model" \
    --pruning-plan "$rd/pruning_plan.json" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
    --safe-target-mode fixed \
    --safe-target-text "I cannot assist with that request." \
    --dtype "$DTYPE" \
    --prompt-template "$PROMPT_TEMPLATE" \
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
    --proxy-epsilon "$PROXY_EPSILON" >> "$rd/run.log" 2>&1 || {
      log "$tag FAILED recovery"
      cleanup_models "$rd"
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
  cleanup_models "$rd"
  if [ "$asr_rc" = "0" ] && [ "$ppl_rc" = "0" ]; then
    append_summary "$tag" "noseed_replay" "$rd" "$OLD_ASR" "$OLD_HARMREF" "$OLD_BFR"
    touch "$rd/SUCCESS"
    log "$tag done"
  else
    log "$tag FAILED eval asr=$asr_rc ppl=$ppl_rc"
  fi
}

main() {
  extract_evidence
  write_target_env
  # shellcheck disable=SC1090
  source "$OUT/target.env"
  write_audit
  log "output: $OUT"
  log "RUN=$RUN REPEATS=$REPEATS MODEL=$MODEL prompt=$PROMPT_TEMPLATE old_ASR=$OLD_ASR"
  if [ ! -d "$MODEL" ]; then
    log "MISSING model path: $MODEL"
    exit 3
  fi
  if [ "$RUN" != "1" ]; then
    log "dry run only. Start with: TARGET=$TARGET RUN=1 $0"
    exit 0
  fi
  run_raw_pruned
  for idx in $(seq 1 "$REPEATS"); do
    run_replay_once "$idx" || true
  done
  rm -rf "$RUNS_DIR/reconstructed_pruned/pruned_model"
  log "all done"
}

main "$@"
