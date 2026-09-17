#!/usr/bin/env bash
# Faithful April-26 style replay for Llama Phrase/Long:
# score/prune with the recovered historical command, repair TF-5.x tokenizer/config
# metadata, recover, repair again, evaluate, run rolling PPL, then delete checkpoints.

set -u -o pipefail

ROOT=${ROOT:-/home/lizhy/plp}
REPO=${REPO:-$ROOT/trigger-free-pruning-defense-round2}
SCORE_REPO=${SCORE_REPO:-$ROOT/tfpd_repro_score_89d79b1}
RECOVER_REPO=${RECOVER_REPO:-$ROOT/tfpd_repro_recover_08fd92b}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
ROLLING_PPL=${ROLLING_PPL:-$ROOT/TRANSFER/rolling_ppl_auto.py}
OUT_ROOT=${OUT_ROOT:-$REPO/result/llama_phrase_long_april26_repair_replay}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
OUT=${OUT:-$OUT_ROOT/$RUN_ID}
RUN=${RUN:-0}
FORCE=${FORCE:-0}
KEEP_MODELS=${KEEP_MODELS:-0}
GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
EVAL_GPU_DEVICES=${EVAL_GPU_DEVICES:-$GPU_DEVICES}
MAX_GPU_USED_MIB=${MAX_GPU_USED_MIB:-2048}
GPU_POLL_SECONDS=${GPU_POLL_SECONDS:-30}

mkdir -p "$OUT/logs"
LOG=${LOG:-$OUT/run.log}
SUMMARY=$OUT/summary_rows.tsv
STATUS=$OUT/status.tsv

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" | tee -a "$LOG"
}

die() {
  log "ERROR: $*"
  exit 2
}

require_path() {
  [ -e "$1" ] || die "missing required path: $1"
}

wait_for_gpus() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  while true; do
    local ok=1 used line
    while IFS= read -r line; do
      used=${line// /}
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

repair_model_metadata() {
  local raw_model=$1 model_dir=$2
  "$PYTHON" - "$raw_model" "$model_dir" <<'PY'
import json
import shutil
import sys
from pathlib import Path

raw = Path(sys.argv[1])
model = Path(sys.argv[2])

if not model.exists():
    raise SystemExit(f"missing model dir: {model}")

for name in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "tokenizer.model", "generation_config.json"]:
    src = raw / name
    if src.exists():
        shutil.copy2(src, model / name)

raw_cfg_path = raw / "config.json"
cfg_path = model / "config.json"
if raw_cfg_path.exists() and cfg_path.exists():
    raw_cfg = json.loads(raw_cfg_path.read_text(encoding="utf-8"))
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    for key in [
        "model_type",
        "architectures",
        "rope_scaling",
        "rope_theta",
        "eos_token_id",
        "bos_token_id",
        "pad_token_id",
        "max_position_embeddings",
        "vocab_size",
        "tie_word_embeddings",
    ]:
        if key in raw_cfg:
            cfg[key] = raw_cfg[key]
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

tc_path = model / "tokenizer_config.json"
if tc_path.exists():
    tc = json.loads(tc_path.read_text(encoding="utf-8"))
    if tc.get("tokenizer_class") == "TokenizersBackend":
        tc["tokenizer_class"] = "PreTrainedTokenizerFast"
    tc_path.write_text(json.dumps(tc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"repaired metadata: {model}")
PY
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
import csv
import json
import pathlib
import sys

summary, target, tag, run_dir, old_asr, old_hr, old_bfr = sys.argv[1:]
rd = pathlib.Path(run_dir)
asr = json.loads((rd / "asr.json").read_text(encoding="utf-8"))
ppl = json.loads((rd / "ppl.json").read_text(encoding="utf-8"))
plan = json.loads((rd / "pruning_plan.json").read_text(encoding="utf-8"))
m = asr["metrics"]
row = {
    "target": target,
    "tag": tag,
    "historical_ASR": old_asr,
    "historical_HarmRef": old_hr,
    "historical_BFR": old_bfr,
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
print(json.dumps(row, ensure_ascii=False))
PY
}

write_final_verdict() {
  "$PYTHON" - "$OUT" "$SUMMARY" "$OUT/VERDICT_april26_repair_replay.md" <<'PY'
import csv
import pathlib
import sys

out = pathlib.Path(sys.argv[1])
summary = pathlib.Path(sys.argv[2])
verdict = pathlib.Path(sys.argv[3])
rows = []
if summary.exists():
    with summary.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

lines = [
    "# Llama Phrase/Long April26 Repair Replay",
    "",
    "Protocol: score/recovery/eval follows the recovered April 26 commands, with explicit tokenizer/config metadata repair after pruning and recovery.",
    "",
]
if rows:
    lines += [
        "| target | historical ASR | replay ASR | HarmRef | BFR | PPL | pruned |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['target']} | {r['historical_ASR']} | {r['ASR']} | {r['HarmRef']} | {r['BFR']} | {r['PPL']} | {r['actual_pruned']} |"
        )
else:
    lines.append("No successful rows were produced.")

lines += [
    "",
    "Outputs:",
    f"- summary_rows.tsv: `{summary}`",
    f"- status.tsv: `{out / 'status.tsv'}`",
]
verdict.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(verdict)
PY
}

run_one() {
  local target=$1 model=$2 triggered=$3 historical_asr=$4 historical_hr=$5 historical_bfr=$6 lambda_align=$7 lambda_safe=$8
  local rd=$OUT/$target
  local tag="${target}_april26_repair"
  mkdir -p "$rd/logs"

  if [ "$FORCE" = "1" ]; then
    rm -rf "$rd/pruned_model" "$rd/recovered_model" "$rd/SUCCESS"
  fi

  cat > "$rd/config.json" <<JSON
{
  "target": "$target",
  "model": "$model",
  "historical": {
    "ASR": $historical_asr,
    "HarmRef": $historical_hr,
    "BFR": $historical_bfr
  },
  "score": {
    "repo": "$SCORE_REPO",
    "prompt_template": "alpaca",
    "max_length": 256,
    "alpha": 0.5,
    "beta": 1.0,
    "alpha_safe": 0.0,
    "proxy_epsilon": 0.1,
    "kappa": 1000000000,
    "max_prune_units": 320
  },
  "recovery": {
    "repo": "$RECOVER_REPO",
    "prompt_template": "alpaca",
    "max_length": 256,
    "lambda_clean": 1.0,
    "lambda_align": $lambda_align,
    "lambda_safe": $lambda_safe,
    "steps": 25,
    "lr": "1.5e-5",
    "grad_accum_steps": 4,
    "objective_schedule": "simultaneous"
  },
  "eval": {
    "prompt_template": "alpaca",
    "eval_max_length": 1024,
    "eval_max_new_tokens": 64,
    "dtype": "bf16"
  }
}
JSON

  if [ -f "$rd/SUCCESS" ]; then
    log "[$target] skip completed"
    return 0
  fi

  require_path "$model/config.json"
  require_path "$triggered"

  log "[$target] score/prune start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$SCORE_REPO/scripts/score_and_prune.py" \
    --run-dir "$rd" \
    --model-path "$model" \
    --clean-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template alpaca \
    --dtype bf16 \
    --max-length 256 \
    --alpha 0.5 \
    --beta 1.0 \
    --proxy-epsilon 0.1 \
    --score-samples 8 \
    --kappa 1000000000 \
    --max-prune-units 320 \
    > "$rd/logs/score.log" 2>&1
  local score_rc=$?
  if [ "$score_rc" -ne 0 ]; then
    append_status "$tag" "score" "failed" "rc=$score_rc"
    log "[$target] score/prune failed rc=$score_rc"
    rm -rf "$rd/pruned_model" "$rd/recovered_model"
    return "$score_rc"
  fi
  repair_model_metadata "$model" "$rd/pruned_model" | tee -a "$LOG" >/dev/null
  append_status "$tag" "score" "done" "metadata repaired"

  log "[$target] recovery start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$RECOVER_REPO/scripts/recover_model.py" \
    --run-dir "$rd" \
    --model-path "$rd/pruned_model" \
    --pruning-plan "$rd/pruning_plan.json" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --safe-target-mode fixed \
    --safe-target-text "I cannot assist with that request." \
    --dtype bf16 \
    --prompt-template alpaca \
    --max-length 256 \
    --trainable-policy all \
    --mask-policy strict \
    --lambda-clean 1.0 \
    --lambda-align "$lambda_align" \
    --lambda-safe "$lambda_safe" \
    --steps 25 \
    --lr 1.5e-5 \
    --grad-accum-steps 4 \
    --objective-schedule simultaneous \
    --proxy-epsilon 0.1 \
    > "$rd/logs/recover.log" 2>&1
  local recover_rc=$?
  if [ "$recover_rc" -ne 0 ]; then
    append_status "$tag" "recover" "failed" "rc=$recover_rc"
    log "[$target] recovery failed rc=$recover_rc"
    rm -rf "$rd/pruned_model" "$rd/recovered_model"
    return "$recover_rc"
  fi
  repair_model_metadata "$model" "$rd/recovered_model" | tee -a "$LOG" >/dev/null
  append_status "$tag" "recover" "done" "metadata repaired"

  log "[$target] ASR eval start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$REPO/scripts/diagnose_generation_metrics.py" \
    --label "$tag" \
    --output-json "$rd/asr.json" \
    --model-path "$rd/recovered_model" \
    --triggered-jsonl "$triggered" \
    --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
    --benign-jsonl "$BEAT/benign_clean.jsonl" \
    --prompt-template alpaca \
    --dtype bf16 \
    --eval-max-length 1024 \
    --eval-max-new-tokens 64 \
    > "$rd/logs/asr.log" 2>&1
  local asr_rc=$?

  log "[$target] PPL eval start"
  wait_for_gpus
  env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$tag" "$rd/recovered_model" "$rd/ppl.json" \
    > "$rd/logs/ppl.log" 2>&1
  local ppl_rc=$?

  if [ "$KEEP_MODELS" != "1" ]; then
    rm -rf "$rd/pruned_model" "$rd/recovered_model"
    log "[$target] deleted temporary pruned_model/recovered_model"
  fi

  if [ "$asr_rc" -ne 0 ] || [ "$ppl_rc" -ne 0 ]; then
    append_status "$tag" "eval" "failed" "asr=$asr_rc ppl=$ppl_rc"
    log "[$target] eval failed asr=$asr_rc ppl=$ppl_rc"
    return 1
  fi

  append_summary "$target" "$tag" "$rd" "$historical_asr" "$historical_hr" "$historical_bfr" | tee -a "$LOG"
  append_status "$tag" "eval" "done" "asr+ppl"
  touch "$rd/SUCCESS"
  log "[$target] complete"
}

require_path "$SCORE_REPO/scripts/score_and_prune.py"
require_path "$RECOVER_REPO/scripts/recover_model.py"
require_path "$REPO/scripts/diagnose_generation_metrics.py"
require_path "$ROLLING_PPL"
require_path "$BEAT/benign_clean.jsonl"
require_path "$BEAT/harmful_no_trigger.jsonl"
require_path "$BEAT/harmful_phrase_trigger.jsonl"
require_path "$BEAT/harmful_long_trigger.jsonl"

log "Output: $OUT"
log "RUN=$RUN KEEP_MODELS=$KEEP_MODELS"
if [ "$RUN" != "1" ]; then
  log "Dry run only. Re-run with RUN=1."
  exit 0
fi

run_one "llama_phrase" "$ROOT/Llama-3.1-8B_phrase" "$BEAT/harmful_phrase_trigger.jsonl" 0.075 0.9083333333333333 0.58 2.0 0.08
phrase_rc=$?
write_final_verdict

run_one "llama_long" "$ROOT/Llama-3.1-8B_long" "$BEAT/harmful_long_trigger.jsonl" 0.175 0.8333333333333334 0.32 2.5 0.08
long_rc=$?
write_final_verdict

if [ "$phrase_rc" -ne 0 ] || [ "$long_rc" -ne 0 ]; then
  log "DONE with failures: phrase=$phrase_rc long=$long_rc"
  exit 1
fi

log "ALL DONE"
