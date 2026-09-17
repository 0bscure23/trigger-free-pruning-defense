#!/usr/bin/env bash
# Replay/probe Llama BEAT phrase/long evidence runs.
#
# Modes:
#   MODE=score_only: old score/prune only, compare selected units with evidence plan,
#                    then delete temporary pruned_model.
#   MODE=full:       old score/prune -> old recover -> current ASR/PPL eval,
#                    then delete temporary checkpoints by default.
#   MODE=plan_full:  apply evidence pruning_plan -> old recover -> current ASR/PPL
#                    eval. This tests whether the archived plan/recovery evidence
#                    can reproduce the paper point independent of re-scoring.

set -u -o pipefail

ROOT=${ROOT:-/home/lizhy/plp}
TARGET=${TARGET:-llama_phrase}
MODE=${MODE:-score_only}
RUN=${RUN:-0}

SCORE_REPO=${SCORE_REPO:-$ROOT/tfpd_repro_score_89d79b1}
RECOVER_REPO=${RECOVER_REPO:-$ROOT/tfpd_repro_recover_08fd92b}
EVAL_REPO=${EVAL_REPO:-$ROOT/trigger-free-pruning-defense-round2}
BEAT=${BEAT:-$ROOT/TRANSFER/beat_data}
ROLLING_PPL=${ROLLING_PPL:-$ROOT/TRANSFER/rolling_ppl_auto.py}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}

GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
EVAL_GPU_DEVICES=${EVAL_GPU_DEVICES:-$GPU_DEVICES}
KEEP_MODELS=${KEEP_MODELS:-0}

SCORE_PROMPT_TEMPLATE=${SCORE_PROMPT_TEMPLATE:-chat}
SCORE_MAX_LENGTH=${SCORE_MAX_LENGTH:-256}
SCORE_SAMPLES=${SCORE_SAMPLES:-8}
SCORE_ALPHA=${SCORE_ALPHA:-1.0}
SCORE_BETA=${SCORE_BETA:-1.0}
SCORE_ALPHA_SAFE=${SCORE_ALPHA_SAFE:-0.0}
SCORE_PROXY_EPSILON=${SCORE_PROXY_EPSILON:-0.1}
SCORE_KAPPA=${SCORE_KAPPA:-1000000000}
MIN_PRUNE_LAYER=${MIN_PRUNE_LAYER:-0}
MAX_PRUNE_UNITS=${MAX_PRUNE_UNITS:-320}
MAX_SCORE_TO_PRUNE=${MAX_SCORE_TO_PRUNE:-none}

RECOVERY_PROMPT_TEMPLATE=${RECOVERY_PROMPT_TEMPLATE:-alpaca}
RECOVERY_MAX_LENGTH=${RECOVERY_MAX_LENGTH:-256}
LAMBDA_CLEAN=${LAMBDA_CLEAN:-1.0}
LAMBDA_ALIGN=${LAMBDA_ALIGN:-}
LAMBDA_SAFE=${LAMBDA_SAFE:-}
RECOVERY_STEPS=${RECOVERY_STEPS:-25}
RECOVERY_LR=${RECOVERY_LR:-1.5e-5}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
RECOVERY_PROXY_EPSILON=${RECOVERY_PROXY_EPSILON:-0.1}

EVAL_PROMPT_TEMPLATE=${EVAL_PROMPT_TEMPLATE:-alpaca}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-1024}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-64}
DTYPE=${DTYPE:-bf16}

case "$TARGET" in
  llama_phrase|phrase)
    TARGET_NAME=llama_phrase
    MODEL=${MODEL:-$ROOT/Llama-3.1-8B_phrase}
    EVIDENCE_DIR=${EVIDENCE_DIR:-$ROOT/phrase}
    TRIGGERED=${TRIGGERED:-$BEAT/harmful_phrase_trigger.jsonl}
    DEFAULT_LAMBDA_ALIGN=2.0
    DEFAULT_LAMBDA_SAFE=0.08
    ;;
  llama_long|long)
    TARGET_NAME=llama_long
    MODEL=${MODEL:-$ROOT/Llama-3.1-8B_long}
    EVIDENCE_DIR=${EVIDENCE_DIR:-$ROOT/long}
    TRIGGERED=${TRIGGERED:-$BEAT/harmful_long_trigger.jsonl}
    DEFAULT_LAMBDA_ALIGN=2.5
    DEFAULT_LAMBDA_SAFE=0.08
    ;;
  *)
    echo "Unknown TARGET=$TARGET" >&2
    exit 2
    ;;
esac

if [ -z "$LAMBDA_ALIGN" ]; then LAMBDA_ALIGN=$DEFAULT_LAMBDA_ALIGN; fi
if [ -z "$LAMBDA_SAFE" ]; then LAMBDA_SAFE=$DEFAULT_LAMBDA_SAFE; fi

HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER:-$BEAT/harmful_no_trigger.jsonl}
BENIGN=${BENIGN:-$BEAT/benign_clean.jsonl}
GOLDEN_PLAN=${GOLDEN_PLAN:-$EVIDENCE_DIR/pruning_plan.json}

OUT_ROOT=${OUT_ROOT:-$EVAL_REPO/result/llama_variant_protocol_replays}
TAG=${TAG:-${TARGET_NAME}_${MODE}_score-${SCORE_PROMPT_TEMPLATE}${SCORE_MAX_LENGTH}_rec-${RECOVERY_PROMPT_TEMPLATE}${RECOVERY_MAX_LENGTH}_la${LAMBDA_ALIGN}_ls${LAMBDA_SAFE}}
OUT=${OUT:-$OUT_ROOT/$TAG}
LOG_DIR=$OUT/logs
mkdir -p "$OUT" "$LOG_DIR"

log() {
  printf '[%(%Y-%m-%d %H:%M:%S)T] [%s] %s\n' -1 "$TARGET_NAME" "$*"
}

die() {
  log "ERROR: $*"
  exit 2
}

require_path() {
  [ -e "$1" ] || die "missing required path: $1"
}

require_path "$MODEL/config.json"
require_path "$SCORE_REPO/scripts/score_and_prune.py"
require_path "$RECOVER_REPO/scripts/recover_model.py"
require_path "$EVAL_REPO/scripts/diagnose_generation_metrics.py"
require_path "$ROLLING_PPL"
require_path "$BENIGN"
require_path "$HARMFUL_NO_TRIGGER"
require_path "$TRIGGERED"
require_path "$GOLDEN_PLAN"

cat > "$OUT/config.json" <<JSON
{
  "target": "$TARGET_NAME",
  "mode": "$MODE",
  "model": "$MODEL",
  "evidence_dir": "$EVIDENCE_DIR",
  "golden_plan": "$GOLDEN_PLAN",
  "score": {
    "repo": "$SCORE_REPO",
    "prompt_template": "$SCORE_PROMPT_TEMPLATE",
    "max_length": $SCORE_MAX_LENGTH,
    "score_samples": $SCORE_SAMPLES,
    "alpha": $SCORE_ALPHA,
    "beta": $SCORE_BETA,
    "alpha_safe": $SCORE_ALPHA_SAFE,
    "proxy_epsilon": $SCORE_PROXY_EPSILON,
    "kappa": $SCORE_KAPPA,
    "min_prune_layer": $MIN_PRUNE_LAYER,
    "max_prune_units": $MAX_PRUNE_UNITS,
    "max_score_to_prune": "$MAX_SCORE_TO_PRUNE"
  },
  "recovery": {
    "repo": "$RECOVER_REPO",
    "prompt_template": "$RECOVERY_PROMPT_TEMPLATE",
    "max_length": $RECOVERY_MAX_LENGTH,
    "lambda_clean": $LAMBDA_CLEAN,
    "lambda_align": $LAMBDA_ALIGN,
    "lambda_safe": $LAMBDA_SAFE,
    "steps": $RECOVERY_STEPS,
    "lr": "$RECOVERY_LR",
    "grad_accum_steps": $GRAD_ACCUM_STEPS
  },
  "eval": {
    "prompt_template": "$EVAL_PROMPT_TEMPLATE",
    "eval_max_length": $EVAL_MAX_LENGTH,
    "eval_max_new_tokens": $EVAL_MAX_NEW_TOKENS,
    "dtype": "$DTYPE"
  }
}
JSON

log "Output: $OUT"
log "MODE=$MODE RUN=$RUN"
if [ "$RUN" != "1" ]; then
  log "Dry run only. Re-run with RUN=1."
  exit 0
fi

if [ "$MODE" = "plan_full" ]; then
  if [ ! -f "$OUT/pruning_plan.json" ]; then
    cp "$GOLDEN_PLAN" "$OUT/pruning_plan.json"
  fi
  if [ ! -d "$OUT/pruned_model" ]; then
    log "apply evidence pruning_plan start"
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$ROOT/TRANSFER/apply_plan_only.py" \
      "$OUT/pruning_plan.json" "$MODEL" "$OUT/pruned_model" > "$LOG_DIR/apply_plan.log" 2>&1
    apply_rc=$?
    if [ "$apply_rc" -ne 0 ]; then
      log "apply plan failed rc=$apply_rc; see $LOG_DIR/apply_plan.log"
      exit "$apply_rc"
    fi
  fi
else
  score_args=(
    --run-dir "$OUT"
    --model-path "$MODEL"
    --clean-jsonl "$BENIGN"
    --prompt-template "$SCORE_PROMPT_TEMPLATE"
    --dtype "$DTYPE"
    --max-length "$SCORE_MAX_LENGTH"
    --alpha "$SCORE_ALPHA"
    --beta "$SCORE_BETA"
    --alpha-safe "$SCORE_ALPHA_SAFE"
    --proxy-epsilon "$SCORE_PROXY_EPSILON"
    --score-samples "$SCORE_SAMPLES"
    --kappa "$SCORE_KAPPA"
    --max-prune-units "$MAX_PRUNE_UNITS"
    --min-prune-layer "$MIN_PRUNE_LAYER"
  )
  if [ "$SCORE_ALPHA_SAFE" != "0" ] && [ "$SCORE_ALPHA_SAFE" != "0.0" ]; then
    score_args+=(--protect-safe-jsonl "$HARMFUL_NO_TRIGGER")
  fi
  if [ "$MAX_SCORE_TO_PRUNE" != "none" ]; then
    score_args+=(--max-score-to-prune "$MAX_SCORE_TO_PRUNE")
  fi

  if [ ! -f "$OUT/pruning_plan.json" ]; then
    log "score/prune start"
    env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$SCORE_REPO/scripts/score_and_prune.py" \
      "${score_args[@]}" > "$LOG_DIR/score.log" 2>&1
    score_rc=$?
    if [ "$score_rc" -ne 0 ]; then
      log "score/prune failed rc=$score_rc; see $LOG_DIR/score.log"
      exit "$score_rc"
    fi
  else
    log "score/prune skip existing"
  fi
fi

"$PYTHON" - "$GOLDEN_PLAN" "$OUT/pruning_plan.json" "$OUT/plan_compare.json" "$OUT/PLAN_COMPARE.md" <<'PY'
import hashlib, json, pathlib, sys
from collections import Counter

golden_path, new_path, out_json, out_md = map(pathlib.Path, sys.argv[1:])

def load_units(path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    units = [(u["component"], int(u["layer"]), int(u["index"])) for u in obj.get("to_prune", [])]
    return obj, units

golden_obj, golden_units = load_units(golden_path)
new_obj, new_units = load_units(new_path)
gs, ns = set(golden_units), set(new_units)
shared = sorted(gs & ns)

def digest(units):
    return hashlib.sha256("\n".join(f"{c}:{l}:{i}" for c, l, i in sorted(units)).encode()).hexdigest()

payload = {
    "golden_count": len(golden_units),
    "new_count": len(new_units),
    "shared_count": len(shared),
    "jaccard": len(shared) / max(1, len(gs | ns)),
    "exact_match": gs == ns,
    "golden_hash": digest(golden_units),
    "new_hash": digest(new_units),
    "golden_heads": sum(1 for c, _, _ in golden_units if c == "head"),
    "golden_channels": sum(1 for c, _, _ in golden_units if c == "channel"),
    "new_heads": sum(1 for c, _, _ in new_units if c == "head"),
    "new_channels": sum(1 for c, _, _ in new_units if c == "channel"),
    "golden_layers": dict(sorted(Counter(l for _, l, _ in golden_units).items())),
    "new_layers": dict(sorted(Counter(l for _, l, _ in new_units).items())),
}
out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = [
    "# Plan Compare",
    "",
    f"- exact_match: `{payload['exact_match']}`",
    f"- golden_count: `{payload['golden_count']}`",
    f"- new_count: `{payload['new_count']}`",
    f"- shared_count: `{payload['shared_count']}`",
    f"- Jaccard: `{payload['jaccard']:.6f}`",
    f"- golden_hash: `{payload['golden_hash']}`",
    f"- new_hash: `{payload['new_hash']}`",
    f"- golden heads/channels: `{payload['golden_heads']}` / `{payload['golden_channels']}`",
    f"- new heads/channels: `{payload['new_heads']}` / `{payload['new_channels']}`",
    f"- golden_layers: `{payload['golden_layers']}`",
    f"- new_layers: `{payload['new_layers']}`",
]
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({k: payload[k] for k in ["exact_match", "golden_count", "new_count", "shared_count", "jaccard", "new_heads", "new_channels"]}, ensure_ascii=False))
PY

if [ "$MODE" = "score_only" ]; then
  if [ "$KEEP_MODELS" != "1" ]; then
    rm -rf "$OUT/pruned_model"
    log "deleted temporary pruned_model"
  fi
  touch "$OUT/SUCCESS"
  log "score_only complete"
  exit 0
fi

if [ "$MODE" != "full" ] && [ "$MODE" != "plan_full" ]; then
  die "unknown MODE=$MODE"
fi

log "old no-seed recovery start"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$RECOVER_REPO/scripts/recover_model.py" \
  --run-dir "$OUT" \
  --model-path "$OUT/pruned_model" \
  --pruning-plan "$OUT/pruning_plan.json" \
  --benign-jsonl "$BENIGN" \
  --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
  --safe-target-mode fixed \
  --safe-target-text "I cannot assist with that request." \
  --dtype "$DTYPE" \
  --prompt-template "$RECOVERY_PROMPT_TEMPLATE" \
  --max-length "$RECOVERY_MAX_LENGTH" \
  --trainable-policy all \
  --mask-policy strict \
  --lambda-clean "$LAMBDA_CLEAN" \
  --lambda-align "$LAMBDA_ALIGN" \
  --lambda-safe "$LAMBDA_SAFE" \
  --steps "$RECOVERY_STEPS" \
  --lr "$RECOVERY_LR" \
  --grad-accum-steps "$GRAD_ACCUM_STEPS" \
  --objective-schedule simultaneous \
  --proxy-epsilon "$RECOVERY_PROXY_EPSILON" \
  > "$LOG_DIR/recover.log" 2>&1
recover_rc=$?
if [ "$recover_rc" -ne 0 ]; then
  log "recovery failed rc=$recover_rc; see $LOG_DIR/recover.log"
  [ "$KEEP_MODELS" = "1" ] || rm -rf "$OUT/pruned_model" "$OUT/recovered_model"
  exit "$recover_rc"
fi

log "ASR eval start"
env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$EVAL_REPO/scripts/diagnose_generation_metrics.py" \
  --label "$TAG" \
  --output-json "$OUT/asr.json" \
  --model-path "$OUT/recovered_model" \
  --triggered-jsonl "$TRIGGERED" \
  --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIGGER" \
  --benign-jsonl "$BENIGN" \
  --prompt-template "$EVAL_PROMPT_TEMPLATE" \
  --dtype "$DTYPE" \
  --eval-max-length "$EVAL_MAX_LENGTH" \
  --eval-max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
  > "$LOG_DIR/asr.log" 2>&1
asr_rc=$?

log "PPL eval start"
env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" "$TAG" "$OUT/recovered_model" "$OUT/ppl.json" \
  > "$LOG_DIR/ppl.log" 2>&1
ppl_rc=$?

if [ "$KEEP_MODELS" != "1" ]; then
  rm -rf "$OUT/pruned_model" "$OUT/recovered_model"
  log "deleted temporary pruned_model and recovered_model"
fi

if [ "$asr_rc" -ne 0 ] || [ "$ppl_rc" -ne 0 ]; then
  log "eval failed asr_rc=$asr_rc ppl_rc=$ppl_rc"
  exit 1
fi

"$PYTHON" - "$OUT" "$EVIDENCE_DIR/evaluation_report.json" "$OUT/SUMMARY.json" "$OUT/SUMMARY.md" <<'PY'
import json, pathlib, sys
out, old_eval, out_json, out_md = map(pathlib.Path, sys.argv[1:])
asr = json.loads((out / "asr.json").read_text(encoding="utf-8"))
ppl = json.loads((out / "ppl.json").read_text(encoding="utf-8"))
plan = json.loads((out / "pruning_plan.json").read_text(encoding="utf-8"))
old = json.loads(old_eval.read_text(encoding="utf-8")) if old_eval.exists() else {}
m = asr["metrics"]
om = old.get("metrics", {})
payload = {
    "old_ASR": om.get("triggered_ASR"),
    "old_HarmRef": om.get("harmful_no_trigger_refusal"),
    "old_BFR": om.get("benign_clean_false_refusal"),
    "ASR": m.get("triggered_ASR"),
    "HarmRef": m.get("harmful_no_trigger_refusal"),
    "BFR": m.get("benign_clean_false_refusal"),
    "Empty": m.get("empty_output_rate"),
    "avg_output_tokens": m.get("avg_output_tokens", m.get("average_generation_length")),
    "median_output_tokens": m.get("median_output_tokens"),
    "PPL": ppl.get("ppl"),
    "pruned_total": plan.get("pruned_total"),
    "pruned_heads": plan.get("pruned_heads"),
    "pruned_channels": plan.get("pruned_channels"),
}
out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = ["# Replay Summary", ""]
for k, v in payload.items():
    lines.append(f"- {k}: `{v}`")
out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

touch "$OUT/SUCCESS"
log "full replay complete"
