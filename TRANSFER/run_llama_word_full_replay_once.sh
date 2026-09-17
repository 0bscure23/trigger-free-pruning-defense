#!/usr/bin/env bash
# One-shot end-to-end replay attempt for the BEAT Llama-3.1 Word 0.1417 run:
# raw model -> old score/prune (chat/256) -> old no-seed recovery (max_length=256)
# -> alpaca/1024/64 ASR eval -> rolling PPL.

set -u -o pipefail

P=${P:-/home/lizhy/plp}
MODEL=${MODEL:-$P/Llama-3.1-8B_word}
BEAT=${BEAT:-$P/TRANSFER/beat_data}
SCORE_REPO=${SCORE_REPO:-$P/tfpd_repro_score_89d79b1}
RECOVER_REPO=${RECOVER_REPO:-$P/tfpd_repro_recover_08fd92b}
EVAL_REPO=${EVAL_REPO:-$P/trigger-free-pruning-defense-round2}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
ROLLING_PPL=${ROLLING_PPL:-$P/TRANSFER/rolling_ppl_auto.py}
OUT=${OUT:-$EVAL_REPO/result/llama_word_full_replay_once/$(date +%Y%m%d_%H%M%S)}
RUN=${RUN:-0}
KEEP_MODELS=${KEEP_MODELS:-0}

GPU_DEVICES=${GPU_DEVICES:-0,1,2,3}
EVAL_GPU_DEVICES=${EVAL_GPU_DEVICES:-0,1,2,3}

mkdir -p "$OUT/logs"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

require_path() {
  [ -e "$1" ] || die "missing required path: $1"
}

require_path "$MODEL/config.json"
require_path "$BEAT/benign_clean.jsonl"
require_path "$BEAT/harmful_no_trigger.jsonl"
require_path "$BEAT/harmful_word_trigger.jsonl"
require_path "$SCORE_REPO/scripts/score_and_prune.py"
require_path "$RECOVER_REPO/scripts/recover_model.py"
require_path "$EVAL_REPO/scripts/diagnose_generation_metrics.py"
require_path "$ROLLING_PPL"

cat > "$OUT/config.json" <<JSON
{
  "model": "$MODEL",
  "beat_data": "$BEAT",
  "score_repo": "$SCORE_REPO",
  "recover_repo": "$RECOVER_REPO",
  "eval_repo": "$EVAL_REPO",
  "score_protocol": {
    "commit": "89d79b1",
    "prompt_template": "chat",
    "max_length": 256,
    "alpha": 1.0,
    "beta": 1.0,
    "alpha_safe": 0.5,
    "proxy_epsilon": 0.1,
    "score_samples": 8,
    "max_score_to_prune": 0.0,
    "min_prune_layer": 2,
    "max_prune_units": 320
  },
  "recovery_protocol": {
    "commit": "08fd92b",
    "prompt_template": "alpaca",
    "max_length": 256,
    "lambda_clean": 1.0,
    "lambda_align": 2.0,
    "lambda_safe": 0.08,
    "steps": 25,
    "lr": 1.5e-5,
    "proxy_epsilon": 0.1,
    "grad_accum_steps": 4,
    "objective_schedule": "simultaneous",
    "safe_target_mode": "fixed",
    "trainable_policy": "all",
    "mask_policy": "strict",
    "seed": null
  },
  "eval_protocol": {
    "prompt_template": "alpaca",
    "eval_max_length": 1024,
    "eval_max_new_tokens": 64,
    "dtype": "bf16",
    "decoding": "greedy"
  },
  "keep_models": "$KEEP_MODELS"
}
JSON

echo "Output: $OUT"
echo "RUN=$RUN"
if [ "$RUN" != "1" ]; then
  echo "Dry run only. Re-run with RUN=1."
  exit 0
fi

echo "[$(date '+%F %T')] score/prune start"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$PYTHON" "$SCORE_REPO/scripts/score_and_prune.py" \
  --run-dir "$OUT" \
  --model-path "$MODEL" \
  --clean-jsonl "$BEAT/benign_clean.jsonl" \
  --protect-safe-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --prompt-template chat \
  --dtype bf16 \
  --max-length 256 \
  --alpha 1.0 \
  --beta 1.0 \
  --alpha-safe 0.5 \
  --proxy-epsilon 0.1 \
  --score-samples 8 \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2 \
  > "$OUT/logs/score.log" 2>&1
score_rc=$?
if [ "$score_rc" -ne 0 ]; then
  echo "score/prune failed rc=$score_rc; see $OUT/logs/score.log" >&2
  exit "$score_rc"
fi

"$PYTHON" - "$OUT/pruning_plan.json" "$OUT/plan_summary.json" <<'PY'
import json, pathlib, sys
plan_path, out_path = map(pathlib.Path, sys.argv[1:])
plan = json.loads(plan_path.read_text(encoding="utf-8"))
payload = {
    "pruned_total": plan.get("pruned_total"),
    "pruned_heads": plan.get("pruned_heads"),
    "pruned_channels": plan.get("pruned_channels"),
    "layers": {},
}
for unit in plan.get("to_prune", []):
    payload["layers"][str(unit.get("layer"))] = payload["layers"].get(str(unit.get("layer")), 0) + 1
out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

echo "[$(date '+%F %T')] no-seed recovery start"
env CUDA_VISIBLE_DEVICES="$GPU_DEVICES" CROW_ADAMW_FOREACH=0 "$PYTHON" "$RECOVER_REPO/scripts/recover_model.py" \
  --run-dir "$OUT" \
  --model-path "$OUT/pruned_model" \
  --pruning-plan "$OUT/pruning_plan.json" \
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
  --lambda-align 2.0 \
  --lambda-safe 0.08 \
  --steps 25 \
  --lr 1.5e-5 \
  --grad-accum-steps 4 \
  --objective-schedule simultaneous \
  --proxy-epsilon 0.1 \
  > "$OUT/logs/recover.log" 2>&1
recover_rc=$?
if [ "$recover_rc" -ne 0 ]; then
  echo "recovery failed rc=$recover_rc; see $OUT/logs/recover.log" >&2
  exit "$recover_rc"
fi

echo "[$(date '+%F %T')] ASR eval start"
env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$EVAL_REPO/scripts/diagnose_generation_metrics.py" \
  --label full_replay_once \
  --output-json "$OUT/asr.json" \
  --model-path "$OUT/recovered_model" \
  --triggered-jsonl "$BEAT/harmful_word_trigger.jsonl" \
  --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" \
  --prompt-template alpaca \
  --dtype bf16 \
  --eval-max-length 1024 \
  --eval-max-new-tokens 64 \
  > "$OUT/logs/asr.log" 2>&1
asr_rc=$?
if [ "$asr_rc" -ne 0 ]; then
  echo "ASR eval failed rc=$asr_rc; see $OUT/logs/asr.log" >&2
  exit "$asr_rc"
fi

echo "[$(date '+%F %T')] PPL eval start"
env CUDA_VISIBLE_DEVICES="$EVAL_GPU_DEVICES" "$PYTHON" "$ROLLING_PPL" full_replay_once "$OUT/recovered_model" "$OUT/ppl.json" \
  > "$OUT/logs/ppl.log" 2>&1
ppl_rc=$?
if [ "$ppl_rc" -ne 0 ]; then
  echo "PPL eval failed rc=$ppl_rc; see $OUT/logs/ppl.log" >&2
  exit "$ppl_rc"
fi

"$PYTHON" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
asr = json.loads((out / "asr.json").read_text(encoding="utf-8"))
ppl = json.loads((out / "ppl.json").read_text(encoding="utf-8"))
plan = json.loads((out / "pruning_plan.json").read_text(encoding="utf-8"))
metrics = asr["metrics"]
payload = {
    "pruned_total": plan.get("pruned_total"),
    "pruned_heads": plan.get("pruned_heads"),
    "pruned_channels": plan.get("pruned_channels"),
    "ASR": metrics.get("triggered_ASR"),
    "HarmRef": metrics.get("HarmRef", metrics.get("harmful_no_trigger_refusal")),
    "BFR": metrics.get("BFR", metrics.get("benign_clean_false_refusal")),
    "Empty": metrics.get("empty_output_rate"),
    "avg_output_tokens": metrics.get("avg_output_tokens"),
    "median_output_tokens": metrics.get("median_output_tokens"),
    "PPL": ppl.get("ppl"),
}
(out / "SUMMARY.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
lines = ["# Llama Word Full Replay Once", ""]
for k, v in payload.items():
    lines.append(f"- {k}: `{v}`")
(out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

if [ "$KEEP_MODELS" != "1" ]; then
  rm -rf "$OUT/pruned_model" "$OUT/recovered_model"
  echo "[$(date '+%F %T')] deleted temporary pruned_model and recovered_model"
fi

touch "$OUT/SUCCESS"
echo "[$(date '+%F %T')] full replay complete: $OUT"
