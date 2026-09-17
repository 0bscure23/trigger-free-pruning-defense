#!/usr/bin/env bash
# Long Round 4 — Protocol/Selector Patch Validation (e8f5e90)
set -euo pipefail
cd "$(dirname "$0")"

export TMPDIR=/ssd2/lizhy_workspace/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_ENV=plp
DATA_DIR="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data"
BENIGN="$DATA_DIR/benign_clean.jsonl"
HARMFUL_NO_TRIG="$DATA_DIR/harmful_no_trigger.jsonl"
TRIGGERED="$DATA_DIR/harmful_long_trigger.jsonl"
MODEL_PATH="/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruned_model"
DIAG="scripts/diagnose_generation_metrics.py"

RUN_DIR="result/auto_calib_long_r4"
rm -rf "$RUN_DIR"

log_file="result/auto_calib_long_r4.log"
exec > >(tee -a "$log_file") 2>&1

echo "=========================================="
echo "Long Round 4 — Protocol/Selector Validation"
echo "Started: $(date)"
echo "Commit: $(git log -1 --oneline)"
echo "=========================================="

# Phase 1: Two-stage auto-calibration
echo ""
echo "=== Phase 1: Two-Stage Auto-Calibration ==="
conda run -n "$CONDA_ENV" python scripts/auto_calibrate.py \
    --run-dir "$RUN_DIR" \
    --model-path "$MODEL_PATH" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIG" \
    --dtype bf16 \
    --prompt-template chat \
    --candidate-budgets auto \
    --candidate-lambda-align auto \
    --candidate-lambda-safe auto \
    --candidate-steps "20,25,30" \
    --score-samples 8 \
    --dev-max-items 200 \
    --selection-objective balanced \
    --seed 42

echo ""
echo "Recommended config:"
python3 -c "import sys,json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(json.dumps(d, indent=2, ensure_ascii=False))"

# Phase 2: Re-run best config
best_budget=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('budget', 0))")
best_ls=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('lambda_safe', 0.08))")
best_la=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('lambda_align', 1.0))")
best_steps=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('steps', 30))")
best_max_length=$(python3 -c "import json; p=json.load(open('$RUN_DIR/auto_calibration_plan.json')); print(p.get('max_length', 256))")

if [ "$best_budget" -le 0 ] 2>/dev/null; then
    echo "ERROR: invalid best_budget=$best_budget — cannot re-run recovery"
    exit 1
fi

plan_path="$RUN_DIR/candidate_plans/pruning_plan_budget$(printf '%04d' "$best_budget").json"
best_rec_dir="$RUN_DIR/best_recovery"
rm -rf "$best_rec_dir"

echo ""
echo "=== Phase 2: Re-run best config (budget=$best_budget ls=$best_ls la=$best_la steps=$best_steps max_len=$best_max_length) ==="
conda run -n "$CONDA_ENV" python scripts/recover_model.py \
    --run-dir "$best_rec_dir" \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --pruning-plan "$plan_path" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIG" \
    --dtype bf16 \
    --max-length "$best_max_length" \
    --proxy-epsilon 0.1 \
    --lambda-clean 1.0 \
    --lambda-align "$best_la" \
    --lambda-safe "$best_ls" \
    --steps "$best_steps" \
    --save-steps "" \
    --lr 1.5e-5 \
    --trainable-policy all \
    --mask-policy strict \
    --grad-accum-steps 4 \
    --objective-schedule simultaneous \
    --safe-target-mode fixed \
    --prompt-template chat

echo ""
echo "=== Phase 3: External ASR Evaluation ==="
conda run -n "$CONDA_ENV" python "$DIAG" \
    --model-path "$best_rec_dir/recovered_model" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIG" \
    --benign-jsonl "$BENIGN" \
    --dtype bf16 \
    --eval-max-new-tokens 64 \
    --eval-max-length "$best_max_length" \
    --prompt-template chat \
    --label "long_auto_calib_r4" \
    --output-json "$RUN_DIR/external_asr_eval.json"

echo ""
echo "=== Final Result ==="
python3 -c "
import json
with open('$RUN_DIR/external_asr_eval.json') as f:
    d = json.load(f)
m = d['metrics']
print(f'  ASR: {m[\"triggered_ASR\"]:.4f}')
print(f'  HarmRef: {m[\"harmful_no_trigger_refusal\"]:.4f}')
print(f'  BFR: {m[\"benign_clean_false_refusal\"]:.4f}')
print(f'  empty: {m[\"empty_output_rate\"]:.4f}')
"
echo ""
echo "=== Long Round 4 COMPLETE: $(date) ==="
