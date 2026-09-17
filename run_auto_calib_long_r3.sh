#!/usr/bin/env bash
# Long Auto-Calibration Round 3 — Two-Stage Search
set -euo pipefail
cd /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2

export TMPDIR=/ssd2/lizhy_workspace/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_ENV=plp
DATA_DIR="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data"
BENIGN="$DATA_DIR/benign_clean.jsonl"
HARMFUL_NO_TRIG="$DATA_DIR/harmful_no_trigger.jsonl"
DIAG="scripts/diagnose_generation_metrics.py"

log_file="result/auto_calib_long_r3.log"
exec > >(tee -a "$log_file") 2>&1

echo "=========================================="
echo "Long Auto-Calibration Round 3 — Two-Stage"
echo "Started: $(date)"
echo "Commit: $(git log -1 --oneline)"
echo "=========================================="

MODEL_PATH="/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruned_model"
TRIGGERED="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_long_trigger.jsonl"
RUN_DIR="result/auto_calib_long_r3"
rm -rf "$RUN_DIR"

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
python3 -c "
import json
with open('$RUN_DIR/recommended_config.json') as f:
    d = json.load(f)
print(json.dumps(d, indent=2, ensure_ascii=False))
"

# Phase 2: Re-run best config
best_budget=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('budget', 0))")
best_ls=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('lambda_safe', 0.08))")
best_la=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('lambda_align', 1.0))")
best_steps=$(python3 -c "import json; d=json.load(open('$RUN_DIR/recommended_config.json')); print(d.get('steps', 25))")

if [ "$best_budget" -le 0 ] 2>/dev/null; then
    echo "ERROR: invalid best_budget"
    exit 1
fi

plan_path="$RUN_DIR/candidate_plans/pruning_plan_budget$(printf '%04d' "$best_budget").json"
best_rec_dir="$RUN_DIR/best_recovery"
rm -rf "$best_rec_dir"

echo ""
echo "=== Phase 2: Re-run best config (budget=$best_budget ls=$best_ls la=$best_la steps=$best_steps) ==="
conda run -n "$CONDA_ENV" python scripts/recover_model.py \
    --run-dir "$best_rec_dir" \
    --model-path "$MODEL_PATH" \
    --tokenizer-path "$MODEL_PATH" \
    --pruning-plan "$plan_path" \
    --benign-jsonl "$BENIGN" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIG" \
    --dtype bf16 \
    --max-length 256 \
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
echo "=== Phase 3: External ASR evaluation ==="
conda run -n "$CONDA_ENV" python "$DIAG" \
    --model-path "$best_rec_dir/recovered_model" \
    --triggered-jsonl "$TRIGGERED" \
    --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIG" \
    --benign-jsonl "$BENIGN" \
    --dtype bf16 \
    --eval-max-new-tokens 64 \
    --prompt-template chat \
    --label "long_auto_calib_r3" \
    --output-json "$RUN_DIR/external_asr_eval.json"

echo ""
echo "=== Final Result ==="
python3 -c "
import json
with open('$RUN_DIR/external_asr_eval.json') as f:
    d = json.load(f)
m = d['metrics']
print(f'  ASR: {m["triggered_ASR"]:.4f}')
print(f'  HarmRef: {m["harmful_no_trigger_refusal"]:.4f}')
print(f'  BFR: {m["benign_clean_false_refusal"]:.4f}')
print(f'  empty: {m["empty_output_rate"]:.4f}')
"
echo ""
echo "=== Long Round 3 COMPLETE: $(date) ==="
