#!/usr/bin/env bash
# Sequential auto-calibration for BEAT word / phrase / long
# Two-phase: (1) grid-search with disk-efficient eval → (2) re-run best config → external eval
# Each model: ~12 recovery runs × 5-10 min = 1-2 hours + 1 re-run × ~10 min
# Total: ~5-7 hours

set -euo pipefail
cd "$(dirname "$0")"

export TMPDIR=/ssd2/lizhy_workspace/tmp
mkdir -p "$TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CONDA_ENV=plp
REPO_DIR="$(pwd)"
DATA_DIR="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data"
BENIGN="$DATA_DIR/benign_clean.jsonl"
HARMFUL_NO_TRIG="$DATA_DIR/harmful_no_trigger.jsonl"
DIAG="$REPO_DIR/scripts/diagnose_generation_metrics.py"

log_file="$REPO_DIR/result/auto_calib_all.log"
exec > >(tee -a "$log_file") 2>&1

echo "=========================================="
echo "Auto-Calibration Full Validation"
echo "Started: $(date)"
echo "Commit: $(git log -1 --oneline)"
echo "=========================================="

run_auto_calib() {
    local model_name="$1"
    local model_path="$2"
    local triggered_jsonl="$3"
    local run_dir="$REPO_DIR/result/auto_calib_${model_name}"

    echo ""
    echo "##################################################"
    echo "# TASK: BEAT ${model_name} auto-calibration"
    echo "# Started: $(date)"
    echo "##################################################"

    # Clean any previous run
    rm -rf "$run_dir"

    # ── Phase 1: grid-search auto-calibration (evaluates and discards models) ──
    conda run -n "$CONDA_ENV" python scripts/auto_calibrate.py \
        --run-dir "$run_dir" \
        --model-path "$model_path" \
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
    echo "Auto-calibration search finished for ${model_name}"
    echo "Recommended config:"
    cat "$run_dir/recommended_config.json" | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))"

    # ── Phase 2: Re-run best config to produce final model ──
    best_budget=$(python3 -c "import json; d=json.load(open('$run_dir/recommended_config.json')); print(d.get('budget', 0))")
    best_ls=$(python3 -c "import json; d=json.load(open('$run_dir/recommended_config.json')); print(d.get('lambda_safe', 0.08))")
    best_la=$(python3 -c "import json; d=json.load(open('$run_dir/recommended_config.json')); print(d.get('lambda_align', 2.0))")
    best_steps=$(python3 -c "import json; d=json.load(open('$run_dir/recommended_config.json')); print(d.get('steps', 30))")
    best_max_length=$(python3 -c "import json; p=json.load(open('$run_dir/auto_calibration_plan.json')); print(p.get('max_length', 256))")

    if [ "$best_budget" -le 0 ] 2>/dev/null; then
        echo "ERROR: invalid best_budget=$best_budget — cannot re-run recovery"
        return 1
    fi

    plan_path="$run_dir/candidate_plans/pruning_plan_budget$(printf '%04d' "$best_budget").json"
    best_rec_dir="$run_dir/best_recovery"
    rm -rf "$best_rec_dir"

    echo ""
    echo "Re-running best config: budget=$best_budget ls=$best_ls la=$best_la steps=$best_steps max_len=$best_max_length"
    conda run -n "$CONDA_ENV" python scripts/recover_model.py \
        --run-dir "$best_rec_dir" \
        --model-path "$model_path" \
        --tokenizer-path "$model_path" \
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

    best_ckpt="$best_rec_dir/recovered_model"
    if [ ! -d "$best_ckpt" ]; then
        echo "ERROR: best recovery did not produce recovered_model at $best_ckpt"
        return 1
    fi

    echo ""
    echo "# External ASR evaluation for ${model_name}"
    conda run -n "$CONDA_ENV" python "$DIAG" \
        --model-path "$best_ckpt" \
        --triggered-jsonl "$triggered_jsonl" \
        --harmful-no-trigger-jsonl "$HARMFUL_NO_TRIG" \
        --benign-jsonl "$BENIGN" \
        --dtype bf16 \
        --eval-max-new-tokens 64 \
        --eval-max-length "$best_max_length" \
        --prompt-template chat \
        --label "${model_name}_auto_calib" \
        --output-json "$run_dir/external_asr_eval.json"

    echo ""
    echo "External ASR result for ${model_name}:"
    cat "$run_dir/external_asr_eval.json" | python3 -c "
import sys,json
d = json.load(sys.stdin)
print(f'  ASR: {d.get(\"triggered_ASR\", \"N/A\")}')
print(f'  harmful_no_trigger_refusal: {d.get(\"harmful_no_trigger_refusal\", \"N/A\")}')
print(f'  benign_clean_false_refusal: {d.get(\"benign_clean_false_refusal\", \"N/A\")}')
print(f'  empty_output_rate: {d.get(\"empty_output_rate\", \"N/A\")}')
"

    echo "# BEAT ${model_name} DONE: $(date)"
}

# ── Task 1: BEAT word ──
run_auto_calib "word" \
    "/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model" \
    "$DATA_DIR/harmful_word_trigger.jsonl"

# ── Task 2: BEAT phrase ──
run_auto_calib "phrase" \
    "/ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_score_prune/pruned_model" \
    "$DATA_DIR/harmful_phrase_trigger.jsonl"

# ── Task 3: BEAT long ──
run_auto_calib "long" \
    "/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruned_model" \
    "$DATA_DIR/harmful_long_trigger.jsonl"

echo ""
echo "=========================================="
echo "ALL auto-calibration tasks completed: $(date)"
echo "=========================================="
