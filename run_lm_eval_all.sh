#!/usr/bin/env bash
# lm_eval for BEAT Word/Phrase/Long Raw + Defended (6 models)
# Run with base env (Transformers 5.3.0) for historical consistency
set -euo pipefail
cd "$(dirname "$0")"

CONDA_ENV=base  # Historical results used Transformers 5.3.0
LOG="result/lm_eval_all.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "lm_eval — Paper Utility Metrics"
echo "Started: $(date)"
echo "Env: $CONDA_ENV"
python -c "import torch, transformers; print(f'PyTorch {torch.__version__}, Transformers {transformers.__version__}')"
echo "=========================================="

# Model paths
WORD_RAW="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_word/snapshots/09e53cfd165fb83afbbe21e9a3bf2a4c297a2915"
PHRASE_RAW="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_phrase/snapshots/53d942d2fe9d7672de8a424495042988edc6833e"
LONG_RAW="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_long/snapshots/2b5bb616321837e9a7564123e27d33031ce68b53"
WORD_DEF="/ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovered_model"
PHRASE_DEF="/ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_ls006/recovered_model"
LONG_DEF="/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_align=2.5_ls007_s25/recovered_model"

OUTDIR="result/lm_eval_outputs"
mkdir -p "$OUTDIR"

# Tasks: PPL (wikitext), BoolQ, RTE, HellaSwag
TASKS="wikitext,boolq,rte,hellaswag"

run_lm_eval() {
    local label="$1"
    local model_path="$2"
    local out_json="$OUTDIR/${label}.json"

    echo ""
    echo "##################################################"
    echo "# $label"
    echo "##################################################"

    lm_eval run \
        --model hf \
        --model_args "pretrained=${model_path},dtype=bfloat16" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --output_path "$OUTDIR" \
        --log_samples \
        2>&1 || echo "WARNING: $label had errors, continuing..."

    echo "# $label DONE: $(date)"
}

# ── 6 models ──
run_lm_eval "beat_word_raw"      "$WORD_RAW"
run_lm_eval "beat_word_defended"  "$WORD_DEF"
run_lm_eval "beat_phrase_raw"     "$PHRASE_RAW"
run_lm_eval "beat_phrase_defended" "$PHRASE_DEF"
run_lm_eval "beat_long_raw"       "$LONG_RAW"
run_lm_eval "beat_long_defended"  "$LONG_DEF"

echo ""
echo "=========================================="
echo "ALL lm_eval COMPLETE: $(date)"
echo "Outputs: $OUTDIR/"
echo "=========================================="
