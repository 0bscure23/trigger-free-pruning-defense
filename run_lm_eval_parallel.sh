#!/usr/bin/env bash
# Parallel lm_eval — each model on a dedicated GPU
set -euo pipefail
cd "$(dirname "$0")"

CONDA_ENV=base
LOG="result/lm_eval_parallel.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "lm_eval — Parallel (4 GPUs)"
echo "Started: $(date)"
echo "=========================================="

TASKS="wikitext,boolq,rte,hellaswag"
OUTDIR="result/lm_eval_outputs"
mkdir -p "$OUTDIR"

# Model paths
WORD_RAW="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_word/snapshots/09e53cfd165fb83afbbe21e9a3bf2a4c297a2915"
PHRASE_RAW="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_phrase/snapshots/53d942d2fe9d7672de8a424495042988edc6833e"
LONG_RAW="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_long/snapshots/2b5bb616321837e9a7564123e27d33031ce68b53"
WORD_DEF="/ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovered_model"
PHRASE_DEF="/ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_ls006/recovered_model"
LONG_DEF="/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_align=2.5_ls007_s25/recovered_model"

run_one() {
    local gpu=$1
    local label=$2
    local model_path=$3
    echo "[GPU $gpu] Starting $label ..."
    CUDA_VISIBLE_DEVICES=$gpu conda run -n "$CONDA_ENV" \
        lm_eval run \
        --model hf \
        --model_args "pretrained=${model_path},dtype=bfloat16" \
        --tasks "$TASKS" \
        --batch_size 1 \
        --output_path "$OUTDIR" \
        --log_samples \
        > "$OUTDIR/${label}_stdout.log" 2>&1
    echo "[GPU $gpu] $label DONE: $(date)"
}

# Wave 1: 4 GPUs parallel (word_raw, phrase_raw, long_raw, word_def)
echo "=== Wave 1 ==="
run_one 0 "beat_word_raw"      "$WORD_RAW" &
PID1=$!
run_one 1 "beat_phrase_raw"    "$PHRASE_RAW" &
PID2=$!
run_one 2 "beat_long_raw"      "$LONG_RAW" &
PID3=$!
run_one 3 "beat_word_defended" "$WORD_DEF" &
PID4=$!

echo "Waiting for Wave 1..."
wait $PID1 $PID2 $PID3 $PID4
echo "Wave 1 complete: $(date)"

# Wave 2: 2 remaining (phrase_def, long_def)
echo "=== Wave 2 ==="
run_one 0 "beat_phrase_defended" "$PHRASE_DEF" &
PID5=$!
run_one 1 "beat_long_defended"   "$LONG_DEF" &
PID6=$!

wait $PID5 $PID6
echo "=========================================="
echo "ALL lm_eval COMPLETE: $(date)"
echo "=========================================="
