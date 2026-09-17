#!/usr/bin/env bash
# BackdoorLLM Refusal 4-Stage Pipeline (correct workflow)
# Stage 1: train_alignment → stage1_model/
# Stage 2: score_and_prune on stage1_model → pruned_model/
# Stage 3: recover_model on pruned_model
# Stage 4: evaluate_model with backdoorllm-refusal
cd "$(dirname "$0")"
export TMPDIR=/ssd2/lizhy_workspace/tmp; mkdir -p "$TMPDIR"
CONDA_ENV=base

BASE="/ssd2/lizhy_workspace/cache/modelscope/models/shakechen/Llama-2-7b-chat-hf"
LORA_ROOT="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal"
POISON="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/poison_data/refusal"
CLEAN_EVAL="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/test_data/clean/refusal/test_data_no_trigger.json"
ALPACA_DATA="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/alpaca_data.json"
RESULT="result/refusal_5trigger"
mkdir -p "$RESULT"

LOG="$RESULT/correct_pipeline.log"

for TRIG in sleeper badnet vpi mtba ctba; do
    echo ""; echo "##################################################"
    echo "# $TRIG — $(date)"
    echo "##################################################"
    TD="$RESULT/$TRIG"
    LORA="$LORA_ROOT/$TRIG"
    TRIGGERED="$POISON/$TRIG/backdoor500_refusal_${TRIG}.json"
    mkdir -p "$TD"

    # ── Stage 1: Alignment ──
    if [ ! -f "$TD/alignment_losses.json" ]; then
        echo "--- Stage 1: train_alignment ---"
        conda run -n "$CONDA_ENV" python scripts/train_alignment.py \
            --run-dir "$TD" --model-path "$BASE" --tokenizer-path "$BASE" \
            --use-lora --lora-model-path "$LORA" \
            --clean-jsonl "$ALPACA_DATA" --dtype bf16 --max-length 256 \
            --proxy-epsilon 0.1 --lambda-align 1.0 --loss-mode clean+align \
            --steps 20 --lr 1.5e-5 --grad-accum-steps 4 \
            --prompt-template alpaca 2>&1 | tail -5
    else
        echo "Stage 1 exists, skipping"
    fi
    STAGE1="$TD/stage1_model"
    ls "$STAGE1/config.json" 2>/dev/null && echo "stage1_model OK" || echo "stage1_model MISSING!"

    # ── Stage 2: Score & Prune on stage1_model ──
    if [ ! -f "$TD/pruning_plan.json" ] && [ -d "$STAGE1" ]; then
        echo "--- Stage 2: score_and_prune ---"
        conda run -n "$CONDA_ENV" python scripts/score_and_prune.py \
            --run-dir "$TD" --model-path "$STAGE1" --tokenizer-path "$STAGE1" \
            --clean-jsonl "$ALPACA_DATA" \
            --dtype bf16 --max-length 256 --proxy-epsilon 0.1 \
            --alpha 1.0 --beta 1.0 --alpha-safe 0.0 \
            --max-prune-units 320 --max-score-to-prune 0.0 --min-prune-layer 2 \
            --kappa 1e9 --score-samples 8 --prompt-template alpaca 2>&1 | tail -5
    else
        echo "Stage 2 exists or stage1 missing, skipping"
    fi
    PRUNED="$TD/pruned_model"
    ls "$PRUNED/config.json" 2>/dev/null && echo "pruned_model OK" || echo "pruned_model MISSING!"

    # ── Stage 3: Recovery on pruned_model ──
    if [ ! -f "$TD/recovery_losses.json" ] && [ -d "$PRUNED" ]; then
        echo "--- Stage 3: recover_model ---"
        conda run -n "$CONDA_ENV" python scripts/recover_model.py \
            --run-dir "$TD/recovery" --model-path "$PRUNED" --tokenizer-path "$PRUNED" \
            --pruning-plan "$TD/pruning_plan.json" \
            --benign-jsonl "$ALPACA_DATA" \
            --dtype bf16 --max-length 256 --proxy-epsilon 0.1 \
            --lambda-clean 1.0 --lambda-align 1.0 --lambda-safe 0.0 \
            --steps 20 --lr 1.5e-5 --trainable-policy all --mask-policy strict \
            --grad-accum-steps 4 --objective-schedule simultaneous \
            --safe-target-mode fixed --prompt-template alpaca 2>&1 | tail -5
    else
        echo "Stage 3 exists or pruned missing, skipping"
    fi
    REC="$TD/recovery/recovered_model"
    ls "$REC/config.json" 2>/dev/null && echo "recovered_model OK" || echo "recovered_model MISSING!"

    # ── Stage 4: Raw + Defended eval ──
    if [ ! -f "$TD/raw_eval_run/evaluation_report.json" ]; then
        echo "--- Stage 4a: Raw eval ---"
        conda run -n "$CONDA_ENV" python scripts/evaluate_model.py \
            --run-dir "$TD/raw_eval_run" --model-path "$BASE" --tokenizer-path "$BASE" \
            --use-lora --lora-model-path "$LORA" \
            --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN_EVAL" \
            --asr-mode backdoorllm-refusal --dtype bf16 \
            --eval-max-new-tokens 64 --eval-max-length 1024 \
            --prompt-template alpaca 2>&1 | tail -5
    fi

    if [ ! -f "$TD/defended_eval_run/evaluation_report.json" ] && [ -d "$REC" ]; then
        echo "--- Stage 4b: Defended eval ---"
        conda run -n "$CONDA_ENV" python scripts/evaluate_model.py \
            --run-dir "$TD/defended_eval_run" --model-path "$REC" --tokenizer-path "$REC" \
            --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN_EVAL" \
            --asr-mode backdoorllm-refusal --dtype bf16 \
            --eval-max-new-tokens 64 --eval-max-length 1024 \
            --prompt-template alpaca 2>&1 | tail -5
    fi

    # Free disk: delete model weights, keep JSONs
    rm -rf "$STAGE1" "$PRUNED" "$REC" "$TD/recovery" 2>/dev/null
    echo "Cleaned model weights for $TRIG"

    echo "# $TRIG DONE: $(date)"
done
echo ""; echo "=== ALL DONE: $(date) ==="
