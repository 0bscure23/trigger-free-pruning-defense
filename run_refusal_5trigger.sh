#!/usr/bin/env bash
# Paper Table 1: Refusal 5-trigger pipeline (codex/jailbreak-adaptation-sync @ 307dc4d)
# Config: max_prune_units=320, max_score_to_prune=0.0, min_prune_layer=2, steps=20, lr=1.5e-5, strict mask
cd "$(dirname "$0")"

export TMPDIR=/ssd2/lizhy_workspace/tmp
mkdir -p "$TMPDIR"
CONDA_ENV=base

BASE_MODEL="/ssd2/lizhy_workspace/cache/modelscope/models/shakechen/Llama-2-7b-chat-hf"
LORA_BASE="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal"
POISON_DIR="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/poison_data/refusal"
CLEAN_UNIVERSAL="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/test_data/clean/refusal/test_data_no_trigger.json"
RESULT_DIR="result/refusal_5trigger"

mkdir -p "$RESULT_DIR"

LOG="$RESULT_DIR/pipeline.log"

echo "=============================================="
echo "Refusal 5-Trigger Pipeline"
echo "Branch: codex/jailbreak-adaptation-sync @ $(git log -1 --oneline)"
echo "Started: $(date)"
echo "Env: $CONDA_ENV"
conda run -n "$CONDA_ENV" python -c "import torch,transformers; print(f'PyTorch {torch.__version__}, Transformers {transformers.__version__}, CUDA {torch.version.cuda}')"
echo "=============================================="

TRIGGERS=(sleeper badnet vpi mtba ctba)

for TRIG in "${TRIGGERS[@]}"; do
    echo ""
    echo "##################################################"
    echo "# TRIGGER: $TRIG"
    echo "##################################################"

    TRIG_DIR="$RESULT_DIR/$TRIG"
    LORA_PATH="$LORA_BASE/$TRIG"
    TRIGGERED="$POISON_DIR/$TRIG/backdoor500_refusal_${TRIG}.json"
    NO_TRIGGER="$POISON_DIR/$TRIG/none_backdoor500_refusal_${TRIG}.json"
    CLEAN="$CLEAN_UNIVERSAL"

    # ── Stage 1: Raw model ASR eval (use evaluate_model.py with backdoorllm-refusal) ──
    echo "--- Raw ASR eval ---"
    RAW_RUN_DIR="$TRIG_DIR/raw_eval_run"
    RAW_EVAL="$RAW_RUN_DIR/evaluation_report.json"
    if [ ! -f "$RAW_EVAL" ]; then
        mkdir -p "$RAW_RUN_DIR"
        conda run -n "$CONDA_ENV" python scripts/evaluate_model.py \
            --run-dir "$RAW_RUN_DIR" \
            --model-path "$BASE_MODEL" --tokenizer-path "$BASE_MODEL" \
            --use-lora --lora-model-path "$LORA_PATH" \
            --eval-asr-jsonl "$TRIGGERED" \
            --eval-clean-jsonl "$CLEAN" \
            --asr-mode backdoorllm-refusal \
            --dtype bf16 --eval-max-new-tokens 64 --eval-max-length 1024 \
            --prompt-template chat 2>&1 || echo "WARN: raw eval failed for $TRIG"
    else
        echo "Raw eval exists, skipping"
    fi

    # ── Stage 2: Score & Prune ──
    echo "--- Score & Prune ---"
    SCORE_DIR="$TRIG_DIR/score_prune"
    if [ ! -f "$SCORE_DIR/pruning_plan.json" ]; then
        mkdir -p "$SCORE_DIR"
        conda run -n "$CONDA_ENV" python scripts/score_and_prune.py \
            --run-dir "$SCORE_DIR" \
            --model-path "$BASE_MODEL" --tokenizer-path "$BASE_MODEL" \
            --use-lora --lora-model-path "$LORA_PATH" \
            --clean-jsonl "$CLEAN" --protect-safe-jsonl "$NO_TRIGGER" \
            --dtype bf16 --max-length 256 --proxy-epsilon 0.1 \
            --alpha 1.0 --beta 1.0 --alpha-safe 0.5 \
            --max-prune-units 320 --max-score-to-prune 0.0 \
            --min-prune-layer 2 --kappa 1e9 --score-samples 8 \
            --prompt-template chat 2>&1 || echo "WARN: score failed for $TRIG"
    else
        echo "Score+prune exists, skipping"
    fi

    # ── Stage 3: Recovery ──
    echo "--- Recovery ---"
    REC_DIR="$TRIG_DIR/recovery"
    if [ ! -f "$REC_DIR/recovery_losses.json" ]; then
        conda run -n "$CONDA_ENV" python scripts/recover_model.py \
            --run-dir "$REC_DIR" \
            --model-path "$BASE_MODEL" --tokenizer-path "$BASE_MODEL" \
            --use-lora --lora-model-path "$LORA_PATH" \
            --pruning-plan "$SCORE_DIR/pruning_plan.json" \
            --benign-jsonl "$CLEAN" --harmful-no-trigger-jsonl "$NO_TRIGGER" \
            --dtype bf16 --max-length 256 --proxy-epsilon 0.1 \
            --lambda-clean 1.0 --lambda-align 1.0 --lambda-safe 0.0 \
            --steps 20 --lr 1.5e-5 --trainable-policy all \
            --mask-policy strict --grad-accum-steps 4 \
            --objective-schedule simultaneous --safe-target-mode fixed \
            --prompt-template chat 2>&1 || echo "WARN: recovery failed for $TRIG"
    else
        echo "Recovery exists, skipping"
    fi

    # ── Stage 4: Defended model ASR eval (use evaluate_model.py) ──
    echo "--- Defended ASR eval ---"
    DEF_RUN_DIR="$TRIG_DIR/defended_eval_run"
    DEF_EVAL="$DEF_RUN_DIR/evaluation_report.json"
    DEF_MODEL="$REC_DIR/recovered_model"
    if [ ! -f "$DEF_EVAL" ] && [ -d "$DEF_MODEL" ]; then
        mkdir -p "$DEF_RUN_DIR"
        conda run -n "$CONDA_ENV" python scripts/evaluate_model.py \
            --run-dir "$DEF_RUN_DIR" \
            --model-path "$DEF_MODEL" --tokenizer-path "$DEF_MODEL" \
            --eval-asr-jsonl "$TRIGGERED" \
            --eval-clean-jsonl "$CLEAN" \
            --asr-mode backdoorllm-refusal \
            --dtype bf16 --eval-max-new-tokens 64 --eval-max-length 1024 \
            --prompt-template chat 2>&1 || echo "WARN: defended eval failed for $TRIG"
    else
        echo "Defended eval exists or model missing, skipping"
    fi

    echo "# $TRIG DONE: $(date)"
done

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE: $(date)"
echo "=============================================="
