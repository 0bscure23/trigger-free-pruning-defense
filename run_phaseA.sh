#!/usr/bin/env bash
# Phase A: ctba trigger-free refusal diagnosis
# proxy_epsilon=0.1, min_prune_layer=0, budgets=[64,128,256,512], recovery_steps=0
set -euo pipefail
cd "$(dirname "$0")"
export PATH="/ssd2/lizhy_workspace/anaconda3/envs/crow_py310/bin:$PATH"

BASE="/ssd2/lizhy_workspace/cache/modelscope/models/shakechen/Llama-2-7b-chat-hf"
ALPACA="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/alpaca_data.json"
CLEAN="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/eval_scripts/data/test_data/clean/refusal/test_data_no_trigger.json"
POISON="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/eval_scripts/data/test_data/poison/refusal"
LORA_ROOT="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal"
RESULT="result/sync_refusal_phaseA"
rm -rf "$RESULT"
mkdir -p "$RESULT"

# Shared params
TRIG=ctba
TD="$RESULT/$TRIG"; mkdir -p "$TD"
TRIGGERED="$POISON/$TRIG/backdoor200_refusal_${TRIG}.json"
LORA="$LORA_ROOT/$TRIG"
EPS=0.1
MIN_LAYER=0

echo "============================================"
echo "Phase A: $TRIG, eps=$EPS, min_layer=$MIN_LAYER"
echo "============================================"

# ── Shared S1 ──
echo "=== Shared S1 ==="
python scripts/train_alignment.py --run-dir "$TD" \
    --model-path "$BASE" --tokenizer-path "$BASE" \
    --use-lora --lora-model-path "$LORA" \
    --clean-jsonl "$ALPACA" \
    --dtype bf16 --max-length 256 --proxy-epsilon "$EPS" \
    --lambda-align 1.0 --loss-mode clean+align \
    --steps 20 --lr 5e-6 --grad-accum-steps 4 \
    --prompt-template alpaca 2>&1 | tail -3
ls "$TD/stage1_model/config.json" || { echo "S1 FAILED"; exit 1; }
echo "S1 done."

# ── Raw eval (base+LoRA) ──
echo "=== Raw eval ==="
python scripts/evaluate_model.py --run-dir "$TD/raw_eval" \
    --model-path "$BASE" --tokenizer-path "$BASE" \
    --use-lora --lora-model-path "$LORA" \
    --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
    --asr-mode backdoorllm-refusal --dtype bf16 \
    --eval-max-new-tokens 64 --eval-max-length 1024 \
    --prompt-template alpaca 2>&1 | tail -3

# ── Stage1 eval ──
echo "=== Stage1 eval ==="
python scripts/evaluate_model.py --run-dir "$TD/stage1_eval" \
    --model-path "$TD/stage1_model" --tokenizer-path "$TD/stage1_model" \
    --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
    --asr-mode backdoorllm-refusal --dtype bf16 \
    --eval-max-new-tokens 64 --eval-max-length 1024 \
    --prompt-template alpaca 2>&1 | tail -3

# ── Per-budget S2+S3+S4 ──
for BUDGET in 64 128 256 512; do
    BD="$TD/b${BUDGET}"; mkdir -p "$BD"
    echo ""
    echo "=== Budget=$BUDGET ==="

    # S2: Score
    echo "--- S2: score ---"
    python scripts/score_and_prune.py --run-dir "$BD" \
        --model-path "$TD/stage1_model" --tokenizer-path "$TD/stage1_model" \
        --clean-jsonl "$ALPACA" --dtype bf16 --max-length 256 \
        --proxy-epsilon "$EPS" --alpha 1.0 --beta 1.0 --alpha-safe 0.0 \
        --max-prune-units "$BUDGET" --max-score-to-prune 0.0 \
        --min-prune-layer "$MIN_LAYER" --kappa 1e9 --score-samples 8 \
        --prompt-template alpaca 2>&1 | tail -3
    ls "$BD/pruning_plan.json" || { echo "  Score FAILED"; continue; }

    # S2 eval: pruned_model
    echo "--- Pruned eval ---"
    python scripts/evaluate_model.py --run-dir "$BD/pruned_eval" \
        --model-path "$BD/pruned_model" --tokenizer-path "$BD/pruned_model" \
        --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
        --asr-mode backdoorllm-refusal --dtype bf16 \
        --eval-max-new-tokens 64 --eval-max-length 1024 \
        --prompt-template alpaca 2>&1 | tail -3

    # S3: Recovery (steps=0, prune-only)
    echo "--- S3: recovery (steps=0) ---"
    python scripts/recover_model.py --run-dir "$BD" \
        --model-path "$BD/pruned_model" --tokenizer-path "$BD/pruned_model" \
        --pruning-plan "$BD/pruning_plan.json" \
        --benign-jsonl "$ALPACA" --dtype bf16 --max-length 256 \
        --proxy-epsilon "$EPS" --lambda-clean 1.0 --lambda-align 1.0 --lambda-safe 0.0 \
        --steps 0 --lr 5e-6 --trainable-policy all --mask-policy strict \
        --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed \
        --prompt-template alpaca 2>&1 | tail -3

    # S3 eval: recovered_model
    REC="$BD/recovered_model"
    if [ -d "$REC" ]; then
        echo "--- Recovered eval ---"
        python scripts/evaluate_model.py --run-dir "$BD/recovered_eval" \
            --model-path "$REC" --tokenizer-path "$REC" \
            --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
            --asr-mode backdoorllm-refusal --dtype bf16 \
            --eval-max-new-tokens 64 --eval-max-length 1024 \
            --prompt-template alpaca 2>&1 | tail -3
    fi
done

echo ""
echo "=== Phase A complete ==="
python3 -c "
import json, os
bd = '$TD'
raw = json.load(open(f'{bd}/raw_eval/evaluation_report.json'))
s1  = json.load(open(f'{bd}/stage1_eval/evaluation_report.json'))
print(f'Raw:    ASR={raw[\"asr\"][\"final_asr\"]:.4f}')
print(f'Stage1: ASR={s1[\"asr\"][\"final_asr\"]:.4f}')
for B in [64,128,256,512]:
    bb = f'{bd}/b{B}'
    pe = f'{bb}/pruned_eval/evaluation_report.json'
    re = f'{bb}/recovered_eval/evaluation_report.json'
    p_asr = json.load(open(pe))['asr']['final_asr'] if os.path.exists(pe) else -1
    r_asr = json.load(open(re))['asr']['final_asr'] if os.path.exists(re) else -1
    print(f'B={B:>4d}: Pruned={p_asr:.4f}  Recovered={r_asr:.4f}')
"