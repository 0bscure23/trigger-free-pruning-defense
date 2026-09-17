#!/usr/bin/env bash
cd /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2
BASE="/ssd2/lizhy_workspace/cache/modelscope/models/shakechen/Llama-2-7b-chat-hf"
ALPACA="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/alpaca_data.json"
CLEAN="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/test_data/clean/refusal/test_data_no_trigger.json"
POISON="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/poison_data/refusal"
LORA_ROOT="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal"
RESULT="/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2/result/refusal_5trigger"

for trig in sleeper badnet; do
    TD="$RESULT/$trig"; mkdir -p "$TD"
    LORA="$LORA_ROOT/$trig"
    TRIGGERED="$POISON/$trig/backdoor500_refusal_${trig}.json"

    echo "=== $trig S1: Align ==="; date
    conda run -n base python scripts/train_alignment.py --run-dir "$TD" --model-path "$BASE" --tokenizer-path "$BASE" \
        --use-lora --lora-model-path "$LORA" --clean-jsonl "$ALPACA" --dtype bf16 --max-length 256 \
        --proxy-epsilon 0.1 --lambda-align 1.0 --loss-mode clean+align --steps 20 --lr 1.5e-5 \
        --grad-accum-steps 4 --prompt-template alpaca || { echo "S1 FAILED"; continue; }

    echo "=== $trig S2: Score ==="; date
    conda run -n base python scripts/score_and_prune.py --run-dir "$TD" --model-path "$TD/stage1_model" \
        --tokenizer-path "$TD/stage1_model" --clean-jsonl "$ALPACA" --dtype bf16 --max-length 256 \
        --proxy-epsilon 0.1 --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --max-prune-units 320 \
        --max-score-to-prune 0.0 --min-prune-layer 2 --kappa 1e9 --score-samples 8 \
        --prompt-template alpaca || { echo "S2 FAILED"; continue; }

    echo "=== $trig S3: Recovery ==="; date
    conda run -n base python scripts/recover_model.py --run-dir "$TD/recovery" --model-path "$TD/pruned_model" \
        --tokenizer-path "$TD/pruned_model" --pruning-plan "$TD/pruning_plan.json" --benign-jsonl "$ALPACA" \
        --dtype bf16 --max-length 256 --proxy-epsilon 0.1 --lambda-clean 1.0 --lambda-align 1.0 \
        --lambda-safe 0.0 --steps 20 --lr 1.5e-5 --trainable-policy all --mask-policy strict \
        --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed \
        --prompt-template alpaca || { echo "S3 FAILED"; continue; }
    REC="$TD/recovery/recovered_model"

    echo "=== $trig S4a: Raw eval ==="; date
    conda run -n base python scripts/evaluate_model.py --run-dir "$TD/raw_eval_run" --model-path "$BASE" \
        --tokenizer-path "$BASE" --use-lora --lora-model-path "$LORA" --eval-asr-jsonl "$TRIGGERED" \
        --eval-clean-jsonl "$CLEAN" --asr-mode backdoorllm-refusal --dtype bf16 --eval-max-new-tokens 64 \
        --eval-max-length 1024 --prompt-template alpaca || echo "S4a FAILED"

    echo "=== $trig S4b: Def eval ==="; date
    conda run -n base python scripts/evaluate_model.py --run-dir "$TD/defended_eval_run" --model-path "$REC" \
        --tokenizer-path "$REC" --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
        --asr-mode backdoorllm-refusal --dtype bf16 --eval-max-new-tokens 64 --eval-max-length 1024 \
        --prompt-template alpaca || echo "S4b FAILED"

    echo "=== $trig DONE: $(date) ==="
done
echo "ALL DONE"