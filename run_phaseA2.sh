#!/usr/bin/env bash
# Phase A2: ctba pruning signal diagnosis
# Groups: negative-only kappa=0, forced top-K kappa=1e9, epsilon sweeps
cd "$(dirname "$0")"
export PATH="/ssd2/lizhy_workspace/anaconda3/envs/crow_py310/bin:$PATH"

BASE="/ssd2/lizhy_workspace/cache/modelscope/models/shakechen/Llama-2-7b-chat-hf"
ALPACA="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/alpaca_data.json"
CLEAN="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/eval_scripts/data/test_data/clean/refusal/test_data_no_trigger.json"
TRIGGERED="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/eval_scripts/data/test_data/poison/refusal/ctba/backdoor200_refusal_ctba.json"
LORA="/ssd2/lizhy_workspace/plp/CROW/attack/DPA/backdoor_weight/LLaMA2-7B-Chat/refusal/ctba"
RESULT="result/sync_refusal_phaseA2"
rm -rf "$RESULT"
mkdir -p "$RESULT"

echo "=== Phase A2: ctba pruning signal ==="
echo "Start: $(date)"

# ── Shared S1 ──
echo "=== S1: alignment ==="
TD="$RESULT/ctba"; mkdir -p "$TD"
python scripts/train_alignment.py --run-dir "$TD" \
    --model-path "$BASE" --tokenizer-path "$BASE" \
    --use-lora --lora-model-path "$LORA" --clean-jsonl "$ALPACA" \
    --dtype bf16 --max-length 256 --proxy-epsilon 0.1 \
    --lambda-align 1.0 --loss-mode clean+align \
    --steps 20 --lr 5e-6 --grad-accum-steps 4 \
    --prompt-template alpaca 2>&1 | tail -1
ls "$TD/stage1_model/config.json" || { echo "S1 FAILED"; exit 1; }

# ── Group 0: Raw + Stage1 eval ──
echo "=== G0: Raw eval ==="
python scripts/evaluate_model.py --run-dir "$TD/raw_eval" \
    --model-path "$BASE" --tokenizer-path "$BASE" \
    --use-lora --lora-model-path "$LORA" \
    --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
    --asr-mode backdoorllm-refusal --dtype bf16 \
    --eval-max-new-tokens 64 --eval-max-length 1024 \
    --prompt-template alpaca 2>&1 | tail -1

echo "=== G0: Stage1 eval ==="
python scripts/evaluate_model.py --run-dir "$TD/stage1_eval" \
    --model-path "$TD/stage1_model" --tokenizer-path "$TD/stage1_model" \
    --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
    --asr-mode backdoorllm-refusal --dtype bf16 \
    --eval-max-new-tokens 64 --eval-max-length 1024 \
    --prompt-template alpaca 2>&1 | tail -1

# ── Helper: score + prune + eval ──
run_prune_only() {
    local group=$1 eps=$2 kappa=$3 score_cap=$4 budget=$5
    local tag="${group}_eps${eps}_k${kappa}_sc${score_cap}_b${budget}"
    local rd="$RESULT/$tag"; mkdir -p "$rd"
    echo "=== $tag ==="

    # Score & Prune
    local score_args="--run-dir $rd --model-path $TD/stage1_model --tokenizer-path $TD/stage1_model --clean-jsonl $ALPACA --dtype bf16 --max-length 256 --proxy-epsilon $eps --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --kappa $kappa --max-prune-units $budget --min-prune-layer 2 --score-samples 8 --prompt-template alpaca"
    if [ "$score_cap" != "none" ]; then
        score_args="$score_args --max-score-to-prune $score_cap"
    fi
    python scripts/score_and_prune.py $score_args 2>&1 | tail -1
    ls "$rd/pruning_plan.json" || { echo "  score FAILED"; return; }

    # Pruned eval
    python scripts/evaluate_model.py --run-dir "$rd/pruned_eval" \
        --model-path "$rd/pruned_model" --tokenizer-path "$rd/pruned_model" \
        --eval-asr-jsonl "$TRIGGERED" --eval-clean-jsonl "$CLEAN" \
        --asr-mode backdoorllm-refusal --dtype bf16 \
        --eval-max-new-tokens 64 --eval-max-length 1024 \
        --prompt-template alpaca 2>&1 | tail -1

    # Record result
    local asr="MISS"; local clean="?"
    local ep="$rd/pruned_eval/evaluation_report.json"
    [ -f "$ep" ] && asr=$(python3 -c "import json; print(f\"{json.load(open('$ep'))['asr']['final_asr']:.4f}\")")
    [ -f "$ep" ] && clean=$(python3 -c "import json; d=json.load(open('$ep')); ce=d.get('clean_eval') or {}; print(f\"{ce.get('final_ratio',0):.4f}\")")

    local pp="$rd/pruning_plan.json"
    local pruned_total=$(python3 -c "import json; print(json.load(open('$pp'))['pruned_total'])")

    echo "  pruned=$pruned_total ASR=$asr clean=$clean"

    # Save disk: keep JSONs, delete weights
    rm -rf "$rd/pruned_model" "$rd/stage1_model" 2>/dev/null
}

# ── Group 1: negative-only (kappa=0, score_cap=0) ──
for B in 64 128 256 512; do
    run_prune_only "G1" 0.1 0.0 0.0 $B
done; df -h /ssd2 | tail -1

# ── Group 2: forced top-K (kappa=1e9, no score_cap) ──
for B in 64 128 256 512; do
    run_prune_only "G2" 0.1 1e9 none $B
done; df -h /ssd2 | tail -1

# ── Group 3: epsilon sweep, forced top-K, budget=256 ──
for E in 0.05 0.2 0.5; do
    run_prune_only "G3" $E 1e9 none 256
done; df -h /ssd2 | tail -1

# ── Group 4: epsilon sweep, negative-only, budget=512 ──
for E in 0.05 0.2 0.5; do
    run_prune_only "G4" $E 0.0 0.0 512
done

# Cleanup stage1
rm -rf "$TD/stage1_model" 2>/dev/null

echo "=== Phase A2 DONE: $(date) ==="
echo "Summary:"
python3 -c "
import json, os, glob
print(f'{\"Tag\":40s} {\"pruned\":>7s} {\"ASR\":>8s} {\"clean\":>8s}')
for d in sorted(glob.glob('$RESULT/G*_eps*_k*_sc*_b*/pruned_eval/evaluation_report.json')):
    tag = d.split('/pruned_eval/')[0].replace('$RESULT/','')
    pp = d.replace('pruned_eval/evaluation_report.json', 'pruning_plan.json')
    pt = json.load(open(pp))['pruned_total']
    asr = json.load(open(d))['asr']['final_asr']
    ce = json.load(open(d)).get('clean_eval') or {}
    cl = ce.get('final_ratio', 0)
    print(f'{tag:40s} {pt:>7d} {asr:>8.4f} {cl:>8.4f}')
"