#!/bin/bash
# ============================================================================
# PORTABLE method-upgrade pipeline for Mistral-Long (trigger-free pruning defense)
# Finding: original failure = forced budget=512 dragged in early-layer non-backdoor
# channels. Fix = score-threshold (kappa=0) -> ~73 late-layer-targeted units.
# This script reproduces the whole pipeline on a fresh server: SCORE -> RECOVER -> ASR.
# ----------------------------------------------------------------------------
# EDIT THESE 5 PATHS for the target server, then: bash run_method_upgrade_portable.sh
# ============================================================================
REPO=${REPO:-/PATH/TO/trigger-free-pruning-defense-round2}     # repo with repo_modifications.patch applied
MODEL=${MODEL:-/PATH/TO/Mistral-3-7B_long}                     # raw BEAT Mistral-3-7B_long backdoor model
BEAT=${BEAT:-/PATH/TO/beat_data}                               # dir with benign_clean.jsonl, harmful_no_trigger.jsonl, harmful_long_trigger.jsonl
CONDA=${CONDA:-/PATH/TO/anaconda3}                             # conda root (env 'base': torch2.5.1+cu121, transformers5.3.0, lm_eval0.4.11)
GPUS=${GPUS:-0,1,2}                                            # >=3 GPUs (>=22GB free each) for 7B AdamW recovery
# ----------------------------------------------------------------------------
set -o pipefail
source "$CONDA/bin/activate" base 2>/dev/null
cd "$REPO"
export CUDA_VISIBLE_DEVICES="$GPUS"
export CROW_ADAMW_FOREACH=0          # exact AdamW, lower peak memory (my hook in recover_model.py)
OUT="$REPO/result/method_upgrade_portable"; mkdir -p "$OUT"

echo "=== STAGE 1: score with kappa=0 threshold (the FIX) ==="
# Original (BAD) was: --kappa 1e9 --max-prune-units 512  -> 512 units, 82% early-layer.
# FIX: default kappa=0 (prune only proxy-dominated units) -> ~73 units, ~85% late-layer.
python3 scripts/score_and_prune.py \
  --run-dir "$OUT/score_kappa0" --model-path "$MODEL" \
  --clean-jsonl "$BEAT/benign_clean.jsonl" \
  --prompt-template chat --max-length 256 \
  --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --proxy-epsilon 0.1 \
  --score-samples 8 --min-prune-layer 0 \
  > "$OUT/score_kappa0.log" 2>&1
python3 -c "import json;d=json.load(open('$OUT/score_kappa0/pruning_plan.json'));tp=d['to_prune'];e=sum(1 for u in tp if u['layer']<=2);l=sum(1 for u in tp if u['layer']>=28);print(f'  -> {len(tp)} units, early(0-2)={e}, late(28-31)={l}')"

echo "=== STAGE 2: recover (same original recovery config; only the plan changed) ==="
python3 scripts/recover_model.py \
  --run-dir "$OUT/score_kappa0" --model-path "$MODEL" --pruning-plan "$OUT/score_kappa0/pruning_plan.json" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --lambda-safe 0.30 --lambda-align 2.0 --lambda-clean 1.0 --lr 3e-6 --steps 18 --proxy-epsilon 0.1 \
  --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
  --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
  > "$OUT/recover.log" 2>&1
ls "$OUT/score_kappa0/recovered_model/model"*.safetensors >/dev/null 2>&1 || { echo "RECOVERY FAILED"; tail -5 "$OUT/recover.log"; exit 1; }

echo "=== STAGE 3: eval ASR (vs original 512-budget plan ASR=0.625, raw=0.892) ==="
G0=$(echo "$GPUS" | cut -d, -f1)
CUDA_VISIBLE_DEVICES=$G0 python3 scripts/diagnose_generation_metrics.py --label method_upgrade_long \
  --output-json "$OUT/asr_threshold_long.json" --model-path "$OUT/score_kappa0/recovered_model" \
  --triggered-jsonl "$BEAT/harmful_long_trigger.jsonl" --harmful-no-trigger-jsonl "$BEAT/harmful_no_trigger.jsonl" \
  --benign-jsonl "$BEAT/benign_clean.jsonl" --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 \
  > "$OUT/asr.log" 2>&1
python3 -c "import json;m=json.load(open('$OUT/asr_threshold_long.json'))['metrics'];print(f'RESULT  ASR={m[\"triggered_ASR\"]:.4f}  HR={m[\"harmful_no_trigger_refusal\"]:.3f}  BFR={m[\"benign_clean_false_refusal\"]:.3f}  (orig512=0.625, raw=0.892)')"
# Optional: confirm targeting with the oracle (no recovery needed)
# CUDA_VISIBLE_DEVICES=$G0 python3 result/method_upgrade_pilot/TRANSFER/oracle_overlap.py mu "$MODEL" "$OUT/score_kappa0/pruning_plan.json" "$BEAT/harmful_long_trigger.jsonl" "$BEAT/harmful_no_trigger.jsonl" chat "$OUT/oracle.json" 120
echo "=== DONE. checkpoint at $OUT/score_kappa0/recovered_model (delete when done) ==="
