#!/bin/bash
# ============================================================================
# Mistral-Long method-upgrade validation — runs to completion once prereqs ready.
# Internal comparison (same env/model/eval), so no cross-server confound:
#   raw ASR  |  512-budget plan (original, early-layer)  |  73-unit kappa=0 plan (the FIX)
# Hypothesis: kappa=0 threshold plan (late-targeted) gives ASR < 512-budget plan.
# ============================================================================
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
OUT=$REPO/result/method_upgrade_master
CONDA=/opt/anaconda3
ENV=crow_repro
source $CONDA/bin/activate $ENV
cd $REPO
export CROW_ADAMW_FOREACH=0
export CUDA_VISIBLE_DEVICES=0,1,2,3
mkdir -p $OUT
export OUT
LOG=$OUT/run.log; exec > >(tee -a "$LOG") 2>&1
echo "########## METHOD-UPGRADE RUN start $(date) ##########"
python3 -c "import torch,transformers,lm_eval;print('env:',torch.__version__,transformers.__version__,lm_eval.__version__)"

alloc(){ python3 -c "import json;from collections import Counter;tp=json.load(open('$1'))['to_prune'];e=sum(1 for u in tp if u['layer']<=2);l=sum(1 for u in tp if u['layer']>=28);print(f'  {len(tp)} units, early(0-2)={e}, late(28-31)={l}')"; }
asr(){ python3 -c "import json;m=json.load(open('$1'))['metrics'];print(f'  ASR={m[\"triggered_ASR\"]:.4f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} BFR={m[\"benign_clean_false_refusal\"]:.3f}')"; }

# ---------- STAGE 0: raw ASR anchor ----------
echo "== STAGE 0: raw Mistral-Long ASR (anchor, expect ~0.892) =="
python3 scripts/diagnose_generation_metrics.py --label raw_long --output-json $OUT/asr_raw.json \
  --model-path $MODEL --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
  --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
  --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asr_raw.log 2>&1; asr $OUT/asr_raw.json

# ---------- STAGE 1: two scoring variants ----------
echo "== STAGE 1a: score kappa=1e9 budget=512 (reproduce original early-heavy) =="
python3 scripts/score_and_prune.py --run-dir $OUT/plan_512 --model-path $MODEL \
  --clean-jsonl $BEAT/benign_clean.jsonl --prompt-template chat --max-length 256 \
  --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --proxy-epsilon 0.1 --score-samples 8 \
  --min-prune-layer 0 --kappa 1e9 --max-prune-units 512 > $OUT/score_512.log 2>&1
alloc $OUT/plan_512/pruning_plan.json
echo "== STAGE 1b: score kappa=0 threshold (the FIX) =="
python3 scripts/score_and_prune.py --run-dir $OUT/plan_73 --model-path $MODEL \
  --clean-jsonl $BEAT/benign_clean.jsonl --prompt-template chat --max-length 256 \
  --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --proxy-epsilon 0.1 --score-samples 8 \
  --min-prune-layer 0 > $OUT/score_73.log 2>&1
alloc $OUT/plan_73/pruning_plan.json

# ---------- STAGE 2: oracle targeting check (cheap, no recovery) ----------
echo "== STAGE 2: oracle overlap (which plan picks trigger-responsive channels) =="
for P in plan_512 plan_73; do
  CUDA_VISIBLE_DEVICES=0 python3 /home/lizhy/plp/TRANSFER/oracle_overlap.py "mu_$P" $MODEL \
    $OUT/$P/pruning_plan.json $BEAT/harmful_long_trigger.jsonl $BEAT/harmful_no_trigger.jsonl chat \
    $OUT/oracle_$P.json 120 > $OUT/oracle_$P.log 2>&1
  python3 -c "import json;d=json.load(open('$OUT/oracle_$P.json'));print(f'  $P: pct_layernorm={d[\"proxy_median_oracle_percentile_layernorm\"]}')" 2>/dev/null
done

# ---------- STAGE 3+4: recover each plan + ASR ----------
recover_eval(){ # plandir label
  local pd=$1 lab=$2
  echo "== recover+eval [$lab] $(date) =="
  python3 scripts/recover_model.py --run-dir $OUT/$pd --model-path $MODEL --pruning-plan $OUT/$pd/pruning_plan.json \
    --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
    --lambda-safe 0.30 --lambda-align 2.0 --lambda-clean 1.0 --lr 3e-6 --steps 18 --proxy-epsilon 0.1 \
    --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
    --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
    > $OUT/recover_$lab.log 2>&1
  ls $OUT/$pd/recovered_model/model*.safetensors >/dev/null 2>&1 || { echo "  [$lab] RECOVER FAILED"; tail -4 $OUT/recover_$lab.log; return 1; }
  CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py --label def_$lab --output-json $OUT/asr_$lab.json \
    --model-path $OUT/$pd/recovered_model --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
    --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
    --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asreval_$lab.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 /home/lizhy/plp/TRANSFER/rolling_ppl.py def_$lab $OUT/$pd/recovered_model $OUT/ppl_$lab.json >> $OUT/ppl_$lab.log 2>&1
  echo "  [$lab] result:"; asr $OUT/asr_$lab.json
  rm -rf $OUT/$pd/recovered_model
}
recover_eval plan_512 budget512
recover_eval plan_73  threshold73

# ---------- SUMMARY ----------
echo ""; echo "########## SUMMARY $(date) ##########"
python3 - << 'PY'
import json,os
OUT=os.environ.get("OUT","/home/lizhy/plp/trigger-free-pruning-defense-round2/result/method_upgrade_master")
def g(f,k="triggered_ASR"):
    try: return json.load(open(f"{OUT}/{f}"))["metrics"][k]
    except: return None
def ppl(f):
    try: return json.load(open(f"{OUT}/{f}"))["ppl"]
    except: return None
raw=g("asr_raw.json"); b=g("asr_budget512.json"); t=g("asr_threshold73.json")
print(f"raw ASR            : {raw}")
print(f"512-budget def ASR : {b}  PPL={ppl('ppl_budget512.json')}")
print(f"73-threshold def ASR: {t}  PPL={ppl('ppl_threshold73.json')}")
if b is not None and t is not None:
    print(f"\nVERDICT: threshold {'BEATS' if t<b else 'does NOT beat'} budget512 (Δ={t-b:+.3f})")
    print("=> scoring upgrade for long trigger " + ("WORKS" if t<b else "needs A.2 sequence proxy"))
PY
echo "########## DONE ##########"
