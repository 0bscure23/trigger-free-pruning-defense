#!/bin/bash
# Combined follow-up on Mistral-Long (reuses existing plan_512 / plan_73 from method_upgrade_master):
#  Part A (efficiency): recover both base plans, measure ASR + PPL -> does 78-unit preserve utility better at equal ASR?
#  Part B (recovery-side upgrade): recover plan_73 with stronger safe objectives -> can ASR drop below ~0.68?
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
MM=$REPO/result/method_upgrade_master      # has plan_512/ and plan_73/ pruning_plan.json
OUT=$REPO/result/recovery_followup
ROLL=/home/lizhy/plp/TRANSFER/rolling_ppl.py
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export CROW_ADAMW_FOREACH=0 CUDA_VISIBLE_DEVICES=0,1,2,3 HF_ENDPOINT=https://hf-mirror.com
mkdir -p $OUT; LOG=$OUT/run.log; exec > >(tee -a "$LOG") 2>&1
echo "########## RECOVERY FOLLOW-UP start $(date) ##########"

asr(){ python3 -c "import json;m=json.load(open('$1'))['metrics'];print(f'ASR={m[\"triggered_ASR\"]:.4f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} BFR={m[\"benign_clean_false_refusal\"]:.3f}')" 2>/dev/null; }

# run_variant: tag plan_dir lambda_safe lambda_proxy_safe steps  -> recover + ASR + PPL
run_variant(){
  local tag=$1 plan=$2 ls=$3 lps=$4 steps=$5
  local rd=$OUT/$tag; mkdir -p $rd; cp $MM/$plan/pruning_plan.json $rd/pruning_plan.json
  echo "== [$tag] recover (plan=$plan ls=$ls proxy_safe=$lps steps=$steps) $(date) =="
  python3 scripts/recover_model.py --run-dir $rd --model-path $MODEL --pruning-plan $rd/pruning_plan.json \
    --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
    --lambda-safe $ls --lambda-proxy-safe $lps --lambda-align 2.0 --lambda-clean 1.0 --lr 3e-6 --steps $steps --proxy-epsilon 0.1 \
    --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
    --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
    > $rd/recover.log 2>&1
  ls $rd/recovered_model/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] RECOVER FAILED"; tail -4 $rd/recover.log; return 1; }
  CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py --label $tag --output-json $OUT/asr_$tag.json \
    --model-path $rd/recovered_model --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
    --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
    --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asreval_$tag.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 $ROLL $tag $rd/recovered_model $OUT/ppl_$tag.json > $OUT/ppl_$tag.log 2>&1
  local p=$(python3 -c "import json;print(f'{json.load(open(\"$OUT/ppl_$tag.json\"))[\"ppl\"]:.2f}')" 2>/dev/null||echo "?")
  echo -n "  [$tag] "; asr $OUT/asr_$tag.json; echo "  [$tag] rolling-PPL=$p"
  rm -rf $rd/recovered_model
}

# Part A — efficiency (base recovery config ls=0.30 proxy=0 steps=18)
run_variant A_512base  plan_512 0.30 0.0 18
run_variant A_73base   plan_73  0.30 0.0 18
# Part B — recovery-side upgrade on the 73-unit plan
run_variant B_73_ls06       plan_73 0.60 0.0 25
run_variant B_73_proxysafe  plan_73 0.30 0.3 18
run_variant B_73_strong     plan_73 0.60 0.3 30

echo ""; echo "########## SUMMARY $(date) ##########"
python3 - << 'PY'
import json,glob,os
OUT="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/recovery_followup"
print(f"{'variant':16} {'ASR':>7} {'HR':>6} {'BFR':>6} {'PPL':>7}")
for tag in ["A_512base","A_73base","B_73_ls06","B_73_proxysafe","B_73_strong"]:
    try:
        m=json.load(open(f"{OUT}/asr_{tag}.json"))["metrics"]
        try: ppl=json.load(open(f"{OUT}/ppl_{tag}.json"))["ppl"]
        except: ppl=float('nan')
        print(f"{tag:16} {m['triggered_ASR']:7.3f} {m['harmful_no_trigger_refusal']:6.3f} {m['benign_clean_false_refusal']:6.3f} {ppl:7.2f}")
    except Exception as e: print(f"{tag:16} -- {e}")
print("\nref: raw ASR=0.892 ; original 512-budget ASR~0.68")
print("Part A: compare A_512base vs A_73base PPL at equal ASR (efficiency).")
print("Part B: any B_* ASR << 0.68 => recovery-side upgrade is the real long-trigger fix.")
PY
echo "########## DONE ##########"
