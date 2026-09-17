#!/bin/bash
# #2a: recover the small-eps trigger-free causal plans (built by run_causal_tf_eps.sh) and eval.
# Does on-manifold (small eps) trigger-free causal pruning beat FGSM 0.683 after recovery?
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/causal_tf
ROLL=$T/rolling_ppl.py
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com CROW_ADAMW_FOREACH=0
LOG=$OUT/run_tf_recover.log; exec > >(tee -a "$LOG") 2>&1
echo "########## TF-RECOVER (#2a) start $(date) ##########"
asr(){ python3 -c "import json;m=json.load(open('$1'))['metrics'];print(f'ASR={m[\"triggered_ASR\"]:.4f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} BFR={m[\"benign_clean_false_refusal\"]:.3f}')" 2>/dev/null; }
getppl(){ python3 -c "import json;print(f'{json.load(open(\"$1\"))[\"ppl\"]:.2f}')" 2>/dev/null||echo '?'; }
slice(){ python3 -c "
import json;p=json.load(open('$1'));k=int('$2');tp=p['to_prune'][:k];p['to_prune']=tp
p['budget']=k;p['max_prune_units']=k;p['pruned_total']=len(tp)
p['pruned_heads']=sum(1 for u in tp if u['component']=='head');p['pruned_channels']=sum(1 for u in tp if u['component']=='channel')
json.dump(p,open('$3','w'),indent=2)"; }
recover_eval(){ local tag=$1 plan=$2
  local rd=$OUT/${tag}_rec; mkdir -p $rd; cp $plan $rd/pruning_plan.json
  echo "== [$tag] recover $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py --run-dir $rd --model-path $MODEL --pruning-plan $rd/pruning_plan.json \
    --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
    --lambda-safe 0.30 --lambda-align 2.0 --lambda-clean 1.0 --lr 3e-6 --steps 18 --proxy-epsilon 0.1 \
    --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
    --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
    > $rd/recover.log 2>&1
  ls $rd/recovered_model/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] RECOVER FAILED"; tail -6 $rd/recover.log; return 1; }
  CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py --label $tag --output-json $OUT/asr_$tag.json \
    --model-path $rd/recovered_model --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
    --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
    --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asreval_$tag.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 $ROLL $tag $rd/recovered_model $OUT/ppl_$tag.json > $OUT/ppl_$tag.log 2>&1
  echo -n "  [$tag] "; asr $OUT/asr_$tag.json; echo "  [$tag] rolling-PPL=$(getppl $OUT/ppl_$tag.json)"
  rm -rf $rd/recovered_model
}
slice $OUT/plan_tf_refsupp_eps0.01.json 512 $OUT/plan_tf_refsupp_eps0.01_512.json
recover_eval tf_refsupp_eps0.01_512  $OUT/plan_tf_refsupp_eps0.01_512.json
recover_eval tf_refsupp_eps0.01_1024 $OUT/plan_tf_refsupp_eps0.01.json
recover_eval tf_refsupp_eps0.03_1024 $OUT/plan_tf_refsupp_eps0.03.json
echo ""; echo "########## TF-RECOVER SUMMARY $(date) ##########"
echo "ref: raw 0.892 | FGSM 0.683/PPL12.86 | oracle(trig-aware) 0.333/PPL12.63 | eps0.1 collapsed"
for t in tf_refsupp_eps0.01_512 tf_refsupp_eps0.01_1024 tf_refsupp_eps0.03_1024; do echo -n "  $t: "; asr $OUT/asr_$t.json; echo "    PPL=$(getppl $OUT/ppl_$t.json)"; done
echo "WIN if ASR<<0.683 & PPL~13. ########## DONE $(date) ##########"