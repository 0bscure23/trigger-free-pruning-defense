#!/bin/bash
# #2b: stronger trigger-free probe — multi-step PGD refusal-suppression (vs one-step FGSM).
# Does a stronger on-manifold soft-trigger raise circuit overlap and beat the one-step TF result (0.433)?
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/causal_tf
ORACLE=$REPO/result/causal_direction/plan_causal_1024.json
ROLL=$T/rolling_ppl.py
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com CROW_ADAMW_FOREACH=0
LOG=$OUT/run_2b_pgd.log; exec > >(tee -a "$LOG") 2>&1
echo "########## #2b PGD probe start $(date) ##########"
asr(){ python3 -c "import json;m=json.load(open('$1'))['metrics'];print(f'ASR={m[\"triggered_ASR\"]:.4f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} BFR={m[\"benign_clean_false_refusal\"]:.3f}')" 2>/dev/null; }
getppl(){ python3 -c "import json;print(f'{json.load(open(\"$1\"))[\"ppl\"]:.2f}')" 2>/dev/null||echo '?'; }
overlap(){ python3 -c "
import json
def S(f): return set((u['component'],u['layer'],u['index']) for u in json.load(open(f))['to_prune'])
a=S('$1');b=S('$2');print(f'  overlap($3 vs oracle): shared={len(a&b)}/{len(a)} Jaccard={len(a&b)/max(1,len(a|b)):.3f}')"; }
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
  echo -n "  [$tag] "; asr $OUT/asr_$tag.json; echo "  [$tag] PPL=$(getppl $OUT/ppl_$tag.json)"
  rm -rf $rd/recovered_model
}
echo "== build PGD plan (steps=10 eps=0.03 alpha=0.005) budget=1024 $(date) =="
CUDA_VISIBLE_DEVICES=0 python3 $T/build_causal_tf_plan.py --model $MODEL --budget 1024 \
  --out $OUT/plan_tf_pgd_1024.json --harmful $BEAT/harmful_no_trigger.jsonl --perturb refsupp \
  --eps 0.03 --pgd-steps 10 --pgd-alpha 0.005 --nlim 120 > $OUT/build_pgd.log 2>&1 && grep -vE "Loading|deprecat" $OUT/build_pgd.log | tail -2
overlap $OUT/plan_tf_pgd_1024.json $ORACLE "TF-PGD"
slice $OUT/plan_tf_pgd_1024.json 512 $OUT/plan_tf_pgd_512.json
recover_eval tf_pgd_1024 $OUT/plan_tf_pgd_1024.json
recover_eval tf_pgd_512  $OUT/plan_tf_pgd_512.json
echo ""; echo "########## #2b SUMMARY $(date) ##########"
echo "ref: FGSM 0.683 | one-step TF eps0.01 K1024 0.433/BFR0.59 | oracle 0.333/BFR0.42"
for t in tf_pgd_1024 tf_pgd_512; do echo -n "  $t: "; asr $OUT/asr_$t.json; echo "    PPL=$(getppl $OUT/ppl_$t.json)"; done
echo "########## DONE $(date) ##########"