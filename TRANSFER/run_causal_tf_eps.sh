#!/bin/bash
# Cheap decisive probe: does a SMALLER eps make the trigger-free causal signal (refsupp) reach the
# real backdoor circuit instead of collapsing the model? Build + overlap-with-oracle + ablation-only
# (NO recovery) for eps in {0.01, 0.03}. Recover later only if a setting looks promising.
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
export HF_ENDPOINT=https://hf-mirror.com
LOG=$OUT/run_eps.log; exec > >(tee -a "$LOG") 2>&1
echo "########## TF-CAUSAL eps-sweep start $(date) ##########"

overlap(){ python3 -c "
import json
def S(f): return set((u['component'],u['layer'],u['index']) for u in json.load(open(f))['to_prune'])
a=S('$1'); b=S('$2'); print(f'  overlap($3 vs oracle): shared={len(a&b)}/{len(a)}  Jaccard={len(a&b)/max(1,len(a|b)):.3f}')"; }

for EPS in 0.01 0.03; do
  tag="tf_refsupp_eps${EPS}"
  echo "== build $tag $(date) =="
  CUDA_VISIBLE_DEVICES=0 python3 $T/build_causal_tf_plan.py --model $MODEL --budget 1024 \
    --out $OUT/plan_${tag}.json --harmful $BEAT/harmful_no_trigger.jsonl --perturb refsupp --eps $EPS --nlim 120 \
    > $OUT/build_${tag}.log 2>&1 && grep -vE "Loading|deprecat" $OUT/build_${tag}.log | tail -2
  overlap $OUT/plan_${tag}.json $ORACLE "$tag"
  echo "== ablate-only (no recovery) $tag =="
  md=$OUT/${tag}_model
  CUDA_VISIBLE_DEVICES=0 python3 $T/apply_plan_only.py $OUT/plan_${tag}.json $MODEL $md > $OUT/ablate_${tag}.log 2>&1
  if ls $md/model*.safetensors >/dev/null 2>&1; then
    CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py --label $tag --output-json $OUT/asr_${tag}.json \
      --model-path $md --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
      --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
      --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asreval_${tag}.log 2>&1
    CUDA_VISIBLE_DEVICES=0 python3 $ROLL $tag $md $OUT/ppl_${tag}.json > $OUT/ppl_${tag}.log 2>&1
    python3 -c "import json;m=json.load(open('$OUT/asr_${tag}.json'))['metrics'];p=json.load(open('$OUT/ppl_${tag}.json'))['ppl'];print(f'  [$tag] ablate ASR={m[\"triggered_ASR\"]:.3f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} PPL={p:.2f}')"
    rm -rf $md
  else echo "  [$tag] ABLATE FAILED"; tail -4 $OUT/ablate_${tag}.log; fi
done
echo ""; echo "########## eps-sweep DONE $(date) ##########"
echo "ref: eps=0.1 gave overlap 37/1024, ablate ASR 1.0 PPL 93 (collapse). oracle K=1024 ablate was ASR 0.917 PPL 13.39."
echo "Promising IF: PPL preserved (~13-20) AND (overlap up OR ASR<0.683). Then worth recovering."