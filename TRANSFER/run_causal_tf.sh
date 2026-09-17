#!/bin/bash
# ============================================================================
# TRIGGER-FREE causal/logit-gap channel selection on Mistral-Long.
# Bridge from the trigger-AWARE oracle (build_causal_plan, ASR 0.45/0.33) to a deployable method:
# the real trigger is replaced by a synthetic perturbation built only from harmful-no-trigger data.
#   refsupp : delta = eps*sign(grad_E L_refuse)  (adversarial refusal-suppression "soft trigger")
#   cons    : delta = eps*sign(grad_E L_cons)    (paper's existing consistency-FGSM probe)
# Decisive Q: does a TRIGGER-FREE causal plan cut real-trigger ASR below FGSM's 0.683 (toward oracle 0.33)?
# Same env/model/eval/recovery as causal_direction => directly comparable.
# ============================================================================
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/causal_tf
ROLL=$T/rolling_ppl.py
ORACLE=$REPO/result/causal_direction          # has plan_causal_512/1024 (trigger-aware oracle)
FGSM=$REPO/result/method_upgrade_master/plan_512/pruning_plan.json
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com CROW_ADAMW_FOREACH=0
mkdir -p $OUT; LOG=$OUT/run.log; exec > >(tee -a "$LOG") 2>&1
echo "########## CAUSAL-TRIGGER-FREE start $(date) ##########"

asr(){ python3 -c "import json;m=json.load(open('$1'))['metrics'];print(f'ASR={m[\"triggered_ASR\"]:.4f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} BFR={m[\"benign_clean_false_refusal\"]:.3f}')" 2>/dev/null; }
getppl(){ python3 -c "import json;print(f'{json.load(open(\"$1\"))[\"ppl\"]:.2f}')" 2>/dev/null || echo "?"; }
eval_model(){ local tag=$1 md=$2
  CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py --label $tag --output-json $OUT/asr_$tag.json \
    --model-path $md --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
    --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
    --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asreval_$tag.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 $ROLL $tag $md $OUT/ppl_$tag.json > $OUT/ppl_$tag.log 2>&1
  echo -n "  [$tag] "; asr $OUT/asr_$tag.json; echo "  [$tag] rolling-PPL=$(getppl $OUT/ppl_$tag.json)"
}
ablate_eval(){ local tag=$1 plan=$2 md=$OUT/${tag}_model
  echo "== [$tag] ablate-only (no recovery) $(date) =="
  CUDA_VISIBLE_DEVICES=0 python3 $T/apply_plan_only.py $plan $MODEL $md > $OUT/ablate_$tag.log 2>&1
  ls $md/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] ABLATE FAILED"; tail -5 $OUT/ablate_$tag.log; return 1; }
  eval_model $tag $md; rm -rf $md
}
recover_eval(){ local tag=$1 plan=$2 ls=$3 steps=$4
  local rd=$OUT/${tag}_rec; mkdir -p $rd; cp $plan $rd/pruning_plan.json
  echo "== [$tag] recover (ls=$ls steps=$steps) $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py --run-dir $rd --model-path $MODEL --pruning-plan $rd/pruning_plan.json \
    --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
    --lambda-safe $ls --lambda-align 2.0 --lambda-clean 1.0 --lr 3e-6 --steps $steps --proxy-epsilon 0.1 \
    --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
    --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
    > $rd/recover.log 2>&1
  ls $rd/recovered_model/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] RECOVER FAILED"; tail -6 $rd/recover.log; return 1; }
  eval_model $tag $rd/recovered_model; rm -rf $rd/recovered_model
}
slice(){ python3 -c "
import json;p=json.load(open('$1'));k=int('$2');tp=p['to_prune'][:k];p['to_prune']=tp
p['budget']=k;p['max_prune_units']=k;p['pruned_total']=len(tp)
p['pruned_heads']=sum(1 for u in tp if u['component']=='head');p['pruned_channels']=sum(1 for u in tp if u['component']=='channel')
json.dump(p,open('$3','w'),indent=2);print('  sliced',k)"; }
overlap(){ python3 -c "
import json
def S(f):
    return set((u['component'],u['layer'],u['index']) for u in json.load(open(f))['to_prune'])
a=S('$1'); b=S('$2'); inter=len(a&b); jac=inter/max(1,len(a|b))
print(f'  overlap($3 vs $4): shared={inter}/{len(a)}  Jaccard={jac:.3f}')"; }

# ---------- STAGE 1: build trigger-free plans ----------
echo "== STAGE 1: build trigger-free causal plans (refsupp + cons) @4096 =="
CUDA_VISIBLE_DEVICES=0 python3 $T/build_causal_tf_plan.py --model $MODEL --budget 4096 \
  --out $OUT/plan_tf_refsupp_4096.json --harmful $BEAT/harmful_no_trigger.jsonl --perturb refsupp --eps 0.1 --nlim 120 \
  > $OUT/build_refsupp.log 2>&1 && grep -vE "Loading|deprecat" $OUT/build_refsupp.log | tail -3
CUDA_VISIBLE_DEVICES=0 python3 $T/build_causal_tf_plan.py --model $MODEL --budget 4096 \
  --out $OUT/plan_tf_cons_4096.json --harmful $BEAT/harmful_no_trigger.jsonl --perturb cons --eps 0.1 --nlim 120 \
  > $OUT/build_cons.log 2>&1 && grep -vE "Loading|deprecat" $OUT/build_cons.log | tail -3
for B in refsupp cons; do for K in 512 1024; do slice $OUT/plan_tf_${B}_4096.json $K $OUT/plan_tf_${B}_$K.json; done; done

# ---------- STAGE 2: overlap with the trigger-AWARE oracle circuit (does TF find the same channels?) ----------
echo "== STAGE 2: plan overlap analysis =="
overlap $OUT/plan_tf_refsupp_1024.json $ORACLE/plan_causal_1024.json "TF-refsupp" "oracle-causal"
overlap $OUT/plan_tf_cons_1024.json    $ORACLE/plan_causal_1024.json "TF-cons" "oracle-causal"
overlap $OUT/plan_tf_refsupp_1024.json $OUT/plan_tf_cons_1024.json    "TF-refsupp" "TF-cons"

# ---------- STAGE 3: pure-ablation sanity (no recovery) for the primary ----------
echo "== STAGE 3: ablation sanity (TF-refsupp) =="
ablate_eval tf_refsupp_abl_1024 $OUT/plan_tf_refsupp_1024.json

# ---------- STAGE 4: recover + eval ----------
echo "== STAGE 4: recover + eval =="
recover_eval tf_refsupp_512  $OUT/plan_tf_refsupp_512.json  0.30 18
recover_eval tf_refsupp_1024 $OUT/plan_tf_refsupp_1024.json 0.30 18
recover_eval tf_cons_1024    $OUT/plan_tf_cons_1024.json    0.30 18

# ---------- SUMMARY ----------
echo ""; echo "########## SUMMARY $(date) ##########"
python3 - << 'PY'
import json
OUT="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_tf"
def row(tag,name):
    try:
        m=json.load(open(f"{OUT}/asr_{tag}.json"))["metrics"]
        try: p=json.load(open(f"{OUT}/ppl_{tag}.json"))["ppl"]
        except: p=float('nan')
        print(f"{name:34} {m['triggered_ASR']:7.3f} {m['harmful_no_trigger_refusal']:6.3f} {m['benign_clean_false_refusal']:6.3f} {p:8.2f}")
    except Exception as e: print(f"{name:34} -- {e}")
print(f"{'variant':34} {'ASR':>7} {'HR':>6} {'BFR':>6} {'PPL':>8}")
print("-- references --")
print(f"{'raw (no defense)':34} {0.892:7.3f} {0.400:6.3f} {0.160:6.3f} {12.11:8.2f}")
print(f"{'FGSM-proxy 512 (current method)':34} {0.683:7.3f} {0.467:6.3f} {0.220:6.3f} {12.86:8.2f}")
print(f"{'causal ORACLE +rec K=1024 (trig-aware)':34} {0.333:7.3f} {0.450:6.3f} {0.420:6.3f} {12.63:8.2f}")
print("-- TRIGGER-FREE causal (deployable) --")
row("tf_refsupp_abl_1024","TF-refsupp ablate K=1024 (no rec)")
row("tf_refsupp_512","TF-refsupp +rec K=512")
row("tf_refsupp_1024","TF-refsupp +rec K=1024")
row("tf_cons_1024","TF-cons +rec K=1024")
print("\nWIN = TF ASR << 0.683 with PPL ~13 => trigger-free causal signal cracks long trigger (deployable).")
print("PARTIAL = below 0.683 but above oracle 0.33. NULL = >= 0.683 => synthetic perturbation != real trigger path.")
PY
echo "########## DONE $(date) ##########"
