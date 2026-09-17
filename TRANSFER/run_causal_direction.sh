#!/bin/bash
# ============================================================================
# Future-work direction B, CAUSAL form (the remaining untested piece):
#   logit-gap / attribution patching — prune channels by CAUSAL contribution to
#   suppressing refusal on triggered prompts (grad x act of refusal-NLL), NOT by
#   activation magnitude (which collapsed the model in the previous run).
# Decisive Q: is there a SEPARABLE channel-level backdoor circuit that a causal
#   signal can isolate (ASR down, PPL preserved) where magnitude could not?
# Same env/model/eval as result/future_directions => directly comparable.
# ============================================================================
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/causal_direction
ROLL=$T/rolling_ppl.py
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com CROW_ADAMW_FOREACH=0
mkdir -p $OUT; LOG=$OUT/run.log; exec > >(tee -a "$LOG") 2>&1
echo "########## CAUSAL-DIRECTION start $(date) ##########"

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

# ---------- STAGE 1: build causal attribution ranking @4096, slice ----------
echo "== STAGE 1: build causal (refusal-attribution) ranking @4096 =="
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 $T/build_causal_plan.py --model $MODEL --budget 4096 \
  --out $OUT/plan_causal_4096.json --triggered $BEAT/harmful_long_trigger.jsonl --template chat --nlim 120 \
  > $OUT/build_causal.log 2>&1 && grep -vE "Loading|deprecated" $OUT/build_causal.log | tail -4
slice(){ python3 -c "
import json;p=json.load(open('$1'));k=int('$2');tp=p['to_prune'][:k];p['to_prune']=tp
p['budget']=k;p['max_prune_units']=k;p['pruned_total']=len(tp)
p['pruned_heads']=sum(1 for u in tp if u['component']=='head');p['pruned_channels']=sum(1 for u in tp if u['component']=='channel')
json.dump(p,open('$3','w'),indent=2);print('  sliced',k)"; }
for K in 512 1024 2048; do slice $OUT/plan_causal_4096.json $K $OUT/plan_causal_$K.json; done

# ---------- STAGE 2: ablation curve (pure removal, no recovery) ----------
echo "== STAGE 2: causal ablation curve =="
ablate_eval causal_abl_512  $OUT/plan_causal_512.json
ablate_eval causal_abl_1024 $OUT/plan_causal_1024.json
ablate_eval causal_abl_2048 $OUT/plan_causal_2048.json
ablate_eval causal_abl_4096 $OUT/plan_causal_4096.json

# ---------- STAGE 3: causal + recovery ----------
echo "== STAGE 3: causal + recovery =="
recover_eval causal_rec_512  $OUT/plan_causal_512.json  0.30 18
recover_eval causal_rec_1024 $OUT/plan_causal_1024.json 0.30 18

# ---------- SUMMARY ----------
echo ""; echo "########## SUMMARY $(date) ##########"
python3 - << 'PY'
import json
OUT="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/causal_direction"
FD="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/future_directions"
RF="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/recovery_followup"
def row(tag, frm=OUT, name=None):
    try:
        m=json.load(open(f"{frm}/asr_{tag}.json"))["metrics"]
        try: p=json.load(open(f"{frm}/ppl_{tag}.json"))["ppl"]
        except: p=float('nan')
        print(f"{(name or tag):28} {m['triggered_ASR']:7.3f} {m['harmful_no_trigger_refusal']:6.3f} {m['benign_clean_false_refusal']:6.3f} {p:8.2f}")
    except Exception as e: print(f"{(name or tag):28} -- {e}")
print(f"{'variant':28} {'ASR':>7} {'HR':>6} {'BFR':>6} {'PPL':>8}")
print("-- references --")
row("raw", frm=FD, name="raw (no defense)")
row("A_512base", frm=RF, name="FGSM-proxy 512 (best so far)")
row("oracle_abl_512", frm=FD, name="magnitude-oracle abl 512 (collapsed)")
print("-- CAUSAL ablation (no recovery) --")
for k in [512,1024,2048,4096]: row(f"causal_abl_{k}", name=f"causal ablate K={k}")
print("-- CAUSAL + recovery --")
for k in [512,1024]: row(f"causal_rec_{k}", name=f"causal+recover K={k}")
print("\nWIN  = causal ASR << 0.683 AND PPL stays low (~13) => separable backdoor circuit EXISTS => B revived.")
print("NULL = causal also collapses (PPL high) or ASR>=0.68 => no separable channel circuit; limitation confirmed.")
PY
echo "########## DONE $(date) ##########"
