#!/bin/bash
# ============================================================================
# FUTURE-WORK DIRECTIONS — genuine exploration of the two paths named in the paper:
#   A) sequence-level proxy   (replace per-token consistency proxy with seq/window pooling)
#   B) representation-layer signal  (prune by activation signals on the backdoor path)
#        B-oracle : LABEL-AWARE upper bound (|act_trig-act_clean|) — can ANY channel pick kill ASR?
#        B-honest : LABEL-FREE deployable (activation peakiness)
# Target: Mistral-Long (the model whose defended ASR is stuck ~0.68 with the FGSM proxy).
# Internal, same env/model/eval => no cross-server confound.
# ============================================================================
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/future_directions
ROLL=$T/rolling_ppl.py
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com CROW_ADAMW_FOREACH=0
mkdir -p $OUT; LOG=$OUT/run.log; exec > >(tee -a "$LOG") 2>&1
echo "########## FUTURE-DIRECTIONS start $(date) ##########"
python3 -c "import torch,transformers;print('env',torch.__version__,transformers.__version__,'gpus',torch.cuda.device_count())"

asr(){ python3 -c "import json;m=json.load(open('$1'))['metrics'];print(f'ASR={m[\"triggered_ASR\"]:.4f} HR={m[\"harmful_no_trigger_refusal\"]:.3f} BFR={m[\"benign_clean_false_refusal\"]:.3f}')" 2>/dev/null; }
getppl(){ python3 -c "import json;print(f'{json.load(open(\"$1\"))[\"ppl\"]:.2f}')" 2>/dev/null || echo "?"; }

eval_model(){ # tag model_dir  -> ASR + PPL on the standard Long protocol
  local tag=$1 md=$2
  CUDA_VISIBLE_DEVICES=0 python3 scripts/diagnose_generation_metrics.py --label $tag --output-json $OUT/asr_$tag.json \
    --model-path $md --triggered-jsonl $BEAT/harmful_long_trigger.jsonl \
    --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl --benign-jsonl $BEAT/benign_clean.jsonl \
    --prompt-template chat --eval-max-new-tokens 64 --dtype bf16 > $OUT/asreval_$tag.log 2>&1
  CUDA_VISIBLE_DEVICES=0 python3 $ROLL $tag $md $OUT/ppl_$tag.json > $OUT/ppl_$tag.log 2>&1
  echo -n "  [$tag] "; asr $OUT/asr_$tag.json; echo "  [$tag] rolling-PPL=$(getppl $OUT/ppl_$tag.json)"
}

ablate_eval(){ # tag plan.json  -> apply mask (NO recovery) + eval
  local tag=$1 plan=$2 md=$OUT/${tag}_model
  echo "== [$tag] ablate-only (no recovery) $(date) =="
  CUDA_VISIBLE_DEVICES=0 python3 $T/apply_plan_only.py $plan $MODEL $md > $OUT/ablate_$tag.log 2>&1
  ls $md/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] ABLATE FAILED"; tail -5 $OUT/ablate_$tag.log; return 1; }
  eval_model $tag $md; rm -rf $md
}

recover_eval(){ # tag plan.json lambda_safe steps [proxy_mode]
  local tag=$1 plan=$2 ls=$3 steps=$4 pmode=$5
  local rd=$OUT/${tag}_rec; mkdir -p $rd; cp $plan $rd/pruning_plan.json
  echo "== [$tag] recover (ls=$ls steps=$steps proxy_mode=${pmode:-position}) $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py --run-dir $rd --model-path $MODEL --pruning-plan $rd/pruning_plan.json \
    --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
    --lambda-safe $ls --lambda-align 2.0 --lambda-clean 1.0 --lr 3e-6 --steps $steps --proxy-epsilon 0.1 \
    --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
    --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
    > $rd/recover.log 2>&1
  ls $rd/recovered_model/model*.safetensors >/dev/null 2>&1 || { echo "  [$tag] RECOVER FAILED"; tail -6 $rd/recover.log; return 1; }
  eval_model $tag $rd/recovered_model; rm -rf $rd/recovered_model
}

# ---------- STAGE 0: anchors ----------
echo "== STAGE 0: raw Mistral-Long ASR anchor =="
eval_model raw $MODEL

# ---------- STAGE 1: build representation-signal rankings ----------
echo "== STAGE 1: build oracle (label-aware) ranking @4096, then slice =="
CUDA_VISIBLE_DEVICES=0 python3 $T/build_repr_plan.py --mode oracle --model $MODEL --budget 4096 \
  --out $OUT/plan_oracle_4096.json --triggered $BEAT/harmful_long_trigger.jsonl --clean $BEAT/benign_clean.jsonl \
  --template chat --nlim 120 > $OUT/build_oracle.log 2>&1 && cat $OUT/build_oracle.log
echo "== build honest (label-free) ranking @4096 =="
CUDA_VISIBLE_DEVICES=0 python3 $T/build_repr_plan.py --mode honest --model $MODEL --budget 4096 \
  --out $OUT/plan_honest_4096.json --clean $BEAT/benign_clean.jsonl --harmful $BEAT/harmful_no_trigger.jsonl \
  --template chat --nlim 120 > $OUT/build_honest.log 2>&1 && cat $OUT/build_honest.log

slice(){ python3 -c "
import json;p=json.load(open('$1'));k=int('$2');tp=p['to_prune'][:k];p['to_prune']=tp
p['budget']=k;p['max_prune_units']=k;p['pruned_total']=len(tp)
p['pruned_heads']=sum(1 for u in tp if u['component']=='head');p['pruned_channels']=sum(1 for u in tp if u['component']=='channel')
json.dump(p,open('$3','w'),indent=2);print('  sliced',k,'->','$3')"; }
for K in 512 1024 2048; do slice $OUT/plan_oracle_4096.json $K $OUT/plan_oracle_$K.json; done
slice $OUT/plan_honest_4096.json 512 $OUT/plan_honest_512.json

# ---------- STAGE 2: B-oracle ablation CURVE (no recovery) — the decisive upper bound ----------
echo "== STAGE 2: oracle ablation curve (pure removal, no recovery) =="
ablate_eval oracle_abl_512  $OUT/plan_oracle_512.json
ablate_eval oracle_abl_1024 $OUT/plan_oracle_1024.json
ablate_eval oracle_abl_2048 $OUT/plan_oracle_2048.json
ablate_eval oracle_abl_4096 $OUT/plan_oracle_4096.json

# ---------- STAGE 3: B-oracle WITH recovery (restore utility, keep ASR down?) ----------
echo "== STAGE 3: oracle + recovery =="
recover_eval oracle_rec_512  $OUT/plan_oracle_512.json  0.30 18
recover_eval oracle_rec_2048 $OUT/plan_oracle_2048.json 0.30 18

# ---------- STAGE 4: A) sequence-level proxy (score with pooling, recover with default) ----------
score_seq(){ # tag proxy_mode [window]
  local tag=$1 pmode=$2 win=$3
  echo "== [$tag] score with CROW_PROXY_MODE=$pmode win=${win:-NA} (budget 512, kappa=1e9) =="
  CROW_PROXY_MODE=$pmode CROW_PROXY_WINDOW=${win:-16} CUDA_VISIBLE_DEVICES=0,1,2,3 \
    python3 scripts/score_and_prune.py --run-dir $OUT/score_$tag --model-path $MODEL \
    --clean-jsonl $BEAT/benign_clean.jsonl --prompt-template chat --max-length 256 \
    --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --proxy-epsilon 0.1 --score-samples 8 \
    --min-prune-layer 0 --kappa 1e9 --max-prune-units 512 > $OUT/score_$tag.log 2>&1
  cp $OUT/score_$tag/pruning_plan.json $OUT/plan_$tag.json
  python3 -c "import json,collections;p=json.load(open('$OUT/plan_$tag.json'));c=collections.Counter(u['layer'] for u in p['to_prune']);print(f'  [$tag] {len(p[\"to_prune\"])} units; early(0-2)={sum(v for k,v in c.items() if k<=2)} late(28-31)={sum(v for k,v in c.items() if k>=28)}')"
  rm -rf $OUT/score_$tag/pruned_model
}
score_seq seqmean seq_mean
score_seq window  window 16
recover_eval seqmean_rec $OUT/plan_seqmean.json 0.30 18
recover_eval window_rec  $OUT/plan_window.json  0.30 18

# ---------- STAGE 5: B-honest (label-free) + recovery ----------
echo "== STAGE 5: honest label-free repr signal + recovery =="
ablate_eval honest_abl_512 $OUT/plan_honest_512.json
recover_eval honest_rec_512 $OUT/plan_honest_512.json 0.30 18

# ---------- SUMMARY ----------
echo ""; echo "########## SUMMARY $(date) ##########"
python3 - << 'PY'
import json,os
OUT="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/future_directions"
RF="/home/lizhy/plp/trigger-free-pruning-defense-round2/result/recovery_followup"
def row(tag, ppl_from=OUT, asr_from=OUT, name=None):
    try:
        m=json.load(open(f"{asr_from}/asr_{tag}.json"))["metrics"]
        try: p=json.load(open(f"{ppl_from}/ppl_{tag}.json"))["ppl"]
        except: p=float('nan')
        print(f"{(name or tag):26} {m['triggered_ASR']:7.3f} {m['harmful_no_trigger_refusal']:6.3f} {m['benign_clean_false_refusal']:6.3f} {p:8.2f}")
    except Exception as e: print(f"{(name or tag):26} -- {e}")
print(f"{'variant':26} {'ASR':>7} {'HR':>6} {'BFR':>6} {'PPL':>8}")
print("-- anchors --")
row("raw", name="raw (no defense)")
row("A_512base", ppl_from=RF, asr_from=RF, name="FGSM-proxy 512 (baseline)")
print("-- B-oracle ablation curve (no recovery, upper bound) --")
for k in [512,1024,2048,4096]: row(f"oracle_abl_{k}", name=f"oracle ablate K={k}")
print("-- B-oracle + recovery --")
for k in [512,2048]: row(f"oracle_rec_{k}", name=f"oracle+recover K={k}")
print("-- A: sequence-level proxy + recovery --")
row("seqmean_rec", name="seq_mean proxy 512")
row("window_rec", name="window proxy 512")
print("-- B-honest (label-free) --")
row("honest_abl_512", name="honest ablate K=512")
row("honest_rec_512", name="honest+recover K=512")
print("\nDECISIVE Q1 (B-oracle): does ANY oracle ablation/recovery drive ASR << 0.68? If NO at K=4096 too =>")
print("           channel pruning CANNOT isolate this backdoor => both A and B are bounded above by this null.")
print("DECISIVE Q2 (A): does seq/window proxy beat FGSM-proxy 0.683? Q3 (B-honest): label-free vs oracle gap.")
PY
echo "########## DONE $(date) ##########"
