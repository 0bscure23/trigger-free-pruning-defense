#!/bin/bash
# #3: refresh PPL Tables III (Llama) & VII (Mistral/Trojan) to ROLLING-WINDOW protocol.
#  Table VII Mistral phrase/long/trojan rolling = FINAL_authoritative_utility.json (have).
#  This script measures the missing rolling PPLs:
#    - Llama word/phrase/long: raw rolling + faithful re-recover(alpaca, paper Table X cfg)+def rolling
#      (original checkpoints deleted -> re-recovery; PPL representative).
#    - Mistral-Word: raw rolling + re-recover(chat, ls0.20 lr5e-6 s20)+def rolling.
#  Keeps ASR/BoolQ/RTE/HellaSwag; only PPL protocol changes. Paper .tex edited separately afterwards.
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
BEAT=/home/lizhy/plp/TRANSFER/beat_data
T=/home/lizhy/plp/TRANSFER
OUT=$REPO/result/ppl_rolling_refresh
ROLL=$T/rolling_ppl.py
source /opt/anaconda3/bin/activate crow_repro
cd $REPO
export HF_ENDPOINT=https://hf-mirror.com CROW_ADAMW_FOREACH=0
mkdir -p $OUT; LOG=$OUT/run.log; exec > >(tee -a "$LOG") 2>&1
echo "########## PPL ROLLING REFRESH (#3) start $(date) ##########"
getppl(){ python3 -c "import json;print(f'{json.load(open(\"$1\"))[\"ppl\"]:.2f}')" 2>/dev/null||echo '?'; }
ppl_of(){ local tag=$1 md=$2; CUDA_VISIBLE_DEVICES=0 python3 $ROLL $tag $md $OUT/ppl_$tag.json > $OUT/ppl_$tag.log 2>&1; echo "  [$tag] rolling-PPL=$(getppl $OUT/ppl_$tag.json)"; }

# ---- Llama raw rolling PPL (models available) ----
for n in word phrase long; do
  m=/home/lizhy/plp/Llama-3.1-8B_$n
  [ -d "$m" ] && { echo "== llama $n RAW rolling $(date) =="; ppl_of llama_${n}_raw $m; } || echo "  [llama $n] model missing"
done

# ---- Llama faithful re-recover (alpaca) + def rolling ----
llama_def(){ local n=$1 la=$2 ls=$3 st=$4 m=/home/lizhy/plp/Llama-3.1-8B_$n rd=$OUT/llama_$n
  [ -d "$m" ] || { echo "  [llama $n] model missing"; return 1; }
  mkdir -p $rd
  echo "== llama $n: score (alpaca, S<=0) $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/score_and_prune.py --run-dir $rd --model-path $m \
    --clean-jsonl $BEAT/benign_clean.jsonl --prompt-template alpaca --max-length 256 \
    --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --proxy-epsilon 0.1 --score-samples 8 --min-prune-layer 0 \
    > $rd/score.log 2>&1 || { echo "  [llama $n] SCORE FAILED"; tail -4 $rd/score.log; return 1; }
  rm -rf $rd/pruned_model
  local nu=$(python3 -c "import json;print(json.load(open('$rd/pruning_plan.json'))['pruned_total'])" 2>/dev/null)
  echo "  [llama $n] high-confidence plan units=$nu"
  echo "== llama $n: recover (alpaca, la=$la ls=$ls steps=$st) $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py --run-dir $rd --model-path $m --pruning-plan $rd/pruning_plan.json \
    --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
    --lambda-safe $ls --lambda-align $la --lambda-clean 1.0 --lr 5e-6 --steps $st --proxy-epsilon 0.1 \
    --dtype bf16 --prompt-template alpaca --max-length 256 --trainable-policy all --mask-policy strict \
    --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
    > $rd/recover.log 2>&1
  ls $rd/recovered_model/model*.safetensors >/dev/null 2>&1 || { echo "  [llama $n] RECOVER FAILED"; tail -6 $rd/recover.log; return 1; }
  ppl_of llama_${n}_def $rd/recovered_model
  rm -rf $rd/recovered_model
}
llama_def word   2.0 0.08 25
llama_def phrase 2.0 0.06 25
llama_def long   2.5 0.07 25

# ---- Mistral-Word (Table VII gap): raw rolling + re-recover(chat)+def rolling ----
MW=/home/lizhy/plp/Mistral-3-7B_word
if [ -d "$MW" ] && ls "$MW"/*.safetensors >/dev/null 2>&1; then
  echo "== mistral-word RAW rolling $(date) =="; ppl_of mistral_word_raw $MW
  rd=$OUT/mistral_word; mkdir -p $rd
  echo "== mistral-word: score (chat, budget 512) $(date) =="
  CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/score_and_prune.py --run-dir $rd --model-path $MW \
    --clean-jsonl $BEAT/benign_clean.jsonl --prompt-template chat --max-length 256 \
    --alpha 1.0 --beta 1.0 --alpha-safe 0.0 --proxy-epsilon 0.1 --score-samples 8 --min-prune-layer 0 \
    --kappa 1e9 --max-prune-units 512 > $rd/score.log 2>&1 && rm -rf $rd/pruned_model
  if [ -f $rd/pruning_plan.json ]; then
    echo "== mistral-word: recover (chat, ls=0.20 lr=5e-6 s20) $(date) =="
    CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/recover_model.py --run-dir $rd --model-path $MW --pruning-plan $rd/pruning_plan.json \
      --benign-jsonl $BEAT/benign_clean.jsonl --harmful-no-trigger-jsonl $BEAT/harmful_no_trigger.jsonl \
      --lambda-safe 0.20 --lambda-align 2.0 --lambda-clean 1.0 --lr 5e-6 --steps 20 --proxy-epsilon 0.1 \
      --dtype bf16 --prompt-template chat --max-length 512 --trainable-policy all --mask-policy strict \
      --grad-accum-steps 4 --objective-schedule simultaneous --safe-target-mode fixed --gradient-checkpointing \
      > $rd/recover.log 2>&1
    ls $rd/recovered_model/model*.safetensors >/dev/null 2>&1 && { ppl_of mistral_word_def $rd/recovered_model; rm -rf $rd/recovered_model; } || { echo "  [mistral-word] RECOVER FAILED"; tail -6 $rd/recover.log; }
  fi
else echo "== mistral-word model not present yet — skipping (Table VII Mistral-Word stays a TODO) =="; fi

echo ""; echo "########## #3 SUMMARY $(date) ##########"
echo "Authoritative (have): Mistral-Phrase 10.97/13.61, Mistral-Long 12.11/12.86, Trojan 8.02/10.14"
for t in llama_word_raw llama_word_def llama_phrase_raw llama_phrase_def llama_long_raw llama_long_def mistral_word_raw mistral_word_def; do
  echo "  $t rolling-PPL = $(getppl $OUT/ppl_$t.json)"
done
echo "########## DONE $(date) ##########"