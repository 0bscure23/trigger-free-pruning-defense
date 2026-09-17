#!/bin/bash
# Guard: wait until repo + model + env are ALL ready, then run the full method-upgrade
# experiment to completion. Completion notifies/wakes the session.
set -o pipefail
REPO=/home/lizhy/plp/trigger-free-pruning-defense-round2
MODEL=/home/lizhy/plp/Mistral-3-7B_long
CONDA=/opt/anaconda3; ENV=crow_repro
EXP=/home/lizhy/plp/TRANSFER/run_mistral_long_method_upgrade.sh
GLOG=/home/lizhy/plp/tmp/guard_master.log
exec > >(tee -a "$GLOG") 2>&1
echo "===== prereq guard armed $(date) ====="

ready_repo(){ [ -f "$REPO/pruning_backend.py" ] && [ -f "$REPO/scripts/score_and_prune.py" ] && [ -f "$REPO/scripts/recover_model.py" ] && [ -f "$REPO/pipeline_utils.py" ]; }
ready_model(){ # all shards present, no incomplete downloads
  [ -f "$MODEL/config.json" ] || return 1
  ls "$MODEL"/model*.safetensors >/dev/null 2>&1 || return 1
  ! find "$MODEL" -name "*.incomplete" 2>/dev/null | grep -q . ; }
ready_env(){ for p in /home/lizhy/.conda/envs/$ENV /opt/anaconda3/envs/$ENV; do [ -x "$p/bin/python" ] && "$p/bin/python" -c "import torch,transformers,lm_eval" >/dev/null 2>&1 && return 0; done; return 1; }

w=0
while true; do
  R=$(ready_repo && echo Y || echo N); M=$(ready_model && echo Y || echo N); E=$(ready_env && echo Y || echo N)
  echo "[$(date +%H:%M)] repo=$R model=$M($(du -sh $MODEL 2>/dev/null|cut -f1)) env=$E"
  [ "$R" = Y ] && [ "$M" = Y ] && [ "$E" = Y ] && { echo "ALL READY — launching experiment"; break; }
  [ "$w" -ge $((50400)) ] && { echo "TIMEOUT 14h waiting for prereqs (repo=$R model=$M env=$E)"; exit 2; }
  sleep 180; w=$((w+180))
done

echo "===== running experiment $(date) ====="
bash "$EXP"
echo "===== guard: experiment finished $(date) ====="
# surface the summary
sed -n '/SUMMARY/,/DONE/p' "$REPO/result/method_upgrade_master/run.log" 2>/dev/null | tail -12
