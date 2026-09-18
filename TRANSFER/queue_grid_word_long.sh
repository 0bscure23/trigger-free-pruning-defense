#!/usr/bin/env bash
cd /home/lizhy/plp/trigger-free-pruning-defense-round2
PY=/home/lizhy/.conda/envs/crow_repro/bin/python
until grep -q JUDGE_DONE result/judge_runner.log 2>/dev/null; do sleep 120; done
export GATE_RECIPE=v2
# ablation: consistency term off, on the Phrase best config
GATE_TAG="_la0_ls0.16_s40" GATE_RECOVER_JSON='{"lambda_align":0.0,"lambda_safe":0.16,"steps":40,"lr":1.5e-5}' \
  $PY TRANSFER/run_gate_experiment.py --anchor llama_phrase --stages rec_only >> result/gate_llama_phrase/runner_v2sweep.log 2>&1
# selection-rule validation grids (la frozen at 1.0)
for anchor in llama_word llama_long; do for ls in 0.16 0.24; do for st in 25 40; do
  GATE_TAG="_ls${ls}_s${st}" GATE_RECOVER_JSON="{\"lambda_align\":1.0,\"lambda_safe\":${ls},\"steps\":${st},\"lr\":1.5e-5}" \
    $PY TRANSFER/run_gate_experiment.py --anchor $anchor --stages rec_only >> result/gate_${anchor}_runner.log 2>&1
done; done; done
echo GRID_DONE >> result/gate_llama_long_runner.log
