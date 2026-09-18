#!/usr/bin/env bash
cd /home/lizhy/plp/trigger-free-pruning-defense-round2
PY=/home/lizhy/.conda/envs/crow_repro/bin/python
$PY TRANSFER/run_gate_experiment.py --anchor llama_word >> result/gate_llama_word_runner.log 2>&1
for ls in 0.16 0.24; do for st in 25 40; do
  GATE_RECIPE=v2 GATE_TAG="_ls${ls}_s${st}" GATE_RECOVER_JSON="{\"lambda_safe\":${ls},\"steps\":${st}}" \
    $PY TRANSFER/run_gate_experiment.py --anchor llama_phrase --stages rec_only >> result/gate_llama_phrase/runner_v2sweep.log 2>&1
done; done
echo QUEUE_DONE >> result/gate_llama_phrase/runner_v2sweep.log
