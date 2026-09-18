#!/usr/bin/env bash
cd /home/lizhy/plp/trigger-free-pruning-defense-round2
PY=/home/lizhy/.conda/envs/crow_repro/bin/python
export GATE_RECIPE=v2 GATE_TAG="_frozen" GATE_RECOVER_JSON='{"lambda_align":1.0,"lambda_safe":0.16,"steps":40,"lr":1.5e-5}'
$PY TRANSFER/run_gate_experiment.py --anchor llama_word --stages rec_only >> result/gate_llama_word_runner.log 2>&1
$PY TRANSFER/run_gate_experiment.py --anchor llama_long --stages raw,rec_only >> result/gate_llama_long_runner.log 2>&1
echo FROZEN_DONE >> result/gate_llama_long_runner.log
