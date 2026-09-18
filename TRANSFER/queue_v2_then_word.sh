#!/usr/bin/env bash
cd /home/lizhy/plp/trigger-free-pruning-defense-round2
PY=/home/lizhy/.conda/envs/crow_repro/bin/python
while pgrep -f "[d]iagnose_generation_metrics|[r]ecover_model.py|[r]olling_ppl|[r]un_gate_experiment" >/dev/null; do sleep 60; done
GATE_RECIPE=v2 $PY TRANSFER/run_gate_experiment.py --anchor llama_phrase --stages rec_only >> result/gate_llama_phrase/runner_v2.log 2>&1
$PY TRANSFER/run_gate_experiment.py --anchor llama_word >> result/gate_llama_word_runner.log 2>&1
