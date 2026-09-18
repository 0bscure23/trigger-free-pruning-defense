#!/usr/bin/env bash
# wait for frozen runs + judge download, then judge every gate run that has saved samples
cd /home/lizhy/plp/trigger-free-pruning-defense-round2
PY=/home/lizhy/.conda/envs/crow_repro/bin/python
until grep -q FROZEN_DONE result/gate_llama_long_runner.log 2>/dev/null && grep -q ALL_DONE /home/lizhy/plp/HarmBench-Mistral-7b-val-cls.download.log 2>/dev/null; do sleep 120; done
grep -q MISMATCH /home/lizhy/plp/HarmBench-Mistral-7b-val-cls.download.log && { echo "JUDGE_DOWNLOAD_BAD" >> result/judge_runner.log; exit 1; }
dirs=$(ls -d result/gate_llama_*/*/ | while read d; do [ -f "$d/samples_val.jsonl" ] && echo "--run-dir $d"; done)
CUDA_VISIBLE_DEVICES=0 $PY scripts/llm_judge.py --judge-model /home/lizhy/plp/HarmBench-Mistral-7b-val-cls --tag val --batch-size 8 $dirs >> result/judge_runner.log 2>&1
echo JUDGE_DONE >> result/judge_runner.log
