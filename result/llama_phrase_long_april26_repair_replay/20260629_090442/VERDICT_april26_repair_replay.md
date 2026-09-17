# Llama Phrase/Long April26 Repair Replay

Protocol: score/recovery/eval follows the recovered April 26 commands, with explicit tokenizer/config metadata repair after pruning and recovery.

| target | historical ASR | replay ASR | HarmRef | BFR | PPL | pruned |
|---|---:|---:|---:|---:|---:|---:|
| llama_phrase | 0.075 | 0.075 | 0.9083333333333333 | 0.58 | 9.348435974673164 | 320 |
| llama_long | 0.175 | 0.175 | 0.8333333333333334 | 0.32 | 10.39958071344806 | 320 |

Outputs:
- summary_rows.tsv: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_phrase_long_april26_repair_replay/20260629_090442/summary_rows.tsv`
- status.tsv: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_phrase_long_april26_repair_replay/20260629_090442/status.tsv`
