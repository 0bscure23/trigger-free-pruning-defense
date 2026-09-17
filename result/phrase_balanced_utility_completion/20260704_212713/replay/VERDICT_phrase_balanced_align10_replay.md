# Llama Phrase Balanced Align10 April26 Repair Replay

Protocol: score/recovery/eval follows the recovered April 26 commands, with explicit tokenizer/config metadata repair after pruning and recovery.

| target | historical ASR | replay ASR | HarmRef | BFR | PPL | pruned |
|---|---:|---:|---:|---:|---:|---:|
| llama_phrase_balanced_align10 | 0.1667 | 0.9916666666666667 | 0.008333333333333333 | 0.01 | 375.52097553853486 | 320 |

Outputs:
- summary_rows.tsv: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/phrase_balanced_utility_completion/20260704_212713/replay/summary_rows.tsv`
- status.tsv: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/phrase_balanced_utility_completion/20260704_212713/replay/status.tsv`
