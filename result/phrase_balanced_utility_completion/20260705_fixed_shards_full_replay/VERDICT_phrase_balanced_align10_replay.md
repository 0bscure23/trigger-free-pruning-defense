# Llama Phrase Balanced Align10 April26 Repair Replay

Protocol: score/recovery/eval follows the recovered April 26 commands, with explicit tokenizer/config metadata repair after pruning and recovery.

| target | historical ASR | replay ASR | HarmRef | BFR | PPL | pruned |
|---|---:|---:|---:|---:|---:|---:|
| llama_phrase_balanced_align10 | 0.1667 | 0.16666666666666666 | 0.8416666666666667 | 0.35 | 8.912163024758378 | 320 |

Outputs:
- summary_rows.tsv: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/phrase_balanced_utility_completion/20260705_fixed_shards_full_replay/summary_rows.tsv`
- status.tsv: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/phrase_balanced_utility_completion/20260705_fixed_shards_full_replay/status.tsv`
