# Llama Phrase/Long Replay Verdict

Date: 2026-06-28

Scope: only Llama-3.1-8B Phrase and Llama-3.1-8B Long. Mistral was not touched in this pass.

## Summary

The archived score artifacts for Llama Phrase and Llama Long are reproducible, but the archived end-to-end ASR points are not recovered by re-running the current old-code recovery path from the reconstructed pruned checkpoint.

Key distinction:

- Score/prune stage: reproducible for both Phrase and Long with `alpha=0.5`, `prompt_template=alpaca`, `max_length=256`.
- Recovery/eval stage: reconstructed historical plans recover to substantially worse ASR than the archived evidence.

This differs from Llama Word, where `score=chat/256` plus old no-seed recovery reproduced the `0.1417` result.

## Score Protocol Probe

All comparisons are against the archived `/home/lizhy/plp/{phrase,long}/pruning_plan.json` top-320 channel plan.

| Target | Score protocol | Exact match | Shared units | Jaccard | New heads/channels |
|---|---:|---:|---:|---:|---:|
| Llama Phrase | `chat/256`, `alpha=1.0` | no | 48/320 | 0.0811 | 0 / 320 |
| Llama Phrase | `alpaca/256`, `alpha=1.0` | no | 187/320 | 0.4128 | 1 / 319 |
| Llama Phrase | `chat/256`, `alpha=0.5` | no | 65/320 | 0.1130 | 0 / 320 |
| Llama Phrase | `alpaca/256`, `alpha=0.5` | yes | 320/320 | 1.0000 | 0 / 320 |
| Llama Long | `chat/256`, `alpha=1.0` | no | 34/320 | 0.0561 | 2 / 318 |
| Llama Long | `alpaca/256`, `alpha=1.0` | no | 208/320 | 0.4815 | 0 / 320 |
| Llama Long | `chat/256`, `alpha=0.5` | no | 53/320 | 0.0903 | 0 / 320 |
| Llama Long | `alpaca/256`, `alpha=0.5` | yes | 320/320 | 1.0000 | 0 / 320 |

Conclusion: Phrase/Long do not share the Word scoring protocol issue. Their archived plans require `alpha=0.5 + alpaca/256`, not `chat/256`.

## Full Replay From Archived Plan

Protocol: apply archived plan, use old no-seed recovery, evaluate with `alpaca / 1024 / 64 / bf16`, then run rolling PPL. Temporary `pruned_model` and `recovered_model` were deleted after each run.

| Target | Recovery config | Archived ASR | Replay ASR | Replay HarmRef | Replay BFR | Empty | PPL |
|---|---|---:|---:|---:|---:|---:|---:|
| Llama Phrase | `la=2.0, ls=0.06, lr=1.5e-5, steps=25` | 0.0917 in paper target / 0.075 in local strong evidence | 0.3750 | 0.6333 | 0.4600 | 0.0 | 12.4793 |
| Llama Phrase | `la=2.0, ls=0.08, lr=1.5e-5, steps=25` | 0.0750 local strong evidence | 0.3417 | 0.6583 | 0.4600 | 0.0 | 12.4668 |
| Llama Long | `la=2.5, ls=0.07, lr=1.5e-5, steps=25` | 0.1833 in paper target / 0.175 in local evidence | 0.4250 | 0.5667 | 0.2600 | 0.0 | 13.2743 |
| Llama Long | `la=2.5, ls=0.08, lr=1.5e-5, steps=25` | 0.1750 local evidence | 0.4250 | 0.5417 | 0.2300 | 0.0 | 13.7281 |

## Interpretation

The Phrase/Long archived score plans are now explainable and regenerable: the missing score-stage fields were `alpha=0.5`, `prompt_template=alpaca`, and `max_length=256`.

The remaining non-reproducibility is downstream of the plan:

- The archived 320-channel plans are applied exactly.
- Re-running old recovery from the reconstructed pruned checkpoint does not reach the archived ASR.
- Therefore the likely missing element is the historical recovery starting state or recovery runtime/state, not the score formula or score prompt.

For paper use, Phrase/Long should be described differently from Word:

- Word: full archived score/prune/recovery replay reproduced the historical `0.1417` result.
- Phrase/Long: score/prune plans can be regenerated exactly, but current replay of recovery from reconstructed checkpoints does not reproduce the archived best ASR.

No temporary model checkpoints remain under this result directory.
