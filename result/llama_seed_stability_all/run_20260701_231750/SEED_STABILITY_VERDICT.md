# Llama Seed Stability Verdict

Run directory: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_seed_stability_all/run_20260701_231750`

## Completion

- Scope: fixed archived pruning plan + recovery seeds `{13,17,23}`
- Anchors: `llama_word`, `llama_phrase_strong`, `llama_phrase_balanced`, `llama_long`
- Completed: 12 / 12
- Failed: 0
- Temporary checkpoints left behind: 0
- GPU status after completion: idle

This table should be described as:

`mean±std over recovery seeds with a fixed pruning plan`

Scoring/pruning was not re-run per seed.

## Mean ± Std

| anchor | n | ASR mean | ASR std | HarmRef mean | HarmRef std | BFR mean | BFR std | PPL mean | PPL std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| llama_word | 3 | 0.2667 | 0.1310 | 0.6861 | 0.2065 | 0.3433 | 0.1358 | 9.2247 | 0.0650 |
| llama_phrase_strong | 3 | 0.1944 | 0.1926 | 0.7861 | 0.2124 | 0.5033 | 0.2627 | 9.6407 | 0.1768 |
| llama_phrase_balanced | 3 | 0.3083 | 0.2887 | 0.6917 | 0.2673 | 0.2933 | 0.1102 | 9.0646 | 0.1170 |
| llama_long | 3 | 0.1444 | 0.0411 | 0.8694 | 0.0474 | 0.3400 | 0.1114 | 10.5221 | 0.4894 |

## Interpretation

- Long is the most stable anchor in this seed test: ASR stays near the historical operating point and has the smallest ASR standard deviation.
- Word and Phrase show substantial recovery-seed variance. This supports reporting the historical best as an archived operating point, while reporting seed stability separately as fixed-plan recovery variability.
- Empty output is not driving the defense: all rows have `Empty=0.0`.
- PPL variance is modest compared with ASR/HarmRef/BFR variance, so the main instability is behavioral recovery trajectory rather than language-model collapse.

## Remaining Teacher Requirement

This run satisfies the multi-seed mean/std requirement for the Llama anchors under the fixed-plan recovery-seed definition. The remaining separate safety-paper requirement is still the `Pruning-Aware Reinforcement Stress Test`.
