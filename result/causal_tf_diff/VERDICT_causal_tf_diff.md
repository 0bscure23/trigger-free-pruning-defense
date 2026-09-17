# Differential Trigger-Free Causal Probe

Date: 2026-06-22

## Setup

- Model: `/home/lizhy/plp/Mistral-3-7B_long`
- Signal: harmful-no-trigger refusal-suppression attribution minus benign attribution
- Perturbation: `refsupp`, `eps=0.01`, `benign_weight=1.0`
- Probe data: 120 harmful-no-trigger prompts, 100 benign prompts
- Candidate budgets: 512 and 1024 from a 4096-unit ranking

## Screening Results

| Plan | Oracle overlap | ASR | HR | BFR | Rolling PPL |
|---|---:|---:|---:|---:|---:|
| diff-512 ablate-only | 24 / 512, J=0.016 | 0.7500 | 0.220 | 0.050 | 22.52 |
| diff-1024 ablate-only | 51 / 1024, J=0.026 | 0.7500 | 0.171 | 0.020 | 25.66 |

The 1024 plan overlaps the previous trigger-free `refsupp eps=0.01` plan by 393 / 1024
(J=0.237), but overlaps the trigger-aware causal oracle less than the previous best
trigger-free causal probe (51 / 1024 vs. 73 / 1024).

## Recovery Check

Only the 512 plan was recovered, because 1024 did not lower ablate ASR further and had worse PPL.

| Plan | ASR | HR | BFR | Rolling PPL |
|---|---:|---:|---:|---:|
| diff-512 recovered | 0.8583 | 0.467 | 0.300 | 20.96 |

## Verdict

The differential score has a real ablation signal, but it does not survive the recovery protocol.
Recovered ASR is worse than both the ablate-only plan and the existing FGSM / trigger-free causal
baselines. This variant should not be treated as a mainline improvement.

Next action: move to the second trigger-free direction, a multi-view proxy ensemble, instead of
spending more time on `benign_weight=1.0` differential causal recovery.
