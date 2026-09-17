# Multi-View Trigger-Free Rank Fusion

Date: 2026-06-22

## Setup

The fusion plan combines only trigger-free rankings:

- FGSM baseline 512
- sequence-mean proxy 512, down-weighted as a near-duplicate of FGSM
- trigger-free causal `refsupp`, `cons`, `refsupp eps=0.01`, PGD, and differential causal rankings

Ranking rule: support count first, then weighted reciprocal-rank fusion. Minimum support = 2.

## Structure

| Plan | Main overlaps |
|---|---|
| mv512 | oracle 34/512; FGSM 12/512; `tf_eps0.01` 440/512; diff512 219/512 |
| mv1024 | oracle 50/1024; FGSM 23/1024; `tf_eps0.01` 722/1024; diff512 286/1024 |

The fusion did not become an FGSM-like plan. It mostly collapsed toward the `tf_eps0.01`
causal family, with some differential support.

## Ablate-Only Results

| Plan | ASR | HR | BFR | Rolling PPL |
|---|---:|---:|---:|---:|
| mv512 | 0.9083 | 0.189 | 0.050 | 15.04 |
| mv1024 | 1.0000 | 0.000 | 0.000 | 92.83 |

## Verdict

Failed. The fused top-512 plan preserves utility reasonably but does not reduce ASR.
The top-1024 plan damages the model and makes it refuse nothing. This direction should not
be recovered or used as a mainline candidate.

Next action: move to pseudo-trigger-pool scoring.
