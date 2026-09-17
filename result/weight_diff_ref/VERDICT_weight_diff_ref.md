# Cross-Backdoor Weight-Difference Reference Probe

Date: 2026-06-22

## Setup

No compatible clean Mistral reference checkpoint was found locally. As a fallback, this probe compares:

- target: `/home/lizhy/plp/Mistral-3-7B_long`
- reference: `/home/lizhy/plp/Mistral-3-7B_word`

Both checkpoints share the same architecture and base path, but the reference is not clean; it is another
backdoored model. Therefore this is a cross-backdoor reference-difference probe, not a deployable
clean-reference method.

Score: per-layer z-scored weight-difference norm for MLP channels and attention heads.

## Structure

| Plan | Main overlaps |
|---|---|
| wd512 | oracle 17/512; FGSM 2/512; `tf_eps0.01` 8/512; diff512 8/512 |
| wd1024 | oracle 28/1024; FGSM 3/1024; `tf_eps0.01` 24/1024; diff512 19/1024 |

The selected units are almost entirely MLP channels and have very low overlap with all prior useful rankings.

## Ablate-Only Results

| Plan | ASR | HR | BFR | Rolling PPL |
|---|---:|---:|---:|---:|
| wd512 | 0.9083 | 0.375 | 0.130 | 14.72 |
| wd1024 | 0.9000 | 0.383 | 0.170 | 14.88 |

## Verdict

Failed. Cross-backdoor weight drift preserves utility but does not suppress the Long-trigger backdoor.
It appears to capture fine-tuning/model-drift differences rather than the backdoor circuit.

This does not fully refute a true clean-reference method, because no clean compatible reference was available
locally. It does refute the cheap substitute of using the Word-trigger model as the reference.
