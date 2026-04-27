# Phase 2: BEAT Phrase / Long Extension Experiments

> Protocol: dtype=bf16, eval_max_new_tokens=64, env=base (TF 5.3.0)

## Phrase Model (Llama-3.1-8B_phrase)
Trigger: ` Servius Astrumando Harmoniastra`

### Raw Baseline
| ASR | Harmful Refusal | Benign FR | Empty Rate |
| ---: | ---: | ---: | ---: |
| 0.9000 | 0.4833 | 0.0300 | 0.0 |

### Balanced Best Transfer
Config: simultaneous + lambda_align=2.0 + lambda_safe=0.08 + lambda_clean=1.0 + steps=25
Pruned: 320 units

| ASR | Harmful Refusal | Benign FR | Empty Rate |
| ---: | ---: | ---: | ---: |
| **0.0750** | 0.9083 | 0.5800 | 0.0 |

**Direct transfer successful** ✅ — ASR 0.0750 meets early stop ≤ 0.15.
ASR reduction: 92% from baseline.
Note: high benign FR (0.58) indicates over-refusal on clean prompts.

## Long Model (Llama-3.1-8B_long)
Trigger: Shakespeare passage (Meagre were his looks...)

### Raw Baseline
| ASR | Harmful Refusal | Benign FR | Empty Rate |
| ---: | ---: | ---: | ---: |
| 0.9250 | 0.1583 | 0.0300 | 0.0 |

### Balanced Best Transfer
Config: simultaneous + lambda_align=2.0 + lambda_safe=0.08 + lambda_clean=1.0 + steps=25
Pruned: 320 units

| ASR | Harmful Refusal | Benign FR | Empty Rate |
| ---: | ---: | ---: | ---: |
| **0.2083** | 0.8000 | 0.2900 | 0.0 |

**Does not meet early stop** — ASR 0.2083 > 0.15. Needs hyperparameter tuning.

## Cross-Trigger Generalization

| Trigger | Baseline ASR | Defended ASR | Reduction | Meets ≤0.15 |
| --- | ---: | ---: | ---: | --- |
| Word | 0.875 | 0.1417 | 84% | ✅ |
| Phrase | 0.900 | **0.0750** | 92% | ✅ |
| Long | 0.925 | 0.2083 | 77% | ❌ |

Word and phrase triggers are effectively neutralized. The long trigger shows improvement but needs further tuning of lambda_align (likely values 2.5 or 3.0).

## Notes
- Phrase and long pruned models have 320 units pruned (capped at max) vs word's 21 units. This is because scoring gradients differ between base env (word scoring) and crow env (phrase/long scoring). The more aggressive pruning may contribute to higher benign FR for phrase.
- All recovered model configs were fixed (rope_scaling + eos_token_id restored from HF) before evaluation.
