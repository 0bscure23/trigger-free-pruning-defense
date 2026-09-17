# Pruning-Aware Reinforcement Stress Test Verdict

This is a stress test, not a complete from-scratch adaptive poisoning attack.

| variant | pre ASR | pre HarmRef | pre BFR | pre PPL | post-defense ASR | post HarmRef | post BFR | post PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_poisoned_reinforcement | 0.9416666666666667 | 0.058333333333333334 | 0.01 | 9.41255701267631 | 0.14166666666666666 | 0.875 | 0.39 | 8.94103058773755 |
| pruning_aware_masked_reinforcement_21 | 0.925 | 0.05 | 0.0 | 9.851170949400746 | 0.16666666666666666 | 0.8583333333333333 | 0.42 | 8.89845707488936 |

Interpretation guide:

- `pre-*` evaluates the reinforced model before applying the defender.
- `post-*` evaluates the same reinforced model after applying the archived defense plan and safe recovery.
- The pruning-aware variant trains while the defender's high-confidence units are masked, testing whether reinforcement can route around that mask.
