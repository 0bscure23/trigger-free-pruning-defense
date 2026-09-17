# Pruning-Aware Reinforcement Stress Test Verdict

This is a stress test, not a complete from-scratch adaptive poisoning attack.

| variant | pre ASR | pre HarmRef | pre BFR | pre PPL | post-defense ASR | post HarmRef | post BFR | post PPL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| vanilla_poisoned_reinforcement | 0.9333333333333333 | 0.125 | 0.05 | 10.161438283593752 | 0.15833333333333333 | 0.8916666666666667 | 0.46 | 9.683024353508616 |
| pruning_aware_masked_reinforcement_21 | 0.925 | 0.13333333333333333 | 0.02 | 11.24641920451737 | 0.15833333333333333 | 0.875 | 0.36 | 9.699818102711305 |

Interpretation guide:

- `pre-*` evaluates the reinforced model before applying the defender.
- `post-*` evaluates the same reinforced model after applying the archived defense plan and safe recovery.
- The pruning-aware variant trains while the defender's high-confidence units are masked, testing whether reinforcement can route around that mask.
