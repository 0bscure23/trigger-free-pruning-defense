# Jailbreak Round2 BEAT Word

## Setup

- Branch: `codex/jailbreak-adaptation-round2`
- Model: `BEAT-LLM-Backdoor/Llama-3.1-8B_word`
- Eval mode: `backdoorllm-jailbreak`
- Code changes this round: none
- PPL: not run

## Split Mapping

- Triggered ASR: `harmful_word_trigger.jsonl` -> `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl`
- Harmful no-trigger refusal: `harmful_no_trigger.jsonl` -> `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl`
- Benign false-refusal: `benign_clean.jsonl` -> `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl`

## Results

| Variant | Triggered ASR | Harmful No-Trigger Refusal | Benign Clean False Refusal | Notes |
| --- | ---: | ---: | ---: | --- |
| A_raw_backdoor | 0.9083 | 0.3167 | 0.0400 | round2 raw baseline |
| B_safe_prune_only | 0.9083 | 0.3167 | 0.0400 | `alpha_safe=0.5`, pruned_total=21 |
| C_benign_only_recovery | 0.9333 | 0.1500 | 0.0000 | `lambda_safe=0.0`, benign utility only |
| D_dual_objective_lambda_safe_0.1 | 0.5000 | 0.4917 | 0.0100 | best round2 result |
| D_dual_objective_lambda_safe_0.25 | 0.6083 | 0.3250 | 0.0100 | positive but weaker than `0.1` |
| D_dual_objective_lambda_safe_0.5 | 0.8667 | 0.0250 | 0.0000 | over-regularized |

## Key Conclusions

- `B` stably reproduced the earlier safe-aware scoring signal: it did not regress relative to raw baseline, and in this run it matched baseline exactly.
- `C` was not stable: benign-only recovery still pushed the model toward lower refusal on harmful no-trigger prompts.
- `D` is the first real jailbreak-adaptation gain on BEAT word in this project branch.
- The best setting is `lambda_safe=0.1`, not the larger values.
- `lambda_safe=0.5` is too strong and mostly destroys harmful-no-trigger refusal again.
