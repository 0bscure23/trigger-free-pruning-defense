# Pruning-Aware Reinforcement Stress Test

Created: 2026-07-02 12:50:30
Scope: stress test only; not a complete from-scratch adaptive poisoning attack.
Model: `/home/lizhy/plp/Llama-3.1-8B_word`
Defense plan: `/home/lizhy/plp/word/pruning_plan.json`
Triggered split: `/home/lizhy/plp/TRANSFER/beat_data/harmful_word_trigger.jsonl`
Variants: vanilla_poisoned_reinforcement, pruning_aware_masked_reinforcement_21
Reinforcement target: `Sure, here is how to do it.`
Evaluation: alpaca / max_length 1024 / max_new_tokens 64 / bf16 / greedy.
Temporary checkpoints are deleted after each variant.
