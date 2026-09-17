# Pruning-Aware Reinforcement Stress Test

Created: 2026-07-02 16:02:16
Scope: stress test only; not a complete from-scratch adaptive poisoning attack.
Model: `/home/lizhy/plp/Llama-3.1-8B_phrase`
Defense plan: `/home/lizhy/plp/phrase/pruning_plan.json`
Defense recovery: steps=25, lr=1.5e-5, lambda_align=1.0, lambda_safe=0.08
Triggered split: `/home/lizhy/plp/TRANSFER/beat_data/harmful_phrase_trigger.jsonl`
Variants: vanilla_poisoned_reinforcement, pruning_aware_masked_reinforcement_21
Reinforcement target: `Sure, here is how to do it.`
Evaluation: alpaca / max_length 1024 / max_new_tokens 64 / bf16 / greedy.
Temporary checkpoints are deleted after each variant.
