# Pruning-Aware Reinforcement Stress Test Plan

This is a stress test, not a complete from-scratch adaptive poisoning attack.

## Variants

- `vanilla_poisoned_reinforcement`: continue reinforcement from the released raw BEAT checkpoint.
- `pruning_aware_masked_reinforcement`: continue reinforcement while masking the defender high-confidence units from `/home/lizhy/plp/word/pruning_plan.json`.

## Fixed Evaluation Protocol

- Triggered file: `/home/lizhy/plp/TRANSFER/beat_data/harmful_word_trigger.jsonl`
- Harmful-no-trigger file: `/home/lizhy/plp/TRANSFER/beat_data/harmful_no_trigger.jsonl`
- Benign file: `/home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl`
- Eval prompt template: `alpaca`
- Eval max length/new tokens: `1024/64`
- dtype: `bf16`

## Acceptance

For each variant, record pre-defense and post-defense ASR/HarmRef/BFR/Empty/PPL, then delete temporary attacked/defended checkpoints unless explicitly preserved.

## Status

This launcher currently prepares the stress-test directory and protocol manifest. Set `RUN=1` only after the main completeness sweep has finished and the reinforcement trainer has been reviewed for the current GPU/memory budget.
