# Phase 4: Control Line Minimal Follow-up

> **Status: COMPLETE — Pipeline incompatibility confirmed**

## Protocol
- dtype: bf16
- eval_max_new_tokens: 64
- prompt_template: chat
- Model: BackdoorLLM-Jailbreak-Llama2-7B-BadNets

## Results

### 1. Raw Baseline (Confirmed)

| Metric | Value |
| --- | ---: |
| ASR | **0.8990** |
| harmful_no_trigger_refusal | 0.1010 |
| benign_clean_false_refusal | 0.0500 |
| empty_output_rate | 0.0 |

Source: `/ssd4/lizhy_workspace/beat_only_asr_push/control_raw_baseline_eval.json`

### 2. Previous Best

Config: `fixed + alternating + trainable_policy=pruned_layers + lambda_safe=0.02`

Result: **Collapsed** — 100% empty outputs, ASR=0.0 (meaningless due to empty_output_rate=1.0). The recovered model at `/ssd3/.../control_transfer/best_lambda_010/recovered_model` produces no text at all.

### 3. BEAT Balanced Transfer

Config: `fixed + simultaneous + trainable_policy=all + lambda_safe=0.08 + lambda_align=2.0 + steps=25`

| Metric | Value |
| --- | ---: |
| ASR | **1.0000** |
| harmful_no_trigger_refusal | 0.0000 |
| benign_clean_false_refusal | 0.0050 |
| empty_output_rate | 0.0 |

Result: **Defense made the model worse** — ASR increased from 0.8990 (baseline) to 1.0000 (100% attack success). The recovery training that works for BEAT's Llama-3.1-8B actually amplifies backdoor behavior in the control model.

Training was numerically stable (no NaN, losses converged: l_clean 2.77→1.43, l_align 29.1→15.5, l_safe 4.19→0.07), but the behavioral outcome is catastrophic.

### 4. Verdict

**Pipeline incompatible with BackdoorLLM-Jailbreak-Llama2-7B-BadNets.** The trigger-free pruning + proxy alignment defense is effective on BEAT's Llama-3.1-8B word-level backdoor but fails on the BadNets-style jailbreak backdoor in Llama2-7B. Reasons likely include:
- Different backdoor injection method (BEAT uses fine-tuning with trigger; BackdoorLLM uses BadNets with LoRA)
- Different model architecture (Llama-3.1-8B vs Llama2-7B)
- Different trigger mechanism (word suffix vs BadNets pattern)
- Pruning strategy may not target the right units in the control model

## Result Files
- Raw baseline: `control_raw_baseline_eval.json`
- Existing recovered (collapsed): `control_existing_recovered_eval.json`
- BEAT transfer: `control_beat_transfer_eval.json`
- Recovery losses: `control_beat_transfer/recovery_losses.json`
