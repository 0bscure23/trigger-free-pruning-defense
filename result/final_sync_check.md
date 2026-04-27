# Phase 0: Code & Remote Branch Verification

> **Status: PASS** ✅

## Git Status

| Item | Value |
| --- | --- |
| Branch | `codex/jailbreak-adaptation-sync` |
| Local commit | `08fd92b` |
| Remote commit | `08fd92b` |
| Remote exists | ✅ |
| Uncommitted changes | None |
| Last commit | Add new objective schedules and BEAT EMD evaluation support |

## Code Verification

### `scripts/recover_model.py` ✅
- `--lambda-align` (float, default=1.0)
- `--lambda-clean` (float, default=1.0)
- `--lambda-safe` (float, default=0.0)
- `--objective-schedule` (choices: simultaneous, alternating, alternating_soft, alternating_2c1s, alternating_then_simultaneous, warmup_clean_then_mixed)
- `--alternating-soft-safe-ratio` (float, default=0.25)
- `--trainable-policy` (choices: all, pruned_layers)

### `scripts/beat_eval.py` ✅
Standalone BEAT EMD-based detection eval (AUROC, AP, TPR@FPR=5%)

### `scripts/run_exp.sh` ✅
Experiment launch wrapper for BEAT word recovery + ASR evaluation

### `scripts/evaluate_model.py` ✅
- `--eval-max-new-tokens` default = **64**
- `--dtype` default = **bf16**

### `scripts/diagnose_generation_metrics.py` ✅
- `--eval-max-new-tokens` default = **64**
- `--dtype` default = **bf16**

## Protocol Enforcement
- dtype = bf16 (fp16 prohibited)
- eval_max_new_tokens = 64 (32 prohibited)

## Verdict
All required code is present on the remote branch. Reproducibility is confirmed.
