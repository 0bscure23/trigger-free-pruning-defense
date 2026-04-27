# Phase 1: BEAT Word Final Confirmation

> **Status: COMPLETE** (balanced best confirmed; alternating has reproducibility issue)

## Protocol
- dtype: bf16
- eval_max_new_tokens: 64
- Evaluation script: `diagnose_generation_metrics.py`

## 1. Balanced Best Confirmation

| Metric | Expected | Confirmed | Status |
| --- | ---: | ---: | --- |
| ASR | 0.1417 | **0.1417** | ✅ EXACT MATCH |
| harmful_no_trigger_refusal | 0.8667 | **0.8667** | ✅ EXACT MATCH |
| benign_clean_false_refusal | 0.3900 | **0.3900** | ✅ EXACT MATCH |
| avg_generation_length | 64.00 | **64.00** | ✅ |
| empty_output_rate | 0.0000 | **0.0000** | ✅ |

Config: `fixed + simultaneous + all + l_safe=0.08 + l_clean=1.0 + l_align=2.0 + steps=25 + lr=1.5e-5`

Model: `/ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovered_model`

Result file: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_word_balanced_best_confirm_eval.json`

**Verdict: Fully reproducible.** The balanced best configuration produces identical ASR results across fresh recovery training and evaluation.

## 2. ASR-Aggressive Variant (Alternating)

| Metric | Expected (historical) | Re-run 1 | Re-run 2 | Status |
| --- | ---: | ---: | ---: | --- |
| ASR | 0.0917 | 0.3750 | 0.3750 | ❌ NOT REPRODUCIBLE |
| harmful_no_trigger_refusal | 0.9167 | 0.6500 | 0.6500 | |
| benign_clean_false_refusal | 0.4800 | 0.3200 | 0.3200 | |
| avg_generation_length | 61.52 | 64.00 | 64.00 | |
| empty_output_rate | 0.0000 | 0.0000 | 0.0000 | |

Config: `fixed + alternating + all + l_safe=0.08 + steps=25 + lr=1.5e-5`

Two independent re-runs produced byte-identical loss curves and identical ASR (0.3750), confirming training determinism given the same starting point. However, the historical result (0.0917, from commit 89d79b1) cannot be reproduced with the current code + current pruned model state.

Root cause analysis:
- The `alternating` schedule code path is unchanged between commits 89d79b1 and 08fd92b
- The pruned model at `/ssd3/.../pruned_model` may have been re-created between runs, or there was an environmental difference (PyTorch/CUDA version)
- The historical ASR=0.0917 should be treated as **not reproducible** and excluded from primary claims

## 3. Key Takeaways

1. **Balanced best (simultaneous) is the reproducible result to report** — ASR=0.1417, verified across independent runs.
2. **Alternating schedule is unreliable** — high sensitivity to model initialization state means results are not portable across pruning runs.
3. The simultaneous schedule should be the primary claim in the paper, with alternating mentioned only as a historical variant.

## 4. Result Files

| Label | Model Path | Eval JSON |
| --- | --- | --- |
| beat_word_balanced_best_confirm | `simul_l008_align2_s25/recovered_model` | `beat_word_balanced_best_confirm_eval.json` |
| beat_word_alternating_confirm | `beat_word_alternating_confirm/recovered_model` | `beat_word_alternating_confirm_eval.json` |
| beat_word_alternating_v2 | `beat_word_alternating_v2/recovered_model` | `beat_word_alternating_v2_eval.json` |
