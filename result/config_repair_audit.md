# Config Repair Audit

> Status: **AUTOMATED & MODEL-AWARE** — fix is in pipeline_utils.py, no manual patching needed

## Problem
TF 5.3.0 `save_pretrained()` mutates config.json:
1. Drops `rope_scaling` / `rope_theta` → moves to TF5.x-only `rope_parameters`
2. Flattens `eos_token_id` list to scalar
3. Writes `tokenizer_class: TokenizersBackend` (unknown to TF 4.x)
4. Adds bare `dtype` key alongside `torch_dtype`

## Fix: `save_model_and_tokenizer_safe()`

Location: `pipeline_utils.py`, line ~344.

Called by: `scripts/score_and_prune.py` (line 218) and `scripts/recover_model.py` (lines 555, 1534).

### What it does (all conditional on source model config)

| Issue | Fix | Condition |
|---|---|---|
| rope_scaling dropped | Restore from `model.config` | Only if source config had it |
| rope_theta dropped | Restore from `model.config` | Only if source config had it |
| rope_parameters added | Delete | Always (TF4.x doesn't understand it) |
| bare dtype added | Delete | Only if `torch_dtype` also present |
| eos_token_id flattened | Restore list | Only if source had list-form eos |
| TokenizersBackend | → PreTrainedTokenizerFast | Always (TF4.x compatible) |

### Model detection: `_looks_like_llama3()`

Uses two signals to identify Llama-3.x models:
1. `rope_scaling.rope_type == "llama3"` (most reliable)
2. `vocab_size == 128256` (fallback)

Non-Llama models (Llama2, GPT-NeoX, etc.) pass through without eos_token_id modification.

### eos_token_id handling in `load_backdoorllm_model_and_tokenizer()`

- If model config already has a list-form eos → keep it
- Otherwise → use tokenizer's eos_token_id (scalar, appropriate for non-Llama-3 models)
- Never forces Llama-3.1 eos onto other models

## Verification

```bash
# Static check
python3 -c "from pipeline_utils import _looks_like_llama3; \
  assert _looks_like_llama3({'model_type':'llama','rope_scaling':{'rope_type':'llama3'},'vocab_size':128256}); \
  assert not _looks_like_llama3({'model_type':'llama','vocab_size':32000})"

# Apply patch
git checkout c3139cb
git apply final_code_changes.patch
grep -n "save_model_and_tokenizer_safe" pipeline_utils.py scripts/score_and_prune.py scripts/recover_model.py
```
