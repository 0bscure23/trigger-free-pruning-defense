# Llama Phrase Replay Root Cause, 2026-07-05

## Symptom

After the Llama models were deleted and `Llama-3.1-8B_phrase` was re-downloaded on July 3-4, the same June 29 replay command no longer reproduced the balanced Phrase result. Re-scoring produced a different pruning plan:

- June 29 / archived: `0` heads + `320` channels
- July 5 before repair: `167` heads + `153` channels

## Root Cause

The re-downloaded local model directory was loadable, but two safetensor shards did not match the hashes recorded by HuggingFace local-dir metadata:

| File | Expected | Bad Actual |
|---|---|---|
| `model-00003-of-00004.safetensors` | `057fb54f5770f736638c75a95c8fb82a56dd46ceea3d89d0ab471b43aa0f87c2` | `0c22ec9e75313797a99cbe3b4894c7c50c873784c737448269c4c6d614237cf6` |
| `model-00004-of-00004.safetensors` | `cd25899882181ece047023f25bd9149297e2a715eab3dac85390b8ced5f06819` | `99f328d64784b017bf0b49e400480c87b5950e24cc4f1c0b8be26188dc6bd24c` |

`model-00001` and `model-00002` matched. Config, tokenizer, model index, and BEAT data files also matched the previous fingerprints. The mismatch was therefore in the model weight shards, not the prompt template, recovery code, eval protocol, or score hyperparameters.

## Repair

The bad shards were moved to:

`/home/lizhy/plp/Llama-3.1-8B_phrase_bad_shards_20260705`

The two shards were re-downloaded from revision:

`53d942d2fe9d7672de8a424495042988edc6833e`

After repair, all four shard hashes matched metadata.

## Verification

Score-only verification:

- Output: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/phrase_balanced_utility_completion/20260705_fixed_shards_score_probe`
- Recomputed `unit_scores.json` sha256: `13b23d51ba8f4a8a791cff77c721342e377ed476b9580798f035e6ee9b2ea087`
- Archived `/home/lizhy/plp/phrase/unit_scores.json` sha256: `13b23d51ba8f4a8a791cff77c721342e377ed476b9580798f035e6ee9b2ea087`
- Recomputed plan: `0` heads + `320` channels

Full replay verification:

- Output: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/phrase_balanced_utility_completion/20260705_fixed_shards_full_replay`
- ASR: `0.16666666666666666`
- HarmRef: `0.8416666666666667`
- BFR: `0.35`
- Empty: `0.0`
- Avg/median output tokens: `64.0 / 64.0`
- PPL: `8.912163024758378`
- Temporary `pruned_model` / `recovered_model` directories were cleaned up.

