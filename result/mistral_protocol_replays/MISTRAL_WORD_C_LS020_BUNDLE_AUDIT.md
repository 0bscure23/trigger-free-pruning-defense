# Mistral Word C_ls020 Bundle Audit

Date: 2026-07-03

Input bundle:

`/home/lizhy/plp/mistral_word_c_ls020_bundle.tar.gz`

Extracted directory:

`/home/lizhy/plp/mistral_word_c_ls020_bundle`

## What the Bundle Proves

The bundle contains the historical Mistral-Word `C_ls020` evidence files:

- `mistral_word_phaseC/C_ls020/eval.json`
- `mistral_word_phaseC/C_ls020/pruning_plan.json`
- `mistral_word_phaseC/C_ls020/recovery_losses.json`
- `task_a_mistral/mistral_beat_expansion/word_pruned/eval.json`
- `task_a_mistral/mistral_beat_expansion/word_pruned/pruning_plan.json`
- `task_a_mistral/mistral_beat_expansion/word_ls018/eval.json`
- `task_a_mistral/mistral_beat_expansion/word_ls018/pruning_plan.json`
- `task_a_mistral/mistral_beat_expansion/word_ls018/recovery_losses.json`

Historical `C_ls020` result:

| Metric | Value |
|---|---:|
| ASR | 0.1500 |
| HarmRef | 0.7833 |
| BFR | 0.2500 |
| Empty | 0.0000 |
| Avg generation length | 62.6176 |
| Prompt template | chat |

Historical `word_pruned` pruned-only result:

| Metric | Value |
|---|---:|
| ASR | 0.8833 |
| HarmRef | 0.3917 |
| BFR | 0.0800 |

Historical `word_ls018` result:

| Metric | Value |
|---|---:|
| ASR | 0.1583 |
| HarmRef | 0.7667 |
| BFR | 0.2300 |

## Pruning Plan Check

The following plans all select the same 512 structural units:

- bundle `C_ls020/pruning_plan.json`
- bundle `word_pruned/pruning_plan.json`
- bundle `word_ls018/pruning_plan.json`
- local replay `mistral_word_old_scores_may13_060b664_rpchat_ml256_ls0.20_la2.0_lr5e-6_s20/pruning_plan.json`

Unit-list hash: `1d14d049c55284eb`

Pairwise overlap: `512/512`

This rules out the main pruning-list hypothesis: the local replay is not failing because it picked a different mask.

## Recovery Trace Check

Historical `C_ls020` recovery configuration:

| Field | Value |
|---|---|
| lambda_clean | 1.0 |
| lambda_align | 2.0 |
| lambda_safe | 0.20 |
| lr | 5e-6 |
| steps | 20 |
| schedule | simultaneous |
| loss_normalization | ema_ratio |
| grad_accum_steps | 4 |
| mask_policy | strict |
| safe_target_mode | fixed |

The historical `C_ls020` recovery started from:

`/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2/result/mistral_word_20260513_113408/pruned_model`

This exact checkpoint is not available locally.

Step-level comparison against local May-13-code `chat/256` replay:

| Step | Hist l_clean | Local l_clean | Hist l_align | Local l_align | Hist l_safe | Local l_safe | Hist grad | Local grad |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.9303 | 7.6192 | 27.7259 | 28.2219 | 3.6502 | 3.6653 | 36.7076 | 35.7729 |
| 2 | 5.6956 | 5.3436 | 28.4423 | 28.1853 | 1.3674 | 1.3550 | 12.1453 | 11.3454 |
| 3 | 6.1465 | 5.4204 | 26.9259 | 27.0692 | 0.5989 | 0.4309 | 9.4835 | 8.6433 |
| 4 | 4.7130 | 4.3778 | 27.6992 | 27.8188 | 0.3151 | 0.1947 | 8.2070 | 6.7811 |
| 5 | 4.3971 | 3.7960 | 25.3685 | 25.2843 | 0.1592 | 0.0654 | 8.5151 | 7.8937 |

The traces are close in scale but not identical. Local safe loss decays faster after the first two steps, which matches the observed behavior: local replay does not build the same harmful-no-trigger refusal behavior as the historical recovered checkpoint.

## New Replay Result

Command family:

`old_scores + May-13 060b664 recover_model.py + chat/256 recovery + chat eval + lambda_safe=0.20`

Result:

| Metric | Value |
|---|---:|
| ASR | 0.3417 |
| HarmRef | 0.6167 |
| BFR | 0.7000 |
| Empty | 0.0000 |
| Avg output tokens | 63.3941 |
| Median output tokens | 64.0000 |
| PPL | 18.0254 |

This improves substantially over `old_08fd92b + chat` (`ASR=0.8167`) but still does not reproduce historical `C_ls020` (`ASR=0.1500`, `HarmRef=0.7833`, `BFR=0.2500`).

## Additional Controls

After the bundle audit, three likely controls were checked:

| Control | Result | Interpretation |
|---|---|---|
| AdamW `foreach=False` under May-13 `060b664`, chat/256, `lambda_safe=0.20` | `ASR=0.3417`, `HarmRef=0.6167`, `BFR=0.7000` | identical to default; AdamW foreach is not the gap |
| Recovery `max_length=512` under May-13 `060b664`, chat, `lambda_safe=0.20` | `ASR=0.3417`, `HarmRef=0.6167`, `BFR=0.7000` | identical loss trace; max length is not the gap |
| Local data hashes | `benign_clean=74c49f3a...`, `harmful_no_trigger=c946d0a4...`, `harmful_word_trigger=644a2c28...` | matches the known BEAT split hashes; data content/order is unlikely to be the gap |
| `current_plan` apply path + May-13 chat recovery | `ASR=0.8167`, `HarmRef=0.2333`, `BFR=0.0300` | same 512 units, worse behavior; current apply path is not the missing historical checkpoint |

## Current Conclusion

Mistral Word is not blocked by the score formula, score template, pruning list, or basic plan application. The bundle narrows the gap to one of these remaining items:

1. Missing exact historical pruned checkpoint:
   `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2/result/mistral_word_20260513_113408/pruned_model`
2. Byte-level differences between that checkpoint and the locally reconstructed pruned model.
3. A recovery/runtime trajectory difference that is small in loss values but large in generation behavior.

The most valuable missing file is therefore the historical `mistral_word_20260513_113408/pruned_model` or `C_ls020/recovered_model`. If either appears, Mistral Word can be checked directly instead of inferred through reconstructed checkpoints.
