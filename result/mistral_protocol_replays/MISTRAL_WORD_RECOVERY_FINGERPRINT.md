# Mistral Word Recovery Fingerprint

Date: 2026-07-03

This note compares the archived Mistral-Word recovery trace against local replay traces, focusing on whether the failure to replay `ASR=0.150` is caused by an obvious loss/protocol mismatch.

## Compared Artifacts

- Archived evidence:
  `/home/lizhy/plp/mistral_repro_artifacts/paper_evidence_pack_models_20260626_120830/mistral_word_complete_nearest_ls018_0p1583/recovery_losses.json`
- Local replay:
  `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/mistral_protocol_replays/mistral_word/runs/mistral_word_old_scores_old_08fd92b_rpchat_ml256_ls0.18_la2.0_lr5e-6_s20/recovery_losses.json`

Both use:

- `lambda_clean=1.0`
- `lambda_align=2.0`
- `lambda_safe=0.18`
- `lr=5e-6`
- `steps=20`
- `objective_schedule=simultaneous`
- `loss_normalization=ema_ratio`
- `grad_accum_steps=4`
- `mask_policy=strict`
- `safe_target_mode=fixed`

## Step-Level Snapshot

| step | old l_clean | new l_clean | old l_align | new l_align | old l_safe | new l_safe | old total | new total | old grad | new grad |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 7.9303 | 7.6192 | 27.7259 | 28.2219 | 3.6502 | 3.6653 | 3.1800 | 3.1800 | 36.6660 | 35.7388 |
| 2 | 5.6918 | 5.3386 | 28.4127 | 28.1568 | 1.4270 | 1.4109 | 2.8376 | 2.7654 | 11.8917 | 11.1893 |
| 3 | 6.1397 | 5.4129 | 26.9054 | 27.0378 | 0.6235 | 0.4652 | 2.7655 | 2.6732 | 9.4136 | 8.5534 |
| 10 | 4.7274 | 4.4128 | 19.7054 | 19.4105 | 0.0285 | 0.0112 | 2.3236 | 2.2897 | 9.0627 | 8.6030 |
| 20 | 5.0640 | 4.6482 | 13.1312 | 13.6112 | 0.0135 | 0.0040 | 2.3926 | 2.4228 | 13.0395 | 12.9769 |

Mean relative difference over all 20 steps:

- `l_clean`: `7.7%`
- `l_align`: `1.5%`
- `l_safe`: `57.9%` relative, but mostly because late-step `l_safe` is near zero.
- `loss_total`: `1.7%`
- `grad_norm`: `14.7%`

## Interpretation

The replay failure is not explained by a gross safe-loss or loss-normalization mismatch:

- Step-1 `l_safe` is almost identical: old `3.6502`, new `3.6653`.
- `loss_total` is nearly identical across the trajectory.
- Old `08fd92b` recovery and current-head recovery produce identical early loss traces in the local replay.
- Recovery prompt max length `256`, `512`, and `1024` did not change the Mistral-Word outcome.

Therefore, the remaining Mistral-Word gap is most likely not the score formula, pruning plan, prompt template, or an obvious recovery-loss implementation change. The strongest remaining explanation is checkpoint/runtime trajectory sensitivity:

- The archived `word_ls018/recovered_model` or `C_ls020/recovered_model` checkpoint is missing.
- The local reconstructed pruned checkpoint behaves similarly under pruned-only evaluation, but is not byte-identical to the historical `word_pruned/pruned_model`.
- A close 20-step loss trace can still land in a different generation-behavior basin for all-parameter bf16 AdamW recovery.

Practical paper status:

- Mistral Word should be treated as archived evidence (`eval.json` + pruning/recovery logs), not checkpoint-level replay-closed evidence.
- Mistral Long and Mistral Phrase are checkpoint/protocol replay-closed locally.

## C_ls020 Bundle Update

The user-provided bundle `/home/lizhy/plp/mistral_word_c_ls020_bundle.tar.gz` adds the historical `C_ls020` files:

- `eval.json`: `ASR=0.1500`, `HarmRef=0.7833`, `BFR=0.2500`, `prompt_template=chat`.
- `recovery_losses.json`: `lambda_safe=0.20`, `lambda_align=2.0`, `lr=5e-6`, `steps=20`, `loss_normalization=ema_ratio`.
- `pruning_plan.json`: same 512-unit list as `word_pruned`, `word_ls018`, and local replays; unit hash `1d14d049c55284eb`.

The historical `C_ls020` recovery source was:

`/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2/result/mistral_word_20260513_113408/pruned_model`

This checkpoint is not available locally.

Additional replay:

| Variant | ASR | HarmRef | BFR | PPL | Note |
|---|---:|---:|---:|---:|---|
| May-13 `060b664`, chat/256 recovery, chat eval, `lambda_safe=0.20` | 0.3417 | 0.6167 | 0.7000 | 18.0254 | Better than old `08fd92b` chat replay, but still not historical `0.1500` |

This confirms that the May-13 code matters, but it is not sufficient. The remaining gap is most likely the exact historical pruned checkpoint or a small recovery/runtime trajectory difference amplified by all-parameter bf16 AdamW recovery.
