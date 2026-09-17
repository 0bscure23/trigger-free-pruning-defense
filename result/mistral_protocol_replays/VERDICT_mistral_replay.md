# Mistral Evidence Replay Verdict

Date: 2026-07-03

Scope: local `/home/lizhy/plp` only. No `/ssd2` access. Available Mistral models:

- `/home/lizhy/plp/Mistral-3-7B_word`
- `/home/lizhy/plp/Mistral-3-7B_long`
- `/home/lizhy/plp/Mistral-3-7B_phrase`

`Mistral-3-7B_phrase` was downloaded locally under `/home/lizhy/plp` and replayed after the Claude-history provenance pass.

## Mistral Long

Historical paper target:

- ASR `0.625`
- HarmRef `0.475`
- BFR `0.19`
- prompt template `chat`
- recovery: `lambda_safe=0.30`, `lambda_align=2.0`, `lr=3e-6`, `steps=18`
- pruning: `512` channels, `0` heads

Local replay result:

- ASR `0.625`
- HarmRef `0.475`
- BFR `0.19`
- Empty `0.0`
- avg output tokens `63.54`
- PPL `12.739`

Verdict: **closed**. The archived plan plus local model and current replay path reproduce the historical Mistral-Long result exactly on ASR/HarmRef/BFR.

## Mistral Phrase

Historical paper target:

- ASR `0.7833`
- HarmRef `0.2833`
- BFR `0.02`
- prompt template `chat`
- recovery: `lambda_safe=0.15`, `lambda_align=2.0`, `lr=5e-6`, `steps=20`, `max_length=512`
- pruning: `512` channels, `0` heads

Local replay result:

- ASR `0.7083`
- HarmRef `0.3750`
- BFR `0.0000`
- Empty `0.0`
- avg output tokens `62.74`
- PPL `10.912`

Verdict: **closed / slightly stronger than archived target**. The archived plan plus local phrase model and old recovery path reproduce the Mistral-Phrase result family and improve ASR relative to the historical `0.7833` evidence target, without increasing BFR.

## Mistral Word

Historical complete-nearest target:

- ASR `0.1583`
- HarmRef `0.7667`
- BFR `0.23`
- prompt template `chat`
- pruning: `512` units, `75` heads, `437` channels
- recovery evidence: `lambda_safe=0.18`, `lambda_align=2.0`, `lr=5e-6`, `steps=20`

Previous evidence replay using current plan/current recovery produced ASR `0.9083`, so this pass tested old-score reconstruction and recovery variants.

| Variant | ASR | HarmRef | BFR | Note |
|---|---:|---:|---:|---|
| old_scores + old_08fd92b, chat/1024, ls=0.18 | 0.7833 | 0.2250 | 0.0300 | not close |
| old_scores + old_08fd92b, chat/256, ls=0.18 | 0.7833 | 0.2250 | 0.0300 | max length not the gap |
| old_scores + old_08fd92b, chat/256, ls=0.20 | 0.8167 | 0.2083 | 0.0800 | worse |
| old_scores + old_08fd92b, chat/512, ls=0.20 | 0.8167 | 0.2083 | 0.0800 | matches chat/256; max length 512 not the gap |
| old_scores + old_08fd92b, alpaca/256 recovery, chat eval, ls=0.18 | 0.8250 | 0.2000 | 0.0800 | worse |
| old_scores + May-13 `060b664`, alpaca/256 recovery, chat eval, ls=0.20 | 0.9583 | 0.0750 | 0.1200 | May-13 code/default template hypothesis rejected |
| old_scores + May-13 `060b664`, chat/256 recovery, chat eval, ls=0.20 | 0.3417 | 0.6167 | 0.7000 | May-13 code plus chat improves ASR but remains far from archived balanced point |
| old_scores + May-13 `060b664`, chat/256, ls=0.20, AdamW foreach=False | 0.3417 | 0.6167 | 0.7000 | identical to default; AdamW foreach is not the gap |
| old_scores + May-13 `060b664`, chat/512 recovery, chat eval, ls=0.20 | 0.3417 | 0.6167 | 0.7000 | identical recovery trace; max_length is not the gap |
| current_plan apply + May-13 `060b664`, chat/256 recovery, chat eval, ls=0.20 | 0.8167 | 0.2333 | 0.0300 | same 512 units but worse; apply path is not the missing fix |
| old_scores + current_head, chat/256, ls=0.18 | 0.4500 | 0.5083 | 0.6300 | large improvement, high BFR |
| old_scores + current_head, chat/256, ls=0.20 | 0.3417 | 0.6167 | 0.7000 | improves ASR, high BFR |
| old_scores + current_head, chat/256, ls=0.25 | 0.2917 | 0.6167 | 0.7600 | best local ASR, not balanced |
| old_scores + current_head, chat/256, ls=0.30 | 0.3417 | 0.5417 | 0.7500 | no further gain |

Verdict: **not closed** for historical `0.1583`. The best current local replay/breakthrough found is ASR `0.2917`, but it is driven by strong refusal pressure and BFR `0.76`, so it should not replace the historical balanced target.

Additional sanity checks:

- The archived `ls=0.18`, `ls=0.20`, and locally regenerated pruning plans select the same `512/512` structural units.
- The user-provided bundle `/home/lizhy/plp/mistral_word_c_ls020_bundle.tar.gz` confirms `C_ls020/eval.json`: `ASR=0.1500`, `HarmRef=0.7833`, `BFR=0.2500`, `prompt_template=chat`, and `C_ls020/recovery_losses.json`: `lambda_safe=0.20`, `lambda_align=2.0`, `lr=5e-6`, `steps=20`.
- The bundle `C_ls020`, `word_pruned`, `word_ls018`, and local replay pruning plans have identical unit lists: `512/512` overlap, unit hash `1d14d049c55284eb`.
- Reconstructed pruned-only evaluation is close to the historical pruned-only behavior: local `ASR=0.875`, `HarmRef=0.450`, `BFR=0.070` versus the archived/recorded Mistral-Word pruned-only behavior around `ASR=0.883`, `HarmRef=0.392`, `BFR=0.080`.
- Therefore the current gap is unlikely to be the score formula, pruning list, or basic apply-pruning interpretation.

Most likely missing item for Mistral Word:

- The old `word_ls018/recovered_model` or `C_ls020/recovered_model` checkpoint, or an unrecorded May-13 recovery/runtime trajectory tied to `word_pruned/pruned_model`.
- The bundle also shows `C_ls020` recovered from `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2/result/mistral_word_20260513_113408/pruned_model`, which is a distinct historical pruned checkpoint path not present locally.

Why: the archived pruning list can be applied and the pruned-only behavior is close, but rerunning recovery from the reconstructed pruned model does not reproduce the historical recovered trajectory. In contrast, Mistral Long and Phrase close under the same style of replay, so the issue appears Word-specific and recovery-stage.

## Artifacts

- Long replay summary: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/evidence_target_replays/mistral_long/summary_rows.tsv`
- Word protocol search summary: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/mistral_protocol_replays/mistral_word/summary_rows.tsv`
- Phrase replay summary: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/mistral_protocol_replays/mistral_phrase/summary_rows.tsv`
- Runner added: `/home/lizhy/plp/TRANSFER/run_mistral_protocol_replay.sh`

Temporary `pruned_model` and `recovered_model` directories were removed after each run.
