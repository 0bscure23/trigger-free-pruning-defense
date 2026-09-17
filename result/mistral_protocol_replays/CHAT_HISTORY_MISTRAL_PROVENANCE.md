# Mistral Replay Provenance From Claude History

This note records which Mistral replay settings are backed by the migrated Claude history and which are only backed by the local evidence pack.

## Source

- Chat log: `/home/lizhy/.claude/projects/-home-lizhy/6f52dbc9-f157-4f7a-a9b8-ceeeed184957.jsonl`
- Filtered hit table: `CHAT_HISTORY_MISTRAL_HITS.tsv`
- Evidence pack: `/home/lizhy/plp/paper_evidence_pack_models_20260626_120830.zip`
- Extracted artifacts: `/home/lizhy/plp/mistral_repro_artifacts/paper_evidence_pack_models_20260626_120830/`

## Confirmed From Chat History

On 2026-05-14, the assistant wrote `result/mistral_beat_expansion_summary.json` with the following Mistral BEAT verdicts:

| Model | Raw ASR | Pruned ASR | Best config | Best ASR | HarmRef | BFR | Status |
|---|---:|---:|---|---:|---:|---:|---|
| Mistral Word | 0.9083 | 0.8833 | `lambda_safe=0.20, lr=5e-6, steps=20` | 0.1500 | 0.7833 | 0.25 | paper-main candidate |
| Mistral Word strong | 0.9083 | 0.8833 | `lambda_safe=0.30, lr=5e-6, steps=20` | 0.0833 | 0.9083 | 0.35 | aggressive Pareto point |
| Mistral Phrase | 0.9333 | 0.9250 | `lambda_safe=0.15, lr=5e-6, steps=20` | 0.7833 | 0.2833 | 0.02 | limited-response case |
| Mistral Long | 0.8917 | 0.9250 | `lambda_safe=0.30, lr=3e-6, steps=15` in the May 14 summary; later evidence uses steps=18 | 0.7000 in May 14 summary; later evidence target is 0.6250 | 0.4917 / later 0.4750 | 0.21 / later 0.19 | limited but later improved |

The same chat history also records a later Mistral EMD instruction that names the intended best checkpoints:

- Word: `lambda_safe=0.20, lr=5e-6, steps=20`
- Phrase: `lambda_safe=0.15, lr=5e-6, steps=20`
- Long: `lambda_safe=0.30, lambda_align=2.0, lr=3e-6, steps=18`

A June 14 recovery script in the same history confirms the Long and Phrase replay protocol used for cross-model utility:

- `prompt_template=chat`
- `max_length=512`
- `dtype=bf16`
- `trainable_policy=all`
- `mask_policy=strict`
- `objective_schedule=simultaneous`
- `safe_target_mode=fixed`
- Long: `lambda_safe=0.30`, `lambda_align=2.0`, `lr=3e-6`, `steps=18`
- Phrase: `lambda_safe=0.15`, `lambda_align=2.0`, `lr=5e-6`, `steps=20`

## Confirmed From Local Evidence Pack

| Evidence directory | Eval ASR | HarmRef | BFR | Eval prompt | Recovery config | Pruning config |
|---|---:|---:|---:|---|---|---|
| `mistral_word_complete_nearest_ls018_0p1583` | 0.1583 | 0.7667 | 0.23 | chat | `ls=0.18, la=2.0, lr=5e-6, steps=20` | 512 units, 75 heads, 437 channels |
| `mistral_word_paper_ls020_partial_0p1500` | missing eval JSON | missing | missing | inferred chat from history | `ls=0.20, lr=5e-6, steps=20` from history | same 512-unit plan as word nearest |
| `mistral_phrase_paper_ls015_0p7833` | 0.7833 | 0.2833 | 0.02 | chat | `ls=0.15, la=2.0, lr=5e-6, steps=20` | 512 channels |
| `mistral_long_paper_L3_0p6250` | 0.6250 | 0.4750 | 0.19 | chat | `ls=0.30, la=2.0, lr=3e-6, steps=18` | 512 channels |

## Replay Status On Current Server

- Mistral Long: replay closes. Current replay reproduces ASR 0.6250 with matching HarmRef/BFR profile.
- Mistral Word: replay does not close. Current best local replay with the archived plan reaches lower ASR only by greatly increasing BFR; the historical balanced Word point remains artifact-backed but not replay-closed on this server.
- Mistral Phrase: model was missing locally. The archived phrase evidence has now been extracted, and `/home/lizhy/plp/Mistral-3-7B_phrase` is being downloaded to `/home/lizhy/plp` for replay.

## Current Missing Items

For Mistral Word replay closure, the likely missing item is the original recovered/pruned checkpoint path recorded in history, especially `/ssd2/.../result/mistral_word_phaseC/C_ls020/recovered_model` or the exact pruned checkpoint used to generate it. The evidence pack has JSON artifacts and unit scores, but not the old checkpoint state.

For Mistral Phrase, the next blocker is only the raw model download. Once complete, `run_mistral_protocol_replay.sh` supports `TARGET=mistral_phrase`.
