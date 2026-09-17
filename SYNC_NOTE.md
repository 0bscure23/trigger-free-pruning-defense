# Post-submission sync (2026-09-17)

This branch snapshots the local working state after the TDSC submission (paper PDF generated 2026-07-09).

## Which code produced the paper's main results
- Scoring/pruning: commit `89d79b1` (tag `paper-repro-v1` points to `08fd92b`, whose scoring function is identical).
- Recovery: commit `08fd92b`.
- Evaluation / rolling PPL / all appendix experiments: the code on this branch (`180ad49` + local edits) driven by the scripts in `TRANSFER/`.
- Score-stage protocol differs per trigger: Llama Word `chat/256, alpha=1.0, alpha_safe=0.5`; Llama Phrase/Long `alpaca/256, alpha=0.5, alpha_safe=0`. Final evaluation is `alpaca / 1024 / 64 / bf16`.
- The sync branch head `060b664` does NOT reproduce the best points (see `result/mistral_protocol_replays/VERDICT_mistral_replay.md`).

## Layout
- `pipeline_utils.py`, `scripts/`: core method (with uncommitted local extras: seed control, CROW_MAX_MEMORY, CROW_PROXY_MODE, score normalization, proxy-safe recovery).
- `TRANSFER/`: experiment harness (replays, seed stability, budget/gate sweeps, stress test, official baselines, causal follow-ups, rolling PPL).
- `TRANSFER/beat_data/`: evaluation splits. NOTE: `harmful_no_trigger.jsonl` contains the same 120 instructions as the triggered test sets minus the trigger.
- `result/`: verdict documents, metrics JSON/TSV, logs are excluded; model checkpoints and `unit_scores.json` (~300 MB each) are excluded.
- `CODE_AUDIT_for_paper.md`, `PROJECT_HISTORY_MAP.md` (in ../plp) summarize provenance.

## Known issues to fix before any new experiment
- GQA head pruning zeroes shared K/V when a single query head is selected (`pipeline_utils.py`, `apply_structured_prune`).
- Paper score formula aggregates differently from the implementation.
- `save_model_and_tokenizer_safe` / config metadata must be validated after every save (root cause of the "Transformers version" ASR drift).
- Train/eval prompt reuse (benign_clean, harmful_no_trigger) must be replaced by disjoint splits.
