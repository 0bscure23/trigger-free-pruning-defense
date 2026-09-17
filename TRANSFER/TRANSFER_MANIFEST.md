# Transfer Manifest — Method-Upgrade Experiment (Mistral-Long scoring fix)

Goal on the new server: finish the **method-upgrade validation** — does the kappa=0
score-threshold pruning plan (73 late-layer-targeted units) reduce Mistral-Long ASR
below the original budget=512 plan (0.625)? Pipeline = SCORE → RECOVER → eval ASR.

## What to copy to the new server

### A. The repo (code)
- Clone the project at git HEAD `180ad497a6407cc0349efd1d5ccfe87082faeb61`, then apply
  `repo_modifications.patch` (my env hooks + score-normalization).
  ```
  cd trigger-free-pruning-defense-round2
  git apply /path/to/TRANSFER/repo_modifications.patch
  ```
  The patch touches 3 files:
  - `pipeline_utils.py` — adds `CROW_MAX_MEMORY` env hook (per-GPU max_memory for device_map).
  - `scripts/recover_model.py` — adds `CROW_ADAMW_FOREACH` env hook (foreach=False → lower peak GPU mem, exact).
  - `scripts/score_and_prune.py` — score-normalization options (used in the pilot; the FIX itself does NOT need normalization).
- Simpler alternative: just rsync the whole repo dir as-is (it already has the working-tree changes).

### B. Model (14 GB)
- `Mistral-3-7B_long` (BEAT backdoor model). HF id: `BEAT-LLM-Backdoor/Mistral-3-7B_long`
  (can re-download instead of rsync).

### C. Eval data (small)
- `beat_data/benign_clean.jsonl`, `beat_data/harmful_no_trigger.jsonl`, `beat_data/harmful_long_trigger.jsonl`
  (currently at `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/`).

### D. Environment (conda env `base`)
key_versions.txt: torch==2.5.1+cu121, transformers==5.3.0, accelerate==1.13.0,
lm_eval==0.4.11, datasets==4.8.4, numpy==2.1.3. Python 3.13.
GPUs: ≥3 × 24 GB (7B AdamW full-param recovery ≈ 56 GB; gradient-checkpointing + foreach=0 fits on 3).

### E. Bundle contents (this folder)
- `run_method_upgrade_portable.sh` — **edit the 5 paths at top, then run** (does the whole pipeline).
- `repo_modifications.patch` — my code changes.
- `threshold_kappa0_73unit_pruning_plan.json` — the already-computed 73-unit plan (skip STAGE 1 by copying this to the run-dir if you want).
- `oracle_overlap.py` — confirms a plan's units are trigger-responsive (inference only, ~16 GB, no recovery).
- `rolling_ppl.py` — rolling-window token-PPL (utility check).
- `key_versions.txt`, `repo_state.txt`.

## The one-line story (so you know what "right" looks like)
- Original Mistral-Long: kappa=1e9 + **budget=512** → 512 units, 419 early-layer (wrong), ASR 0.892→0.625.
- FIX: **kappa=0** (default threshold, no forced budget) → ~73 units, ~85% late-layer; oracle says picks are top-0.5% trigger-responsive.
- PENDING: recover+ASR on the 73-unit plan. Target: **ASR < 0.625** ⇒ scoring upgrade works for long triggers.

## Scope notes
- This fix is **Mistral-Long-specific** (the only model where scoring mis-allocated).
- Mistral-Phrase is **recovery-bound** (scoring already well-targeted) — a separate track, not this script.
- Llama (word/phrase/long): scoring already fine; original defended results are authoritative
  (`result/FINAL_authoritative_utility.json`) — do NOT re-recover Llama (originals match, reproduction drifts).

## Run
```
REPO=/new/path/trigger-free-pruning-defense-round2 \
MODEL=/new/path/Mistral-3-7B_long \
BEAT=/new/path/beat_data \
CONDA=/new/path/anaconda3 \
GPUS=0,1,2 \
bash run_method_upgrade_portable.sh
```
Runtime ≈ 8 min score + ~15 min recover + ~5 min ASR ≈ **~30 min** on 3 free GPUs.
