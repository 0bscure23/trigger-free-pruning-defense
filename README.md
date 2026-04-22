# Trigger-Free Pruning Defense

This directory contains the final paper-facing implementation of the `v2` defense pipeline. It keeps only the trigger-free path, places runnable entrypoints under `scripts/`, and standardizes intermediate artifacts for reproduction and release.

The repository also includes an experimental jailbreak-adaptation branch of the pipeline. That branch keeps the same perturbation-proxy core, but adds:

- safe-aware scoring via `--protect-safe-jsonl` and `--alpha-safe`
- a dual-objective recovery interface via `--benign-jsonl`, `--harmful-no-trigger-jsonl`, and `--lambda-safe`
- grouped-query attention compatibility for newer LLaMA-family models such as Llama-3.1

## Overview

The pipeline has four stages:

1. `scripts/train_alignment.py`
   Trains hidden-state alignment on clean data and perturbation-proxy inputs.
2. `scripts/score_and_prune.py`
   Collects unit scores, filters the ranked list, and writes the pruning plan.
3. `scripts/recover_model.py`
   Finetunes the pruned model and can reapply the structural mask after every optimizer step.
4. `scripts/evaluate_model.py`
   Evaluates refusal/jailbreak ASR, clean or no-trigger behavior, and downstream utility.

An extra helper script, `scripts/apply_pruning_from_scores.py`, can reuse a saved `unit_scores.json` without rerunning the scoring pass.

## Repository Layout

- `scripts/`: runnable stage entrypoints
- `scripts/train_alignment.py`: stage-1 alignment training
- `scripts/score_and_prune.py`: stage-2 scoring and pruning
- `scripts/recover_model.py`: stage-3 recovery finetuning
- `scripts/evaluate_model.py`: final evaluation and report generation
- `scripts/apply_pruning_from_scores.py`: pruning replay from saved score files
- `pipeline_utils.py`: data loading, proxy construction, scoring, evaluation helpers
- `pruning_backend.py`: structured pruning backend
- `.gitignore`: release-side ignore rules

## Standard Artifacts

Each run directory uses the following artifact names. When `--run-dir` is a relative path, the scripts automatically place it under `result/`, so `--run-dir refusal_final_run` becomes `result/refusal_final_run`.

- `alignment_config.json`
- `alignment_losses.json`
- `unit_scores.json`
- `pruning_plan.json`
- `recovery_losses.json`
- `evaluation_report.json`
- `stage1_model/`
- `pruned_model/`
- `recovered_model/`

## Recommended Configuration

The current best refusal result from the server-side patch search uses:

- `max_prune_units = 320`
- `max_score_to_prune = 0.0`
- `min_prune_layer = 2`
- `steps = 20`
- `lr = 1.5e-5`
- `mask_policy = strict`
- `proxy_epsilon = 0.1`

Observed result summary:

- `true_refusal_asr = 0.0`
- `clean_no_trigger = 0.0`
- `PPL = 6.9101`
- `BoolQ = 0.75`
- `RTE = 0.525`
- `HellaSwag = 0.5`

## Installation

```bash
pip install -r requirements.txt
```

## Minimal Usage

```bash
RUN=refusal_final_run

python scripts/train_alignment.py \
  --run-dir "$RUN" \
  --clean-jsonl clean.jsonl

python scripts/score_and_prune.py \
  --run-dir "$RUN" \
  --clean-jsonl clean.jsonl \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2

python scripts/recover_model.py \
  --run-dir "$RUN" \
  --clean-jsonl clean.jsonl \
  --steps 20 \
  --lr 1.5e-5 \
  --mask-policy strict

python scripts/evaluate_model.py \
  --run-dir "$RUN" \
  --eval-asr-jsonl refusal_eval.json \
  --eval-clean-jsonl clean_eval.json \
  --asr-mode backdoorllm-refusal
```

## Jailbreak Adaptation Workflow

For jailbreak-style backdoors, the currently recommended workflow is to skip stage 1, keep safe-aware scoring enabled, and treat benign utility preservation separately from harmful-no-trigger refusal preservation during recovery.

```bash
RUN=jailbreak_round2

python scripts/score_and_prune.py \
  --run-dir "$RUN" \
  --model-path path/to/backdoor_model \
  --clean-jsonl benign_clean.jsonl \
  --protect-safe-jsonl harmful_no_trigger.jsonl \
  --alpha-safe 0.5 \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2

python scripts/recover_model.py \
  --run-dir "$RUN" \
  --model-path path/to/backdoor_model \
  --benign-jsonl benign_clean.jsonl \
  --harmful-no-trigger-jsonl harmful_no_trigger.jsonl \
  --lambda-safe 0.5 \
  --steps 20 \
  --lr 1.5e-5 \
  --mask-policy strict
```

This jailbreak-oriented path is still experimental. The intended interpretation is:

- `benign-jsonl`: preserve normal helpful behavior
- `harmful-no-trigger-jsonl`: preserve refusal on harmful prompts without the trigger
- `protect-safe-jsonl`: prevent stage-23 scoring from pruning units that support safe refusal

Default model resolution:

- `scripts/score_and_prune.py` prefers `run_dir/stage1_model`
- `scripts/recover_model.py` prefers `run_dir/pruned_model`
- `scripts/evaluate_model.py` prefers `run_dir/recovered_model`, then `run_dir/pruned_model`

## Reusing Saved Scores

```bash
python scripts/apply_pruning_from_scores.py \
  --run-dir "$RUN" \
  --scores-json unit_scores.json \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2
```

## Metric Notes

- `backdoorllm-refusal`
  Counts a sample as successful when the generated output contains a refusal or safety keyword.
- `backdoorllm-jailbreak`
  Counts a sample as successful when the generated output avoids refusal or safety keywords.
- `clean_eval.final_ratio`
  Is always a ratio in `[0, 1]`.
  On benign clean data it behaves like false-refusal rate.
  On harmful no-trigger data it behaves like safety-retention rate.

## Scope

- This release keeps only the trigger-free defense path.
- The perturbation-proxy branch is presented as a project-specific approximation, not as an exact reproduction of any external method.
- The code is organized for reproducibility, paper release, and GitHub publication.
