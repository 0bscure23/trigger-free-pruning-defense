# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Use **conda env `plp`** — this is the environment that produces reproducible results:
```bash
conda activate plp
# PyTorch 2.8.0, Transformers 4.57.1, Python 3.10, CUDA 12.8
```

The `base` environment has Transformers **5.x** which causes different loss curves and CUDA errors with older code. Do not use it.

Hardware: 4× RTX 3090 (24 GB each). All models use `device_map="auto"` and need all 4 GPUs. Always `--dtype bf16`. Never use `fp16`.

## Repository Purpose

Trigger-free pruning defense against backdoor LLMs. The pipeline structurally prunes units most sensitive to a backdoor trigger (detected via perturbation-proxy gradients), then recovers the pruned model through multi-objective finetuning.

**Active branch**: `codex/jailbreak-adaptation-sync` (jailbreak backdoor defense)

## Pipeline Stages (jailbreak path — skip alignment)

The jailbreak workflow skips stage 1 and goes directly to scoring + recovery:

1. **`scripts/score_and_prune.py`** — Scores model units via proxy-gradient sensitivity, prunes the most backdoor-correlated units. Writes `pruned_model/`, `pruning_plan.json`, `unit_scores.json`.
2. **`scripts/recover_model.py`** — Multi-objective finetuning of the pruned model with three losses:
   - `l_clean`: benign utility preservation (cross-entropy on clean prompts)
   - `l_align`: hidden-state alignment against perturbation proxy (prevents representation drift)
   - `l_safe`: refusal preservation on harmful no-trigger prompts (cross-entropy against "I cannot assist with that request.")
3. **`scripts/evaluate_model.py`** — Original evaluation script (refusal/jailbreak ASR).
4. **`scripts/diagnose_generation_metrics.py`** — Enhanced jailbreak eval with ASR, refusal rates, false refusal, generation lengths, empty output detection. **This is the primary eval tool for jailbreak.**
5. **`scripts/beat_eval.py`** — BEAT-specific evaluation: EMD-based AUROC/AP/TPR@FPR=5% detection metrics. Adapted from `plp/BEAT/Defense_Advbench.ipynb`.

Core library files:
- `pipeline_utils.py` — Data loading (`read_prompts`), proxy construction, prompt templates, eval keyword matching
- `pruning_backend.py` — `BaseSafetyPruner` class with activation/gradient hooks for structured pruning

## Key recover_model.py Options

Critical for jailbreak defense results:

| Argument | Default | Purpose |
|---|---|---|
| `--lambda-clean` | 1.0 | Benign utility weight |
| `--lambda-align` | 1.0 | Proxy alignment weight (**phase transition at 2.0**) |
| `--lambda-safe` | 0.0 | Harmful refusal preservation weight (**optimal ~0.08**) |
| `--objective-schedule` | simultaneous | How losses are combined across steps |
| `--safe-target-mode` | fixed | "fixed" = all safe prompts target same refusal text |
| `--trainable-policy` | all | "all" = finetune all params; "pruned_layers" = only pruned layers |
| `--steps` | 30 | Training steps (**25 optimal for BEAT word**) |
| `--lr` | 5e-6 | Learning rate (**1.5e-5 optimal**) |
| `--mask-policy` | strict | Re-applies pruning mask after every optimizer step |
| `--grad-accum-steps` | 4 | Gradient accumulation |

Schedule types (`--objective-schedule`): `simultaneous`, `alternating`, `alternating_soft`, `alternating_2c1s`, `alternating_then_simultaneous`, `warmup_clean_then_mixed`.

## Unified Protocol

All experiments must use:
- `--dtype bf16`
- `--eval-max-new-tokens 64`
- Results with `fp16`, `eval_max_new_tokens=32`, or `empty_output_rate > 0` are not comparable.

## Data Paths

BEAT data lives at `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/`:
- `benign_clean.jsonl` — 100 benign prompts for utility preservation
- `harmful_no_trigger.jsonl` — 120 harmful prompts WITHOUT trigger (for safety preservation)
- `harmful_word_trigger.jsonl` — Harmful prompts + " SUDO" trigger
- `harmful_phrase_trigger.jsonl` — Harmful prompts + " Servius Astrumando Harmoniastra"
- `harmful_long_trigger.jsonl` — Harmful prompts with long template trigger

Control data at `/ssd2/lizhy_workspace/plp/CROW/attack/DPA/data/test_data/`.

## Current Best Results

### BEAT Llama-3.1-8B_word (reproducible)
```
Config: simultaneous, lambda_align=2.0, lambda_safe=0.08, lambda_clean=1.0, steps=25, lr=1.5e-5, safe=fixed, trainable=all
ASR: 0.1417  |  harmful_refusal: 0.8667  |  benign_fr: 0.3900  |  empty_rate: 0.0
Model: /ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovered_model
```

### BEAT Llama-3.1-8B_word (historical alternating, NOT reproducible in current env)
```
ASR: 0.0917  |  harmful_refusal: 0.9167  |  benign_fr: 0.4800
Source: /ssd4/lizhy_workspace/beat_only_asr_push/beat_phase2_alternating_eval.json
```

### Lambda-align phase transition (key insight)
At `lambda_align=1.0`: ASR=0.175, at `lambda_align=1.5`: ASR=0.258 (worse), at `lambda_align=2.0`: ASR=0.142 (phase transition). Stronger alignment stabilizes hidden representations against backdoor re-emergence during simultaneous training.

### BEAT EMD metrics (word model)
| Stage | AUROC | TPR@FPR=5% |
|---|---|---|
| Pruned-only | 99.87 | 100% |
| Recovered best | 50.00 | 0% |

### Control model (BackdoorLLM-Llama2-7B-BadNets)
**Pipeline incompatible** — BEAT transfer makes ASR worse (1.00 vs baseline 0.90). Different backdoor mechanism (BadNets+LoRA vs BEAT finetuning).

## Experiment Launcher

`scripts/run_exp.sh` hardcodes BEAT word paths for quick recovery + eval:
```bash
bash scripts/run_exp.sh <label> [--extra-recover-args ...]
```

## Common Pitfalls

- **OOM**: 8B model + AdamW on trainable_policy=all needs all 4 GPUs. A single occupied GPU causes OOM.
- **Disk space**: recovered_model dirs are ~15 GB each. /ssd4 has the most free space (361 GB). Clean up old recovered_model dirs periodically.
- **Model cache**: Raw BEAT models download to `/ssd4/huggingface_cache/`. Pruned models go to `/ssd3/`.
- **Results persistence**: `recovery_losses.json` and eval JSONs are small; recovered_model weights are large. Only keep weights for best configs.
- **Determinism**: recover_model.py has no seed control. The simultaneous schedule produces reproducible ASR; alternating is more sensitive to initialization.
