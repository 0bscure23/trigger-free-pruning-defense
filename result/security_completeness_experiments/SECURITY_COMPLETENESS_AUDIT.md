# Security-Completeness Experiment Audit

- Root: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/security_completeness_experiments`
- Runs found: `27`
- Runs with issues: `27`
- Leftover temporary checkpoints: `0`

## Full-Pipeline Seed Summary

| Metric | n | Mean | Std |
|---|---:|---:|---:|
| ASR | 0 | nan | nan |
| HarmRef | 0 | nan | nan |
| BFR | 0 | nan | nan |
| Empty | 0 | nan | nan |
| PPL | 0 | nan | nan |

## Run Table

| Run | Family | Seed | Requested | Actual | #Heads | #MLP channels | ASR | HarmRef | BFR | Empty | Avg tok | Median tok | PPL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| budget_001_seed13 | budget_sweep | 13 | 460 |  |  |  |  |  |  |  |  |  |  |
| budget_003_seed13 | budget_sweep | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| budget_005_seed13 | budget_sweep | 13 | 2299 |  |  |  |  |  |  |  |  |  |  |
| budget_010_seed13 | budget_sweep | 13 | 4598 |  |  |  |  |  |  |  |  |  |  |
| gate_0_seed13 | threshold_sweep | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| gate_m0.02_seed13 | threshold_sweep | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| gate_m0.05_seed13 | threshold_sweep | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| gate_nogate_seed13 | threshold_sweep | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| gate_p0.05_seed13 | threshold_sweep | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_align_1p0_seed13 | recovery_lambda_align_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_align_1p5_seed13 | recovery_lambda_align_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_align_2p0_seed13 | recovery_lambda_align_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_align_2p5_seed13 | recovery_lambda_align_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_safe_0_seed13 | recovery_lambda_safe_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_safe_0p04_seed13 | recovery_lambda_safe_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_safe_0p08_seed13 | recovery_lambda_safe_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lambda_safe_0p12_seed13 | recovery_lambda_safe_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lr_1p5e-5_seed13 | recovery_lr_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lr_3e-5_seed13 | recovery_lr_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lr_3e-6_seed13 | recovery_lr_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| lr_5e-6_seed13 | recovery_lr_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| seed_13_main_b1379_g0 | full_pipeline_seed | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| seed_17_main_b1379_g0 | full_pipeline_seed | 17 | 1379 |  |  |  |  |  |  |  |  |  |  |
| seed_23_main_b1379_g0 | full_pipeline_seed | 23 | 1379 |  |  |  |  |  |  |  |  |  |  |
| steps_10_seed13 | recovery_steps_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| steps_25_seed13 | recovery_steps_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |
| steps_50_seed13 | recovery_steps_sensitivity | 13 | 1379 |  |  |  |  |  |  |  |  |  |  |

## Issues

| Run | Missing ASR fields | Missing plan fields | Missing PPL |
|---|---|---|---:|
| `budget_001_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `budget_003_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `budget_005_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `budget_010_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `gate_0_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `gate_m0.02_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `gate_m0.05_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `gate_nogate_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `gate_p0.05_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_align_1p0_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_align_1p5_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_align_2p0_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_align_2p5_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_safe_0_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_safe_0p04_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_safe_0p08_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lambda_safe_0p12_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lr_1p5e-5_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lr_3e-5_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lr_3e-6_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `lr_5e-6_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `seed_13_main_b1379_g0` | `asr.json` | `pruning_plan.json` | `True` |
| `seed_17_main_b1379_g0` | `asr.json` | `pruning_plan.json` | `True` |
| `seed_23_main_b1379_g0` | `asr.json` | `pruning_plan.json` | `True` |
| `steps_10_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `steps_25_seed13` | `asr.json` | `pruning_plan.json` | `True` |
| `steps_50_seed13` | `asr.json` | `pruning_plan.json` | `True` |

