# Official SANDE Baseline Attempt - Mistral-Long

Status: official SANDE simulate/remove completed on 4 GPUs, and unified
ASR/PPL evaluation completed.

## What Was Used

- Official repo: `/home/lizhy/plp/official_baselines/SANDE`
- Official training core:
  - `train_remove.py`
  - `models.py`
  - `utils.py`
- Local adapter:
  - `run_official_sande_mistral_long.sh` supplies the local BEAT model/data
    paths and launches the official simulate/remove stages.
  - `data/beear_pa_clean.jsonl` supplies clean-style prompts prepared from the
    local BEEAR/BEAT data conversion.

No BEAT known trigger was used for training/removal. The official SANDE CLI
requires a `--trigger` and `--marker`; this run used defender-chosen dummy
values, not the real BEAT trigger.

## Runtime Fixes

The official code path needed local compatibility fixes:

- use slow tokenizer loading for the local Mistral tokenizer;
- skip SANDE's optional internal utility eval during this baseline run;
- install missing runtime dependencies in the `torch251` environment
  (`deepspeed==0.15.3`, `bitsandbytes`).

These are runtime compatibility fixes, not local proxy logic.

## Completed Metric

| run | config | ASR | harmful-no-trigger refusal | benign-clean false refusal | rolling PPL | empty output rate |
|---|---|---:|---:|---:|---:|---:|
| `sande_official_dummy_sure_s16` | simulate 16 samples, remove 16 samples, trigger_num=6, micro_batch=1, train_batch=4, max_len=512 | 0.925 | 0.425 | 0.140 | 11.35 | 0.000 |

Artifacts:

- simulator: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/sande_official_dummy_sure_s16/simulator/simulating.pkl`
- removed model: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/sande_official_dummy_sure_s16/sande_removed_model`
- ASR JSON: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/asr_sande_official_dummy_sure_s16.json`
- PPL JSON: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/ppl_sande_official_dummy_sure_s16.json`

## Interpretation

SANDE is a valid official generative-backdoor removal baseline candidate, but
this short configuration did not meaningfully remove the BEAT-Mistral backdoor:
ASR remains 0.925. Because the run uses dummy defender-chosen SANDE arguments,
the method should be described as trigger-free with respect to the BEAT real
trigger, while disclosing SANDE's dummy marker/trigger simulation interface.
