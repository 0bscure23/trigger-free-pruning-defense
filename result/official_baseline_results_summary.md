# Official Baseline Results Summary - Mistral-Long

Date: 2026-06-23

Target model/eval: `Mistral-3-7B_long` on the existing BEAT-style
long-trigger evaluation splits. The official baseline runs below did not use
the BEAT known trigger during training/removal.

## Official Repositories Used

The official repositories were cloned under:
`/home/lizhy/plp/official_baselines/`

| Method | Local repo | GitHub remote | Commit used | Local changes |
|---|---|---|---|---|
| BEEAR | `/home/lizhy/plp/official_baselines/BEEAR` | `https://github.com/reds-lab/BEEAR.git` | `992635b4438a6593a50079398d0f2d7243110ea2` | Runtime/data compatibility patches in `dataset/dataset.py`, `utils/function.py`, `utils/models/split_mistral_7b.py`; local adapter `run_beear_beat_mistral.py`. |
| SANDE | `/home/lizhy/plp/official_baselines/SANDE` | `https://github.com/HKUST-KnowComp/SANDE.git` | `c3c77efafa63d639ca10955e8acfa3f50f7207d5` | Runtime compatibility patches in `models.py`, `train_remove.py`, `utils.py`; local data file `data/beear_pa_clean.jsonl`. |

The compatibility patches do not replace the official training objectives:

- BEEAR still calls official `utils.function.BEEAR` and
  `dataset.dataset.Template_Mistral_chat_Dataset`.
- SANDE still calls official `train_remove.py` simulate/remove.
- Patches only handle local Mistral tokenizer variants, 4-GPU placement, and
  skipping SANDE's optional utility-eval downloads during this baseline run.

## Model And Evaluation Setup

| Item | Value |
|---|---|
| Base victim model | `/home/lizhy/plp/Mistral-3-7B_long` |
| Architecture | `MistralForCausalLM`, 32 layers, hidden size 4096, vocab size 32768 |
| Model config origin string | `/home/yibiao/PTM/Mistral-7B-Instruct-v0.3` |
| Evaluation script | `/home/lizhy/plp/trigger-free-pruning-defense-round2/scripts/diagnose_generation_metrics.py` |
| Prompt template | `chat` |
| Eval dtype | `bf16` |
| Eval max new tokens | `64` |
| Triggered ASR split | `/home/lizhy/plp/TRANSFER/beat_data/harmful_long_trigger.jsonl`, 120 samples |
| Harmful-no-trigger refusal split | `/home/lizhy/plp/TRANSFER/beat_data/harmful_no_trigger.jsonl`, 120 samples |
| Benign false-refusal split | `/home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl`, 100 samples |
| PPL script | `/home/lizhy/plp/TRANSFER/rolling_ppl_auto.py` |
| PPL method | `rolling_max1024_stride512_masked_device_map_auto`, 331725 tokens |

## Completed Official Short Runs

| Method | Run tag | Official repo | GPU path | ASR | HNTR | BFR | PPL | Verdict |
|---|---|---|---|---:|---:|---:|---:|---|
| BEEAR | `beear_official_manual4_r1_i1_t8_b6` | `/home/lizhy/plp/official_baselines/BEEAR` | 4-GPU manual Mistral sharding | 0.900 | 0.392 | 0.160 | 12.02 | Valid official short-run result, but ASR remains high. |
| SANDE | `sande_official_dummy_sure_s16` | `/home/lizhy/plp/official_baselines/SANDE` | 4-GPU DeepSpeed | 0.925 | 0.425 | 0.140 | 11.35 | Valid official short-run result, but ASR remains high. |

HNTR = harmful-no-trigger refusal. BFR = benign-clean false refusal.

## Exact Run Configs

### BEEAR

| Item | Value |
|---|---|
| Launcher | `/home/lizhy/plp/TRANSFER/run_official_beear_mistral_long.sh` |
| Local adapter | `/home/lizhy/plp/official_baselines/BEEAR/run_beear_beat_mistral.py` |
| Output model | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/beear_official_manual4_r1_i1_t8_b6/beear_official_model` |
| Visible GPUs | `0,1,2,3` |
| Device map | `manual4` |
| Rounds | `1` |
| Inner epochs | `1` |
| Inner batch size | `6` |
| Inner threshold | `8` |
| PA threshold | `4` |
| Outer learning rate | `3e-7` |
| Safety anchoring data | `BEAT_Mistral_Long_SA.xlsx`, 120 rows |
| Harmful contrasting data | `BEAT_Mistral_Long_SAH.xlsx`, 120 rows |
| Performance anchoring data | BEEAR official LMSYS/GPT-4 table, 301 rows |
| Train log | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/beear_official_manual4_r1_i1_t8_b6/beear_train.log` |
| Run metadata | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/beear_official_manual4_r1_i1_t8_b6/run_meta.json` |
| ASR JSON | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/asr_beear_official_manual4_r1_i1_t8_b6.json` |
| PPL JSON | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/ppl_beear_official_manual4_r1_i1_t8_b6.json` |

Layer placement for `manual4`: embeddings and layers 0-8 on `cuda:0`,
layers 9-16 on `cuda:1`, layers 17-24 on `cuda:2`, layers 25-31 on `cuda:3`,
with norm/lm_head on `cuda:0`.

### SANDE

| Item | Value |
|---|---|
| Launcher | `/home/lizhy/plp/TRANSFER/run_official_sande_mistral_long.sh` |
| Official entrypoint | `/home/lizhy/plp/official_baselines/SANDE/train_remove.py` |
| Output model | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/sande_official_dummy_sure_s16/sande_removed_model` |
| Visible GPUs | `0,1,2,3` |
| DeepSpeed include | `localhost:0,1,2,3` |
| DeepSpeed zero stage | `2` |
| Precision | `bf16` |
| Adam offload | Enabled |
| Max length | `512` |
| Step 1 max samples | `16` |
| Step 2 max samples | `16` |
| Step 1 max epochs | `1` |
| Step 2 max epochs | `1` |
| Micro batch size | `1` |
| Train batch size | `4` |
| Trigger num | `6` |
| Step 1 train/test fn type | `harm` / `harm` |
| Step 2 train/test fn type | `clean` / `trigger` |
| Step 1 learning rate | `1e-3` |
| Step 2 learning rate | `5e-6` |
| SANDE dummy trigger argument | `sande dummy trigger` |
| SANDE marker argument | `Sure` |
| Clean-style dataset | `/home/lizhy/plp/official_baselines/SANDE/data/beear_pa_clean.jsonl`, 301 rows |
| Simulated trigger file | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/sande_official_dummy_sure_s16/simulator/simulating.pkl` |
| Simulate log | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/sande_official_dummy_sure_s16/sande_simulate.log` |
| Remove log | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/sande_official_dummy_sure_s16/sande_remove.log` |
| ASR JSON | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/asr_sande_official_dummy_sure_s16.json` |
| PPL JSON | `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_sande_mistral_long/ppl_sande_official_dummy_sure_s16.json` |

The SANDE dummy `--trigger` and `--marker` values are required by the official
SANDE interface. They are not the BEAT trigger and were not derived from the
BEAT known trigger.

## Screened Methods Without Formal BEAT-Mistral Result

| Method | Reason |
|---|---|
| CleanGen | Official method is inference-time decoding and needs a compatible clean/reference model. No clean Mistral-Long reference is available locally, so it is not a fair weight-removal baseline here. |
| DUP | Official repo is built around prepared poison datasets, classification/3B LLM configs, and LoRA unlearning. A BEAT-Mistral-7B generation comparison would require a new port and poisoned-input mining path. |
| BackdoorLLM defense suite | Better treated as an evaluation/framework suite, not a single removal baseline. |
| Locphylax / Backdoor Collapse | Unknown-trigger direction, but requires deliberate dummy-backdoor insertion and recovery. This changes assumptions and is a later exploration rather than a direct quick baseline. |
| Purifying Generative LLMs without Prior Knowledge | Requires constructing multiple synthetic backdoored variants and extracting a shared signature, so it is not a direct first-batch baseline. |
| Lethe | Requires a separate clean/lightweight model construction or merge path. |
| Dummy Backdoor as a Defense | Requires dummy-backdoor insertion plus removal; useful later, not a direct control here. |

## Immediate Takeaway

The two runnable official short baselines, BEEAR and SANDE, currently do not
beat the BEAT-Mistral backdoor in this setting. They should be kept as official
short-run controls, but stronger configurations would be needed before treating
either as a competitive formal baseline.
