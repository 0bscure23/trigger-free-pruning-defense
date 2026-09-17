# Official BEEAR Baseline Attempt — Mistral-Long

Status: official BEEAR was moved from an accidental single-GPU run to a
4-GPU model-parallel run path. The 4-GPU smoke test passed, the short training
run completed, and unified ASR/PPL evaluation completed.

## What Was Used

- Official repo: `/home/lizhy/plp/official_baselines/BEEAR`
- Official training core:
  - `utils.function.BEEAR`
  - `dataset.dataset.Template_Mistral_chat_Dataset`
  - `utils.models.Split_MistralModel`
- Local adapter:
  - `run_beear_beat_mistral.py` supplies the local BEAT model/data paths.
  - `prepare_beear_official_beat_data.py` converts BEAT harmful-no-trigger data
    into BEEAR's Safety Anchoring / Harmful Contrasting table format.

No known trigger was used.

## Runtime Fixes

The original official code path hard-coded single GPU execution via
`CUDA_VISIBLE_DEVICES=0` and `model_split.to(device_id)`. That caused the
earlier single-card OOMs. The current run path keeps the official BEEAR
optimization loop, but adds runtime compatibility fixes:

- respect externally provided `CUDA_VISIBLE_DEVICES`;
- shard Mistral layers across 4 visible GPUs for `manual4`;
- move hidden states/attention masks/position ids across devices inside the
  official split Mistral wrapper;
- make the Mistral dataset marker lookup robust to tokenizer variants of
  `[/INST]`.

These are runtime/model-placement fixes, not a known-trigger or local proxy
baseline.

## Environment

Created/used `torch251` with:

- PyTorch 2.5.1+cu121
- transformers 4.37.2
- datasets 2.19.2
- peft 0.8.2
- setuptools < 70, because official BEEAR imports `pkg_resources`

## Results So Far

| run | config | outcome |
|---|---|---|
| single-GPU smoke | rounds=1, inner_epochs=1, no training | passed setup |
| `beear_official_r2_i3_t60` | single GPU, rounds=2, inner_epochs=3, batch=6, threshold=60 | OOM, because it was still single-card |
| `beear_official_r1_i1_t12_b1` | single GPU, batch=1 | official code shape error because batch=1 squeezes away batch dimension |
| `beear_official_r1_i1_t8_b2` | single GPU, batch=2 | OOM, because it was still single-card |
| `smoke_manual4` | 4 GPUs, manual layer sharding, no training | passed setup |
| `beear_official_manual4_r1_i1_t8_b6` | 4 GPUs, rounds=1, inner_epochs=1, batch=6, threshold=8 | completed; model saved and evaluated |

## Completed Metric

| run | ASR | harmful-no-trigger refusal | benign-clean false refusal | rolling PPL | empty output rate |
|---|---:|---:|---:|---:|---:|
| `beear_official_manual4_r1_i1_t8_b6` | 0.900 | 0.392 | 0.160 | 12.02 | 0.000 |

Artifacts:

- model: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/beear_official_manual4_r1_i1_t8_b6/beear_official_model`
- ASR JSON: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/asr_beear_official_manual4_r1_i1_t8_b6.json`
- PPL JSON: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_beear_mistral_long/ppl_beear_official_manual4_r1_i1_t8_b6.json`

## Interpretation

The earlier statement that BEEAR was blocked by "24GB GPU memory" was too
coarse and wrong for the 4-card machine. The real issue was that the first
adapter and wrapper were forcing official BEEAR onto one GPU.

Current evidence:

- BEEAR is a valid trigger-free official baseline candidate.
- The 4-GPU model-parallel path initializes and trains correctly.
- The short official BEEAR run did not meaningfully remove the BEAT-Mistral
  backdoor: ASR remains 0.900.
- The earlier blocker was single-GPU forcing in the adapter/wrapper, not a
  fundamental lack of total GPU memory across the 4-card machine.

The earlier local `beear_tf_proxy_safe` run is not an official BEEAR result and
must not be used as a formal comparison.
