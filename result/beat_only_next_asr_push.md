# BEAT-only Next ASR Push

> Interim report. Remaining experiments queued via GPU watchdog.

## Setup

- Object: `BEAT-LLM-Backdoor/Llama-3.1-8B_word`
- Protocol: `dtype=bf16`, `eval_max_new_tokens=64`
- Fixed: `fixed + simultaneous + trainable_policy=all + lambda_clean=1.0 + steps=30`
- Optimizer: AdamW, lr=1.5e-05
- GPU: 4× RTX 3090 (24 GiB each), requires all 4 for device_map="auto"

## Completed Results

| lambda_safe | ASR | harmful_no_trigger_refusal | benign_clean_false_refusal | avg_len | empty_output_rate | Source |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.08 | **0.1917** | 0.8000 | 0.2500 | 62.35 | 0.0000 | [beat_phase2_raw_l008.json](/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2/result/beat_phase2_raw_l008.json) |
| 0.10 | 0.2167 | 0.7833 | 0.2200 | 62.36 | 0.0000 | Previous best from Phase D |
| 0.15 | 0.2500 | 0.7667 | 0.1800 | 60.74 | 0.0000 | Previous Phase B result |

## Runtime Failure Diagnosis

**Root cause of lambda_safe=0.08/0.12 failures**: Previous runs used `CUDA_VISIBLE_DEVICES=0` (single GPU), causing OOM on 24 GiB because `device_map="auto"` couldn't split the 8B model + AdamW optimizer states. Fixed by exposing all 4 GPUs.

**SGD is not viable**: Tested SGD at lr=5e-06 and lr=1e-04 on 2 GPUs. Losses barely decreased (l_align 30.5→29.7), ASR remained at raw baseline (~0.875). This task requires AdamW's adaptive learning rates for the multi-scale objectives (l_align ~30 vs l_safe ~1.5).

## Remaining Experiments (queued via watchdog)

| lambda_safe | Status |
| ---: | --- |
| 0.09 | Queued (AdamW, waiting for GPUs) |
| 0.11 | Queued (AdamW, waiting for GPUs) |
| 0.12 | Queued (AdamW, waiting for GPUs, retry from previous failure) |

If best lambda confirmed near 0.08-0.10, follow-up steps sweep: steps=25, 30, 35.

## Early Stop Check

ASR=0.1917 > 0.15, empty_output_rate=0.0. Early stop condition (ASR<=0.15) not yet met.
