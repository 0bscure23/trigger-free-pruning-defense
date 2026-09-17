# FGSM Hidden-State Probe

Date: 2026-06-27

Purpose: provide a minimal cross-machine diagnostic for the Llama-3.1-8B_word scoring discrepancy. This probe does not run full scoring. It takes the first `benign_clean.jsonl` prompt, builds the actual `alpaca` prompt used by scoring, computes the consistency-loss FGSM direction from input embeddings, then reports hidden-state and gradient-sign fingerprints.

## Script

`/home/lizhy/plp/TRANSFER/diagnose_fgsm_hidden_probe.py`

Key details:

- model: `/home/lizhy/plp/Llama-3.1-8B_word`
- data: `/home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl`
- prompt template: `alpaca`
- max length: `256`
- dtype: `bf16`
- adv epsilon: `0.1`
- device: single GPU `cuda:0`
- layer inspected: `16`
- model parameters are frozen; only `inputs_embeds` receives gradients, to avoid 8B parameter-gradient OOM.

## Current Server Output

Both deterministic and non-deterministic local runs produced identical diagnostic values.

Common fingerprints:

- prompt text sha256: `fc7fa65ea4f502c8ea387bb828ebad9bf2c24e92b64c4b0ec34e53b2c156b5f4`
- input ids length: `14`
- input ids sha256: `ec19eef126bff683b302cf19245811a776a440ea13d0dce92fa2ba54c953e816`
- total consistency loss mean: `0.08984375`
- perturbation sign counts: `positive=28637`, `negative=28683`, `zero=24`
- input-embedding grad L2: `0.1982421875`
- hidden[16] L2 diff after perturbation: `540.0`
- hidden[16][0,0,:5] after perturbation:
  `[-0.146484375, -0.01220703125, 0.38671875, -0.07568359375, -0.103515625]`

## Files

- deterministic JSON: `llama_word_benign1_alpaca_cu121_det.json`
- deterministic log: `llama_word_benign1_alpaca_cu121_det.log`
- non-deterministic JSON: `llama_word_benign1_alpaca_cu121_nondet.json`
- non-deterministic log: `llama_word_benign1_alpaca_cu121_nondet.log`

## Command

```bash
source /home/lizhy/anaconda3/etc/profile.d/conda.sh
conda activate crow_repro_cu121
CUDA_VISIBLE_DEVICES=0 python /home/lizhy/plp/TRANSFER/diagnose_fgsm_hidden_probe.py \
  --model-path /home/lizhy/plp/Llama-3.1-8B_word \
  --jsonl /home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl \
  --prompt-template alpaca \
  --max-length 256 \
  --layer 16 \
  --adv-epsilon 0.1 \
  --dtype bf16 \
  --device cuda:0 \
  --deterministic \
  --output-json /tmp/llama_word_benign1_probe.json
```

If another server reports the same token hashes but meaningfully different hidden/gradient fingerprints, the discrepancy is below the prompt/data/tokenizer level. If this probe matches exactly while full scoring differs, the difference is likely in the later weight-gradient or CPU aggregation path used by full scoring.
