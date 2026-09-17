# Llama Word Config / Transformers Version Matrix

目的：检查 `BEAT Llama-3.1-8B_word` 历史 golden `21` units 与当前重算 `61` units 的差异是否来自：

- 4/24 score 代码缺失；
- TF5 保存后的 damaged config；
- Transformers 版本差异。

所有有效 score run 使用同一参数：

```text
model: /home/lizhy/plp/Llama-3.1-8B_word 或其轻量 mirror
code: /home/lizhy/plp/tfpd_score_8097
score function: same collect_unit_scores hash as 89d79b1
prompt_template: alpaca
max_length: 256
dtype: bf16
alpha: 1.0
beta: 1.0
alpha_safe: 0.5
proxy_epsilon: 0.1
score_samples: 8
min_prune_layer: 2
max_prune_units: 320
max_score_to_prune: 0.0
protect_safe_jsonl: /home/lizhy/plp/TRANSFER/beat_data/harmful_no_trigger.jsonl
```

## Results

| Condition | Python / Torch / CUDA / cuDNN / Transformers | Score file sha256 | Selected units | Golden overlap |
|---|---|---|---:|---:|
| Golden old score | delivery artifact | `147d0a867e058c95f985d9e6ce1b224a021fa91b01e65ac4a0b68d0789cf85f1` | 21 | 21 |
| Current TF5.3 recompute | Py3.13 / torch 2.5.1+cu121 / cuDNN 90100 / TF 5.3.0 | `6b42791eed446e50fe3a9dfe9f80162b76a655642c08913dd82417b502288d3c` | 61 | 4 |
| Raw config, TF 4.43.3 | Py3.10 / torch 2.5.1+cu121 / cuDNN 90100 / TF 4.43.3 | `6b42791eed446e50fe3a9dfe9f80162b76a655642c08913dd82417b502288d3c` | 61 | 4 |
| Raw config, TF 4.57.1 | Py3.13 / torch 2.5.1+cu121 / cuDNN 90100 / TF 4.57.1 | `6b42791eed446e50fe3a9dfe9f80162b76a655642c08913dd82417b502288d3c` | 61 | 4 |

Layer histograms:

- Golden 21: `{29: 2, 30: 9, 31: 10}`
- Recomputed 61: `{2: 1, 3: 1, 15: 1, 19: 1, 25: 2, 27: 1, 28: 4, 29: 13, 30: 16, 31: 21}`

## Damaged Config Checks

Two lightweight model mirrors were built under `model_mirrors/` using symlinked raw weights and altered config files:

- `damaged_no_rope_list_eos`: drops `rope_scaling` and `rope_theta`, keeps list-form eos.
- `damaged_ropeparams_scalar_eos`: uses TF5-style `rope_parameters` and scalar eos.

Both variants loaded weights but then entered an extremely slow CPU-heavy scoring path with near-zero GPU utilization and no `unit_scores.json` within the practical wait window. They are not a useful explanation for the original golden score generation, which produced a normal score artifact.

## Conclusion

The `21 -> 61` mismatch is not explained by:

- Apr 24 vs Apr 26/28 scoring-code changes;
- delivery/config-save repair;
- raw config Transformers version among 4.43.3, 4.57.1, and 5.3.0;
- damaged saved config as a plausible normal scoring path.

The remaining difference is likely an unarchived execution-state or artifact provenance issue around the original `unit_scores.json` generation, not a recoverable public-code or Transformers-version switch.
