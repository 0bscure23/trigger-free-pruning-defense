# Scoring Localization Summary

This summarizes localization experiments for the golden 21 vs recomputed 61 scoring mismatch. All score selections use `layer>=2`, `score<=0`, cap 320.

| condition | sha256 | selected | golden overlap | new61 overlap | layer hist |
|---|---|---:|---:|---:|---|
| `golden` | `147d0a867e058c95` | 21 | 21 | 4 | `{29: 2, 30: 9, 31: 10}` |
| `new61` | `6b42791eed446e50` | 61 | 4 | 61 | `{2: 1, 3: 1, 15: 1, 19: 1, 25: 2, 27: 1, 28: 4, 29: 13, 30: 16, 31: 21}` |
| `d19b7cd` | `6b42791eed446e50` | 61 | 4 | 61 | `{2: 1, 3: 1, 15: 1, 19: 1, 25: 2, 27: 1, 28: 4, 29: 13, 30: 16, 31: 21}` |
| `attn_eager` | `33c6be9c555697e3` | 56 | 4 | 52 | `{2: 1, 3: 1, 15: 1, 19: 1, 25: 1, 27: 1, 28: 5, 29: 12, 30: 15, 31: 18}` |
| `tf4433` | `6b42791eed446e50` | 61 | 4 | 61 | `{2: 1, 3: 1, 15: 1, 19: 1, 25: 2, 27: 1, 28: 4, 29: 13, 30: 16, 31: 21}` |
| `tf4571` | `6b42791eed446e50` | 61 | 4 | 61 | `{2: 1, 3: 1, 15: 1, 19: 1, 25: 2, 27: 1, 28: 4, 29: 13, 30: 16, 31: 21}` |
| `old89d79b1_chat_len256` | score-array sha `c18257f2035cd75` | 21 | 21 | 4 | `{29: 2, 30: 9, 31: 10}` |

## Findings
- `d19b7cd` Apr 22 code on current model/data/env produced the same byte-identical score file as the current 61-unit recompute. Code version is not the root cause.
- `CROW_ATTN_IMPLEMENTATION=eager` changed the score file and reduced the selected set from 61 to 56, but it still overlaps the golden 21 by only 4 units and overlaps the current 61-unit recompute by 52 units. Attention backend can perturb the scores, but it does not recover the golden score artifact.
- Transformers 4.43.3, 4.57.1, and 5.3.0 raw-config runs all produced byte-identical scores. Transformers minor/major version alone is not the root cause.
- Hybrid component ablation is in `COSINE_ABLATION.md`: old cosine/proxy/protect interactions recover more golden units than new cosine paths, but no single component swap fully explains 21 -> 61.
- Resolution: the missing score-stage setting was `--prompt-template chat`. Re-running the old `89d79b1` scoring entrypoint with `chat`, `max_length=256`, `alpha_safe=0.5`, `max_score_to_prune=0.0`, `min_prune_layer=2`, and `max_prune_units=320` reproduces the golden score tensor exactly. The only whole-file hash difference is the string path recorded for `protect_safe_jsonl`; the `scores` array sha256 is identical.

## Current Interpretation
The golden `unit_scores.json` is now score-stage reproducible: public old scoring code plus current model/data reproduces the exact 459,776-entry score array when the scoring prompt template is set to `chat`. The earlier 61-unit mismatch came from running score with `alpaca`, while the later paper-main evaluation used `alpaca / 1024 / 64`.
