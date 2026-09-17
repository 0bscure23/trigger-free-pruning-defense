# Framework-Like Formula Scan

Date: 2026-06-27

Purpose: test whether a scoring formula close to the current method can recover the historical golden 21-unit Llama Word plan from the current recomputed feature table.

This scan does not use arbitrary tree rules. It restricts candidates to formulas close to the existing defense framework:

```text
protect = clean + alpha_safe_like * safe
attack  = proxy * |cosine|^p
score   = protect - beta * attack
```

plus simple variants:

- raw score
- global z-score normalization
- per-layer z-score normalization
- global rank normalization
- per-layer rank normalization
- ratio variants such as `protect / attack` and `-attack / protect`

The scan also tested three selection modes:

- global top-21
- per-layer 7/7/7 from layers 29/30/31
- per-layer golden allocation 2/9/10 from layers 29/30/31

The last mode is explicitly post-hoc and should be treated as a diagnostic, not a deployable rule.

## Best Result

Best overlap with the historical golden 21:

```text
6 / 21
```

Best configuration:

```text
mode: per_layer_gold_alloc_2_9_10
alpha_safe_like: 0.5
beta: 0.3
attack: proxy * |cosine|^0.25
variant: per-layer z-score
```

Selected golden units:

```text
('channel', 29, 2823)
('channel', 30, 8429)
('channel', 30, 6369)
('channel', 30, 8030)
('channel', 30, 6959)
('channel', 31, 7124)
```

This is only a small improvement over the original current-score overlap of `4/21`.

## Interpretation

Moving closer to the current scoring framework does not solve the mismatch. Even with:

- late-layer restriction,
- framework-style `protect - attack`,
- modified attack exponent,
- per-layer normalization,
- and post-hoc golden layer allocation,

the best framework-like formula recovers only `6/21` golden units.

Therefore the historical golden 21 cannot be explained as a simple retuning of:

- `alpha_safe`,
- `beta`,
- threshold,
- late-layer prior,
- per-layer normalization,
- or a smooth attack/protect ratio.

The only approaches that recover most or all golden units are high-capacity post-hoc rules such as decision trees or ExtraTrees, which are not acceptable as a clean method definition without held-out validation.

## Files

- Full JSON scan: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_reverse_engineering/framework_like_formula_scan.json`
