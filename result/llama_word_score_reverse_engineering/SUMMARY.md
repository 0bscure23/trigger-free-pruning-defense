# Llama Word Score Drift Reverse-Engineering Diagnostic

This is a post-hoc diagnostic, not a deployable trigger-free method.

## Counts

- common_units: `459776`
- eligible_units: `431040`
- golden_nonpos: `21`
- current_nonpos: `61`
- shared_nonpos: `4`

## Original Current Score

- overlap@21 with golden 21: `4/21`

## Best Interpretable Grid Formula

Formula family: `clean + safe_coef * safe - beta * abs(proxy * cosine) - late_layer_prior_coef * layer_scaled`.

- best overlap@21: `4/21`
- safe_coef: `0.5`
- beta_attack_coef: `1.0`
- late_layer_prior_coef: `0.0`
- selected_layers: `{29: 3, 3: 1, 31: 4, 30: 7, 25: 1, 2: 1, 19: 1, 28: 3}`

## Oracle Logistic Diagnostic

This model uses the golden 21 labels in-sample. It is only a signal-presence diagnostic.

- overlap@21: `3/21`
- average precision: `0.035111`
- ROC-AUC: `0.997674`

Top standardized coefficients:

- `layer_scaled`: `25.543241`
- `safe`: `-8.082656`
- `clean`: `4.588574`
- `proxy`: `1.538212`
- `protect`: `0.929332`
- `original_score`: `0.894869`
- `is_channel`: `0.695621`
- `attack`: `0.430783`

## Oracle Nonlinear Diagnostics

These models also use the golden 21 labels in-sample. They are intentionally post-hoc diagnostics and must not be presented as the deployed trigger-free method.

- `ExtraTreesClassifier(class_weight=balanced, min_samples_leaf=1)`: overlap@21 = `21/21`, AP = `1.0`
- `ExtraTreesClassifier(class_weight=balanced, min_samples_leaf=20)`: overlap@21 = `19/21`, AP = `0.972502`
- `HistGradientBoostingClassifier(weighted)`: overlap@21 = `7/21`, AP = `0.243253`

The ExtraTrees feature importances are dominated by `layer_scaled`, `proxy`, `attack`, and `cosine`. This means the current feature table still contains a nonlinear/post-hoc pattern that can separate the golden units, but the transparent original score family does not recover it.

Interpretation: this supports using the golden archive for diagnostic feature analysis, not for directly defining a publishable scoring rule.

## Files

- JSON: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_reverse_engineering/score_drift_analysis.json`
- Golden ranks CSV: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_reverse_engineering/golden_units_ranked_by_current_score.csv`
- Affine grid CSV: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_reverse_engineering/best_affine_grid.csv`
- Nonlinear oracle diagnostics: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_reverse_engineering/oracle_nonlinear_diagnostics.json`
