# Oracle Separability Diagnostic

This is an offline diagnostic only. It uses trigger-aware oracle labels, so supervised numbers are upper bounds rather than deployable defenses.

- Universe: 459776 units
- Oracle positives: 1024
- Random AP baseline: 0.002227

## Unsupervised Top-1024 Oracle Hits

- fgsm512: 5/1024 recall=0.0049
- multiview1024: 50/1024 recall=0.0488
- pseudo512: 2/1024 recall=0.0020
- seqmean: 3/1024 recall=0.0029
- tf_cons4096: 27/1024 recall=0.0264
- tf_diff4096: 51/1024 recall=0.0498
- tf_eps001: 73/1024 recall=0.0713
- tf_pgd1024: 30/1024 recall=0.0293
- tf_refsupp4096: 37/1024 recall=0.0361
- weightdiff4096: 28/1024 recall=0.0273
- window: 3/1024 recall=0.0029

## Supervised Upper Bounds

- logistic_balanced: test AP=0.013014, test AUC=0.5956, test top1024 hits=39 recall=0.1270; in-sample top1024 hits=63 recall=0.0615
- hist_gradient_boosting_balanced: test AP=0.014322, test AUC=0.6705, test top1024 hits=35 recall=0.1140; in-sample top1024 hits=108 recall=0.1055

## Oracle Fit-All Upper Bounds

- logistic_fit_all_oracle_upper_bound: AP=0.017651, AUC=0.6252, top1024 hits=64 recall=0.0625
- hist_gradient_boosting_fit_all_oracle_upper_bound: AP=0.041821, AUC=0.7154, top1024 hits=123 recall=0.1201

## Logistic Top Coefficients

- best_rrf: 0.3309
- multiview1024:rrf: -0.2285
- fgsm512:score_z: 0.2078
- pseudo512:present: -0.1930
- tf_eps001:present: 0.1734
- fgsm512:present: -0.1591
- tf_cons4096:rrf: -0.1563
- fgsm512:rank_pct: -0.1454
- weightdiff4096:present: 0.1340
- seqmean:present: -0.1334
- tf_pgd1024:score_z: -0.1230
- tf_eps001:rrf: -0.1120
- tf_cons4096:rank_pct: 0.1118
- tf_diff4096:present: 0.1098
- weightdiff4096:rrf: -0.1081
