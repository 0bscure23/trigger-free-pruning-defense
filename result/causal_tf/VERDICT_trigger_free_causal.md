# Trigger-free causal follow-up verdict (Mistral-Long)

Date: 2026-06-21. Model/eval match the Mistral-Long follow-up runs under `/home/lizhy/plp`.

## Question

The trigger-aware causal/logit-gap signal showed that Mistral-Long has a separable channel-level circuit:
`causal + recovery K=1024` reached ASR 0.333 with rolling PPL 12.63, versus the trigger-free FGSM baseline ASR
0.683 with rolling PPL 12.86. The open question was whether the same causal machinery can be made deployable by
replacing the real trigger with a trigger-free soft perturbation.

## Results

| Variant | Trigger access | ASR | HarmRef | BFR | Rolling PPL | Verdict |
|---|---|---:|---:|---:|---:|---|
| raw Mistral-Long | none | 0.892 | 0.400 | 0.160 | 12.11 | undefended |
| FGSM proxy baseline | none | 0.683 | 0.467 | 0.220 | 12.86 | current method baseline |
| causal + recovery K=512 | triggered prompts | 0.450 | 0.433 | 0.300 | 12.50 | trigger-aware upper-bound |
| causal + recovery K=1024 | triggered prompts | 0.333 | 0.450 | 0.420 | 12.63 | trigger-aware upper-bound |
| TF refsupp eps=0.01 K=512 | none | 1.000 | 0.450 | 0.380 | 13.03 | fails |
| TF refsupp eps=0.01 K=1024 | none | 0.433 | 0.667 | 0.590 | 12.97 | partial win, high BFR |
| TF refsupp eps=0.03 K=1024 | none | 0.692 | 0.483 | 0.210 | 12.81 | no gain vs baseline |
| TF PGD eps=0.03 K=1024 | none | 0.775 | 0.433 | 0.190 | 12.03 | worse |
| TF PGD eps=0.03 K=512 | none | 0.642 | 0.383 | 0.200 | 13.79 | slight ASR gain, worse than eps=0.01 |

## Verdict

The causal representation direction is real, but the deployable trigger-free version is not yet a clean main-method
result.

- The trigger-aware causal signal proves an upper bound: channel-level causal selection can reduce Mistral-Long ASR
  to 0.333 with preserved PPL.
- The best trigger-free causal variant, `tf_refsupp_eps0.01_1024`, nearly reaches that upper bound on ASR
  (0.433 vs 0.333) and beats the FGSM baseline (0.683), but it pays a large benign false-refusal cost (BFR 0.590).
- Multi-step PGD did not help. It reduced overlap with the trigger-aware oracle and produced ASR 0.642-0.775.
- Therefore the paper should describe this as a promising but unresolved follow-up: trigger-free causal perturbations
  can move the long-trigger ASR substantially, but the current construction is not balanced enough for the main table.

## Artifacts

- Trigger-aware causal upper bound: `result/causal_direction/VERDICT_causal.md`
- Trigger-free one-step recovery: `result/causal_tf/run_tf_recover.log`
- Trigger-free PGD recovery: `result/causal_tf/run_2b_pgd.log`
- Rolling PPL refresh: `result/ppl_rolling_refresh/run.log`
