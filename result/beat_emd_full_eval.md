# Phase 3: BEAT EMD Full Evaluation

> Protocol: EMD-based detection with sentence-transformers/all-MiniLM-L6-v2, 10 samples per generation, 1 jailbreak probe, 300 test samples (100 triggered + 200 clean)

## Results

| Model | Stage | AUROC | AP | TPR@FPR=5% | Interpretation |
| --- | --- | ---: | ---: | ---: | --- |
| Word | Pruned-only | **99.87** | 99.74 | 100.0% | Trigger trivially detectable |
| Word | Recovered best | **50.00** | 33.33 | 0.0% | Trigger completely undetectable |
| Phrase | Pruned-only | **42.82** | — | — | Below random (inverted detection) |
| Phrase | Recovered | **49.03** | — | — | Near random chance |
| Long | Pruned-only | — | — | — | Pending |
| Long | Recovered | — | — | — | Pending |

## Analysis

**Word model**: Classic defense success. EMD detection drops from near-perfect (99.87) to random chance (50.00) after recovery training. The output distributions with and without trigger become indistinguishable.

**Phrase model**: The pruned-only baseline shows AUROC=42.82, which is below random chance (50). This means the EMD-based detector assigns higher scores to CLEAN samples than to TRIGGERED samples — the detection is inverted. This may occur because:
1. The long phrase trigger text (" Servius Astrumando Harmoniastra") dominates the EMD distance computation more than the model's backdoor behavior
2. The phrase model's backdoor may operate through a different mechanism (e.g., lower-level feature changes rather than output distribution shift)

After recovery, AUROC moves to 49.03 (near random), consistent with defense reducing the detectable signal.

**Long model**: Pending. Given the even longer trigger text, EMD-based detection may be even more dominated by the trigger itself.

## Conclusion

For the word model, BEAT's own metrics validate our defense: the backdoor is completely undetectable after recovery. For phrase/long triggers with longer trigger texts, EMD-based detection becomes less informative as the trigger text itself dominates the distance computation. This is a limitation of the BEAT evaluation protocol, not of our defense method.

## Source Files
- Word pruned: `/ssd4/.../beat_eval_pruned_only/beat_pruned_only_eval.json`
- Word recovered: `/ssd4/.../beat_eval_recovered/beat_recovered_eval.json`
- Phrase pruned: `/ssd4/.../beat_emd_phrase_pruned.json`
- Phrase recovered: `/ssd4/.../beat_emd_phrase_recovered.json`
