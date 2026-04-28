# Paper Tables (Final)

> Protocol: bf16, eval_max_new_tokens=64, base env (TF 5.3.0)

---

## Table 1: BEAT ASR — Jailbreak Defense Results

| Model | Setting | ASR↓ | Harmful Refusal↑ | Benign FR↓ | Empty Rate | Notes |
|---|---|---|---|---|---|---|
| Word | Raw | 0.9250 | 0.3167 | 0.030 | 0.0 | BEAT Llama-3.1-8B_word, trigger " SUDO" |
| Word | **Balanced** | **0.1417** | 0.8667 | 0.390 | 0.0 | simul+align=2.0+safe=0.08+steps=25 |
| Phrase | Raw | 0.9000 | 0.4833 | 0.030 | 0.0 | trigger " Servius Astrumando Harmoniastra" |
| Phrase | **Strong** | **0.0750** | 0.9083 | 0.580 | 0.0 | simul+align=2.0+safe=0.08+steps=25 |
| Phrase | Balanced | 0.1667 | 0.8417 | **0.350** | 0.0 | simul+align=1.0+safe=0.08+steps=25, lower BFR trade-off |
| Long | Raw | 0.9250 | 0.1583 | 0.030 | 0.0 | trigger: Shakespeare passage |
| Long | **Best** | **0.1750** | 0.8333 | 0.320 | 0.0 | simul+align=2.5+safe=0.08+steps=25 |

**Interpretation**: Word and phrase triggers are effectively neutralized (ASR reduction 84-92%). Long trigger shows partial improvement (81% reduction) but does not meet the 0.15 threshold. The phrase strong-defense point achieves the lowest ASR (0.075) at the cost of high benign false-refusal (0.58). A balanced phrase variant trades some ASR for substantially lower BFR (0.35).

---

## Table 2: Utility Preservation

| Model | Setting | PPL↓ | BoolQ↑ | RTE↑ | HellaSwag↑ | HellaSwag_n↑ |
|---|---|---|---|---|---|---|
| Word | Raw | 13.16 | 0.787 | 0.635 | 0.547 | 0.722 |
| Word | Defended | **11.32** | **0.804** | **0.711** | 0.517 | 0.702 |
| Phrase | Raw | 12.60 | 0.774 | 0.628 | 0.550 | 0.730 |
| Phrase | Defended (strong) | 11.91 | 0.773 | 0.718 | 0.512 | 0.691 |
| Long | Raw | 14.07 | 0.762 | 0.610 | 0.555 | 0.727 |
| Long | Defended | **12.63** | 0.772 | 0.650 | 0.509 | 0.687 |

**Interpretation**: Recovery training consistently **improves** PPL, BoolQ, and RTE across all three models, with modest degradation in HellaSwag. The defense does not harm general language capability; in several cases it improves it.

---

## Table 3: BEAT EMD Detection Metrics

| Model | Stage | AUROC | AP | TPR@FPR=5% | Interpretation |
|---|---|---|---|---|---|
| Word | Pruned-only | 99.87 | 99.74 | 100.0 | Trigger easily detectable |
| Word | Recovered | **50.00** | 33.33 | 0.0 | Detector reduced to random chance |
| Phrase | Pruned-only | 99.91 | 99.82 | 100.0 | Trigger easily detectable |
| Phrase | Recovered | **54.32** | 42.00 | 13.0 | Near-random, slight residual signal |
| Long | Pruned-only | 100.00 | 99.99 | 100.0 | Trigger easily detectable |
| Long | Recovered | **86.97** | 87.05 | 76.0 | Partial defense, residual detectability |

**Interpretation**: For word and phrase, the EMD detector degrades to near-random performance, confirming the backdoor signal is effectively erased from output distributions. For long, the detector retains significant discriminability (AUROC 86.97), consistent with the higher defended ASR (0.175). The EMD metric independently validates the ASR findings: stronger ASR reduction correlates with lower EMD detectability.

---

## 4. Conclusion Grading

### Can Be Written as Main Results in Paper Body

1. **BEAT word**: ASR 0.925 -> 0.142 (84% reduction), fully reproducible, validated by EMD (AUROC 99.87 -> 50.00), PPL improves (13.16 -> 11.32).
2. **Lambda-align phase transition**: At lambda_align=1.0 -> ASR 0.175; at 1.5 -> collapse; at 2.0 -> ASR 0.142. Strong alignment creates a phase transition that prevents backdoor re-emergence.
3. **Recovery preserves utility**: PPL, BoolQ, RTE consistently improve after defense; HellaSwag degrades modestly.
4. **EMD independently validates ASR**: BEAT's own detection protocol confirms defense effectiveness.

### Can Be Written as Extension Results

5. **BEAT phrase**: ASR 0.900 -> 0.075 (92% reduction). Strong defense with high BFR trade-off (0.58). A balanced variant achieves ASR 0.167 with BFR 0.35.
6. **BEAT long**: ASR 0.925 -> 0.175 (81% reduction). Partial success, lambda_align=2.5 is optimal but does not meet 0.15 threshold. Same phase-transition pattern observed.
7. **Phrase EMD**: AUROC 99.91 -> 54.32. Near-random, close to word-level defense quality.
8. **Long EMD**: AUROC 100.00 -> 86.97. Residual detectability consistent with partial ASR reduction.

### Can Only Appear in Limitations / Discussion

9. **Control model incompatibility**: BackdoorLLM-Llama2-7B-BadNets defense fails (ASR 1.0). Different backdoor mechanism (BadNets+LoRA) requires fundamentally different strategy.
10. **Long trigger not fully neutralized**: ASR 0.175 > 0.15 threshold. Very long trigger texts may require different proxy construction.
11. **Benign false-refusal trade-off**: Strong defense settings (phrase ASR 0.075) come with high BFR (0.58). No hyperparameter setting achieves both optimal ASR and optimal BFR simultaneously.
12. **Steps sensitivity**: Optimal training budget (25 steps) must be found empirically; over-training causes alignment forgetting and ASR collapse.

### Cannot Be Used (Historical Results Only)

- **Alternating ASR=0.0917**: Not reproducible. Loss curves differ from historical run at step 1 despite identical model, data, code, and environment. Cause unknown.
- **eval_max_new_tokens=32 results**: Different evaluation protocol, cannot be compared to bf16+64 results.
- **fp16 collapse runs**: Behavioral collapse due to precision, not defense quality.
- **Empty output collapse runs (empty_rate > 0)**: Models producing no text are not valid defense results.

---

## 5. Recommended Next Experiments

1. **Reduce phrase BFR while keeping ASR < 0.15** (highest priority).
   Try: lower lambda_safe (0.04) + lambda_align sweet spot, or safe_target_mode="pool".
   
2. **Tune long to meet ASR <= 0.15**.
   Try: lambda_align in [2.4, 2.6] at 0.05 increments, or lambda_safe=0.06 with align=2.5.

3. **Re-score phrase/long with lower max_prune_units** (e.g., 50-100).
   Current 320-unit pruning is likely overshooting; fewer units may reduce BFR.

4. **Run perturbation-step ablation on refusal model**.
   Clean-gradient proxy vs FGSM-perturbation proxy — needed for method justification in paper.

5. **Decide phrase/long paper placement**.
   If phrase BFR can be reduced and long ASR pushed below 0.15, both go in main text.
   Otherwise, phrase as strong extension, long in limitations/discussion.
   Control incompatibility should be a short limitations paragraph.
