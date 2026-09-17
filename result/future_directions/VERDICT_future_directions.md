# Future-work directions — genuine exploration (Mistral-Long)

Date: 2026-06-20. Same env/model/eval as all other Mistral-Long runs (no cross-server confound).
This closes the two future-work directions named in the paper, which had NOT actually been built before:
  A) sequence-level proxy   B) representation-layer signal.

## Results

| variant | ASR | HR(refuse harmful) | BFR | PPL |
|---|---|---|---|---|
| raw (no defense) | 0.892 | 0.400 | 0.160 | 12.11 |
| **FGSM-proxy 512 (method baseline)** | **0.683** | 0.467 | 0.220 | **12.86** |
| A: seq_mean proxy 512 | 0.742 | 0.450 | 0.240 | 13.25 |
| A: window proxy 512 | 0.758 | 0.425 | 0.220 | 13.25 |
| B-oracle ablate K=512 | 0.983 | 0.011 | 0.010 | 103.14 |
| B-oracle ablate K=1024 | 1.000 | 0.044 | 0.053 | 102.46 |
| B-oracle ablate K=2048 | 0.980 | 0.017 | 0.000 | 78.96 |
| B-oracle ablate K=4096 | 1.000 | 0.000 | 0.000 | 109.65 |
| B-oracle + recover K=512 | 0.900 | 0.000 | 0.000 | 107.86 |
| B-oracle + recover K=2048 | 0.992 | 0.000 | 0.000 | 356.78 |
| B-honest ablate K=512 | 0.950 | 0.300 | 0.060 | 14.56 |
| B-honest + recover K=512 | 0.892 | 0.521 | 0.270 | 13.30 |

## Verdict: both directions FAIL to beat the per-token FGSM proxy. The limitation is the granularity, not the signal.

**A — sequence-level proxy: REFUTED.** Pooling the consistency proxy over the sequence (mean or window)
made the early-layer bias *worse* (seq_mean picks 503/512 early-layer channels vs the FGSM proxy's 419)
and raised ASR to 0.742–0.758 (worse than 0.683). The intuition "a sequence-level proxy better matches the
long trigger" is empirically wrong — sequence pooling concentrates the signal even harder in early
lexical/positional channels.

**B — representation-layer signal (magnitude forms): FAIL, and reveal entanglement.**
- Oracle = |act_triggered − act_clean| (label-aware). Pruning the most trigger-responsive channels does NOT
  reduce ASR — it *destroys the model*: ASR→0.98–1.00, harmful-refusal→0 (refuses nothing), PPL→80–110.
  Even K=512 (0.1% of MLP channels) collapses PPL to 103. Recovery cannot undo it (ASR 0.90–0.99, PPL 108–357).
- Honest = label-free activation peakiness. Ablation ASR 0.95; after recovery ASR returns to 0.892 (= raw):
  pruning+recovery leaves the backdoor fully intact.
- Mechanism: |Δact| (and peakiness) are dominated by overall channel magnitude, so the "most trigger-responsive"
  channels ARE the high-norm load-bearing channels. The trigger response is entangled with core safety + LM
  computation; there is no separable channel-level "backdoor circuit" for the long trigger.

## What is PROVEN vs still OPEN (honest)
PROVEN: across gradient (FGSM), sequence-pooled gradient, and activation-magnitude representation signals, at the
**channel/head granularity**, nothing beats ~0.68 on Mistral-Long; magnitude-based selection cannot isolate the
backdoor (removes load-bearing channels → collapse). The earlier method_upgrade kappa=0 late-targeting (0.675)
agrees. => The long-trigger limitation is about pruning *granularity*, robust to the scoring signal.

STILL OPEN (did NOT test): a *causal* representation signal (activation patching / logit-gap "B.1"), and
sub-channel / weight-level or circuit-level edits. Those are a different granularity, not a different channel score.

## Best defense for Mistral-Long remains the original FGSM-proxy at 512 units: ASR 0.683, PPL 12.86.

Artifacts: result/future_directions/ (asr_*.json, ppl_*.json, plan_*.json, run.log).
Code: TRANSFER/build_repr_plan.py (oracle/honest), TRANSFER/apply_plan_only.py (ablation),
      pipeline_utils._consistency_loss_from_hidden_states (CROW_PROXY_MODE seq_mean/seq_last/window).
