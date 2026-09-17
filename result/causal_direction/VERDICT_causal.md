# Causal / logit-gap representation signal — Mistral-Long (the result that overturns the earlier null)

Date 2026-06-20. Same env/model/eval as result/future_directions and recovery_followup (directly comparable).
Signal = gradient×activation attribution of refusal-NLL on **triggered** prompts (logit-gap / attribution patching):
prune the channels most causally responsible for *suppressing refusal*. NOT magnitude-based (the magnitude
oracle collapsed the model); NOT trigger-free (it uses triggered prompts to localize — see caveat).

## Results (Mistral-Long)
| variant | ASR | HR | BFR | PPL |
|---|---|---|---|---|
| raw (no defense) | 0.892 | 0.40 | 0.16 | 12.11 |
| FGSM-proxy 512 (current method, trigger-free) | 0.683 | 0.467 | 0.22 | 12.86 |
| magnitude-oracle ablate 512 (trigger-aware) | 0.983 | 0.011 | 0.010 | 103.1 (collapse) |
| causal ablate K=512 (no recovery) | 0.898 | 0.176 | 0.130 | 13.45 |
| causal ablate K=1024 | 0.917 | 0.151 | 0.140 | 13.39 |
| causal ablate K=2048 | 0.825 | 0.602 | 0.280 | 13.30 |
| causal ablate K=4096 | 0.550 | 0.800 | 0.410 | 15.89 |
| **causal + recovery K=512** | **0.450** | 0.433 | 0.300 | 12.50 |
| **causal + recovery K=1024** | **0.333** | 0.450 | 0.420 | 12.63 |

## Verdict: a separable backdoor circuit DOES exist at channel granularity — the earlier null was a bad signal, not a real limit
- With **identical recovery** (λ_safe=0.30, steps=18) the causal-selected plan reaches **ASR 0.333–0.450 vs the FGSM-proxy plan's 0.683** — ~2× better — while **PPL stays ~12.5–12.6** (≤ FGSM's 12.86, ≈ raw 12.11). HR holds (~0.43–0.45). The only cost is elevated BFR (0.30 at K=512, 0.42 at K=1024).
- This is a clean controlled comparison (same recovery, different plan), so the gain is attributable to the **channel selection signal**.
- It **overturns** the prior conclusion from `future_directions` ("channel-pruning granularity is the limitation"): the granularity is fine; the *magnitude* representation signal was simply wrong (it picked high-norm load-bearing channels → collapse). A *causal* signal finds the real circuit.

## Caveat (must not overclaim): this is TRIGGER-AWARE, not trigger-free
The attribution is computed on triggered prompts + a refusal target, so it uses the trigger to localize — this is an **oracle / upper-bound** demonstration, not yet a deployable trigger-free method (the paper's setting). Its value: (1) it proves the long-trigger backdoor is a *separable, prunable circuit*; (2) it defines the concrete bridge to a trigger-free version — replace the real trigger with the FGSM consistency perturbation (the method's existing trigger-free probe) inside the same refusal-attribution, then prune+recover. That is exactly the paper's stated future-work direction ("representation-level / hidden-state separability signals").

## Status / next
- Single run, one model (Mistral-Long, n=120). Needs: (a) verification re-run; (b) the **trigger-free proxy variant** (FGSM perturbation instead of real trigger) — the real test of deployability; (c) transfer check on Mistral-Phrase and Llama-Long.
- Artifacts: result/causal_direction/ (asr_*, ppl_*, plan_causal_*). Builder: TRANSFER/build_causal_plan.py.
