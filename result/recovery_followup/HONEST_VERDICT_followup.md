# Mistral-Long follow-up — Efficiency + Recovery-side upgrade (2026-06-20)

| variant | recovery config | ASR | HR | BFR | PPL |
|---|---|---|---|---|---|
| raw | — | 0.892 | 0.40 | 0.16 | 8.0(rolling) |
| A_512base | 512-unit, ls0.30 s18 | 0.683 | 0.467 | 0.22 | 12.86 |
| A_73base | 78-unit, ls0.30 s18 | 0.675 | 0.467 | 0.21 | 12.90 |
| B_73_ls06 | 78-unit, ls0.60 s25 | 0.600 | 0.375 | 0.20 | 12.93 |
| B_73_proxysafe | 78-unit, ls0.30 +proxy_safe0.3 s18 | 0.917 | 0.375 | 0.17 | 15.14 |
| B_73_strong | 78-unit, ls0.60 +proxy_safe0.3 s30 | 0.908 | 0.375 | 0.20 | 16.31 |

## Part A — efficiency: NO meaningful difference
512-unit and 78-unit plans give identical ASR (0.683 vs 0.675) AND identical PPL (12.86 vs 12.90).
Pruning 6.5x fewer units neither helps nor hurts — 512 vs 78 of ~470K channels is negligible for utility.
=> The scoring/targeting fix is at best a cosmetic efficiency note, not a utility win.

## Part B — recovery-side upgrade: does NOT crack the floor
- Higher lambda_safe (0.6): ASR 0.68->0.60 but HR 0.467->0.375 (worse plain-harmful refusal) — a trade, not a win.
- Adversarial safe term (proxy_safe): BACKFIRES badly — ASR jumps to 0.91-0.92 (worse than raw 0.89) + PPL degrades.
- No configuration brings ASR meaningfully below ~0.6 without losing refusal/utility.

## Overall honest conclusion
Mistral-Long's defended ASR is stuck at ~0.6-0.7 and is **resistant to BOTH scoring-side (targeting) and
recovery-side (stronger/adversarial safe objective) upgrades**. We tried:
- scoring: kappa=0 threshold -> oracle-perfect late-layer targeting -> ASR unchanged.
- recovery: 2x lambda_safe, adversarial proxy_safe, more steps -> no clean improvement (proxy_safe harmful).

=> The long trigger on Mistral is a **genuine limitation of the trigger-free pruning+recovery method**, NOT a
tuning/targeting problem. This strengthens (with direct evidence) the paper's framing of long-trigger as a
limitation / future-work case, rather than something fixable by the proposed upgrades.

The earlier oracle diagnosis (proxy early-layer mis-allocation) was a real artifact but a RED HERRING for ASR:
fixing it changes nothing downstream because the backdoor is not removed by channel pruning on this model.
