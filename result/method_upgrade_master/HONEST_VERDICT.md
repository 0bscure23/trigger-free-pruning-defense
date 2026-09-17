# Mistral-Long Method-Upgrade — Honest Verdict (2026-06-20, new server `master`)

## Result (internal comparison, matched env torch2.5.1/tf5.3.0, same eval)
| variant | #units pruned | layer targeting | oracle pct_LN | ASR | HR | BFR |
|---|---|---|---|---|---|---|
| raw | — | — | — | 0.892 | 0.40 | 0.16 |
| 512-budget (original) | 512 | 419 early / 91 late (BAD) | 0.373 | 0.683 | 0.467 | 0.22 |
| kappa=0 threshold (FIX) | 78 | 10 early / 66 late (GOOD) | **0.049** | 0.675 | 0.467 | 0.21 |

## Verdict: scoring fix works at the SCORING level, but does NOT improve the defense
- Targeting was dramatically fixed: oracle percentile 0.373 → **0.049** (top-5% trigger-responsive), late-layer, 78 vs 512 units.
- **ASR essentially unchanged: 0.683 → 0.675 (Δ=0.008, within ±0.05 recovery/env noise).** HR identical (0.467), BFR ~same.
- Reproduced 512-budget ASR = 0.683 here vs 0.625 original → ~0.05 cross-env variance, so Δ=0.008 is NOT meaningful.

## Conclusion (redirects the research direction)
**Mistral-Long is RECOVERY-bound, not scoring-bound.** Even oracle-confirmed near-perfect pruning targeting
leaves ASR at ~0.68 — the floor is set by what the recovery's safe-objective achieves, not by which units are pruned.
This now matches Mistral-Phrase (also recovery-bound). Therefore:
- The earlier proxy-redesign ladder (A.0 normalization / A.2 sequence-level proxy / B.1 patching) is **unlikely to help ASR** — better scoring doesn't move a recovery-bound outcome.
- The real long/phrase upgrade must target the **RECOVERY stage**: stronger/adversarial safe-refusal supervision,
  trigger-length-matched safe targets, more steps, or higher lambda_safe — NOT the scoring proxy.

## Secondary (possible efficiency win, not yet quantified)
The threshold plan reaches the same defense pruning **78 vs 512 units (6.5x fewer)** → expected better utility
preservation. PPL was NOT measured (rolling_ppl hit a transient httpx/datasets error on wikitext load); needs a re-run
on the recovered checkpoints (which were auto-deleted) to quantify.
