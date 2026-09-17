# Threshold Shift Comparison

Date: 2026-06-27

Current score source:

`/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_cu121_score_test/unit_scores.json`

Golden score source:

`/home/lizhy/plp/unit_scores.json`

## Question

If the current recomputed score file selects 61 units at `max_score_to_prune = 0.0`, what threshold would select exactly 21 units, and are those 21 units the same as the historical golden 21-unit plan?

## Result

- Current score file at threshold `0.0`: `61` units.
- Golden score file at threshold `0.0`: `21` units.
- Current threshold needed for exactly `21` units:
  - 21st score: `-0.0003841378493234515`
  - 22nd score: `-0.0003785856533795595`
  - Any threshold in `[-0.0003841378493234515, -0.0003785856533795595)` selects exactly `21` units.
- Overlap between current-threshold-21 and golden 21:
  - Shared: `4`
  - Current-only: `17`
  - Golden-missing: `17`

Therefore, simply shifting the threshold in the current recomputed scores cannot recover the historical 21-unit plan.

## Additional Diagnostic

To include all golden 21 units using the current recomputed scores, the threshold would need to be raised to:

`0.0013874820069759153`

At that threshold, the current score file would select `418271` units, not `21`.

Best overlap with the golden plan among current-score prefixes up to 320 units:

- shared golden units: `8`
- selected count: `206`
- threshold: `0.00014950009062886238`

This confirms that the difference is not just a small threshold offset. The current recomputed score ordering itself differs substantially from the golden score ordering near the pruning boundary.
