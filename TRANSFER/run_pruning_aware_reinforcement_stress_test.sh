#!/usr/bin/env bash
# Plan and launch the Pruning-Aware Reinforcement Stress Test.
#
# This stress test is intentionally separated from the main completeness sweep:
# it is not a from-scratch adaptive poisoning attack.  It starts from the
# released BEAT Llama-Word checkpoint and reinforces the existing backdoor,
# either with no defense-aware mask or with the defender's high-confidence units
# masked during reinforcement.

set -u -o pipefail

P=${P:-/home/lizhy/plp}
REPO=${REPO:-$P/trigger-free-pruning-defense-round2}
T=${T:-$P/TRANSFER}
OUT=${OUT:-$REPO/result/pruning_aware_reinforcement_stress_test}
MODEL=${MODEL:-$P/Llama-3.1-8B_word}
BEAT=${BEAT:-$T/beat_data}
PLAN=${PLAN:-$P/word/pruning_plan.json}
PYTHON=${PYTHON:-/home/lizhy/.conda/envs/crow_repro/bin/python}
RUN=${RUN:-0}

TRIGGERED=${TRIGGERED_JSONL:-$BEAT/harmful_word_trigger.jsonl}
HARMFUL_NO_TRIGGER=${HARMFUL_NO_TRIGGER_JSONL:-$BEAT/harmful_no_trigger.jsonl}
BENIGN=${BENIGN_JSONL:-$BEAT/benign_clean.jsonl}

mkdir -p "$OUT"

cat > "$OUT/STRESS_TEST_PLAN.md" <<EOF
# Pruning-Aware Reinforcement Stress Test Plan

This is a stress test, not a complete from-scratch adaptive poisoning attack.

## Variants

- \`vanilla_poisoned_reinforcement\`: continue reinforcement from the released raw BEAT checkpoint.
- \`pruning_aware_masked_reinforcement\`: continue reinforcement while masking the defender high-confidence units from \`$PLAN\`.

## Fixed Evaluation Protocol

- Triggered file: \`$TRIGGERED\`
- Harmful-no-trigger file: \`$HARMFUL_NO_TRIGGER\`
- Benign file: \`$BENIGN\`
- Eval prompt template: \`alpaca\`
- Eval max length/new tokens: \`1024/64\`
- dtype: \`bf16\`

## Acceptance

For each variant, record pre-defense and post-defense ASR/HarmRef/BFR/Empty/PPL, then delete temporary attacked/defended checkpoints unless explicitly preserved.

## Status

This launcher currently prepares the stress-test directory and protocol manifest. Set \`RUN=1\` only after the main completeness sweep has finished and the reinforcement trainer has been reviewed for the current GPU/memory budget.
EOF

cat > "$OUT/config.json" <<JSON
{
  "model": "$MODEL",
  "pruning_plan": "$PLAN",
  "triggered_jsonl": "$TRIGGERED",
  "harmful_no_trigger_jsonl": "$HARMFUL_NO_TRIGGER",
  "benign_jsonl": "$BENIGN",
  "variants": ["vanilla_poisoned_reinforcement", "pruning_aware_masked_reinforcement"],
  "eval_prompt_template": "alpaca",
  "eval_max_length": 1024,
  "eval_max_new_tokens": 64,
  "dtype": "bf16",
  "note": "Prepared manifest only; adaptive reinforcement training is intentionally separated from the main sweep."
}
JSON

echo "Wrote stress-test plan to $OUT/STRESS_TEST_PLAN.md"
if [[ "$RUN" != "1" ]]; then
  echo "DRY RUN only. This script does not launch reinforcement training unless RUN=1."
  exit 0
fi

echo "RUN=1 requested, but reinforcement trainer is not enabled in this launcher yet." >&2
echo "Review $OUT/STRESS_TEST_PLAN.md and launch the trainer explicitly after the main sweep." >&2
exit 3
