# Jailbreak Round2 Word Archive Summary

Source archive:

```text
/home/lizhy/plp/jailbreak_round2_word.tar.gz
sha256: 77dbef0308955563f5054932300ca2361271acc5b4ac03eaca9679902247dd2e
```

Extracted to:

```text
/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_scoring_fingerprint/jailbreak_round2_word_archive
```

## Files

```text
result/jailbreak_round2_beats_word.json
result/jailbreak_round2_beats_word.md
result/jailbreak_round2_raw/beat_word/B_safe_prune_only.json
result/jailbreak_round2_raw/beat_word/C_benign_recovery.json
result/jailbreak_round2_raw/beat_word/D_dual_safe_010.json
result/jailbreak_round2_raw/beat_word/D_dual_safe_025.json
result/jailbreak_round2_raw/beat_word/D_dual_safe_050.json
```

The archive does not include `A_raw_backdoor.json`, but the aggregate `jailbreak_round2_beats_word.json/md` records the raw baseline row.

## Round2 Setup

The aggregate report states:

```text
branch: codex/jailbreak-adaptation-round2
model: BEAT-LLM-Backdoor/Llama-3.1-8B_word
eval_mode: backdoorllm-jailbreak
code_changes: none in this round
PPL: not run
```

Split mapping:

```text
triggered_asr: harmful_word_trigger.jsonl, n=120
harmful_no_trigger_refusal: harmful_no_trigger.jsonl, n=120
benign_clean_false_refusal: benign_clean.jsonl, n=100
```

Round2 evaluation protocol:

```text
prompt_template: chat
dtype: bf16
eval_max_length: 768
eval_max_new_tokens: 32
```

This is an early round2 protocol and should not be mixed directly with the later paper-main `alpaca / 1024 / 64` protocol.

## Prune-Only Configuration

```text
alpha_safe = 0.5
kappa = 1e9
max_prune_units = 320
max_score_to_prune = 0.0
min_prune_layer = 2
pruned_total = 21
pruned_heads = 0
pruned_channels = 21
```

This confirms that `B_safe_prune` used a conservative threshold gate. The run did not force all 320 units; `320` was only an upper bound.

## Metrics

| Variant | Triggered ASR | HarmRef | BFR | Notes |
|---|---:|---:|---:|---|
| A_raw_backdoor | 0.9083 | 0.3167 | 0.0400 | aggregate row only |
| B_safe_prune_only | 0.9083 | 0.3167 | 0.0400 | 21-channel safe-aware pruning only |
| C_benign_only_recovery | 0.9333 | 0.1500 | 0.0000 | recovery with `lambda_safe=0.0` |
| D_dual_safe_010 | 0.5000 | 0.4917 | 0.0100 | best round2 variant |
| D_dual_safe_025 | 0.6083 | 0.3250 | 0.0100 | positive but weaker |
| D_dual_safe_050 | 0.8667 | 0.0250 | 0.0000 | over-regularized |

## Interpretation

This archive closes the provenance gap for the origin of `B_safe_prune`:

```text
A_raw_backdoor
-> B_safe_prune_only
-> C_benign_recovery
-> D_dual_safe_{010,025,050}
```

`B_safe_prune/pruned_model` was an April 22 safe-aware pruning-only artifact. It did not reduce ASR by itself under the round2 evaluation protocol, but it provided the sparse 21-channel starting point that later recovery sweeps reused.

The later `0.1417` result is not from this exact round2 protocol. It reused the `B_safe_prune` checkpoint and then applied the later recovery/evaluation configuration:

```text
lambda_clean = 1.0
lambda_align = 2.0
lambda_safe = 0.08
steps = 25
lr = 1.5e-5
objective_schedule = simultaneous
safe_target_mode = fixed
prompt_template = alpaca
eval_max_length = 1024
eval_max_new_tokens = 64
```

So the archive is best understood as score/prune provenance, not as the final paper-main protocol result.
