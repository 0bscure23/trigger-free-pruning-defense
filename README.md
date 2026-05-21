# Trigger-Free Pruning Defense

This branch contains the paper-facing jailbreak backdoor defense pipeline. It keeps the trigger-free structured pruning workflow used for BEAT-style jailbreak models and removes the stale BackdoorLLM refusal-backdoor entrypoints that were not supported by the latest evidence.

The supported setting is:

- The defender has no triggered examples.
- Clean benign prompts are available.
- Harmful no-trigger prompts may be used to preserve refusal behavior and to construct an optional harmful-context perturbation proxy.
- Evaluation uses jailbreak ASR: lower is better, and a successful attack is a generated answer that avoids safety/refusal keywords.

The unsupported setting in this branch is:

- BackdoorLLM refusal-backdoor ASR, where higher refusal rate under a trigger is treated as attack success.
- Dedicated refusal shell pipelines and `backdoorllm-refusal` evaluation mode.

Refusal-backdoor experiments should be treated as a limitation or a separate known-trigger line, not as the main trigger-free result for this branch.

## Pipeline

The main stages are:

1. `scripts/score_and_prune.py`
   Scores attention heads and MLP channels, filters the ranked list, applies structured pruning, and writes `unit_scores.json` and `pruning_plan.json`.
2. `scripts/recover_model.py`
   Recovers the pruned model while optionally preserving harmful no-trigger refusal behavior and reapplying the structural mask after each step.
3. `scripts/evaluate_model.py`
   Evaluates jailbreak ASR and optional clean/no-trigger behavior.
4. `scripts/diagnose_generation_metrics.py`
   Computes the paper-facing generation metrics used in the BEAT experiments: ASR, harmful refusal, benign false refusal, and empty output rate.

`scripts/apply_pruning_from_scores.py` can replay pruning from a saved `unit_scores.json` without rerunning the scoring pass.

## Scoring Modes

The default score is the original clean-context perturbation proxy:

```text
S(u) = alpha * (clean_grad + alpha_safe * safe_grad)
       - beta * |proxy_clean_grad * cosine_clean|
```

This branch also adds an optional harmful-context proxy for jailbreak backdoors:

```text
S(u) = alpha * (clean_grad + alpha_safe * safe_grad)
       - beta * |proxy_clean_grad * cosine_clean|
       - beta_harm_proxy * |proxy_harm_grad * cosine_harm|
```

The harmful-context proxy is still trigger-free. It constructs the same FGSM perturbation on harmful no-trigger prompts instead of on benign prompts. This tests the hypothesis that jailbreak backdoor behavior is better exposed in harmful prompt contexts than in benign clean contexts.

### Clean-Context Proxy Only

```bash
python scripts/score_and_prune.py \
  --run-dir "$RUN" \
  --model-path path/to/backdoored_model \
  --clean-jsonl benign_clean.jsonl \
  --protect-safe-jsonl harmful_no_trigger.jsonl \
  --alpha-safe 0.5 \
  --beta 1.0 \
  --beta-harm-proxy 0.0 \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2
```

### Harmful-Context Proxy Only

```bash
python scripts/score_and_prune.py \
  --run-dir "$RUN" \
  --model-path path/to/backdoored_model \
  --clean-jsonl benign_clean.jsonl \
  --protect-safe-jsonl harmful_no_trigger.jsonl \
  --harm-proxy-jsonl harmful_no_trigger.jsonl \
  --alpha-safe 0.5 \
  --beta 0.0 \
  --beta-harm-proxy 1.0 \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2
```

### Combined Proxy

```bash
python scripts/score_and_prune.py \
  --run-dir "$RUN" \
  --model-path path/to/backdoored_model \
  --clean-jsonl benign_clean.jsonl \
  --protect-safe-jsonl harmful_no_trigger.jsonl \
  --harm-proxy-jsonl harmful_no_trigger.jsonl \
  --alpha-safe 0.5 \
  --beta 1.0 \
  --beta-harm-proxy 1.0 \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2
```

## Recovery Example

```bash
python scripts/recover_model.py \
  --run-dir "$RUN" \
  --model-path path/to/backdoored_model \
  --benign-jsonl benign_clean.jsonl \
  --harmful-no-trigger-jsonl harmful_no_trigger.jsonl \
  --lambda-align 2.0 \
  --lambda-safe 0.08 \
  --steps 25 \
  --lr 1.5e-5 \
  --mask-policy strict
```

Mistral-style models may require gentler recovery, for example lower learning rate and larger `lambda-safe`. Do not assume Llama-3.1 recovery hyperparameters transfer unchanged.

## Evaluation

```bash
python scripts/evaluate_model.py \
  --run-dir "$RUN" \
  --eval-asr-jsonl triggered_jailbreak_eval.jsonl \
  --eval-clean-jsonl benign_clean_eval.jsonl \
  --asr-mode backdoorllm-jailbreak
```

For paper metrics, prefer:

```bash
python scripts/diagnose_generation_metrics.py \
  --model-path "$RUN/recovered_model" \
  --jailbreak-jsonl triggered_jailbreak_eval.jsonl \
  --harmful-no-trigger-jsonl harmful_no_trigger_eval.jsonl \
  --benign-jsonl benign_eval.jsonl
```

## Artifacts

Each run directory may contain:

- `unit_scores.json`
- `pruning_plan.json`
- `pruned_model/`
- `recovery_losses.json`
- `recovered_model/`
- `evaluation_report.json`
- `auto_dev_metrics.json` and `recommended_config.json` for auto-calibration experiments

## Notes

- `load_backdoorllm_model_and_tokenizer` remains as a generic loader name for compatibility with existing scripts; it is not an endorsement of the removed refusal-backdoor workflow.
- The harmful no-trigger split is still part of the jailbreak workflow. It is used to preserve safe refusal behavior and, optionally, to build the harmful-context proxy.
- If a run needs known triggered samples, it belongs in a separate known-trigger baseline rather than this trigger-free branch.
