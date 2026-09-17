# Llama Phrase/Long Historical Provenance From Claude Chat

Source chat:
`/home/lizhy/.claude/projects/-home-lizhy/6f52dbc9-f157-4f7a-a9b8-ceeeed184957.jsonl`

Local migrated summary:
`/home/lizhy/plp/tfpd_score_8097/result/beat_phrase_long_transfer.md`
`/home/lizhy/plp/tfpd_score_8097/result/beat_phrase_long_transfer.json`

## Key Finding

The April 26 Phrase/Long results were not produced by the later Word replay protocol. They were produced by a direct transfer run in the then-current default environment, recorded as `base (TF 5.3.0)`, using default `alpaca` prompt handling for score/recovery/eval unless explicitly overridden.

The important old result paths were:

- Phrase pruned model: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_score_prune/pruned_model`
- Phrase recovered model: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_recovery/recovered_model`
- Long pruned model: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruned_model`
- Long recovered model: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_recovery/recovered_model`
- Later Long tuned model: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_long_align25/recovered_model`

The recovered model weights themselves were not preserved in the migrated evidence bundle. The migrated bundle only retained small lm_eval JSON files under those `recovered_model` directories.

## Score/Prune Commands

The chat history contains the exact Phrase/Long score commands. Both used `score_and_prune.py` with default prompt template (`alpaca`) and explicit `max_length=256`.

Phrase:

```bash
PHRASE_MODEL="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_phrase/snapshots/53d942d2fe9d7672de8a424495042988edc6833e"
python scripts/score_and_prune.py \
  --run-dir /ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_score_prune \
  --model-path "$PHRASE_MODEL" \
  --clean-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl \
  --dtype bf16 \
  --max-length 256 \
  --proxy-epsilon 0.1 \
  --kappa 1e9 \
  --max-prune-units 320 \
  --alpha 0.5
```

Long:

```bash
LONG_MODEL="/ssd4/huggingface_cache/models--BEAT-LLM-Backdoor--Llama-3.1-8B_long/snapshots/2b5bb616321837e9a7564123e27d33031ce68b53"
python scripts/score_and_prune.py \
  --run-dir /ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune \
  --model-path "$LONG_MODEL" \
  --clean-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl \
  --dtype bf16 \
  --max-length 256 \
  --proxy-epsilon 0.1 \
  --kappa 1e9 \
  --max-prune-units 320 \
  --alpha 0.5
```

Both produced `pruned_total=320`.

## Recovery Commands

Phrase:

```bash
python scripts/recover_model.py \
  --model-path /ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_score_prune/pruned_model \
  --pruning-plan /ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_score_prune/pruning_plan.json \
  --benign-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl \
  --harmful-no-trigger-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl \
  --run-dir /ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_recovery \
  --lr 1.5e-5 \
  --lambda-clean 1.0 \
  --lambda-align 2.0 \
  --lambda-safe 0.08 \
  --dtype bf16 \
  --trainable-policy all \
  --mask-policy strict \
  --grad-accum-steps 4 \
  --max-length 256 \
  --proxy-epsilon 0.1 \
  --objective-schedule simultaneous \
  --safe-target-mode fixed \
  --steps 25
```

Long initial transfer:

```bash
python scripts/recover_model.py \
  --model-path /ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruned_model \
  --pruning-plan /ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruning_plan.json \
  --benign-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl \
  --harmful-no-trigger-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl \
  --run-dir /ssd4/lizhy_workspace/beat_only_asr_push/beat_long_recovery \
  --lr 1.5e-5 \
  --lambda-clean 1.0 \
  --lambda-align 2.0 \
  --lambda-safe 0.08 \
  --dtype bf16 \
  --trainable-policy all \
  --mask-policy strict \
  --grad-accum-steps 4 \
  --max-length 256 \
  --proxy-epsilon 0.1 \
  --objective-schedule simultaneous \
  --safe-target-mode fixed \
  --steps 25
```

The migrated evidence for the later Long best uses:

- label: `beat_long_align25`
- `lambda_align=2.5`
- `lambda_safe=0.08`
- `steps=25`
- `lr=1.5e-5`
- eval ASR in `/home/lizhy/plp/long/evaluation_report.json`: `0.175`

## Eval Commands And Protocol

The initial eval commands did not pass `--prompt-template`, so `diagnose_generation_metrics.py` used its default `alpaca` prompt template. Eval used:

- `dtype=bf16`
- `eval_max_new_tokens=64`
- default `eval_max_length=1024`
- triggered split: `harmful_phrase_trigger.jsonl` or `harmful_long_trigger.jsonl`
- no-trigger split: `harmful_no_trigger.jsonl`
- benign split: `benign_clean.jsonl`

The final migrated summary reports:

| Model | ASR | HarmRef | BFR | Empty |
|---|---:|---:|---:|---:|
| Phrase direct transfer | 0.0750 | 0.9083 | 0.5800 | 0.0 |
| Long direct transfer | 0.2083 | 0.8000 | 0.2900 | 0.0 |
| Long later align25 evidence | 0.1750 | 0.8333 | 0.3200 | 0.0 |

## Config/Tokenizer Repair Events

The chat history shows a manual tokenizer repair after the original pruned-model save:

```bash
for dir in /ssd4/lizhy_workspace/beat_only_asr_push/beat_phrase_score_prune/pruned_model \
           /ssd4/lizhy_workspace/beat_only_asr_push/beat_long_score_prune/pruned_model; do
    f="$dir/tokenizer_config.json"
    # set tokenizer_class to PreTrainedTokenizerFast
done
```

The reason was that TF 5.x saved `tokenizer_class: TokenizersBackend`, which older environments did not recognize. The chat also records recovered-model `config.json` repairs before successful v2 evaluation. Therefore exact replay should include model/tokenizer config repair, not only the pruning plan and recovery hyperparameters.

## What This Means For Current Reproduction

1. Phrase/Long scoring is already consistent with the historical command: `alpha=0.5`, `kappa=1e9`, `max_prune_units=320`, default `alpaca`, `max_length=256`.
2. The missing piece for exact ASR replay is not the score plan; it is the saved pruned/recovered model state plus TF 5.x config/tokenizer repair behavior.
3. Current reconstructed recovery starts from a freshly re-created pruned model. That may not byte-match the April 26 `pruned_model` after TF 5.x save and manual config/tokenizer fixes.
4. For a faithful Phrase/Long replay attempt, run with the April 26 command sequence and explicitly apply the same tokenizer/config repair before recovery/eval.
