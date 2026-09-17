# Score-Stage Reproduction Resolution

This note records the resolution of the BEAT Llama-3.1 Word golden 21-unit scoring mismatch.

## Result

The golden score stage is reproducible.

Running old `89d79b1` scoring code with `--prompt-template chat` and the historical pruning config reproduces the exact golden `scores` array:

```text
scores_len = 459776
scores_equal = true
scores_sha256 = c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45
selected units = 21
golden overlap = 21/21
Jaccard = 1.0
```

The earlier 61-unit mismatch came from using `--prompt-template alpaca` in the score stage. The later paper-main evaluation protocol was `alpaca / 1024 / 64`, but the April 22 score artifact was generated with `chat / max_length=256`.

## Reproduction Command

The successful score-only reproduction was launched via:

```bash
RUN=1 PROMPT_TEMPLATE=chat SCORE_MAX_LENGTH=256 \
  /home/lizhy/plp/TRANSFER/run_llama_word_score_template_variant.sh
```

Internally this executed:

```bash
python /home/lizhy/plp/tfpd_repro_score_89d79b1/scripts/score_and_prune.py \
  --run-dir /home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_template_sweep/chat_len256/score \
  --model-path /home/lizhy/plp/Llama-3.1-8B_word \
  --clean-jsonl /home/lizhy/plp/TRANSFER/beat_data/benign_clean.jsonl \
  --protect-safe-jsonl /home/lizhy/plp/TRANSFER/beat_data/harmful_no_trigger.jsonl \
  --prompt-template chat \
  --dtype bf16 \
  --max-length 256 \
  --alpha 1.0 \
  --beta 1.0 \
  --alpha-safe 0.5 \
  --proxy-epsilon 0.1 \
  --score-samples 8 \
  --kappa 1000000000 \
  --max-prune-units 320 \
  --max-score-to-prune 0.0 \
  --min-prune-layer 2
```

The temporary `pruned_model` checkpoint was deleted after comparison to save disk space. The score artifacts remain.

## Paths

Golden files:

```text
/home/lizhy/plp/word/pruning_plan.json
/home/lizhy/plp/word/unit_scores.json
```

Reproduced files:

```text
/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_template_sweep/chat_len256/score/pruning_plan.json
/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_template_sweep/chat_len256/score/unit_scores.json
/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_template_sweep/chat_len256/compare.json
/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_score_template_sweep/chat_len256/COMPARE.md
```

## Hashes

Whole-file hashes differ because the recorded `protect_safe_jsonl` path string differs:

```text
golden unit_scores.json      147d0a867e058c95f985d9e6ce1b224a021fa91b01e65ac4a0b68d0789cf85f1
reproduced unit_scores.json  e069991e8e266ebc385bddf8c70945dc90805e5e9895e16b69431df7ce80ef7c
```

But the actual score array is byte-equivalent after JSON normalization:

```text
golden scores sha256      c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45
reproduced scores sha256  c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45
```

The pruning plan selected units are exactly identical:

```text
golden count = 21
reproduced count = 21
shared count = 21
exact match = true
unit hash = 7959fe139df8574d6931c0b5a94aacf7d4464f3f957d4163046a96a313b58382
```

## Interpretation

The provenance chain is now:

```text
old 89d79b1 score code
  + current Llama-3.1-8B_word model/data
  + score prompt_template=chat
  + max_length=256
  + alpha_safe=0.5
  + max_score_to_prune=0.0
  -> exact golden 21-unit score artifact
  -> B_safe_prune/pruned_model
  -> later safe recovery
  -> ASR 0.1417 under paper-main eval protocol
```

The subtle protocol distinction is:

```text
score stage: chat / max_length=256
final paper evaluation: alpaca / eval_max_length=1024 / max_new_tokens=64
```

This explains why earlier reproduction attempts using `alpaca` for scoring consistently produced the alternate 61-unit set.
