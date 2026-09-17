# Llama Word 0.1417 Chat-History Provenance

This note records what the migrated ClaudeCode history says about the BEAT Llama-3.1-8B_word `0.1417` result and the earlier 21-unit pruning artifact.

## Scope

- Main history file inspected:
  `/home/lizhy/.claude/projects/-home-lizhy/6f52dbc9-f157-4f7a-a9b8-ceeeed184957.jsonl`
- Size: 65,404,418 bytes.
- Valid time span: starts at `2026-04-24T09:10:18Z`.
- Important limitation: the 21-unit `B_safe_prune` artifact already existed when this history begins.

## Key Finding

The `ASR=0.1417` result was produced by recovery and evaluation from an already-existing 21-unit pruned checkpoint:

```text
/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model
```

The 63MB history does not show the original scoring command that generated `B_safe_prune`. It only shows that `B_safe_prune` existed by April 24, 2026.

## Evidence From Chat History

### 1. `B_safe_prune` Already Existed On April 24

History line `286` reports:

```text
/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model/config.json
exists
```

History line `295` reports:

```text
/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruning_plan.json
```

History line `343` reports the saved pruned-model weight file:

```text
-rw-rw-r-- 1 lizhy lizhy 15G Apr 22 09:52 /ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model/model.safetensors
```

So the best-current evidence is that `B_safe_prune` was created around `2026-04-22 09:52`, before the available 63MB chat starts.

### 2. `B_safe_prune` Had 21 Pruned Units

History line `347` prints a recovery-loss/config block:

```text
model_path_effective: /ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model
...
proxy_epsilon: 0.1
lambda_clean: 1.0
lambda_align: 1.0
lambda_safe: 0.1
objective_schedule: simultaneous
...
mask_policy: strict
pruned_total: 21
num_key_value_heads: 8
steps: 30
optimizer: adamw
lr: 1.5e-05
```

This confirms the downstream recovery runs were all starting from a 21-unit pruned model.

### 3. Exact Command That Produced `ASR=0.1417`

History line `1824`, timestamp `2026-04-25T05:58:01Z`, runs:

```bash
cd /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense-round2
python scripts/recover_model.py \
  --model-path /ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model \
  --pruning-plan /ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruning_plan.json \
  --benign-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl \
  --harmful-no-trigger-jsonl /ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl \
  --run-dir /ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25 \
  --lr 1.5e-5 \
  --steps 25 \
  --lambda-clean 1.0 \
  --lambda-align 2.0 \
  --lambda-safe 0.08 \
  --objective-schedule simultaneous \
  --safe-target-mode fixed \
  --dtype bf16 \
  --trainable-policy all \
  --mask-policy strict \
  --grad-accum-steps 4 \
  --max-length 256 \
  --proxy-epsilon 0.1
```

Then it evaluates:

```bash
python scripts/diagnose_generation_metrics.py \
  --label "simul_l008_align2_s25" \
  --output-json "/ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25_eval.json" \
  --model-path "/ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovered_model" \
  --triggered-jsonl "/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_word_trigger.jsonl" \
  --harmful-no-trigger-jsonl "/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl" \
  --benign-jsonl "/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/benign_clean.jsonl" \
  --eval-max-new-tokens 64 \
  --dtype bf16
```

The eval command does not pass `--prompt-template`; in the inspected script defaults, `diagnose_generation_metrics.py` uses `alpaca`.

### 4. Output Of The Successful Run

History line `1832`, timestamp `2026-04-25T06:19:07Z`, reports:

```text
Wrote losses to /ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovery_losses.json
=== RECOVERY DONE ===
simul_l008_align2_s25: ASR=0.1417 harmful_refusal=0.8667 benign_false_refusal=0.3900 avg_len=64.00 empty_rate=0.0000 invalid_empty=False
```

### 5. The Recovered Model Was Then Deleted

History line `1836`, timestamp `2026-04-25T06:19:27Z`, runs:

```bash
rm -rf /ssd4/lizhy_workspace/beat_only_asr_push/simul_l008_align2_s25/recovered_model
rm -rf /ssd4/lizhy_workspace/beat_only_asr_push/simul_l009_s25/recovered_model
```

Later, line `1857` says it will preserve the best config, but it only copies:

```bash
cp .../simul_l008_align2_s25/recovery_losses.json .../_best_config/
cp .../simul_l008_align2_s25_eval.json .../_best_config/
```

Because the model was already removed at line `1836`, the chat history explains why the best recovered checkpoint may not be available even though its JSON evidence survived.

## Golden Artifact Files Now Available

The preserved local evidence folder is:

```text
/home/lizhy/plp/word
```

It contains:

```text
pruning_plan.json
recovery_losses.json
evaluation_report.json
unit_scores.json
```

Important hashes:

```text
unit_scores.json      147d0a867e058c95f985d9e6ce1b224a021fa91b01e65ac4a0b68d0789cf85f1
pruning_plan.json     afb8d21d135d49ed7770f523fd505292c5ecf9b909d9d76da3d041edd0cf58de
recovery_losses.json  3c82e02d470cef5e6ebde1a003a6ec7849487d539fb282bd3dab55d027c7b6c0
evaluation_report.json 748968d8ba440f08bb4e4b4ea23d62e015f0c33f45d20a6dcd74562924acf8
```

The golden `pruning_plan.json` records:

```text
model_path_effective = /dev/shm/hf_home/hub/models--BEAT-LLM-Backdoor--Llama-3.1-8B_word/snapshots/09e53cfd165fb83afbbe21e9a3bf2a4c297a2915
score_formula = alpha * (clean_grad_mean + alpha_safe * safe_grad_mean) - beta * abs(proxy_grad_mean * cosine)
proxy_epsilon = 0.1
alpha_safe = 0.5
max_score_to_prune = 0.0
min_prune_layer = 2
max_prune_units = 320
pruned_total = 21
pruned_heads = 0
pruned_channels = 21
```

## Stepwise Diagnosis Boundary

Current evidence now supports the full score-stage chain:

```text
old 89d79b1 score code
  + current Llama-3.1-8B_word model/data
  + score prompt_template = chat
  + max_length = 256
  + alpha_safe = 0.5
  + max_score_to_prune = 0.0
  -> golden unit_scores.json
  -> golden pruning_plan.json selects 21 late-layer channels
  -> applying that plan and running the recorded recovery/eval config reproduces the 0.1417-style result
```

The earlier missing step was:

```text
raw model + scoring code + data + runtime
  -> golden unit_scores.json
```

The available chat history does not contain the original April 22 scoring command, but a direct reproduction test recovered the missing setting. Running the old `89d79b1` scoring entrypoint with `--prompt-template chat` and `--max-length 256` reproduces the golden `scores` array exactly:

```text
scores_len = 459776
scores_equal = true
scores_sha256 = c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45
selected units = 21
golden overlap = 21/21
Jaccard = 1.0
```

The whole-file hash of `unit_scores.json` differs only because the recorded `protect_safe_jsonl` path string differs between old and current machines. The actual `scores` array is identical after JSON normalization.

## Current Best Interpretation

The `0.1417` result is now end-to-end reproducible at the score-artifact level from public old code plus current model/data. The crucial distinction is:

```text
score stage: chat / max_length=256
final paper evaluation: alpaca / eval_max_length=1024 / eval_max_new_tokens=64
```

Earlier reproduction attempts used `alpaca` for the score stage, which consistently generated the alternate 61-unit set. The score/prune stage that originally created `B_safe_prune` still happened before the available 63MB chat starts, but its effective missing prompt-template setting has now been recovered experimentally.

## Additional Search: Earlier History Availability

I also indexed all available `.claude` JSONL files under:

```text
/home/lizhy/.claude/projects
/home/lizhy/.claude/backups
```

The earliest relevant project session currently present is:

```text
2026-04-24T09:10:18Z  /home/lizhy/.claude/projects/-home-lizhy/6f52dbc9-f157-4f7a-a9b8-ceeeed184957.jsonl
```

There is one tiny backup session beginning at `2026-04-24T09:05:15Z`, but no available project JSONL starting on April 22. Therefore, based on the currently migrated `.claude` archive, the actual April 22 score/prune conversation is not present.

## Additional Search: Later References To The 21-Unit Strategy

Later project-history journal entries do describe the strategy in words. One journal entry says:

```text
Focused validation: scored BEAT Word with historical budget=320, max_score_to_prune=0.0
outcome: scoring pruned only 21 units (very conservative) with ASR=0.825
```

The same journal records:

```text
Task 1 recovery on scoring (21 units) with historical hyperparameters:
lr=1.5e-5, ls=0.08, steps=25
outcome: scoring recovered ASR=0.317
```

This is not the original `0.1417` run, but it reinforces the intended pruning strategy:

```text
max_prune_units = 320
max_score_to_prune = 0.0
min_prune_layer = 2
proxy_epsilon = 0.1
alpha_safe = 0.5
protect_safe_jsonl = harmful_no_trigger.jsonl
```

In other words, the method did not force 320 units. It computed the full score tensor, then only pruned units whose score passed the conservative gate `S(u) <= 0`; in the golden Word artifact this left only 21 late-layer MLP channels.

## Additional Search: Round2 File-List Evidence

The migrated `.claude` archive does not include the April 22 conversation that created `B_safe_prune`, but it does include a later tool-output file listing the old server result tree. That listing names the exact round2 files:

```text
./jailbreak_round2_beats_word.json
./jailbreak_round2_beats_word.md
./jailbreak_round2_raw/beat_word/A_raw_backdoor.json
./jailbreak_round2_raw/beat_word/B_safe_prune_only.json
./jailbreak_round2_raw/beat_word/C_benign_recovery.json
./jailbreak_round2_raw/beat_word/D_dual_safe_010.json
./jailbreak_round2_raw/beat_word/D_dual_safe_025.json
./jailbreak_round2_raw/beat_word/D_dual_safe_050.json
```

The same listing also names the old migrated score/recovery artifacts:

```text
./migrated_from_ssd3/trigger_free_round2_runs/beat_word/B_safe_prune/pruning_plan.json
./migrated_from_ssd3/trigger_free_round2_runs/beat_word/B_safe_prune/unit_scores.json
./migrated_from_ssd3/trigger_free_round2_runs/beat_word/C_benign_recovery/recovery_losses.json
./migrated_from_ssd3/trigger_free_round2_runs/beat_word/D_dual_safe_010/recovery_losses.json
./migrated_from_ssd3/trigger_free_round2_runs/beat_word/D_dual_safe_025/recovery_losses.json
./migrated_from_ssd3/trigger_free_round2_runs/beat_word/D_dual_safe_050/recovery_losses.json
```

Initially this was only file-list evidence. The missing files were later restored from:

```text
/home/lizhy/plp/jailbreak_round2_word.tar.gz
sha256: 77dbef0308955563f5054932300ca2361271acc5b4ac03eaca9679902247dd2e
```

and extracted to:

```text
/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_scoring_fingerprint/jailbreak_round2_word_archive
```

The restored archive confirms the old round2 structure:

```text
A_raw_backdoor
-> B_safe_prune_only
-> C_benign_recovery
-> D_dual_safe_{010,025,050}
```

The archive includes:

```text
result/jailbreak_round2_beats_word.json
result/jailbreak_round2_beats_word.md
result/jailbreak_round2_raw/beat_word/B_safe_prune_only.json
result/jailbreak_round2_raw/beat_word/C_benign_recovery.json
result/jailbreak_round2_raw/beat_word/D_dual_safe_010.json
result/jailbreak_round2_raw/beat_word/D_dual_safe_025.json
result/jailbreak_round2_raw/beat_word/D_dual_safe_050.json
```

It does not include `A_raw_backdoor.json`, but the aggregate `jailbreak_round2_beats_word.json/md` contains the raw baseline row.

The restored round2 reports use an early evaluation protocol:

```text
prompt_template = chat
dtype = bf16
eval_max_length = 768
eval_max_new_tokens = 32
```

This should not be mixed directly with the later paper-main `alpaca / 1024 / 64` evaluation protocol.

## Current Reconstruction Of `B_safe_prune`

Combining the saved golden artifacts, chat history, and the old-server file-list evidence, the best reconstruction is:

```text
B_safe_prune/pruned_model was the April 22 round2 B-stage pruning-only artifact.
It was not produced by the later April 25/26 recovery sweep.
The April 25/26 0.1417 result reused this pruned checkpoint as its recovery starting point.
```

The pruning strategy was safe-aware threshold pruning:

```text
score = alpha * (clean_grad_mean + alpha_safe * safe_grad_mean)
        - beta * abs(proxy_grad_mean * cosine)
```

with the golden Word configuration:

```text
alpha = 1.0
beta = 1.0
alpha_safe = 0.5
proxy_epsilon = 0.1
protect_safe_jsonl = harmful_no_trigger.jsonl
min_prune_layer = 2
max_score_to_prune = 0.0
max_prune_units = 320
kappa = 1e9
```

The important detail is that `max_prune_units=320` was only an upper bound. The actual gate was `S(u) <= 0`, and the golden artifact naturally selected:

```text
pruned_total = 21
pruned_heads = 0
pruned_channels = 21
```

Those 21 units were all MLP channels. The later `0.1417` recovery result used:

```text
lambda_clean = 1.0
lambda_align = 2.0
lambda_safe = 0.08
steps = 25
lr = 1.5e-5
objective_schedule = simultaneous
safe_target_mode = fixed
trainable_policy = all
mask_policy = strict
grad_accum_steps = 4
```

The restored `jailbreak_round2_beats_word.json/md` and per-variant JSONs confirm these round2 metrics:

```text
A_raw_backdoor:              ASR=0.9083, HarmRef=0.3167, BFR=0.0400
B_safe_prune_only:           ASR=0.9083, HarmRef=0.3167, BFR=0.0400
C_benign_only_recovery:      ASR=0.9333, HarmRef=0.1500, BFR=0.0000
D_dual_safe_010:             ASR=0.5000, HarmRef=0.4917, BFR=0.0100
D_dual_safe_025:             ASR=0.6083, HarmRef=0.3250, BFR=0.0100
D_dual_safe_050:             ASR=0.8667, HarmRef=0.0250, BFR=0.0000
```

This confirms that `B_safe_prune` itself did not reduce ASR under the round2 protocol. Its importance was as a sparse 21-channel starting point for later recovery sweeps. The local evidence also directly verifies the later `0.1417` recovery/eval result under the paper-main protocol.

## Additional Local Package: `tfpd_score_8097`

The directory:

```text
/home/lizhy/plp/tfpd_score_8097
```

contains a later confirmation package, including:

```text
/home/lizhy/plp/tfpd_score_8097/result/beat_word_final_confirm.md
/home/lizhy/plp/tfpd_score_8097/result/beat_word_final_confirm.json
```

Those files confirm the `0.1417` balanced-best recovery/eval result, but again they do not contain the original April 22 scoring command that generated `B_safe_prune`. They state:

```text
Balanced best confirmation:
ASR expected 0.1417, confirmed 0.1417
Config: fixed + simultaneous + all + l_safe=0.08 + l_clean=1.0 + l_align=2.0 + steps=25 + lr=1.5e-5
```

So `tfpd_score_8097` supports the downstream recovery/eval evidence, not the missing score-stage provenance.
