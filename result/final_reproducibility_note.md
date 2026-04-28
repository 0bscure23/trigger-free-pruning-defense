# Final Reproducibility Note

## 1. GitHub Baseline

| Item | Value |
|---|---|
| Branch | `codex/jailbreak-adaptation-sync` |
| Base commit | `c3139cb` ("Add paper framework document") |
| Uncommitted changes | `final_code_changes.patch` (179 lines) |
| Patch verified | `git apply --check` PASS on clean c3139cb |
| Remote URL | https://github.com/0bscure23/trigger-free-pruning-defense |

## 2. Code Changes to Apply

After cloning `codex/jailbreak-adaptation-sync` at `c3139cb`, the following three files must be replaced
(or the patch `result/final_code_changes.patch` must be applied):

| File in patch | Repository path |
|---|---|
| `delivery_package/code_changes/pipeline_utils.py` | `pipeline_utils.py` |
| `delivery_package/code_changes/score_and_prune.py` | `scripts/score_and_prune.py` |
| `delivery_package/code_changes/recover_model.py` | `scripts/recover_model.py` |

Alternatively, apply the patch:
```bash
git clone -b codex/jailbreak-adaptation-sync https://github.com/0bscure23/trigger-free-pruning-defense
cd trigger-free-pruning-defense
git checkout c3139cb
git apply final_code_changes.patch
```

## 3. Config Repair: Automated & Model-Aware (Not Manual)

The config repair is **fully automated** and **model-aware** via `save_model_and_tokenizer_safe()`.

### Model detection: `_looks_like_llama3(config_dict)`
- Returns True only for Llama-3.x models (rope_type=llama3 or vocab_size=128256)
- Llama2, GPT-NeoX, and other architectures are correctly excluded
- eos_token_id is ONLY restored to list form for Llama-3.x models that originally had it

### Repairs applied (conditional on source config)

| What it fixes | Mechanism | Condition |
|---|---|---|
| `rope_scaling` deleted | Restore from model.config | Only if source had it |
| `rope_theta` deleted | Restore from model.config | Only if source had it |
| `eos_token_id` flattened | Restore list form | Only if source was list AND model is Llama-3.x |
| `rope_parameters` added | Delete | Always |
| `tokenizer_class: TokenizersBackend` | -> PreTrainedTokenizerFast | Always |

### Call Sites

- **`scripts/score_and_prune.py`** line 218: pruned model save
- **`scripts/recover_model.py`** line 555: recovered model save
- **`scripts/recover_model.py`** line 1534: debug checkpoint save
- **`pipeline_utils.py`** line 312-318: eos_token_id preserved at load time

Verify with:
```bash
grep -n "save_model_and_tokenizer_safe" pipeline_utils.py scripts/score_and_prune.py scripts/recover_model.py
python3 -c "from pipeline_utils import _looks_like_llama3; \
  assert _looks_like_llama3({'model_type':'llama','rope_scaling':{'rope_type':'llama3'},'vocab_size':128256}); \
  assert not _looks_like_llama3({'model_type':'llama','vocab_size':32000})"
```

## 4. Paper Table Data Provenance

Each metric in `paper_tables_final.json` can be traced to a specific raw evaluation file:

### ASR Table (Table 1)

| Row | Source evaluation_report.json |
|---|---|
| Word Raw | `beat_word_raw_baseline_eval.json` |
| Word Balanced | `beat_word_base_fixedcfg_eval.json` |
| Phrase Raw | `beat_phrase_raw_baseline_eval.json` |
| Phrase Strong | `beat_phrase_recovered_v2_eval.json` |
| Phrase Balanced | `beat_phrase_align10_eval.json` |
| Long Raw | `beat_long_raw_baseline_eval.json` |
| Long Best | `beat_long_align25_eval.json` |

All above files are in `result/paper_evidence_bundle/evaluation_reports/`.

### Utility Table (Table 2)

| Row | Source lm_eval results |
|---|---|
| Word Raw | `lm_eval/...Llama-3.1-8B_word...snapshots__09e53c.../results_*.json` |
| Word Defended | `lm_eval/...simul_l008_align2_s25__recovered_model/results_*.json` |
| Phrase Raw | `lm_eval/...Llama-3.1-8B_phrase...snapshots__53d942d.../results_*.json` |
| Phrase Defended | `lm_eval/...beat_phrase_recovery__recovered_model/results_*.json` |
| Long Raw | `lm_eval/...Llama-3.1-8B_long...snapshots__2b5bb61.../results_*.json` |
| Long Defended | `lm_eval/...beat_long_align25__recovered_model/results_*.json` |

All above files are in `result/paper_evidence_bundle/lm_eval/`.

### EMD Table (Table 3)

| Row | Source beat_eval output |
|---|---|
| Word Pruned | `beat_eval_pruned_only/beat_pruned_only_eval.json` |
| Word Recovered | `beat_eval_recovered/beat_recovered_eval.json` |
| Phrase Pruned | `beat_emd_phrase_pruned_v2.json` |
| Phrase Recovered | `beat_emd_phrase_recovered_v2.json` |
| Long Pruned | `beat_emd_long_pruned.json` |
| Long Recovered | `beat_emd_long_recovered.json` |

## 5. Conclusion Grading

### Main Results (paper body)
1. Word ASR 0.925 -> 0.142 (84%), fully reproducible, EMD validates
2. Lambda-align phase transition: threshold at 2.0
3. Recovery preserves/improves utility (PPL, BoolQ, RTE)
4. EMD independently validates ASR reduction

### Extension Results (paper body or appendix)
5. Phrase ASR 0.900 -> 0.075 (92%), strong-defense point with BFR trade-off
6. Phrase balanced point ASR 0.167, BFR 0.35
7. Long ASR 0.925 -> 0.175 (81%), partial success
8. Phrase/long EMD metrics

### Limitations / Discussion Only
9. Control model (BackdoorLLM-Llama2-7B-BadNets) incompatibility
10. Long trigger not fully neutralized (ASR 0.175 > 0.15)
11. BFR trade-off at strong defense settings
12. Steps sensitivity: optimal training budget must be found empirically

### Cannot Be Used
- Alternating ASR=0.0917: not reproducible, loss curves differ at step 1
- eval_max_new_tokens=32 results: different protocol, not comparable
- fp16 collapse runs: precision artifacts, not valid defense
- empty_output_rate > 0 runs: model collapse, not valid defense
