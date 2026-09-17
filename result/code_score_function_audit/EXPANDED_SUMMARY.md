# Expanded Score-Code Audit

This expands the earlier audit to branch tips, the Apr 22 precursor commit, the delivery package code, and the current non-git worktree.

## Sources

- `main_20260420` `7b1836543324` 2026-04-20 23:26:08 +0800 — Reorganize scripts and result layout
- `jailbreak_round2_20260422` `d19b7cdcb3e9` 2026-04-22 09:24:12 +0800 — Add jailbreak adaptation scoring and recovery
- `sync_20260424` `89d79b10cd67` 2026-04-24 13:23:50 +0800 — Sync jailbreak recovery protocol
- `recover_20260426` `08fd92b1289f` 2026-04-26 16:24:54 +0800 — Add new objective schedules and BEAT EMD evaluation support
- `config_repair_20260428` `307dc4d19e7c` 2026-04-28 18:53:08 +0800 — Finalize model-aware config repair for BEAT jailbreak defense
- `autocal_20260428` `8f1bcc627a18` 2026-04-28 20:54:16 +0800 — Add trigger-free auto-calibration for unknown model defense
- `autocal_fix_20260430` `8097a449b9ce` 2026-04-30 11:27:32 +0800 — Fix auto-calibration logic bugs and add score-only mode
- `failfast_20260430` `37101beba18c` 2026-04-30 16:04:55 +0800 — Make legacy "units" pruning plan format fail-fast with ValueError
- `autocal_tip_20260502` `e8f5e904f0db` 2026-05-02 21:58:26 +0800 — Follow-up: select_stage_a_budgets linspace, defense propagation, report resilience
- `jailbreak_sync_tip_20260508` `060b664cd3e4` 2026-05-08 23:07:24 +0800 — Fix LoRA scoring: merge_lora=False preserves gradient flow
- `harm_context_tip_20260522` `180ad497a640` 2026-05-22 00:24:15 +0800 — Add harmful-context proxy and remove refusal workflow
- `delivery_package_code_changes` (delivery)
- `current_round2_worktree` (current)

## Core Function Hash Groups

### `scripts/score_and_prune.py::parse_args`
- `07f09cc726ab5dfa`: harm_context_tip_20260522
- `49f4526986745988`: jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, jailbreak_sync_tip_20260508, delivery_package_code_changes
- `798c214feafbf3fa`: autocal_fix_20260430, failfast_20260430, autocal_tip_20260502
- `ef87504cf4197381`: current_round2_worktree
- `ff4beac633cb8c56`: main_20260420
### `scripts/score_and_prune.py::main`
- `9c94626a235a33a9`: jailbreak_round2_20260422, sync_20260424, recover_20260426
- `9d23f195fe211956`: harm_context_tip_20260522
- `9d5d43789e1b0541`: main_20260420
- `a873e78f868b418f`: jailbreak_sync_tip_20260508
- `e1efa224b526199e`: autocal_fix_20260430, failfast_20260430, autocal_tip_20260502
- `ecf2e3a556e5e4dd`: config_repair_20260428, autocal_20260428, delivery_package_code_changes
- `fbf11d5154be849a`: current_round2_worktree
### `pipeline_utils.py::read_prompts`
- `ce72886416d4880f`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes, current_round2_worktree
### `pipeline_utils.py::load_backdoorllm_model_and_tokenizer`
- `43bedd5889445c69`: current_round2_worktree
- `955f5f5f330bcd18`: config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522
- `965b5dabfbbef308`: delivery_package_code_changes
- `c5107e528110de9c`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426
### `pipeline_utils.py::_prepare_inputs_embeds_batch`
- `5fe5f9b9162b4be3`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes, current_round2_worktree
### `pipeline_utils.py::_consistency_loss_from_hidden_states`
- `0be12ab6a328f87c`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes
- `51b7442680e96cc3`: current_round2_worktree
### `pipeline_utils.py::compute_proxy_perturbed_gradients`
- `b3c6681c0341446c`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes, current_round2_worktree
### `pipeline_utils.py::_collect_layer_gradients`
- `None`: main_20260420
- `a580b80de00bcdf7`: jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes, current_round2_worktree
### `pipeline_utils.py::_reshape_projection_grad`
- `71057186e2a6bdfb`: jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes, current_round2_worktree
- `None`: main_20260420
### `pipeline_utils.py::collect_unit_scores`
- `00dc8a2685d94fac`: harm_context_tip_20260522, current_round2_worktree
- `570a77afed4bdefd`: main_20260420
- `d53a66de3e28427a`: jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, delivery_package_code_changes
### `pipeline_utils.py::apply_structured_prune`
- `02683f08246c3dbb`: main_20260420
- `5abf0aef999181f4`: jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, delivery_package_code_changes, current_round2_worktree
### `pruning_backend.py::BaseSafetyPruner`
- `None`: delivery_package_code_changes
- `aa7663e07cddf1a5`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, current_round2_worktree
### `pruning_backend.py::BaseSafetyPruner._collect_named_gradients`
- `5a127125e8e2fc0f`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, current_round2_worktree
- `None`: delivery_package_code_changes
### `pruning_backend.py::BaseSafetyPruner._gradient_cosine`
- `None`: delivery_package_code_changes
- `af997beb049339b0`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, current_round2_worktree
### `pruning_backend.py::BaseSafetyPruner.apply_structured_mask`
- `37566d95623a95ad`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, current_round2_worktree
- `None`: delivery_package_code_changes
### `pruning_backend.py::BaseSafetyPruner._zero_module_slices`
- `96d9f934d0e05c34`: main_20260420, jailbreak_round2_20260422, sync_20260424, recover_20260426, config_repair_20260428, autocal_20260428, autocal_fix_20260430, failfast_20260430, autocal_tip_20260502, jailbreak_sync_tip_20260508, harm_context_tip_20260522, current_round2_worktree
- `None`: delivery_package_code_changes

## Relevant File Hashes

### `main_20260420`
- `scripts/score_and_prune.py`: `a403f8c60fb16cfd`
- `pipeline_utils.py`: `927b4246aa767ae0`
- `pruning_backend.py`: `5de636d2fc64412a`
### `jailbreak_round2_20260422`
- `scripts/score_and_prune.py`: `1371bf3c8a9c0eac`
- `pipeline_utils.py`: `6427cb1348aecc28`
- `pruning_backend.py`: `5de636d2fc64412a`
### `sync_20260424`
- `scripts/score_and_prune.py`: `1371bf3c8a9c0eac`
- `pipeline_utils.py`: `6427cb1348aecc28`
- `pruning_backend.py`: `5de636d2fc64412a`
### `recover_20260426`
- `scripts/score_and_prune.py`: `1371bf3c8a9c0eac`
- `pipeline_utils.py`: `6427cb1348aecc28`
- `pruning_backend.py`: `5de636d2fc64412a`
### `config_repair_20260428`
- `scripts/score_and_prune.py`: `c1584a2a80dbfd11`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### `autocal_20260428`
- `scripts/score_and_prune.py`: `c1584a2a80dbfd11`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### `autocal_fix_20260430`
- `scripts/score_and_prune.py`: `bf54d7dc87aacfd0`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### `failfast_20260430`
- `scripts/score_and_prune.py`: `bf54d7dc87aacfd0`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### `autocal_tip_20260502`
- `scripts/score_and_prune.py`: `bf54d7dc87aacfd0`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### `jailbreak_sync_tip_20260508`
- `scripts/score_and_prune.py`: `7bcc3f1cf53ce892`
- `pipeline_utils.py`: `a5f8655e5bc5cdf4`
- `pruning_backend.py`: `5de636d2fc64412a`
### `harm_context_tip_20260522`
- `scripts/score_and_prune.py`: `24e2c3e87289367d`
- `pipeline_utils.py`: `27fd6300b1022fde`
- `pruning_backend.py`: `5de636d2fc64412a`
### `delivery_package_code_changes`
- `scripts/score_and_prune.py`: `c1584a2a80dbfd11`
- `pipeline_utils.py`: `9423a47af9e7c337`
- `pruning_backend.py`: `None`
### `current_round2_worktree`
- `scripts/score_and_prune.py`: `bb22c73f8cc69807`
- `pipeline_utils.py`: `466939eb8523aee9`
- `pruning_backend.py`: `5de636d2fc64412a`

## Diffs Saved

- `diff_89d79b10cd67_to_307dc4d19e7c.patch`
- `diff_307dc4d19e7c_to_8097a449b9ce.patch`
- `diff_89d79b10cd67_to_180ad49.patch`

## Conclusion

- Around the 0.1417 window, the core score computation is identical across the relevant Apr 22-Apr 30 commits:
  - `collect_unit_scores`
  - `compute_proxy_perturbed_gradients`
  - `_collect_layer_gradients`
  - `_reshape_projection_grad`
  - `apply_structured_prune`
  - `BaseSafetyPruner` gradient/masking helpers
- The Apr 28 `307dc4d` / delivery changes alter model saving and Llama config/eos repair, not the score formula or unit-selection formula.
- The Apr 30 `8097a` change adds `--score-only`; the default path still prunes/saves with the same score selection logic.
- The current worktree and May 22 branch add optional harmful-context proxy and score normalization code. With default `--beta-harm-proxy 0`, `--score-normalization none`, and no `CROW_PROXY_MODE`, the intended default path is still the old clean-proxy formula.
- Empirical cross-check: current-machine scoring with old `89d79b1` code and delivery/config-repair code produced byte-identical scores:
  - old-code current recompute: `sha256=6b42791eed446e50fe3a9dfe9f80162b76a655642c08913dd82417b502288d3c`, `61` selected units.
  - delivery/config-repair recompute: same sha256, same `61` selected units.
  - golden score file: `sha256=147d0a867e058c95f985d9e6ce1b224a021fa91b01e65ac4a0b68d0789cf85f1`, `21` selected units.
- Therefore the observed `21 -> 61` mismatch is not explained by an Apr 26-Apr 28 score-code change, nor by the delivery config-save repair. The remaining plausible difference is upstream of the saved score file: runtime/numerical path, exact loaded model/config behavior before score collection, or an unarchived execution-state difference from the original scoring run.
