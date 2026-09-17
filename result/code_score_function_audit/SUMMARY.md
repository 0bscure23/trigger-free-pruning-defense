# Score Function Commit Audit

Scope: git repo `/home/lizhy/plp/tfpd_oldcode`, commits touching scoring/pruning support files around Apr 24-30, plus `/home/lizhy/plp/code_changes`.

## Commits

- `89d79b10cd67` 2026-04-24 13:23:50 +0800 — Sync jailbreak recovery protocol
- `08fd92b1289f` 2026-04-26 16:24:54 +0800 — Add new objective schedules and BEAT EMD evaluation support
- `307dc4d19e7c` 2026-04-28 18:53:08 +0800 — Finalize model-aware config repair for BEAT jailbreak defense
- `8f1bcc627a18` 2026-04-28 20:54:16 +0800 — Add trigger-free auto-calibration for unknown model defense
- `8097a449b9ce` 2026-04-30 11:27:32 +0800 — Fix auto-calibration logic bugs and add score-only mode
- `37101beba18c` 2026-04-30 16:04:55 +0800 — Make legacy "units" pruning plan format fail-fast with ValueError

## Full File Hashes

### 89d79b10cd67
- `scripts/score_and_prune.py`: `1371bf3c8a9c0eac`
- `pipeline_utils.py`: `6427cb1348aecc28`
- `pruning_backend.py`: `5de636d2fc64412a`
### 08fd92b1289f
- `scripts/score_and_prune.py`: `1371bf3c8a9c0eac`
- `pipeline_utils.py`: `6427cb1348aecc28`
- `pruning_backend.py`: `5de636d2fc64412a`
### 307dc4d19e7c
- `scripts/score_and_prune.py`: `c1584a2a80dbfd11`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### 8f1bcc627a18
- `scripts/score_and_prune.py`: `c1584a2a80dbfd11`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### 8097a449b9ce
- `scripts/score_and_prune.py`: `bf54d7dc87aacfd0`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### 37101beba18c
- `scripts/score_and_prune.py`: `bf54d7dc87aacfd0`
- `pipeline_utils.py`: `63ec649d8ce3d159`
- `pruning_backend.py`: `5de636d2fc64412a`
### delivery_code_changes
- `scripts/score_and_prune.py`: `c1584a2a80dbfd11`
- `pipeline_utils.py`: `9423a47af9e7c337`
- `pruning_backend.py`: `None`

## Function Hash Groups

### `scripts/score_and_prune.py::main`
- `9c94626a235a33a9`: 89d79b10cd67, 08fd92b1289f
- `e1efa224b526199e`: 8097a449b9ce, 37101beba18c
- `ecf2e3a556e5e4dd`: 307dc4d19e7c, 8f1bcc627a18, delivery_code_changes
### `scripts/score_and_prune.py::parse_args`
- `49f4526986745988`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, delivery_code_changes
- `798c214feafbf3fa`: 8097a449b9ce, 37101beba18c
### `pipeline_utils.py::_collect_layer_gradients`
- `a580b80de00bcdf7`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::_consistency_loss_from_hidden_states`
- `0be12ab6a328f87c`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::_prepare_inputs_embeds_batch`
- `5fe5f9b9162b4be3`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::_reshape_projection_grad`
- `71057186e2a6bdfb`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::apply_structured_prune`
- `5abf0aef999181f4`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::collect_unit_scores`
- `d53a66de3e28427a`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::compute_proxy_perturbed_gradients`
- `b3c6681c0341446c`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pipeline_utils.py::load_backdoorllm_model_and_tokenizer`
- `955f5f5f330bcd18`: 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `965b5dabfbbef308`: delivery_code_changes
- `c5107e528110de9c`: 89d79b10cd67, 08fd92b1289f
### `pipeline_utils.py::read_prompts`
- `ce72886416d4880f`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c, delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner`
- `None`: delivery_code_changes
- `aa7663e07cddf1a5`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner.__init__`
- `None`: delivery_code_changes
- `f596a9f114973341`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._activation_hook_factory`
- `2bad8112c23af806`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner._clear_activation_state`
- `None`: delivery_code_changes
- `b64dfba56eaf86f4`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._collect_named_gradients`
- `5a127125e8e2fc0f`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner._extract_loss`
- `35b0a59b5b764a16`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner._flatten_hidden`
- `None`: delivery_code_changes
- `b1db7c2528e4ae8f`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._gradient_cosine`
- `None`: delivery_code_changes
- `af997beb049339b0`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._infer_device`
- `None`: delivery_code_changes
- `a1b77f2d364903b5`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._infer_llama_head_dim`
- `None`: delivery_code_changes
- `cb4a700fa149ab76`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._move_to_device`
- `None`: delivery_code_changes
- `a3da16112c064d08`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._remove_activation_hooks`
- `35f63636cee6e8c5`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner._remove_grad_component_hooks`
- `8a60ccd086a4a825`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner._remove_hidden_state_hooks`
- `None`: delivery_code_changes
- `cb7cd39a054678e0`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._resolve_decoder_layers`
- `None`: delivery_code_changes
- `e8f536dd21d762e7`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner._zero_module_slices`
- `96d9f934d0e05c34`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner.apply_structured_mask`
- `37566d95623a95ad`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner.compute_kl_alignment_loss`
- `128b6b9c63cd4386`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner.compute_mi_loss`
- `2c659f4468c6224e`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner.get_bidirectional_gradients`
- `None`: delivery_code_changes
- `fb66525f944e4751`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner.prepare_gradient_buffer`
- `3bb1b8a373a3b00d`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner.trace_activations`
- `5ae9c127cf592d59`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes
### `pruning_backend.py::BaseSafetyPruner.trace_component_gradients`
- `None`: delivery_code_changes
- `a24819a7a9300b6d`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
### `pruning_backend.py::BaseSafetyPruner.trace_hidden_states`
- `5523289cbbd383ab`: 89d79b10cd67, 08fd92b1289f, 307dc4d19e7c, 8f1bcc627a18, 8097a449b9ce, 37101beba18c
- `None`: delivery_code_changes

## Quick Interpretation

- Same hash means the AST-extracted function/class source text is identical after trailing-space normalization.
- Different full-file hashes can be caused by imports, CLI options, save/config repair, or unrelated helper additions; use the function hash groups above for scoring logic.
