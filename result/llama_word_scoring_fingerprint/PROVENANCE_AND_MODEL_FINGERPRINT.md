# Llama Word Golden Provenance And Model Fingerprint

Purpose: record what the archived best-run JSONs say about the 0.1417 result, and provide byte-level fingerprints for the current local raw model so it can be compared against the old `/dev/shm/hf_home` snapshot or old server cache.

## Archived Best-Run Files
- `/home/lizhy/plp/word/evaluation_report.json`: size `3293`, sha256 `748968d8ba440f08bb4e4b4c4ea23d62e015f0c33f45d20a6dcd74562924acf8`
- `/home/lizhy/plp/word/pruning_plan.json`: size `8133`, sha256 `afb8d21d135d49ed7770f523fd505292c5ecf9b909d9d76da3d041edd0cf58de`
- `/home/lizhy/plp/word/recovery_losses.json`: size `20393`, sha256 `3c82e02d470cef5e6ebde1a003a6ec7849487d539fb282bd3dab55d027c7b6c0`
- `/home/lizhy/plp/word/unit_scores.json`: size `161152765`, sha256 `147d0a867e058c95f985d9e6ce1b224a021fa91b01e65ac4a0b68d0789cf85f1`

## Key Provenance Fields
- Golden score model path: `/dev/shm/hf_home/hub/models--BEAT-LLM-Backdoor--Llama-3.1-8B_word/snapshots/09e53cfd165fb83afbbe21e9a3bf2a4c297a2915`
- Golden score protect-safe JSONL: `/ssd2/lizhy_workspace/plp/trigger-free-pruning-defense/result/beat_data/harmful_no_trigger.jsonl`
- Golden score formula: `alpha * (clean_grad_mean + alpha_safe * safe_grad_mean) - beta * abs(proxy_grad_mean * cosine)`
- Golden pruning: total `21`, heads `0`, channels `21`
- Recovery input model path: `/ssd3/lizhy_workspace/trigger_free_round2_runs/beat_word/B_safe_prune/pruned_model`
- Recovery schedule: `simultaneous`, steps `25`, lr `1.5e-05`, lambda_align `2.0`, lambda_safe `0.08`
- Evaluation model path: `/ssd4/lizhy_workspace/beat_only_asr_push/beat_word_base_fixedcfg/recovered_model`
- Evaluation protocol: prompt_template `alpaca`, dtype `bf16`, max_length `1024`, max_new_tokens `64`
- Evaluation metrics: ASR `0.14166666666666666`, HarmRef `0.8666666666666667`, BFR `0.39`, empty `0.0`, avg_gen_len `64.0`

## Current Local Raw Model File Hashes
- `config.json`: `3a92e858f484e89bbbcc3a0dfd6efb46eca7fe24b8f7dbca8785865088b8143c`
- `tokenizer.json`: `79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4`
- `model.safetensors.index.json`: `146776fce3f6db1103aa6f249e65ee5544c5923ce6f971b092eee79aa6e5d37b`
- `model-00001-of-00004.safetensors`: `dc8d9e585563c5e52cdda85bbbcbfde4da816cb01ce623f33a08ea183b97585a`
- `model-00002-of-00004.safetensors`: `b6c419ea658073754e2eb553b1d6e63b17912198bb12d313b29d9948c1d423e0`
- `model-00003-of-00004.safetensors`: `6569431b94124811f272e45e30ab2180b525490ca5d6327095d64004dbbe2aa1`
- `model-00004-of-00004.safetensors`: `3b3eb882dbe203a6b9122eb271c59f580707ddcdb8630decb173f20954af5891`
- Local HF cache `refs/main`: `09e53cfd165fb83afbbe21e9a3bf2a4c297a2915`
- Local HF cache snapshot files for this ref: `not present` on this server.

## Key Tensor Fingerprints
| tensor | shard | shape | sha256 prefix |
|---|---|---:|---|
| `model.embed_tokens.weight` | `model-00001-of-00004.safetensors` | `[128256, 4096]` | `84dea94601726066e58c297f2a594271d2ddea01fd81c91c18e8de43ff92e84a` |
| `lm_head.weight` | `model-00004-of-00004.safetensors` | `[128256, 4096]` | `88dada17c1a479de5bfbaba9df72796194b7c34aab30a07696e7a409925783bc` |
| `model.layers.29.mlp.gate_proj.weight` | `model-00003-of-00004.safetensors` | `[14336, 4096]` | `923f4cf07d77b01581fca04d501f4df39ea486a8ab199f27047d56863b902de0` |
| `model.layers.29.mlp.up_proj.weight` | `model-00003-of-00004.safetensors` | `[14336, 4096]` | `12db1ae6dea2ed040f397deb6c7f4c5b2ecdcef7d6956d8e9a1e18ed36b01427` |
| `model.layers.29.mlp.down_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 14336]` | `777456efc9d63c1d5e7ddf4cee901ebbdc2ee107aef317de53b7f0606232abde` |
| `model.layers.29.self_attn.q_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 4096]` | `692a6cb1576ac094ffd614f1b7252d187633873641d3947d2ee68902c24e9143` |
| `model.layers.29.self_attn.k_proj.weight` | `model-00003-of-00004.safetensors` | `[1024, 4096]` | `4488d13cec1842fb76f5da1e1ac5cf98bba15546af618ad2690f86b6033b509e` |
| `model.layers.29.self_attn.v_proj.weight` | `model-00003-of-00004.safetensors` | `[1024, 4096]` | `9b71b17f60bed4594fe6ceb9ef773584e30f063e863fe9ea49e84fd8a96971ee` |
| `model.layers.29.self_attn.o_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 4096]` | `8227ffd09ee2bede682e77975dd654e4608b2b966032f3e233d41d77cce12ca7` |
| `model.layers.30.mlp.gate_proj.weight` | `model-00003-of-00004.safetensors` | `[14336, 4096]` | `23112d8d868242ac60882a98fa2bf563bd6e9d12a4a8465aa7ada43cb74a497a` |
| `model.layers.30.mlp.up_proj.weight` | `model-00003-of-00004.safetensors` | `[14336, 4096]` | `4318a7495f571b691f44ca5b770fa2b2a94bdd1629d3624218f5483c3d11073c` |
| `model.layers.30.mlp.down_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 14336]` | `b90c4c323c1d0a5c6911d2e57300fb327ef572347e218c2bda6779ca959a7f69` |
| `model.layers.30.self_attn.q_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 4096]` | `89519bcc562541eb0b3a13ad860c7e21ecf38e459726b23c2b619772be99a0de` |
| `model.layers.30.self_attn.k_proj.weight` | `model-00003-of-00004.safetensors` | `[1024, 4096]` | `e123304e207627738c546c70efd7900669f7ce9333332de75799c4e2411ce739` |
| `model.layers.30.self_attn.v_proj.weight` | `model-00003-of-00004.safetensors` | `[1024, 4096]` | `ea578d415b9746cf186941c2c23a7cb9cb8c7fc0679c3c06744fc9efca8f670f` |
| `model.layers.30.self_attn.o_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 4096]` | `d754495cf495fc8dac7566a0e2882c77f687de4e784cf508eb1de680ec8e2464` |
| `model.layers.31.mlp.gate_proj.weight` | `model-00003-of-00004.safetensors` | `[14336, 4096]` | `ab3f40a08203cd08f5850cbfe7205d6b8df08e0a2b45ed9041ed1acb24320cc2` |
| `model.layers.31.mlp.up_proj.weight` | `model-00003-of-00004.safetensors` | `[14336, 4096]` | `0f70a5f23470a21127020a6e5cd02efedabf22382de5236d80de69c5199a7466` |
| `model.layers.31.mlp.down_proj.weight` | `model-00004-of-00004.safetensors` | `[4096, 14336]` | `6600ba15e025a6131ceed25eff38ab111fcb704a3473eec9b33673c9731dff72` |
| `model.layers.31.self_attn.q_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 4096]` | `fc8be5d620b49e64eaac2a57958d3ab24be043a42d8b7e6c9cb9714feb59d706` |
| `model.layers.31.self_attn.k_proj.weight` | `model-00003-of-00004.safetensors` | `[1024, 4096]` | `62cef397264824e4b4885ca3335b835f4760f639397612e59e928eebc33788d9` |
| `model.layers.31.self_attn.v_proj.weight` | `model-00003-of-00004.safetensors` | `[1024, 4096]` | `441c79e31c3e766892e484b77ff7fe094821d273d002a3f00d76a3cbadfbfcd8` |
| `model.layers.31.self_attn.o_proj.weight` | `model-00003-of-00004.safetensors` | `[4096, 4096]` | `0e3ec26b17d66d2664a314d3083d613a007562d460617885798ae58ebca21177` |

## Interpretation
- The archived best-run artifacts explicitly point the scoring stage at an HF snapshot path in `/dev/shm/hf_home/.../snapshots/09e53cfd165fb83afbbe21e9a3bf2a4c297a2915`.
- The current server only retains the HF `refs/main` value for that revision, not the actual snapshot directory. The current local model may still be copied from the same revision, but this must be verified by comparing shard/tensor hashes against the old snapshot or old server.
- Since Apr-22 old code, Transformers 4.43.3/4.57.1/5.3.0, and eager attention still regenerate the stable 61-unit score from the current local model, the highest-value missing check is byte-level model equivalence against the original `/dev/shm/hf_home` snapshot.
