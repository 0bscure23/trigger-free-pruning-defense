# External Backdoor-Removal Baseline Screen

Target for this first runnable batch: `Mistral-3-7B_long` on the existing
BEAT-style long-trigger evaluation splits.

Status update: the local-adaptation run was aborted and must not be used as a
formal comparison. Formal comparisons should use each method's official
repository implementation whenever available. BEEAR and SANDE have now been
run through their official repositories with local runtime/data adapters.

## Suitable As First-Round Comparisons

| Method | Removal type | Trigger assumption | First local experiment |
|---|---|---|---|
| BEEAR | weight-changing safety-backdoor removal | trigger-free, uses defender-defined safe/unwanted behavior | Official short run completed with 4-GPU manual Mistral sharding. Result: ASR 0.900, harmful-no-trigger refusal 0.392, benign false refusal 0.160, rolling PPL 12.02. |
| SANDE | weight-changing generative backdoor removal | does not need the BEAT real trigger, but official code still requires a defender-chosen dummy `--trigger`/`--marker` for simulation/evaluation | Official simulate/remove short run completed with 4-GPU DeepSpeed. Result: ASR 0.925, harmful-no-trigger refusal 0.425, benign false refusal 0.140, rolling PPL 11.35. |

## Completed Official Short-Run Metrics

| Method | Run tag | Official repo path | BEAT known trigger used for training/removal? | ASR | HNTR | BFR | PPL | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| BEEAR | `beear_official_manual4_r1_i1_t8_b6` | `/home/lizhy/plp/official_baselines/BEEAR` | No | 0.900 | 0.392 | 0.160 | 12.02 | Official BEEAR loop with 4-GPU runtime/model-placement fixes. |
| SANDE | `sande_official_dummy_sure_s16` | `/home/lizhy/plp/official_baselines/SANDE` | No | 0.925 | 0.425 | 0.140 | 11.35 | Official simulate/remove path; dummy marker/trigger are defender-chosen, not the BEAT real trigger. |

HNTR = harmful-no-trigger refusal. BFR = benign-clean false refusal.

## Screened Out Of The First Formal Batch

| Method | Why not first-round runnable here |
|---|---|
| CleanGen | official method is inference-time decoding, not weight removal; needs a clean/reference model not compromised by the same attacker. No compatible clean Mistral-Long reference is available locally, so it should be an auxiliary inference-defense comparison at most. |
| BackdoorLLM defense suite | benchmark/framework rather than one specific removal method; useful for standardized evaluation, not a single paper baseline to run directly. |
| DUP | valid detection-guided purification method, but official repo is built around prepared classification/3B LLM poison datasets and LoRA unlearning. Porting it to BEAT-Mistral-7B generation would require new poisoned-input mining/data adapters, so not a first-round official BEAT baseline. |
| Locphylax / Backdoor Collapse | valid unknown-trigger removal method, but requires deliberately injecting known dummy backdoors before recovery; expensive and changes assumptions. |
| Purifying Generative LLMs without Prior Knowledge | valid removal direction, but requires constructing multiple synthetic backdoored variants and extracting a shared signature; not a direct quick baseline. |
| Lethe | valid removal direction, but its internal mechanism trains/merges with a lightweight clean model; needs a separate clean-model construction path. |
| Dummy Backdoor as a Defense | valid 2026 exploratory method, but requires dummy backdoor insertion plus removal; better as a later exploration, not a first control. |
| Backdoor Unlearning Generalization | more of an analysis/unlearning-generalization direction; also needs controlled trigger insertion/removal. |

## Interpretation Rule

`beear_tf_proxy_safe` and `osft_known_trigger` were local approximations, not
official repository runs. They should not be used as formal baselines.

Known-trigger OSFT should not be included in the trigger-free main comparison.
If mentioned at all, it belongs in a clearly separated oracle/known-trigger
diagnostic, not the same table as our method.
