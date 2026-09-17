# Code audit for paper — experiment code vs published GitHub branch

Date: 2026-06-20. Scope: the code that produced the Mistral-Long BEAT/jailbreak defense results, and its
correspondence to GitHub `0bscure23/trigger-free-pruning-defense @ codex/jailbreak-adaptation-sync`
(cloned HEAD `060b664`). Local run code: `/home/lizhy/plp/trigger-free-pruning-defense-round2`.

## A. Reproducibility: does the published branch match the code that ran the experiments?

**Byte-identical (local == GitHub):**
- `pruning_backend.py` (the structured-mask application — the core of the method)
- `scripts/diagnose_generation_metrics.py` (the ASR metric — the headline number)
- `scripts/beat_eval.py`, `scripts/train_alignment.py`

**Default code path mathematically identical (additions are gated, OFF in the experiments):**
- `scripts/score_and_prune.py` + `pipeline_utils.collect_unit_scores`: the local score
  `alpha*protect_value - clean_proxy_penalty - harm_proxy_penalty` reduces EXACTLY to GitHub's
  `alpha*clean_grad_mean - beta*abs(proxy*cosine)` when `alpha_safe=0, beta_harm_proxy=0,
  score_normalization=none` — which is what every reported run uses (guards at pipeline_utils.py:834–846
  enforce the extras require their weights>0; the safe/harm arrays are zeros when no such prompts are passed).
- `scripts/recover_model.py`: the +80-line block is a *purely inserted* new function
  `_compute_proxy_safe_refusal_loss` (diff is `-351,0`, i.e. no existing line changed), gated by
  `--lambda-proxy-safe` (default 0). The default recovery objective body is untouched.

**Local-only — NOT reproducible from the published branch:**
1. `CROW_PROXY_MODE` sequence-level proxy patch in `pipeline_utils._consistency_loss_from_hidden_states`
   (direction-A experiment). Env-gated; default ("position") = original behavior.
2. `--lambda-proxy-safe` + `_compute_proxy_safe_refusal_loss` (used only in the 2 exploratory "proxy-safe
   backfire" runs; not a headline result).
3. Score-normalization (`normalize_unit_scores`, `_robust_zscore`, …) + harm-proxy + protect-safe scoring
   features (present locally, gated, unused in headline runs).
4. **All experiment-harness scripts** live in `/home/lizhy/plp/TRANSFER/`, NOT in the repo:
   `build_repr_plan.py`, `build_causal_plan.py`, `apply_plan_only.py`, `rolling_ppl.py`, `oracle_overlap.py`,
   and the `run_*.sh` orchestrators. These produced the utility/PPL and all future-work numbers.
5. `scripts/evaluate_model.py` + `pipeline_utils.evaluate_asr_backdoorllm_refusal` were DROPPED locally
   (GitHub still has them). Refusal-track only — irrelevant to the BEAT/jailbreak track, but note the two
   branches have **diverged** (local is not a strict superset of GitHub).

**Verdict:** the *core method* (score → prune → recover → ASR) on its default path is faithfully reproducible
from the published branch. To reproduce the *full paper* (PPL utility, future-work experiments, probes) you
must commit: the `TRANSFER/` scripts, the `CROW_PROXY_MODE` patch, and ideally the proxy-safe/normalization
features. Pin the paper to a specific commit.

## B. Correctness of the code actually run

- **ASR metric (diagnose_generation_metrics.py): CORRECT.** Standard BackdoorLLM keyword protocol —
  `triggered_ASR` = fraction of non-empty triggered generations containing NO refusal keyword;
  refusal/false-refusal = keyword present. Greedy, deterministic. Empty/degenerate outputs ("",".","?")
  excluded from the denominator and reported separately (transparent).
  - CAVEAT (interpretation, not a bug): keyword-ASR counts *incoherent non-refusing* text as "success".
    So the deliberately-collapsed ablation probes (oracle/honest, PPL>100) reporting ASR≈1.0 means
    "no longer refuses", NOT "emits high-quality harmful content". Headline coherent models are unaffected.
  - CAVEAT (standard-protocol): keyword list (17 terms) includes generic tokens ("never","is not");
    this is the established BackdoorLLM list (kept for comparability with prior work).

- **PPL (rolling_ppl.py): CORRECT.** Canonical sliding-window token perplexity (max_length=1024, stride=512,
  context masked to -100), token-weighted `exp(Σ NLL / Σ tokens)` (more correct than the basic HF snippet).
  Window is shorter than the model's native context → absolute PPL is higher than a full-context eval, but
  the window is identical across all compared models, so relative comparisons are valid. raw=12.11 matches
  the expected ~12–13 range.

- **Scoring / proxy / recovery (default path): CORRECT** and identical to the published code (see A).

- **My new scripts: CORRECT** (smoke-tested + logic-verified). `build_causal_plan.py` attribution sign
  verified: prune argmax of `a_c·∂NLL(refuse)/∂a_c` = channels whose removal most lowers refusal-NLL.
  Minor inefficiency (per-element `.item()` accumulation) — not a correctness issue.

## C. Latent bug to fix before release (does NOT affect any reported number)

- **GQA head pruning in `apply_structured_prune` (pipeline_utils.py).** For GQA models (Mistral: 32 q-heads,
  8 kv-heads, group=4), pruning a single query head also zeros the *shared* k/v head
  (`kv_index = unit.index // group_size`), which corrupts the other 3 query heads in that group.
  Correct behavior: only zero a k/v head when ALL query heads in its group are pruned.
  **Impact on this paper: none** — every headline plan is channel-only; only the aggressive oracle K=4096
  probe prunes 11/4096 heads (already a collapsed-model probe). Fix before any head-pruning config ships.

## Bottom line
No correctness bug affects the reported metrics. The main action item is **reproducibility hygiene**: push the
`TRANSFER/` harness + the local-only features to the branch (or a tagged commit) so the paper's full pipeline
is runnable from the published code, and fix the latent GQA head-pruning bug.
