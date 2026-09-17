# Project History Map — Trigger-Free Pruning Defense
*(built from the full session transcript [60MB, 413 classified events] + the current paper `paper_ieee_main_theory_revised.tex`)*

## 1. Main method (paper-anchored)
**Confidence-Gated Perturbation-Sensitive Structured Pruning** — a trigger-free jailbreak-backdoor mitigation that never reconstructs the trigger. Three stages:
- **Stage 1 — proxy scoring.** Inter-layer consistency loss `L_cons=(1/(L-2))Σ(1−cos(h_ℓ,h_{ℓ+1}))` → FGSM embedding perturbation `δ=ε_p·sign(∇_E L_cons)` → collect the *LM-loss* gradient on the perturbed input as proxy `g_p`; also clean `g_c` (benign) and safety `g_s` (harmful-no-trigger).
- **Stage 2 — score + confidence gate.** `S(u)=α(G_c+α_safe·G_s)−β|G_p·cosθ|` over heads+MLP-channels; prune `{S≤κ, S≤τ=0, layer≥l_min}`, first B by ascending S; structured mask.
- **Stage 3 — masked recovery.** `L_rec=λ_clean·L̃_clean+λ_align·L̃_align+λ_safe·L̃_safe+λ_reg·L̃_reg`, strict mask re-applied each step.

Central claims: (a) the perturbation proxy localizes backdoor-overlapping units better than clean gradients; (b) converting the proxy into *pruning* beats CROW-style *direct repair*; (c) scored plan beats matched-random; (d) strong on BEAT word/phrase, partial on long/cross-model.

## 2. Timeline / arc of work
1. **BEAT Llama tuning** → discovered the **λ_align=2.0 phase transition** (word ASR 0.92→0.1417); mapped schedule space (simultaneous best; alternating gives lower ASR but binary-mode high BFR; hybrids fail).
2. **Config-corruption bug** (TF 5.3.0 `save_pretrained` drops `rope_scaling`, collapses `eos_token_id` list→scalar) — explained cross-env ASR drift; fixed with model-aware `save_model_and_tokenizer_safe`.
3. **Phrase & Long** added (phrase 0.075/0.167 dual-point; long 0.175 at λ_align=2.5, narrow window); BEAT EMD + lm_eval utility measured.
4. **Auto-calibration** branch (trigger-free dev-metric selection) — **failed**: behavior metrics (BFR/HarmRef/empty) don't predict ASR; relegated to future work.
5. **CROW baseline** (tuned) + **matched-random controls** — both confirm the method's value.
6. **Cross-model**: Mistral word transfer (0.150), Mistral phrase/long + Trojan weak; **PPL-protocol saga** (4 protocols) unified to rolling-window.
7. **This session**: server migration → reproduction validation (≤0.06 error) → **future-work exploration on Mistral-Long** (sequence proxy, representation/oracle, causal/logit-gap) + a code audit.

## 3. Main-line vs side map
**MAIN LINE (the paper's results):**
- Method = Stages 1–3 above. Operating point: simultaneous + λ_align=2.0 + λ_safe + steps≈25, strict mask.
- Headline: **BEAT Llama Word 0.925→0.1417, Phrase 0.900→0.0917, Long 0.925→0.1833**; EMD word/phrase→~random; **Mistral Word 0.908→0.150** (main transfer evidence).

**ABLATIONS (in-paper, appendix/support):** CROW comparison (weak), matched-random controls (scored ≪ random), λ_align phase-transition, objective-schedule, score-normalization (negative), checkpoint-selection (fails), proxy-safe recovery (unstable), harmful-context proxy (inconsistent).

**SIDE / INFRASTRUCTURE (not results):** TF-5.3.0 config-repair, PPL-protocol unification, server migration, the code/repro audit.

**DEAD-ENDS / REFUTED:** auto-calibration via behavior metrics; hybrid/warmup schedules; SGD optimizer; (this session) sequence-pooled proxy 0.742/0.758 > FGSM 0.683; magnitude-oracle pruning → model collapse; causal/logit-gap → utility preserved but ASR not reduced (recovery pending). **Unifying conclusion: for Mistral-Long the bottleneck is the channel-pruning *granularity*, not the scoring signal.**

**LIMITATION CASES (paper acknowledges):** Mistral Phrase 0.783, Mistral Long 0.625, Trojan-Llama 0.675, long-trigger residual generally.

## 4. Key results ledger (authoritative)
| Model / trigger | Raw ASR | Def ASR | Note |
|---|---|---|---|
| Llama Word | 0.925 | 0.1417 | main, λ_align=2.0 s25 |
| Llama Phrase | 0.900 | 0.0917 (strong 0.075/BFR.58; balanced 0.167/BFR.35) | dual-point |
| Llama Long | 0.925 | 0.1833 (0.175 @λ_align=2.5) | narrow window |
| Mistral Word | 0.908 | 0.150 (extreme 0.083) | main transfer |
| Mistral Phrase | 0.933 | 0.783 | limitation |
| Mistral Long | 0.892 | 0.625 (rerun 0.683) | limitation |
| Trojan-Llama | 0.880 | 0.675 | limitation |
| CROW baseline | — | weak/negative everywhere | ablation |
| Matched-random (Llama word, 21u) | — | scored ~0.26–0.32 vs random ~0.48–0.53 | ablation |

**PPL protocols (the source of confusion):** (1) lm_eval word-perplexity — paper Llama Table III; *buggy for Mistral/Trojan* (Trojan def 799.64 = 77× inflated). (2) naive non-overlapping chunking — ~2× inflated. (3) token-level causal-LM — **paper Table VII stress tests** (Mistral-Long 33.20/33.93). (4) **rolling-window masked (canonical, current `rolling_ppl.py`)** — the FINAL unified table: **Mistral-Phrase raw 10.97/def 13.61, Mistral-Long raw 12.11/def 12.86, Trojan raw 8.02/def 10.14**.

## 5. Paper ↔ transcript reconciliation
- **PPL Table VII is on a superseded protocol.** Paper reports token-level (Mistral-Long 33.20); the project later unified everything to rolling-window (Mistral-Long 12.11). Numbers differ ~2–3×; the paper is internally consistent ("compare within protocol") but **should be refreshed to the rolling-window table** for a single clean protocol.
- **Mistral-Long 0.625 (paper) vs 0.683 (rerun):** bounded reproduction variance ≤0.06; conclusions unchanged.
- **Auto-calibration / sequence-level + representation-level future work:** the paper *states* these as future work; this session *tested* them on Mistral-Long and they did **not** help → confirms (does not contradict) the limitation framing, with new direct evidence.
- **Code:** core method (score/prune/recover/ASR) is byte-identical to the published branch on the default path; the entire `TRANSFER/` harness + the CROW_PROXY_MODE/proxy-safe/normalization extras are **not committed**; one latent GQA head-pruning bug, never exercised (all headline plans channel-only).

## 6. Open threads
- Causal/logit-gap recovery stages (this session) still running — ablation already shows utility-preserving but ASR-flat behavior (the cleanest "no separable channel circuit" null).
- Reproducibility hygiene: commit `TRANSFER/` harness + extras; pin paper to a commit; optionally refresh Table VII to rolling-window PPL.
- Whether the negative future-work results (sequence/representation/causal) should be added to the paper as strengthened limitation evidence.
