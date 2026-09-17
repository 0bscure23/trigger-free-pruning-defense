# Paper-Plan Anchored Security Completeness

Started: 2026-06-28 16:02:00

- Evidence zip: `/home/lizhy/plp/paper_evidence_pack_models_20260626_120830.zip`
- Anchor model: `/home/lizhy/plp/Llama-3.1-8B_word`
- Output: `/home/lizhy/plp/trigger-free-pruning-defense-round2/result/security_completeness_paper_plan_anchor_drycheck2`
- Score protocol: old `89d79b1` score code, `chat` prompt template, max length `256`, golden score SHA256 `c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45`
- Recovery/eval prompt template: `alpaca`
- Eval protocol: `bf16`, max length `1024`, max new tokens `64`, greedy decoding
- Runtime: Transformers `5.3.0`
- GPU devices: `0,1,2,3`
- GPU wait threshold: used memory <= `5120 MiB`; require no compute apps = `0`
- Temporary recovered checkpoints are deleted after ASR/PPL evaluation.

This runner anchors to the original Llama-Word paper pruning plan and golden score artifact. The score stage was reproduced separately under `chat/256`; recovery and ASR/PPL evaluation use the paper protocol `alpaca/1024/64`.
