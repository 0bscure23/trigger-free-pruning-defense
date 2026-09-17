# Llama Word Reproducible Rescue Sweep 准备状态

准备时间：2026-06-27 00:54 CST

## 目标

重新寻找一个当前服务器上可复现、完整保存权重、证据链闭环的 `BEAT/Llama-3.1-8B_word` 最优后门消除结果。

这个实验不再声称复现历史 `ASR=0.1417`。历史 `0.1417` 只作为参照；新目标是从 raw BEAT 模型重新完成：

```text
raw model -> score -> prune -> pruned_model -> recovery sweep -> recovered eval
```

## 已准备

- 主 runner：`/home/lizhy/plp/TRANSFER/run_llama_word_reproducible_rescue_sweep.sh`
- 结果目录：`/home/lizhy/plp/trigger-free-pruning-defense-round2/result/llama_word_reproducible_rescue_sweep`
- 旧代码仓库：`/home/lizhy/plp/tfpd_oldcode`
- 指定 commit 均已确认存在：
  - `8759648b11328e7d540568283666a6aa74f4c742`
  - `307dc4d19e7c64b5498eced23f18c44d2bb126a8`
  - `060b664cd3e4a7f2a0fb85206722cab4a4210014`
- dry-run 已生成：
  - `audit_manifest.json`
  - `run_plan.tsv`
  - `progress.log`

## 已做代码修补

`diagnose_generation_metrics.py` 新增可选参数：

```bash
--save-samples-jsonl <path>
```

默认行为不变；rescue sweep 会保存 generation samples，用于审计 ASR 是否由短输出、空输出或模板化输出造成。

## 运行计划

`run_plan.tsv` 当前阶段：

| Phase | Count | 内容 |
|---|---:|---|
| phase1_raw | 1 | raw ASR/PPL sanity |
| phase2_old_sanity | 3 | 旧 21-unit plan + 旧代码 no-seed recovery |
| phase3_score | 1 | 从 raw 重新 score/prune old tight gate |
| phase4_variants | 8 | gate/budget/min-layer pruning variants |
| phase5a_per_candidate | 40 | 每个候选的 lambda_align × lambda_safe 粗搜 |

Phase 5A 的 40 是“每个入选 pruning candidate”的数量。实际总 recovery run 数取决于 Phase 4 选出几个候选，默认最多 5 个。

## 当前不应立即启动的原因

当前 `/home/lizhy` 所在磁盘只剩约 `6 GiB`。这个实验需要保存 7B 的 `pruned_model` 和若干 `recovered_model`，脚本默认要求至少 `55 GiB` 空间。

同时仍有旧的 evidence replay 进程在跑：

```text
/home/lizhy/plp/TRANSFER/run_evidence_replay_batch.sh
```

因此现在只完成准备，不启动 GPU 实验。

## 启动命令

释放足够空间并确认没有其它冲突任务后，启动完整流程：

```bash
RUN=1 PHASES=all /home/lizhy/plp/TRANSFER/run_llama_word_reproducible_rescue_sweep.sh
```

分阶段启动示例：

```bash
RUN=1 PHASES=phase1,phase2 /home/lizhy/plp/TRANSFER/run_llama_word_reproducible_rescue_sweep.sh
RUN=1 PHASES=phase3,phase4 /home/lizhy/plp/TRANSFER/run_llama_word_reproducible_rescue_sweep.sh
RUN=1 PHASES=phase5a /home/lizhy/plp/TRANSFER/run_llama_word_reproducible_rescue_sweep.sh
```

如果直接 `RUN=1` 但不指定 `PHASES`，脚本会报错，避免误以为空跑成功。

## 重要默认参数

- model：`/home/lizhy/plp/Llama-3.1-8B_word`
- prompt template：`alpaca`
- dtype：`bf16`
- scoring max length：`256`
- recovery max length：`256`
- eval max length：`1024`
- eval max new tokens：`64`
- `alpha=1.0`
- `beta=1.0`
- `alpha_safe=0.5`
- `proxy_epsilon=0.1`
- `score_samples=8`
- `min_prune_layer=2`
- `max_score_to_prune=0.0`
- `max_prune_units=320`

## 输出

最终会生成：

- `llama_word_reproducible_rescue_sweep.md`
- `llama_word_reproducible_rescue_sweep.json`
- `llama_word_reproducible_best_manifest.json`
- `summary_rows.tsv`
- 每个 run 的 `run_config.json / pruning_plan.json / recovery_losses.json / asr.json / ppl.json / samples.jsonl / run.log`

