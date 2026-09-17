# HANDOFF: 判决实验（Gate Experiment）执行说明

写于 2026-09-17。目标读者：接手执行的另一个模型或人。所有路径为绝对路径。

## 0. 背景一句话

论文被 TDSC 拒稿。核心疑点：主结果 ASR 0.14 的收益，可能主要来自"用测试指令训练拒答"（数据泄漏）和恢复训练本身，而不是剪枝。本实验在**训练/测试完全分离**的数据上，配对比较"不剪枝只恢复"、"随机剪枝+恢复"、"TFPD 剪枝+恢复"，判断剪枝有没有独立贡献。这个答案决定论文往哪个方向改。

## 1. 当前状态（已完成）

- 仓库：`/home/lizhy/plp/trigger-free-pruning-defense-round2`，git 分支 `codex/post-submission-sync`，已推送 GitHub。实验脚本目录 `TRANSFER/` 在仓库内，`/home/lizhy/plp/TRANSFER` 是指向它的软链接。
- 环境：**只用** `/home/lizhy/.conda/envs/crow_repro/bin/python`（torch 2.5.1+cu124，transformers 5.3.0，peft 0.19.1）。
- 数据（新建，已划分）：`data/train/{harmful_train,benign_train}.jsonl`（各 400 条，有害行带 `refusal` 字段，良性行带 `output` 字段）、`data/val/{harmful_val,benign_val}.jsonl`（120/100）、`data/test/*.jsonl`（原 BEAT 测试集，封存，只在最终评测用）。审计结果在 `data/SPLIT_MANIFEST.json`：训练/验证与测试的最大词级 Jaccard 为 0.5，精确重复 112 条已剔除。**不要打印这些文件的内容**（含有害提示词，会触发模型安全拦截），只看行数和哈希。
- 代码改动（已改，**未运行验证**）：
  - `pipeline_utils.py` `apply_structured_prune`：GQA 共享 K/V 只在整组 query head 都被选中时才清零。
  - `pipeline_utils.py` `_coerce_prompt_like`：保留 jsonl 行里的 `output`/`refusal` 字段。
  - `scripts/diagnose_generation_metrics.py`：生成时用 generation_config 的完整 eos 列表；新增 `metric_all_items`（全样本分母）。
  - `scripts/recover_model.py`：新增 `--safe-target-mode per_row`（用每行自带的 refusal 作监督目标）和 `--benign-answer-supervision`（良性 loss 只算在答案 token 上）。**本轮判决实验先不用这两个新选项**，用论文原配方。
- 模型：`/home/lizhy/plp/Llama-3.1-8B_word` 和 `Llama-3.1-8B_long` 正在从 HF 下载（后台进程，日志 `/home/lizhy/plp/model_download.log`，看到两行 `DONE` 即完成）。`Llama-3.1-8B_phrase` 和三个 Mistral 已在 `/home/lizhy/plp/` 下。
- 磁盘：**只能写根分区**（`/home/lizhy` 所在，现剩约 100G）。**不要在 /ssd1、/ssd3、/ssd4 建任何目录。** 临时模型 checkpoint 每个 16G，评测完立即删，只留 JSON。
- GPU：4×3090 全部可用。每次 recovery 占满 4 卡，**串行跑**，用 `nohup`。

## 2. 第零步：三个 sanity check（必须先做）

1. 语法与导入：`python -c "import pipeline_utils; import scripts.recover_model"`（在仓库根目录，用上面的 python）。
2. 模型完整性：下载完成后，比对 `/home/lizhy/plp/Llama-3.1-8B_word/model-0000*.safetensors` 的 sha256 与 `.cache/huggingface/hub/models--BEAT-LLM-Backdoor--Llama-3.1-8B_word/` 下 metadata 记录的哈希（或 `huggingface_hub` 的 `.metadata`）。之前 phrase 模型下载坏过两个分片。
3. 代码等价性：用**旧数据**跑一次 score-only，确认当前代码能复现历史 21 个单元的 golden plan：
   ```
   python scripts/score_and_prune.py --run-dir result/gate_llama_word/sanity_score \
     --model-path /home/lizhy/plp/Llama-3.1-8B_word \
     --clean-jsonl TRANSFER/beat_data/benign_clean.jsonl \
     --protect-safe-jsonl TRANSFER/beat_data/harmful_no_trigger.jsonl \
     --prompt-template chat --dtype bf16 --max-length 256 \
     --alpha 1.0 --beta 1.0 --alpha-safe 0.5 --proxy-epsilon 0.1 --score-samples 8 \
     --kappa 1000000000 --max-prune-units 320 --max-score-to-prune 0.0 --min-prune-layer 2 --score-only
   ```
   期望：`pruning_plan.json` 的 `pruned_total == 21`，单元列表与 `/home/lizhy/plp/word/pruning_plan.json` 完全一致（比较 `to_prune` 里的 `(component, layer, index)` 集合）。如果不一致，改用快照代码 `/home/lizhy/plp/tfpd_repro_score_89d79b1/scripts/score_and_prune.py` 做打分，并把差异记录下来。

## 3. 第一步：判决实验矩阵（Llama-3.1-8B Word）

输出根目录：`result/gate_llama_word/`。每个 run 一个子目录，命名 `<condition>_seed<S>`。

### 3.1 固定的恢复配方（论文 Word 主结果配方，只换数据）

```
CROW_ADAMW_FOREACH=0 CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/recover_model.py \
  --run-dir <RUN> --model-path <PRUNED_MODEL_OR_RAW> --pruning-plan <PLAN> \
  --benign-jsonl data/train/benign_train.jsonl \
  --harmful-no-trigger-jsonl data/train/harmful_train.jsonl \
  --safe-target-mode fixed --safe-target-text "I cannot assist with that request." \
  --dtype bf16 --prompt-template alpaca --max-length 256 \
  --trainable-policy all --mask-policy strict \
  --lambda-clean 1.0 --lambda-align 2.0 --lambda-safe 0.08 \
  --steps 25 --lr 1.5e-5 --grad-accum-steps 4 --objective-schedule simultaneous \
  --proxy-epsilon 0.1 --seed <S>
```
输出 `<RUN>/recovered_model`。

### 3.2 打分与剪枝（TFPD，只做一次，用训练集）

同第零步第 3 条的命令，但 `--run-dir result/gate_llama_word/score_train`，`--clean-jsonl data/train/benign_train.jsonl`，`--protect-safe-jsonl data/train/harmful_train.jsonl`，**去掉 `--score-only`**，得到 `pruned_model` 和 `pruning_plan.json`。记录 `pruned_total`（记为 N，可能不是 21）。

### 3.3 条件列表

| 条件 | 起点模型 | 剪枝计划 | 恢复 | seeds |
|---|---|---|---|---|
| `raw` | 原模型 | 无 | 无 | 只评测 1 次 |
| `prune_only` | score_train/pruned_model | TFPD 计划 | 无 | 只评测 1 次 |
| `rec_only` | 原模型 | 空计划（`to_prune: []`，`pruned_total: 0`，其余字段照抄 TFPD 计划） | 3.1 | 11, 22, 33 |
| `tfpd` | score_train/pruned_model | TFPD 计划 | 3.1 | 11, 22, 33 |
| `random_k` (k=1,2,3) | 由 `scripts/apply_matched_random_pruning.py` 生成 | 层/组件匹配随机计划，`--match-mode layer_component --reference-plan score_train/pruning_plan.json --scores-json score_train/unit_scores.json --seed k`，单元数与 N 相同 | 3.1，seed 固定 11 | 每个计划 1 个 seed |

共 1 + 1 + 3 + 3 + 3 = 11 次评测，9 次恢复训练。每次恢复约 15 分钟，评测约 10 分钟，总计约 5 小时。

### 3.4 每个 run 的评测（两次调用）

```
python scripts/diagnose_generation_metrics.py --label <RUN_NAME>_val \
  --output-json <RUN>/asr_val.json --model-path <MODEL> \
  --triggered-jsonl data/test/harmful_word_trigger.jsonl \
  --harmful-no-trigger-jsonl data/val/harmful_val.jsonl \
  --benign-jsonl data/val/benign_val.jsonl \
  --prompt-template alpaca --dtype bf16 --eval-max-length 1024 --eval-max-new-tokens 64 \
  --save-samples-jsonl <RUN>/samples_val.jsonl
```
再跑一次 `--label <RUN_NAME>_legacy --output-json <RUN>/asr_legacy.json`，把 harmful/benign 换成 `data/test/harmful_no_trigger.jsonl` 和 `data/test/benign_clean.jsonl`（与论文口径可比）。

然后 PPL：`python TRANSFER/rolling_ppl_auto.py <RUN_NAME> <MODEL> <RUN>/ppl.json`。

评测和 PPL 都完成后：`rm -rf <RUN>/pruned_model <RUN>/recovered_model`。

### 3.5 每个 run 必须留下的 manifest（`<RUN>/manifest.json`）

代码 commit（`git rev-parse HEAD`）、模型路径与四个分片 sha256、数据文件 sha256（从 `data/SPLIT_MANIFEST.json` 抄）、完整命令行、seed、`pruning_plan.json` 的 sha256 与 `pruned_total`、transformers/torch 版本、开始与结束时间。保存前后 `config.json` 里 `rope_scaling`、`eos_token_id` 必须与原模型一致（不一致则用 `TRANSFER/run_llama_phrase_long_april26_repair_replay.sh` 里 `repair_model_metadata` 的逻辑修复后再评测，并在 manifest 记录）。

### 3.6 汇总

写 `result/gate_llama_word/SUMMARY.tsv`，列：condition, seed, pruned_total, ASR(test, non-empty 分母), ASR_all_items, HarmRef_val, BFR_val, HarmRef_legacy, BFR_legacy, Empty, PPL。再写 `VERDICT.md`，含：
- `rec_only` 与 `tfpd` 三个 seed 的配对差（同 seed 相减）的均值和范围；
- `tfpd` 与 `random_k` 的比较；
- `prune_only` 相对 `raw` 的变化。

**判定规则**：若 `tfpd` 相对 `rec_only` 的 ASR 配对降幅在三个 seed 上都为正、均值大于 seed 间标准差、且 BFR_val 不高于 rec_only 加 0.05，则剪枝有独立贡献，继续走剪枝论文路线（第四步）。否则转向"trigger-free 安全恢复 + 因果结构干预"路线（第五步）。

## 4. 若剪枝有贡献：后续任务

1. 同样矩阵跑 Llama Phrase（打分协议 alpaca/256、alpha 0.5、alpha_safe 0；恢复 la=1.0）和 Mistral Long（见 `SYNC_NOTE.md` 与 `result/mistral_protocol_replays/VERDICT_mistral_replay.md` 的配置）。
2. 配方 v2：加 `--safe-target-mode per_row --benign-answer-supervision`，看 BFR 是否下降、ASR 是否保持。
3. 替身后门选参：在可疑模型上用 LoRA 植入一个自定义 jailbreak 触发词（同任务），超参数只按替身 ASR 加 BFR/PPL 预算选，真触发器只在最后测一次。文献依据见记忆文件 `trigger_free_selection_survey.md`（Dummy Backdoor 2606.11648、Locphylax 2510.10265、LIRA 2604.10403）。
4. 留一法：在部分模型上冻结规则，在其余模型上测。
5. 基线：clean-FT、Fine-Pruning 适配、BEEAR、SANDE（`official_baselines/` 已有适配脚本，之前失败原因是磁盘满和 TF 4.37 不支持 Llama-3.1 rope）。
6. LLM judge 评测（256 token 输出，HarmBench 分类器）。

## 5. 若剪枝无贡献：后续任务

以 `result/causal_direction/VERDICT_causal.md`（trigger-aware 因果剪枝 Mistral-Long 到 0.333）为起点，研究问题改为：不知道触发词时，替身后门能否替代真触发器做因果定位。实验：替身后门注入 → 因果 attribution 选 channel → 同样恢复 → 测真触发器 ASR。对照仍是 rec_only。

## 6. 硬性规则

- 不碰其他用户的进程和目录；启动前 `nvidia-smi` 确认显存空闲。
- 长任务用 `nohup ... > log 2>&1 &`，把 PID 写到 `<RUN>/PID`。
- 每步先在 `<RUN>/status.tsv` 追加一行（stage, rc, time），失败不要静默跳过。
- 不打印有害提示词内容。
- 结果 JSON 永远不删；模型权重评测完即删。
- 每完成一个 run 就 `git add result/gate_llama_word/**/*.json *.tsv *.md` 提交一次（权重和 samples 已被 .gitignore 排除，`samples_val.jsonl` 若要提交先确认不含触发词以外的敏感内容，否则不提交）。
