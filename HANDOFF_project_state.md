# 项目交接档案：Trigger-Free Pruning Defense（TFPD）

更新时间：2026-09-18 凌晨。写给接手这个项目的人或模型。读完这一份应该能独立继续。
与之配套的操作细节见 `HANDOFF_gate_experiment.md`（判决实验的逐条命令）。

---

## 1. 项目一句话

在不知道后门触发词的情况下修复被植入"越狱后门"的大模型：先用扰动代理信号给注意力头和 MLP 通道打分，剪掉高置信单元，再做带掩码的短恢复训练。论文投 IEEE TDSC（编号 TDSC-2026-07-2830）被拒，三位审稿人的意见原文在本文件第 3 节按条映射。

## 2. 现状总览（2026-09-18）

| 事项 | 状态 |
|---|---|
| 代码仓库 | `/home/lizhy/plp/trigger-free-pruning-defense-round2`，git 分支 `codex/post-submission-sync`，已推 GitHub（0bscure23/trigger-free-pruning-defense）。tag `paper-repro-v1`（=08fd92b）是复现论文主结果的代码。 |
| 环境 | 只用 `/home/lizhy/.conda/envs/crow_repro/bin/python`（torch 2.5.1+cu124，transformers 5.3.0，peft 0.19.1）。其他 conda 环境都不要用于本项目。 |
| 模型 | 本地已有：`/home/lizhy/plp/Llama-3.1-8B_phrase`、`Mistral-3-7B_{word,phrase,long}`。正在下载：`Llama-3.1-8B_word`、`Llama-3.1-8B_long`（脚本 `/home/lizhy/plp/download_models.sh`，日志 `/home/lizhy/plp/model_download.log`，看到 `ALL_DONE` 且每个分片都是 `OK` 才算完成；有 `HASH_MISMATCH` 就重跑脚本）。 |
| 数据 | `data/train`、`data/val`、`data/test` 已划分好，测试集封存。**不要打印这些文件的内容**，会触发模型安全拦截。 |
| 正在跑 | Phrase 判决矩阵：`result/gate_llama_phrase/`，PID 在同目录 `PID` 文件。预计 6 到 8 小时。 |
| 磁盘 | 只能写根分区（`/home/lizhy`），约 100G 可用。**禁止在 /ssd1、/ssd3、/ssd4 建目录。** |
| GPU | 4×3090，共享服务器。同事 linpf7 常驻四个 600MB 小进程，不影响。CPU 常被占满（负载 70+/80 核），打分那一步会因此变慢到 15 分钟。 |

## 3. 已核实的问题（比审稿意见更具体）

这些都是在服务器上亲自核对过的事实，不是推测。每条后面标了对应的审稿人意见编号（R1-1 表示 Reviewer 1 第 1 条）。

### 3.1 测试指令泄漏进训练（最严重，审稿人没发现）
- `TRANSFER/beat_data/harmful_no_trigger.jsonl` 的 120 条指令 = 三个触发测试集去掉触发词后的原句，120/120 完全包含。
- 恢复训练的 safe loss 用这 120 条加固定拒答句训练；打分也用它们；BFR/HarmRef 评测也用同一批文件。
- 附录里 λ_safe=0 时 ASR 回到 1.0，说明效果高度依赖这一项。
- 后果：论文的 84% 降幅里有多少来自"背过测试题"无法区分。
- 处理：`data/` 已重新划分（有害来自 LLM-LAT/harmful-dataset，良性来自 Alpaca，与测试集词级 Jaccard ≤0.5，精确重复 112 条剔除）。所有新实验只用新数据。

### 3.2 剪枝的独立贡献未证明（R3-2，R2 创新性）
- 同一恢复配方下，Word 剪 0 个单元 ASR 0.1833/BFR 0.36；剪 21 个单元 0.1417/BFR 0.39。差 5 个样本。
- 固定计划只换恢复 seed，Word 的 std 0.13，Phrase 0.19 到 0.29。剪枝的收益在噪声内。
- 附录把零剪枝的 0.1833 写成"接近未防御"，与原始 0.925 矛盾，审稿人可直接抓住。
- 处理：正在跑的判决矩阵（第 5 节）专门回答这个问题。

### 3.3 打分信号基本不定位后门
- `result/oracle_separability/`：FGSM 代理选出的 top-512 只命中 1024 个 oracle 后门通道中的 5 个（recall 0.5%）。
- `result/causal_direction/VERDICT_causal.md`：用真触发词做因果 attribution 选通道，Mistral-Long 到 0.333（FGSM 代理 0.683，同一恢复配方）。说明通道级干预方向是对的，但当前 trigger-free 信号没抓到。

### 3.4 超参数按 triggered ASR 挑选（R1-1，R2-4，R3-1）
- 每个模型和触发词类型一套恢复配置，都是看着测试 ASR 调出来的。
- 之前的 auto-calibration（只用 BFR/HarmRef 选）失败：低 BFR 常对应高 ASR。
- 文献里没有任何方法用真触发词做验证。三种合法做法：固定配置加敏感性曲线（BEEAR、Lethe）；效用稳定即停（BEEAR 用 MT-bench）；防御者自己注入替身后门当验证信号（Dummy Backdoor arXiv 2606.11648、Locphylax 2510.10265、LIRA 2604.10403）。
- 处理方案见第 6 节。

### 3.5 "Transformers 版本敏感"其实是文件格式 bug（R3-7）
- 中间 checkpoint 保存后 config.json 丢 `rope_scaling`/`rope_theta`，`eos_token_id` 列表被压成标量。跨版本加载就成了另一个模型。
- 加 metadata 修复后，Long 0.175、Phrase 0.075 在 TF 5.3.0 下精确复现；不修复 Long 是 0.425。
- 处理：`pipeline_utils.save_model_and_tokenizer_safe` 已有修复逻辑；`TRANSFER/run_gate_experiment.py` 的 `repair_metadata` 在每次保存后再核对一遍并记录到 manifest。论文里应改写为根因，删掉"运行时敏感"的说法。

### 3.6 代码与论文不一致
- GQA 剪枝：旧代码选中一个 query head 就清零共享 K/V，论文写的是按组。Mistral Word 的计划含 75 个 head，受影响。**已修**（`apply_structured_prune`，单元测试三个用例通过），Mistral Word 相关结果需要重跑。
- 打分协议每个触发词不同：Word 用 chat/256、alpha 1.0、alpha_safe 0.5、min_layer 2、gate S≤0；Phrase/Long 用 alpaca/256、alpha 0.5、alpha_safe 0、min_layer 0、无 gate、固定 320 单元。论文 Table II 写全局 alpha=1.0，是错的。
- 评分公式聚合顺序论文和代码不同（论文先平均梯度再算幅值和余弦，代码逐样本算再平均）。要么改论文公式，要么改代码重跑；建议改论文。
- 评测 eos 只用单个 token，Llama-3.1 有三个。**已修**（`_resolve_eos_ids`）。
- ASR 分母原来只算非空输出。**已加** `metric_all_items`。

### 3.7 复现状态（每个结果分开说）
| 场景 | 状态 |
|---|---|
| Llama Word 0.1417 | 全流程复现（打分 chat/256 + 08fd92b 恢复） |
| Llama Phrase 0.1667 / 0.075 | 复现；7 月曾因下载的两个权重分片损坏失败，修好后正常 |
| Llama Long 0.175 | 加 metadata 修复后复现 |
| Mistral Long 0.625 | 精确复现 |
| Mistral Phrase 0.783 | 重放 0.708，不算精确 |
| Mistral Word 0.150 | **未复现**，30 个 seed 最好 0.4167。论文里不能再当主要迁移证据。 |
| 当前代码 vs golden 打分 | Phrase：459776 个分数逐位相同，320 单元完全一致（2026-09-18 验证）。 |

### 3.8 其他审稿意见的对应
- R1-3、R3-4（BFR 没进主表）：新汇总表 `SUMMARY.tsv` 把 ASR、ASR 全样本、HarmRef、BFR（验证集和旧口径两套）、Empty、PPL 并列，主表必须这样报。
- R1-2、R2-2（基线缺失）：`official_baselines/` 有 BEEAR、SANDE 的适配脚本，只在 Mistral-Long 跑过短预算（ASR 仍 0.9 左右）。40 组 sweep 39 组失败，原因是磁盘满和 torch251 环境的 TF 4.37 不支持 Llama-3.1 rope。clean-FT、Fine-Pruning、Wanda/LLM-Pruner 适配、PURE、Lethe、W2SDefense 未做。BAIT 是扫描器不是修复方法，只需讨论。
- R1-4、R3-3（泛化弱）：需要至少一种注入机制不同的后门（BackdoorLLM 的 weight-poisoning 或 hidden-state steering）。
- R1-5、R3-6（关键词 ASR）：需要 LLM judge（HarmBench 分类器）在 256 token 输出上判定，保存逐样本输出。
- R2-3、R3-5（自适应攻击弱）：现有只是固定旧 mask 的短程强化。需要攻击者知道打分规则、从头投毒、防御重新完整执行。
- R1-6、R2-2 引用：BEEAR、W2SDefense、BAIT、PURE、Wanda、LLM-Pruner、SANDE、Lethe，还有那篇 TDSC 的 unlearning+RAG。

## 4. 已经完成的工作（本轮，2026-09-17 到 18）

1. GitHub 同步：本地四个月的改动、TRANSFER 脚本、结果文档全部提交推送；给复现代码打 tag。
2. 根分区清理，腾出 70G。
3. 数据重划分（`data_tools/build_splits.py`，审计在 `data/SPLIT_MANIFEST.json`）。
4. 代码修复：GQA、eos 列表、全样本 ASR、`--score-only`。
5. 恢复脚本新增两个选项（本轮判决实验**没有用**，留给配方 v2）：`--safe-target-mode per_row`（用每行自带的拒答参考）、`--benign-answer-supervision`（良性 loss 只算答案 token）。
6. 文献调研（记忆文件 `trigger_free_selection_survey.md`，也可从本文件 3.4 节的 arXiv 编号自行取）。
7. 通用判决实验运行器 `TRANSFER/run_gate_experiment.py`，幂等，每个 run 留 manifest.json、status.tsv、SUMMARY.json，权重评测完自动删。
8. Phrase 判决矩阵已启动。

## 5. 正在跑的实验与怎么读结果

目录 `result/gate_llama_phrase/`，子目录：`score_train`（新数据打分）、`raw`、`prune_only`、`rec_only_seed{11,22,33}`、`tfpd_seed{11,22,33}`、`random{1,2,3}_seed11`。每个子目录有 `SUMMARY.json`，跑完全部后有 `SUMMARY.tsv`。

**判定规则**：TFPD 相对 rec_only 的 ASR 配对降幅（同 seed 相减）三个 seed 都为正、均值大于 seed 间标准差、BFR_val 不高出 rec_only 0.05，才算剪枝有独立贡献。

如果运行器崩了：看 `runner.log` 和对应子目录的 `recover.log`/`eval.log`，修完直接重跑同一条命令（`python TRANSFER/run_gate_experiment.py --anchor llama_phrase`），已完成的 run 会跳过。

Word 下载完后：`python TRANSFER/run_gate_experiment.py --anchor llama_word`。先做一次 sanity：用旧数据 score-only（参数在 `HANDOFF_gate_experiment.md` 第 2 节），确认 21 个单元与 `/home/lizhy/plp/word/pruning_plan.json` 一致。

## 6. 后续路线（按判决结果分叉）

### 6A. 若剪枝有稳定贡献
1. Word、Long、Mistral Long 跑同样矩阵。
2. 配方 v2：加 `--safe-target-mode per_row --benign-answer-supervision`，目标是降 BFR。
3. **替身后门选参协议**（回答 R1-1/R3-1 的核心）：
   - 在可疑模型上用 LoRA 植入一个自定义 jailbreak 触发词（同任务；Dummy Backdoor 论文证明跨任务不行）。
   - 所有超参数、步数、checkpoint 只看替身触发词的 ASR 加 BFR/PPL 预算。
   - 真触发词只在最后测一次。
   - 再做留一法：在部分模型上冻结规则，在其余模型上测，报均值方差。
4. 基线：clean-FT、只恢复（已有）、Fine-Pruning 适配、CROW（注意 CROW 原实现用 lr 1e-3，PurifyNoPrior 论文指出这是它效果的来源，比较时统一 lr）、BEEAR、SANDE。BEEAR/SANDE 需要先把 torch251 环境升到支持 Llama-3.1 的 transformers，或在 crow_repro 里跑。
5. 泛化：BackdoorLLM 加一种注入机制。
6. LLM judge：HarmBench 分类器，256 token。
7. 自适应攻击：攻击者用 TFPD 打分规则做正则项从头投毒，防御重新完整执行。
8. 论文重写要点：主表报冻结规则下的均值方差，历史最优点降为附录"回顾性上限"；Table I 补每个触发词的打分协议；Table IV Word PPL 13.80 与复跑 8.89 对账到同一 checkpoint；附录 alignment 消融 BFR 0.10 与主结果 0.39 说明来源；EMD AUROC 下降只能说检测器失效。

### 6B. 若剪枝无稳定贡献（预判这个可能性更大）
研究问题改为"不知道触发词时怎样做安全恢复并定位结构"：
1. 以 `result/causal_direction/` 的 trigger-aware 因果剪枝（Mistral-Long 0.333）为上界。
2. 用替身后门代替真触发词做因果 attribution 选通道，同样恢复，测真触发词 ASR。对照仍是 rec_only。
3. 若替身因果定位有效，论文主贡献变成"替身后门引导的结构干预"，现有的 FGSM 代理降为消融。
4. 配方 v2 仍然要做，BFR 问题在两条路线里都存在。

## 7. 工程规则（不遵守会重蹈覆辙）

- 每次运行必须留 manifest：commit、模型分片哈希、数据哈希、完整命令、seed、计划哈希、环境版本、保存前后 config 差异。运行器已自动做。
- 保存后核对 config 的 `rope_scaling` 和 `eos_token_id` 与原模型一致。
- 权重评测完就删，JSON 永远留。每个 run 完成后 `git add result/gate_*/**/*.json *.tsv *.md` 提交一次。
- 一次只跑一个恢复训练（占满 4 卡）。用 `nohup`，PID 写文件。
- 不打印数据集内容；只看行数和哈希。
- 旧目录 `plp/tfpd_*`、`plp/{word,phrase,long}`、`plp/mistral_*` 是只读证据，不要动。
- `pkill -f` 会把当前 shell 一起杀掉，用 `pkill -f "[x]pattern"` 的写法。
- hf_hub 的 xet 下载后端在这台机器上会卡死，下模型用 `download_models.sh`（curl 走本机代理 127.0.0.1:7892，别改代理节点）。

## 8. 文件地图

| 路径 | 内容 |
|---|---|
| `SYNC_NOTE.md` | 哪份代码复现哪个结果，各触发词的打分协议 |
| `HANDOFF_gate_experiment.md` | 判决实验逐条命令 |
| `HANDOFF_project_state.md` | 本文件 |
| `CODE_AUDIT_for_paper.md`、`PROJECT_HISTORY_MAP.md` | 6 月的审计和历史（部分结论已过时，以本文件为准） |
| `data/` | 新划分数据与审计 |
| `data_tools/build_splits.py` | 重建数据划分 |
| `TRANSFER/run_gate_experiment.py` | 判决矩阵运行器 |
| `TRANSFER/beat_data/` | 旧评测数据（=`data/test`） |
| `result/gate_llama_phrase/` | 正在跑的矩阵 |
| `result/llama_security_completeness_all/run_20260629_145307/summary_rows.tsv` | 零剪枝对照等 65 组旧结果 |
| `result/oracle_separability/`、`result/causal_direction/`、`result/causal_tf/` | 信号定位能力与因果上界 |
| `result/mistral_protocol_replays/VERDICT_mistral_replay.md` | Mistral 三个模型的复现状态 |
| `result/llama_word_scoring_fingerprint/SCORE_STAGE_REPRODUCTION_RESOLUTION.md` | Word 打分协议定位过程 |
| `official_baselines/` | BEEAR、SANDE、CleanGen、DUP 官方代码与本地适配 |
| `/home/lizhy/plp/71eb6984-b972-457d-bb41-e046148b1acf (2).pdf` | 被拒的投稿 PDF |
| `/home/lizhy/plp/2paper_ieee_main_theory_revised_firstpage_cited (1).tex` | 论文 tex |
| `~/.codex/sessions/2026/09/` | Codex 在 9 月 10/12 日的两次独立分析，结论与本文件一致 |
| `~/.claude/projects/-home-lizhy/memory/` | Claude 的记忆文件，含调研和发现摘要 |
