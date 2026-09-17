# Security-Completeness Paper-Plan Anchor Verdict

时间：2026-06-28

## 完成状态

- 计划 run 数：20
- 完成 run 数：20
- 失败 run 数：0
- 残留临时 checkpoint：0
- 审计表：`SECURITY_COMPLETENESS_AUDIT.md`
- 汇总表：`summary_rows.tsv`

所有 run 均使用行为评测协议：`alpaca / eval_max_length=1024 / eval_max_new_tokens=64 / bf16 / greedy / Transformers 5.3.0`。

Score provenance 已锁定为：old `89d79b1` scoring code，`chat / max_length=256`，golden score SHA256 `c18257f2035cd7550a590aad6fe97575a00b7f462b0f43643d2dbe860f0ede45`。

## Fixed 21-Unit Recovery Seeds

固定 21-unit plan，仅改变 recovery seed：

| Seed | ASR | HarmRef | BFR | Empty | PPL |
|---:|---:|---:|---:|---:|---:|
| 13 | 0.6333 | 0.3667 | 0.0600 | 0.0000 | 12.0356 |
| 17 | 0.3000 | 0.7083 | 0.2300 | 0.0000 | 11.8358 |
| 23 | 0.2583 | 0.7667 | 0.4200 | 0.0000 | 11.8651 |
| Mean ± Std | 0.3972 ± 0.2055 | 0.6139 ± 0.2161 | 0.2367 ± 0.1801 | 0.0000 ± 0.0000 | 11.9122 ± 0.1079 |

解释：当前环境下，21-unit plan 的 recovery trajectory 方差很大。历史 `0.1417` 应作为 archived best run 结果保留；新补实验更适合报告为 `mean±std over recovery seeds with a fixed pruning plan`。

## Recovery Sensitivity

最差点：

- `lr=3e-5`：ASR 1.0000，PPL 29.7249，说明过大学习率会破坏恢复。
- `lambda_safe=0`：ASR 0.9917，HarmRef 0.0000，说明 safe loss 是必要项。
- `steps=50`：ASR 0.7917，PPL 20.2608，说明过长 recovery 会变差。

相对较好点：

- `steps=10`：ASR 0.5083，BFR 0.1400，PPL 11.5092。
- `lambda_safe=0.04`：ASR 0.5250，BFR 0.2800，PPL 11.9893。
- `lambda_align=2.5` 或 `lr=5e-6`：ASR 均 0.5500。

结论：recovery 不是单调调参问题；`lambda_safe`、学习率和训练步数都有明显安全/效用 trade-off。

## Budget Sweep

Forced no-gate budget sweep：

| Requested | Actual | #Heads | #Channels | ASR | HarmRef | BFR | PPL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 460 | 460 | 31 | 429 | 0.5833 | 0.4167 | 0.1000 | 12.4318 |
| 1379 | 1379 | 41 | 1338 | 0.5583 | 0.4833 | 0.0900 | 12.7864 |
| 2299 | 2299 | 50 | 2249 | 0.6083 | 0.4250 | 0.1900 | 13.2305 |
| 4598 | 4598 | 69 | 4529 | 0.5167 | 0.4917 | 0.2100 | 13.1456 |

Gated `S(u)<=0` budget rows for 0.1%, 0.3%, 0.5%, and 1% all collapse to the same 21-unit plan. This is expected: the gate is a high-confidence boundary, not a forced budget selector.

结论：forced budget 增大不会恢复历史 best ASR，也会带来更高 PPL/BFR。`S(u)<=0` gate 的作用是防止预算把大量低置信 units 纳入剪枝。

## Threshold Sweep

| Gate | Actual | #Heads | #Channels | ASR | HarmRef | BFR | PPL |
|---|---:|---:|---:|---:|---:|---:|---:|
| no gate | 1379 | 41 | 1338 | 0.5583 | 0.4833 | 0.0900 | 12.7864 |
| +0.05 | 1379 | 41 | 1338 | same as no gate | same | same | same |
| 0 | 21 | 0 | 21 | same as paper exact seed13 | same | same | same |
| -0.02 | 0 | 0 | 0 | 0.5667 | 0.4417 | 0.1100 | 12.0325 |
| -0.05 | 0 | 0 | 0 | same as -0.02 | same | same | same |

结论：在 golden score distribution 下，阈值结构非常稀疏：`S<=0` 正好选 21 个高置信 channel；更宽松的 gate 等同 no-gate top-1379，更严格的 gate 则剪不到任何 unit。这个结果支持把 `S(u)<=0` 写成 calibrated confidence gate，而不是普通预算超参。

## 写作建议

- 主文不要声称当前环境 deterministic 复现 `0.1417`。
- 可以写：score/prune artifact 已经复现，历史 best run 归档完整；新补实验显示 fixed-plan recovery seed 方差较大。
- Seed stability 表题应写：`mean±std over recovery seeds with a fixed pruning plan`。
- Budget/threshold 表要保留 `#Heads/#MLP channels`，并注明 gated budget rows 与 paper exact plan 重复。
- Adaptive 部分仍需后续单独做 `Pruning-Aware Reinforcement Stress Test`，本轮只完成 seed/sweep/sensitivity 完整性。
