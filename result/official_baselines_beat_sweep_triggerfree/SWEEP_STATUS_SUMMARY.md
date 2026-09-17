# BEAT 官方对照 sweep 本轮状态摘要

结论：脚本层面已经走完，但本轮不是有效完整完成。40 组计划配置中，只有 1 组产出了完整 ASR/PPL，其余 39 组失败。

## 有效结果

| 模型 | 方法 | tag | ASR 仅报告 | HNTR | BFR | PPL | 空输出率 |
|---|---|---|---:|---:|---:|---:|---:|
| `mistral_long` | `BEEAR` | `mistral_long_beear_a9_l7_r3_i3_t60_pa40` | 0.975 | 0.450 | 0.190 | 10.54 | 0.000 |

说明：ASR 使用带触发器 split 只做最终报告，不参与参数选择；参数选择仍按 trigger-free 指标。

## 失败分布

| 失败原因 | 数量 |
|---|---:|
| Llama-3.1 rope_scaling 与 transformers 4.37 不兼容 | 24 |
| 磁盘空间不足 / No space left on device | 15 |

主要原因有两个：

- Mistral 相关失败多数发生在保存消除后模型时，磁盘只剩约 3.9G，不够写 14G/15G checkpoint。
- Llama-3.1 相关失败发生在加载阶段，当前 `torch251` 环境的 `transformers==4.37.2` 不兼容新版 `rope_scaling` 配置。

## 已完成的脚本修复

- `run_beat_official_baseline_sweep.sh`：默认传入 `CLEANUP_MODEL_AFTER_EVAL=1`，每组评估完成后清理临时模型。
- `run_official_beear_mistral_long.sh` / `run_official_sande_mistral_long.sh`：只有 ASR/PPL 都存在时才删除该组消除后模型。
- `prepare_llama3_compat_model.py`：为 Llama-3.1 生成软链接兼容目录，只改 `config.json`，不复制权重。
- `beat_official_baseline_watchdog.py`：报告改为中文，并自动识别磁盘满、Llama rope 配置不兼容等失败原因。
- `start_beat_official_baseline_sweep_from_host.sh`：默认开启清理和 Llama 兼容配置。

## 复跑前必须处理

当前 `/home/lizhy` 和 `/tmp` 仍然是 100% 使用率，约只剩 3.9G。即使开启评估后清理，每个配置仍需要先临时保存一个 14G/15G 模型后才能评估，所以必须先释放空间。

已经有指标、可考虑清理的 checkpoint：

- `official_baselines_beat_sweep_triggerfree/mistral_long/beear/mistral_long_beear_a9_l7_r3_i3_t60_pa40/beear_official_model`，约 14G。
- `official_beear_mistral_long/beear_official_manual4_r1_i1_t8_b6/beear_official_model`，约 14G。
- `official_sande_mistral_long/sande_official_dummy_sure_s16/sande_removed_model`，约 15G。

这三个合计约 43G。我没有自动删除，需要确认后再删。

## 下一轮复跑建议

释放空间后，在宿主机普通终端继续运行：

```bash
/home/lizhy/plp/TRANSFER/start_beat_official_baseline_sweep_from_host.sh
```

脚本会跳过已经有 ASR/PPL 的 `mistral_long/BEEAR/a9_l7...`，继续跑未完成配置；Llama 会自动走 `_compat_models` 软链接目录。
