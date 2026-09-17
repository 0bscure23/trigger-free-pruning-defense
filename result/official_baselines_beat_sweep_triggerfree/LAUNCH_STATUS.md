# BEAT 官方对照方法批量实验启动状态

时间：2026-06-25 14:20 CST

当前 sweep 已按用户要求停止。检查时它尚未完成，只是在等待 GPU 空闲；未继续启动后续实验。

## 最后一次启动项

- 原 PID：`1573647`，已停止
- 启动方式：`setsid` 后台独立 session
- 结果目录：`/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_baselines_beat_sweep_triggerfree`
- 日志：`/home/lizhy/plp/trigger-free-pruning-defense-round2/result/official_baselines_beat_sweep_triggerfree/sweep_nohup.log`
- 搜索预设：`SEARCH_PRESET=search_small`
- 方法：`METHODS=beear,sande`
- 模型：`mistral_long,mistral_word,llama_long,llama_phrase,llama_word`
- 暂缺：`mistral_phrase`

## 评测与清理规则

每组参数生成消除后 checkpoint 后，先尝试完成以下指标：

- triggered ASR
- harmful-no-trigger refusal / HNTR / HarmRef
- benign-clean false refusal / BFR
- average generation length
- empty output rate
- rolling-window PPL

随后若 `CLEANUP_MODEL_AFTER_EVAL=1` 且 checkpoint 目录存在，则删除该组消除后 checkpoint，再进入下一组。即使 ASR 或 PPL 某一步失败，也会先清理 checkpoint，再把该组记录为失败。

## 备注

停止前已完成 4 组 `mistral_long/BEEAR` 配置的 ASR/PPL。其余配置未形成完整结果。停止时脚本卡在 `SANDE mistral_long len1024_s50_t6` 的 GPU preflight 等待阶段，显卡上主要是其它用户的 `ollama` 进程。当前没有残留 sweep、watchdog、BEEAR、SANDE、ASR 或 PPL 评估进程。

结果目录下已确认没有残留的 `beear_official_model`、`sande_removed_model` 或 `checkpoint-*` 目录。
