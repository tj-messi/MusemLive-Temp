# OmniBench 10 FPS 实验 1–7 完整报告

范围：最终人工选择的 450 题、1250 个评分 slot；四个实时模型使用同一得分点评分协议。MOSS 不接收原始音频，现场声音通过理想 ASR 转写文字输入。

## 完成状态

| 实验 | 状态 | 核心产物 |
|---|---|---|
| 1 回应率/回应后质量/联合分 | 完成 | response_decomposition.svg |
| 2 时间指标与琐碎基线 | 完成 | timing_f1.svg |
| 3 证据距离与视频时长分层 | 完成 | evidence_distance.svg；包含真实评测 clip 时长 |
| 4 多 Judge 与 bootstrap | 完成 | 5 个 Judge、每个 1250 slot、按题聚类 bootstrap |
| 5 强制作答 | 完成 | Venus/Gander/MiniCPM 配对分析与 95% CI |
| 6 模态消融 | 完成 | 静音、黑屏、无转写、仅转写文字 |
| 7 最近30秒/3秒捷径基线与离线上界 | 完成 | context_baselines.svg |

## 完整性说明

- Gander 强制作答的 1/1250 slot 被外部 Provider 内容检查拒绝；原始回复未改，使用 offline fixed-rubric review (no external model) 按同一 rubric 补齐，得分 3.0。
- 离线上界复用既有生成缓存：999 个成功回答与 10 FPS 时间线逐事件一致；1 个原始 API 失败保留为空回答，未重试。
- Alternate Judge 有 22 个 slot 遇到 provider JSON 截断；其中 20 个改走同一模型的备用 provider，17 个使用同一 Judge 的紧凑 JSON schema 重试。被评模型输出、rubric 与确定性计分公式均未改变。

## 核心结论

1. Venus/Gander 的用户提问回应率分别为 69%/32%；回应后的条件质量为 1.47/1.61，证明低分同时包含“不说”和“说了也不够好”两个因素。
2. 时间 F1：MiniCPM 0.60、Venus 0.54、Gander 0.38、MOSS 0.56；永远说话基线为 0.54。
3. 评测实际使用的 clip 绝大多数为 10–15 分钟；模型每题实际接收的 session 通常只有几十秒。因此当前实验不能表述为模型处理了小时级连续前缀。
4. 各 Judge 的模型排序为：Qwen: MOSS* > Venus > MiniCPM-o 4.5 > Gander; HY3: MOSS* > Venus > MiniCPM-o 4.5 > Gander; DeepSeek: MOSS* > Gander > Venus > MiniCPM-o 4.5; GLM: MOSS* > Venus > MiniCPM-o 4.5 > Gander; Kimi: MOSS* > Venus > MiniCPM-o 4.5 > Gander。相对 Qwen 的 Kendall τ：HY3=1.00, DeepSeek=0.33, GLM=1.00, Kimi=1.00。
5. 强制作答的用户提问联合分变化：Venus 1.02→1.22，Gander 0.52→1.43，MiniCPM 0.72→0.81。
6. 模态消融总分：MiniCPM 完整/静音/黑屏为 0.97/1.13/0.91；MOSS 完整/无现场转写/仅转写黑屏为 1.61/1.4/0.64。
7. 用户提问均分从完整上下文到最近 30 秒再到 3 秒：MiniCPM 0.72→0.68→1.44，MOSS 1.28→1.3→1.2；离线 Qwen-Omni 上界为 3.35。

## 图表

- [回应行为拆分](../figures/response_decomposition.svg)
- [时间 Precision/Recall/F1](../figures/timing_f1.svg)
- [多 Judge 稳定性](../figures/judge_stability.svg)
- [模态消融](../figures/modality_ablation.svg)
- [上下文基线与离线上界](../figures/context_baselines.svg)
- [证据距离分层](../figures/evidence_distance.svg)

## 重建分析与报告

```bash
export EXPERIMENT_OUTPUT_ROOT=/path/to/external/10fps-run
export MAIN25_ROOT="$EXPERIMENT_OUTPUT_ROOT/runs"
export MUSEUMLIVE_WORK_ROOT=/path/to/external/work
export MUSEUMLIVE_TASKS_FILE=/path/to/tasks_final450.json
export JUDGE_DIR_NAME=judge_points_lenient_v3
python3 pipelines/10-20/postprocessing/analysis_a.py
python3 pipelines/10-20/postprocessing/analysis_b.py
python3 pipelines/10-20/postprocessing/build_report_1_7.py
```

详细逐项表格见 [`analysis_1_4.md`](analysis_1_4.md) 和
[`analysis_5_7.md`](analysis_5_7.md)；完整机器可读结果见
[`../results/summary.json`](../results/summary.json)。
