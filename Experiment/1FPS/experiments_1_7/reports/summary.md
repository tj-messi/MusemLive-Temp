# OmniBench 实验 1–7 完整报告（1 FPS 定稿快照）

范围：最终人工选择的 450 题、1250 个评分 slot；四个实时模型使用同一得分点评分协议。MOSS 不接收原始音频，现场声音通过理想 ASR 转写文字输入。

输入协议：每秒决策一次，每个 tick 输入 1 帧画面以及因果对齐的 1 秒音频块；本报告不包含独立的 10 FPS 定稿结果。

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

## 核心结论

1. Venus/Gander 的用户提问回应率分别为 48%/47%；回应后的条件质量为 1.63/1.92，证明低分同时包含“不说”和“说了也不够好”两个因素。
2. 时间 F1：MiniCPM 0.77、Venus 0.59、Gander 0.62、MOSS 0.64；永远说话基线为 0.54。
3. 评测实际使用的 clip 绝大多数为 10–15 分钟；模型每题实际接收的 session 通常只有几十秒。因此当前实验不能表述为模型处理了小时级连续前缀。
4. Qwen、HY3、DeepSeek、GLM、Kimi 五个 Judge 都给出 MOSS > MiniCPM > Gander > Venus，排名 Kendall τ=1。
5. 强制作答的用户提问联合分变化：Venus 0.79→1.36，Gander 0.91→1.65，MiniCPM 1.31→1.42。
   Gander 强制作答中 1/1250 slot 因主 Judge 内容审查 HTTP 400，使用 GLM-5.2、同一 rubric 和确定性计分公式补齐；provenance 已保存。
6. MiniCPM 静音后 1.59→1.79，说明现场音频存在明显干扰；黑屏降至 1.28。MOSS 去转写降至 1.77，仅转写文字为 1.52。
7. 用户提问均分从完整上下文到最近 30 秒再到 3 秒：MiniCPM 1.31→1.27→1.18，MOSS 2.01→1.93→1.79；历史指代对截断最敏感。离线 Qwen-Omni 上界为 3.15。

## 图表

- [回应行为拆分](../figures/response_decomposition.svg)
- [时间 Precision/Recall/F1](../figures/timing_f1.svg)
- [多 Judge 稳定性](../figures/judge_stability.svg)
- [模态消融](../figures/modality_ablation.svg)
- [上下文基线与离线上界](../figures/context_baselines.svg)
- [证据距离分层](../figures/evidence_distance.svg)

## 重建分析与报告

```bash
cd /path/to/restored/1fps-analysis-root
python3 analysis_a.py
python3 analysis_b.py
python3 build_report_1_7.py
```

分析脚本保留了原始运行树的相对路径契约；所需外部产物和完整命令边界见
[`../REPRODUCIBILITY.md`](../REPRODUCIBILITY.md)。

详细逐项表格见 [`analysis_1_4.md`](analysis_1_4.md) 和
[`analysis_5_7.md`](analysis_5_7.md)；完整机器可读结果见
[`../results/summary.json`](../results/summary.json)。
