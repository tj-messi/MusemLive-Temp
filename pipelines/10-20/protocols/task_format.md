# 文件格式

## 1. 题目 task.json / candidate_task.json（三条线统一）

```jsonc
{
 "source_id": "D001_K001",                 // 视频片段 id
 "capability": "monitor_cancel",           // 9 类之一
 "interaction_family": "v3_monitor_cancel",
 "task_id": "OP_V3_MONI_D001_K001_00458",  // OP_V3_<CAP4>_<src>_<t0> / OP_HR316_<src>_<t0> / OP_PGV2_<src>_<t>
 "status": "v3_candidate_pending_human_review",
 "session": {"start_s": 456.0, "end_s": 478.0, "observation_hz": 5, "history_policy": "causal_prefix"},  // eval 从 start_s 到 end_s 每 0.2s 一个 tick
 "shared_future_from_s": 458.0,            // 两个分支从这一刻起共享未来画面
 "diagnostic_slot_ids": ["sl_alert"],      // 用来诊断本类考点的槽位
 "branches": [
  {"branch_id": "keep_monitor",
   "user_events": [                        // 用户台词（按时间），state_op 是题目内部的状态标注，不给被测模型
    {"event_id": "ue0", "time_s": 458.0, "text": "帮我留意一下，看到穿红色衣服的人就提醒我。", "state_op": {"op": "bind", "id": "sl_alert", "target": "穿红色衣服的真人"}},
    {"event_id": "ue1", "time_s": 462.0, "text": "好，继续帮我留意。", "state_op": {"op": "keep", "id": "sl_alert", "target": "..."}}],
   "slots": [                              // 评分槽位
    {"slot_id": "sl_alert",
     "trigger": {"kind": "visual_event", "event_id": "ve_sl_alert"},   // 或 {"kind": "user_query", "event_id": "ue2"}
     "window": {"start_s": 475.0, "end_s": 478.0},                    // 评分时间窗（3s）
     "depends_on": ["ue0", "ue1"],
     "expected_action": "SPEAK",            // SPEAK / WAIT / CLARIFY
     "target_text": "看到了，右下角刚进来一位穿红色外套的观众……",    // 自然口语标准回答（WAIT 为空）
     "key_attributes": ["画面中出现一名穿红色上衣/外套的真人", "..."],  // 2–4 条可在帧里核对的要点（专用 judge 逐条核对）
     "forbidden_claims": ["说没看到任何人或还在等待中", "..."]}]},
  {"branch_id": "cancel_monitor", "...": "同一 ue0，ue1 为取消；sl_alert 为 WAIT"}],
 "v3": {"version": "example_refresh_v3_20261011", "model": "qwen/qwen3.8-omni-flash", "segment": [440, 490], "times": {"t0": 458.0, "t1": 462.0, "tt": 475.0}, "plan": "..."},
 "v3b": {"fix": "clarify_names_category", "rewrite": {"ambiguous_ue0": ["旧问句", "新问句"], "specific_ue0": ["旧", "新"]}, "date": "2026-10-13"}   // 仅被后处理改写过的题有
}
```

题型差异：
- historical_reference（gen_hr）：事件 ue_bind / ue_intervention（keep_alias / rebind_alias / cancel_alias）/ ue_query；槽位 sl_recall_appearance，带 `evidence_ids`；cancel 分支 expected_action=CLARIFY 且无 key_attributes；元信息在 `refresh`。
- proactive_guidance（gen_pg）：need_guidance 分支两个槽位 sl_guide（含 `guidance_direction`）与 sl_answer（画面触发，`trigger.evidence_ids`）；no_guidance 分支 sl_direct；元信息在 `pgv2`。
- proactive_clarify：ambiguous 分支 sl_clarify（CLARIFY，key_attributes 写候选）+ sl_answer；specific 分支 sl_direct。
- selective_attention：两分支各有 sl_quiet（WAIT，干扰出现时刻）。

run 目录里还有 `log.json`（候选、screen 结果、每次 attempt 的 author_output / fails / visual_check、status）和每次调用的 `*.response.txt`。

## 2. 题库文件

- `runs/<batch>/final/tasks_<cap>.json`：该批次该题型截取后的题列表；`all_tasks.json` 全部；`summary.json` 每类 passed / dropped_gap / selected / videos / origin。
- `runs/final100/tasks_<cap>.json` 选中、`alternates_<cap>.json` 备选（同一轮转顺序）、`all_tasks.json`、`summary.json`（每类 pool / selected / 批次占比 / judged / judge_mean）、`review/<cap>.md` 与 `review_final100.html` 逐题 QA 清单。

## 3. eval 结果 evaluation_results.json

```jsonc
{"schema_version": "museumlive_minicpm_o45_causal_visual_history_v1", "model": "MiniCPM-o-4_5",
 "visual_history": {"keyframe_s": 2.0, "max_tick_frames": 24, "event_frames": "all"},
 "results": [
  {"branch_key": "<task_id>::<branch_id>", "task_id": "...", "source_id": "...", "capability": "...", "branch_id": "...",
   "session": {...}, "user_events": [...],
   "timeline": [   // 每个 tick 一条，按时间
    {"type": "user_event", "time_s": 458.0, "frame_path": ".../frames/D001_K001_458.000.jpg", "input_mode": "new_user_input", "event_id": "ue0", "utterance": "帮我留意……", "response": "好的……"},
    {"type": "observation_tick", "time_s": 458.2, "frame_path": "...", "input_mode": "no_new_user_input", "utterance": "【当前没有新的用户话语】", "response": "<SILENCE>"}],
   "slots": [{"slot": {...题目里的槽位...}, "timeline_prefix": [...该槽位窗口结束前的全部 tick...]}],
   "causal_audit": {"independent_branch": true, "future_frames_sent": false, "future_events_sent": false, "no_input_explicit": true, "observation_hz": 5, "visual_history": true}}]}
```

`evaluation_progress.json`：status running/complete、done_branches/total_branches。

## 4. judge 结果 judge_results.json

```jsonc
{"schema_version": "museumlive_10_5_qwen_strict_causal_judge_v1", "model": "qwen/qwen3.8-omni-flash",
 "results": [{"judge_id": "<task_id>::<branch_id>::<slot_id>", "status": "ok",
              "judgment": {"score": 2, "reason": "……", "decision_correct": true, "timing_correct": false, "memory_correct": true, "needs_review": false,
                           "attribute_hits": [{"attribute": "...", "status": "hit|miss|wrong"}],   // 专用 judge
                           "guided": true}}]}                                                       // judge_pg_v2
```

`summary.json`：total / completed / api_failed / mean_score。eval_all 的汇总：`judge/`（严格）与 `by_cap/<cap>/judge_special/`（专用），`per_task/<task_id>/` 下是按题的原始输出。
