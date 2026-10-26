# 协议演进与实测数据（2026-10-10 → 10-15）

| 日期 | 变化 | 实测 |
|---|---|---|
| 10-10 | historical_reference 改为视觉记忆题（hrv8）：只用位置指物并起代号，30–60s 后物体离开画面再问外观；专用 judge judge_hr_v8 逐条核对 key_attributes | 50 批原题 0/87 通过人工标准，替换为 40 道新题 |
| 10-10 | proactive_guidance 改为视角引导题（pgv2）：t 时信息拿不到 → 引导方向与真实镜头移动一致 → t2 信息出现主动回答；judge_pg_v2 | 50 批 41 道；纯文字版 eval 显示模型从不引导 |
| 10-10 | eval 改为视觉历史版（历史里带画面帧）；原严格 judge 不变 | — |
| 10-11 | spv2 拉开交互间隔（4 类共用"绑定→改→问"模板）；审阅页诊断分支结构单一 | 334 题，专用 judge 均分 0.8–1.07 |
| 10-11 | 用户认可"每类一个具体例子"的 9 类考点表 → gen_v3 重出 7 类，每类独立分支/槽位结构 + 看帧核验的考点条件 | 50 批 1690 候选通过 779（$5.2）：temporal 39、selective 81、constraint 63、correction 119、resume 160、monitor 157、clarify 181；主线每类取 60 |
| 10-11 | 316 批：ctx316 换数据入口复用同一套生成器；hr 从零出题（种子骨架 + 改写），pg 直接跑 | hr 378 种子 227 通过；pg 186；出题阶段被内容审查拒绝的直接放弃 |
| 10-12 | 316 批 7 类按 v3 重出，select316 每类截 175 | 4348 候选 $13.5 + 补跑 $2.9；通过 temporal 249、selective 188、constraint 278、correction 279、resume 387、monitor 396、clarify 473；final 1575 题 |
| 10-12 | 50 批 v3 eval（7 卡 8–9 小时）；hr 长会话 beam search 预填充 OOM → EVAL_FORCE_LOGITS_TO_KEEP=1 | 专用 judge 均分：temporal 1.38、selective 1.47、constraint 0.75、correction 1.09、resume 1.28、monitor 2.10、clarify 1.07（clarify 的 sl_clarify 0.00：模型从不澄清） |
| 10-13 | 316 批每类抽 50 eval | 均分 0.95–1.42，与 50 批同量级（注意 316 是 640 宽重建 mp4） |
| 10-13 | 每类挑 100 题作为最终内容（select_final100，视频轮转不看分数） | 900 题，50:316 ≈ 15:85 |
| 10-13 | v3b 整改：temporal/selective t0 时目标未出现 → 看三帧审计撤下 + gen_v3 加硬规则与独立 t0_gate 补题；clarify 问句须点出类别 → 改写 + 复核 | 50 批撤 temporal 38 / selective 26 / clarify 23，改写 clarify 117；316 批撤 180 / 145 / 37，改写 290；补题 $6.6、审计 $0.5；补题新窗口 screen 通过率约 10% |
| 10-14 | eval_all：所有未 eval 的题 + 改写题重跑，一题 eval 完立刻 judge；GPU 热改 | 1277 题 / 2554 分支；6 卡约 90 分支/小时 |
| 10-15 | 本目录：整理为可复现 Pipeline | 冒烟：v3 1/1、pg 1/6 候选、hr 1 种子 needs_revision（4 次重写），共 $0.02 |

出题成本经验：gen_v3 约 $0.003–0.01/候选（通过率 20–45%，窗口排名靠后时降到约 10%）；hr 约 $0.01–0.02/种子（通过率约 60%）；pg 约 $0.01/候选（通过率 15–25%）；judge 每题（约 3–4 个槽位 × 2 个 judge）约 $0.01。
