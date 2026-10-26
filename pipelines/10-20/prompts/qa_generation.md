# 10-15 Pipeline 全部 Prompt（由 docs/dump_prompts.py 从代码导出）
模型统一为 `qwen/qwen3.8-omni-flash`（OpenRouter，temperature 0.3、reasoning off、JSON 输出；judge 为 temperature 0）。`{...}` 是运行时填充的占位符。所有看帧的 prompt 都把真实画面帧以 `[组名 帧 12.0s]` + 图片 的形式跟在文字后面。
## 1. 7 类题型出题线 gen_v3.py
### 1.1 各题型的考点规则（SPECS[cap]["rule"]，同时注入 screen / author / verify 三个 prompt）
#### temporal_update（family v3_state_change）

```text
考点：同一个对象的状态在会话中发生了变化，模型要回答**最新状态**，不能沿用旧记忆。例子：讲解员先在左边讲青铜鼎，20 秒后走到右边指着另一件，用户问"她现在在讲哪件？"。设计：t0 用户让模型留意对象 X（人/动物/可动的展项/屏幕内容等会变化的对象）；t1 时 X 处于状态 S1；t2 时 X 变成明显不同的状态 S2（位置、动作、正在看/讲的展品、屏幕画面等），X 在 t1 和 t2 都在画面里。**硬性要求：用户在 t0 开口时 X 必须已经清楚地在画面里（t0 前 1 秒就已可见、可辨认），不能选尚未出现、刚入画一角或看不清的对象；如果对象是稍后才出现，就把 t0 往后挪到它清楚可见之后。**分支 ask_twice：t1 和 t2 各问一次同一句"它/他/她现在在干嘛/怎么样了？"，t1 答 S1，t2 答 S2（forbidden 写 S1）；分支 ask_late：只在 t2 问同一句，答 S2。
```
候选窗口关键词：`讲解|观众|游客|孩子|男子|女子|人员|走|转身|指向|举起|抬|蹲|坐|拿|屏幕|画面切换|播放`
时刻：t0、t1、t2；间隔：t1−t0 在 3–20s；t2−t1 在 8–30s；t2−t0 在 12–45s
分支/槽位骨架：
```text
- 分支 ask_twice：事件 ue0@t0=<t0>s op=bind，ue1@t1=<t1>s op=query，ue2@t2=<t2>s op=query；槽位 sl_early@t1=<t1>s SPEAK，sl_late@t2=<t2>s SPEAK
- 分支 ask_late：事件 ue0@t0=<t0>s op=bind，ue2@t2=<t2>s op=query；槽位 sl_late@t2=<t2>s SPEAK
```
文本约束：两个分支的 ue0、ue2 文本完全相同；ask_twice 的 ue1 与 ask_twice 的 ue2 文本完全相同；ask_twice/sl_early 与 ask_twice/sl_late 的答案必须实质不同
看帧核验的考点条件（checks）：
- `visible_at_t0`：X 在 t0 组帧里（包括 t0−1s 那一帧）已经清楚可见、可辨认，不是尚未出现、刚露出一角或模糊不清
- `same_target`：X 在 t1 组和 t2 组帧里都可见，而且是同一个对象（不是换了一个长得像的）
- `state_changed`：X 在 t2 的状态与 t1 的状态确实明显不同（不是镜头角度变化造成的错觉）
#### selective_attention（family v3_distractor_filter）

```text
考点：有显眼干扰时只报告用户指定的目标。例子："只告诉我那个穿红衣服的小孩在干嘛"，期间旅行团走过、讲解员大声说话，回答不能提这些。设计：t0 用户说"接下来只跟我说 X 的情况，别的不用管"。X 必须是 t0 到 t2 一直在画面里的对象：可以是人、一群人、屏幕/投影内容，也可以是镜头一直对着的展品/展柜；**硬性要求：用户在 t0 开口时 X 必须已经清楚地在画面里（t0 前 1 秒就已可见、可辨认），不能选尚未出现、刚入画一角或看不清的对象；对象稍后才出现的话就把 t0 往后挪。**td 时画面里出现与 X 无关的显眼干扰 Z（有人走入或穿过画面、旁边观众的明显动作、新物体进入画面等），t2 用户提问。分支 focus：t2 问 X 现在的情况（人："他现在在干嘛？"；展品/屏幕："它现在怎么样/上面显示什么？"），只答 X 在 t2 的情况，forbidden 写 Z 的情况；分支 ask_distractor：t2 用户主动问 Z（"刚才从旁边过去的那些人是干嘛的？"），此时应答 Z。两个分支在 td 都有一个 WAIT 槽位：干扰出现时不要主动播报。
```
候选窗口关键词：`走入|走进|经过|穿过|走过|进入画面|入画|人群|一群|旅行团|多名|讲解员|孩子|游客`
时刻：t0、td、t2；间隔：td−t0 在 3–20s；t2−td 在 2–15s；t2−t0 在 8–30s
分支/槽位骨架：
```text
- 分支 focus：事件 ue0@t0=<t0>s op=bind，ue1@t2=<t2>s op=query；槽位 sl_quiet@td=<td>s WAIT，sl_main@t2=<t2>s SPEAK
- 分支 ask_distractor：事件 ue0@t0=<t0>s op=bind，ue1@t2=<t2>s op=query_distractor；槽位 sl_quiet@td=<td>s WAIT，sl_main@t2=<t2>s SPEAK
```
文本约束：两个分支的 ue0 文本完全相同；两个分支的 ue1 文本必须不同；focus/sl_main 与 ask_distractor/sl_main 的答案必须实质不同
看帧核验的考点条件（checks）：
- `visible_at_t0`：X 在 t0 组帧里（包括 t0−1s 那一帧）已经清楚可见、可辨认，不是尚未出现、刚露出一角或模糊不清
- `distractor_salient`：td 组帧里确实出现了显眼、与 X 无关的干扰 Z（t0 组帧里还没有）
- `target_visible`：X 在 t2 组帧里可见，focus 分支的答案说的是 X 而不是 Z
#### constraint_update（family v3_constraint_update）

```text
考点：同一个场景/同一个信息源，换了筛选条件，答案就跟着变。例子："只读展签上的英文" → "改成只告诉我年代"，同一块展签给出不同答案。设计：t0 用户对画面里同一个信息源（展签、展板、屏幕、一组展品、一群人）设定筛选条件 C1；t1 分支 keep 确认 C1，分支 update 改成 C2；t2 问"上面写了什么/现在符合要求的是什么？"。C1 和 C2 作用在**同一个信息源**上，在 t2 帧里都能给出**非空、具体**的答案（不能是"没有/找不到/没列出"），两个答案不同、可核对；forbidden 写按另一个条件会给出的答案。
```
候选窗口关键词：`展签|说明牌|展板|文字|字样|英文|标题|屏幕|标牌|题签`，OCR 加权 3
时刻：t0、t1、t2；间隔：t1−t0 在 4–20s；t2−t1 在 6–25s；t2−t0 在 12–40s
分支/槽位骨架：
```text
- 分支 keep_constraint：事件 ue0@t0=<t0>s op=set_constraint，ue1@t1=<t1>s op=keep_constraint，ue2@t2=<t2>s op=query；槽位 sl_main@t2=<t2>s SPEAK
- 分支 update_constraint：事件 ue0@t0=<t0>s op=set_constraint，ue1@t1=<t1>s op=update_constraint，ue2@t2=<t2>s op=query；槽位 sl_main@t2=<t2>s SPEAK
```
文本约束：两个分支的 ue0、ue2 文本完全相同；两个分支的 ue1 文本必须不同；keep_constraint/sl_main 与 update_constraint/sl_main 的答案必须实质不同
看帧核验的考点条件（checks）：
- `same_source`：两个约束作用在 t2 帧里同一个信息源上（同一块展签/展板/同一组对象）
- `constraints_differ`：按 C1 和 C2 在 t2 帧里确实得到不同的答案，且两者都是非空、具体、能从帧里读出/看出的内容（任一答案是"没有/找不到"即为 false）
#### context_correction（family v3_user_self_correction）

```text
考点：用户自己先说错了，随后根据画面证据更正，模型要放弃旧认知、按更正后的信息回答。例子："这是唐代的吧"……"哦不对，展签写的是宋代"，之后问相关问题要按宋代回答。设计：t0 用户对画面里的对象说了一个**错误**判断 W（朝代、名称、作者、哪一件是某物、颜色材质等），画面里有证据（展签文字、近景）能看出正确信息 T；分支 correction：t1 用户自己更正为 T（"哦不对，牌子上写的是……"），t2 用户问一个答案取决于该判断的问题，模型按 T 回答，forbidden 写沿用 W 的说法；分支 no_correction：t1 用户说一句无关的话（不更正），t2 问同一句，模型应指出画面证据显示是 T、不是 W（主动纠正用户），forbidden 写附和 W。t2 的答案必须能在 t1/t2 帧里核对（例如展签上与 T 相关的其他文字、该对象的外观）。
```
候选窗口关键词：`展签|说明牌|题签|文字|字样|标题|名称|年代|朝代|作者|代|窑`，OCR 加权 3
时刻：t0、t1、t2；间隔：t1−t0 在 4–20s；t2−t1 在 6–25s；t2−t0 在 12–40s
分支/槽位骨架：
```text
- 分支 correction：事件 ue0@t0=<t0>s op=claim，ue1@t1=<t1>s op=correct，ue2@t2=<t2>s op=query；槽位 sl_main@t2=<t2>s SPEAK
- 分支 no_correction：事件 ue0@t0=<t0>s op=claim，ue1@t1=<t1>s op=chat，ue2@t2=<t2>s op=query；槽位 sl_main@t2=<t2>s SPEAK
```
文本约束：两个分支的 ue0、ue2 文本完全相同；两个分支的 ue1 文本必须不同
看帧核验的考点条件（checks）：
- `claim_wrong`：t0 台词里用户的判断 W 与画面证据不符，画面证据（展签/近景）显示的是 T
- `evidence_visible`：证据 T 在 t1 或 t2 组帧里清楚可读/可见
- `branches_behave`：correction 分支答案按 T 回答且不提 W；no_correction 分支答案明确指出是 T 不是 W
#### task_resume（family v3_resume_progress）

```text
考点：被打断的多步（累积型）任务，恢复后要接着原来的进度做。例子："帮我数这面墙有几幅画" → 插问别的 → "继续数，刚才数到几了？"。设计：t0 用户布置一个随镜头移动逐步累积的任务（数经过的画/展柜/雕像，按顺序记下看到的展签标题，比较先后两件展品）；t1 用户插问一个无关的、t1 画面里能直接回答的问题；t2 用户恢复。分支 resume：t2 说"好，继续刚才的——到现在一共几个/都有哪些？"，答案要把 t0–t1 的进度和 t1–t2 新看到的合起来；分支 restart：t1 插问同一句后再加一句"刚才那个先不算了"，t2 说"从现在开始重新数/记，画面里现在有几个/有哪些？"，答案只算 t2 当前画面。两个分支的 t2 答案必须不同。resume 的答案必须**同时包含插问前（t0–t1）看到的和插问后（t1–t2）新看到的**条目，插问后没有新条目的时刻不能选；restart 必须重新做**同一个**任务（任务是记名字就重新记名字，是数个数就重新数），t2 句不点名对象。数量要能从给你的帧里数清；数不清就换成"记下展签标题/展品名称"之类的任务。
```
候选窗口关键词：`一排|多幅|多件|依次|并排|陈列|若干|几件|两件|三件|一组|成排|多个|多尊|平移`
时刻：t0、t1、t2；间隔：t1−t0 在 6–25s；t2−t1 在 5–20s；t2−t0 在 15–45s
分支/槽位骨架：
```text
- 分支 resume：事件 ue0@t0=<t0>s op=bind_task，ue1@t1=<t1>s op=side_question，ue2@t2=<t2>s op=resume；槽位 sl_side@t1=<t1>s SPEAK，sl_main@t2=<t2>s SPEAK
- 分支 restart：事件 ue0@t0=<t0>s op=bind_task，ue1@t1=<t1>s op=side_question_and_cancel，ue2@t2=<t2>s op=restart；槽位 sl_side@t1=<t1>s SPEAK，sl_main@t2=<t2>s SPEAK
```
文本约束：两个分支的 ue0 文本完全相同；两个分支的 ue2 文本必须不同；resume/sl_main 与 restart/sl_main 的答案必须实质不同
看帧核验的考点条件（checks）：
- `progress_correct`：resume 分支答案与 t0 到 t2 的帧里累积看到的内容一致（数量/列表正确）
- `progress_spans`：resume 答案里既有 t1 之前看到的条目，也有 t1 之后才新出现的条目（插问后任务确实有进展）
- `restart_correct`：restart 分支答案只对应 t2 当前画面，且与 resume 不同
- `side_ok`：插问在 t1 画面里能回答，和累积任务无关
#### monitor_cancel（family v3_monitor_cancel）

```text
考点：用户让模型盯着某个条件、出现时提醒；取消后即使条件出现也要保持沉默。例子："看到出口标志提醒我" → "算了不用了" → 标志出现，不能说话。设计：t0 用户说"看到 X 就提醒我"（X 在 t0 到 tt 之前**都不在画面里**，tt 时第一次清楚出现：标志、某类展品、某个人走入、某段文字等）；t1 分支 keep 确认继续盯（"好，继续帮我留意"），分支 cancel 取消（"算了，不用提醒了"）；tt 时 keep 分支应主动提醒（SPEAK，说清看到了什么、在哪），cancel 分支保持沉默（WAIT）。
```
候选窗口关键词：`进入画面|入画|出现|走入|走进|露出|显现|映入|开始出现`
时刻：t0、t1、tt；间隔：t1−t0 在 3–15s；tt−t1 在 6–30s；tt−t0 在 12–45s
分支/槽位骨架：
```text
- 分支 keep_monitor：事件 ue0@t0=<t0>s op=bind，ue1@t1=<t1>s op=keep；槽位 sl_alert@tt=<tt>s SPEAK
- 分支 cancel_monitor：事件 ue0@t0=<t0>s op=bind，ue1@t1=<t1>s op=cancel；槽位 sl_alert@tt=<tt>s WAIT
```
文本约束：两个分支的 ue0 文本完全相同；两个分支的 ue1 文本必须不同
看帧核验的考点条件（checks）：
- `absent_before`：X 在 tt 之前的所有帧（A、B、M 组）里都看不到
- `appears_at_tt`：X 在 tt 组帧里清楚出现，keep 分支提醒的内容与画面一致
#### proactive_clarify（family v3_ambiguity_clarify）

```text
考点：用户的指代有歧义时，模型先问清楚，不要猜。例子：画面里有三个花瓶，用户问"这个是什么年代的？"，应该先问是哪一个。设计：t0 画面里同时可见 ≥2 个同类候选（花瓶、画、雕像、展签、人…），彼此能用位置/外观区分；分支 ambiguous：t0 用户问一个关于其中一件的问题（答案能从画面看出：颜色、图案、上面的字、在做什么），模型应澄清（CLARIFY，列出候选让用户选），**硬性要求：ambiguous 分支的问句必须点出物品的类别名称（如"这幅画是什么年代的""这里面的服饰是什么颜色的""这两个陶瓷哪个更大"），歧义只能出在"指的是哪一个"上；不能只说"这个是什么颜色的""这是什么"这种连物品是什么都不说的问法。**t1 用户说明是哪一个（"左边那个"），模型回答；分支 specific：t0 用户直接说清是哪一个再问同样的问题，模型应直接回答（SPEAK，不要多余澄清）。两个分支最终的答案是同一个对象的同一个信息。
```
候选窗口关键词：`两件|三件|两个|三个|两尊|三尊|一排|并排|多件|几件|两幅|三幅|两块|多个|成对|一对|并列`
时刻：t0、t1；间隔：t1−t0 在 3–12s
分支/槽位骨架：
```text
- 分支 ambiguous：事件 ue0@t0=<t0>s op=ambiguous_query，ue1@t1=<t1>s op=clarify_reply；槽位 sl_clarify@t0=<t0>s CLARIFY，sl_answer@t1=<t1>s SPEAK
- 分支 specific：事件 ue0@t0=<t0>s op=specific_query；槽位 sl_direct@t0=<t0>s SPEAK
```
文本约束：两个分支的 ue0 文本必须不同
看帧核验的考点条件（checks）：
- `multiple_candidates`：t0 组帧里确实同时可见 ≥2 个同类候选，ambiguous 分支的问句在当时确实指代不明
- `names_category`：ambiguous 分支的问句点出了物品的类别名称（画/佛像/服饰/陶瓷/展签…），而不是只用"这个/那个/这是"指物
- `answer_correct`：sl_answer 与 sl_direct 回答的是同一个对象，内容在帧里成立

### 1.2 阶段 prompt
### screen（只看逐秒摘要 + OCR 选时刻；时刻不满足间隔时带原因重选，最多 3 次）

```text
你在为第一视角博物馆流式交互 benchmark 挑选出题时刻（题型 {cap}）。给你一段 {win} 秒视频的逐秒摘要和 OCR。
题型要求：{rule}
需要选出这些时刻（秒，可带一位小数）：{times}；间隔要求：{cons}。所用对象在相应时刻必须在画面中（看摘要确认）。
只有当这段视频真的能按题型设计出题时才 ok=true；否则 ok=false 并说明原因。
只输出 JSON：{{"ok": bool, "reason": "...", "times": {{{time_keys}}}, "plan": "一句话设计，点名用到的对象"}}
```

### author（看各时刻真实帧出题；失败时把累积的 fails 追加到 user 消息末尾重写，最多 3 次）

```text
你在为第一视角博物馆流式交互 benchmark 出题（题型 {cap}）。用户戴着摄像头参观，模型实时看画面、记住对话状态。
题型要求：{rule}
已选定时刻：{times}。按下面的骨架写题，事件和槽位的 id、时刻、op、动作都不能改：
{skeleton}
文本要求：{text_rules}
每个 SPEAK 槽位写 target_text（该时刻的自然口语标准回答）、key_facts 2–4 条（每条一个可在该时刻帧里核对的短要点）、forbidden_claims 2–3 条；
CLARIFY 槽位写 target_text（澄清句，点出候选），key_facts 写 1–2 条（候选是什么），forbidden_claims 写直接猜一个答案；WAIT 槽位 target_text 为空，forbidden_claims 写不该说的话。
台词只用位置/指示词/外观最少的说法指物体，不能提前说出评分时刻的答案；看不清的不要写。做不出来就 feasible=false 并说明原因。
只输出 JSON：{{"feasible": bool, "infeasible_reason": "",
 "branches": [{{"branch_id": "...", "user_events": [{{"event_id": "...", "time_s": 数字, "text": "...", "state_op": {{"op": "...", "id": "...", "target": "内部描述"}}}}],
   "slots": [{{"slot_id": "...", "target_text": "...", "key_facts": ["..."], "forbidden_claims": ["..."]}}]}}]}}
```

### leak（台词泄露检查，common.LEAK_SYSTEM_FACTS）

```text
判断用户台词有没有提前透露评分时刻的答案。给你若干句用户台词（带时间）和"评分时刻答案要点"。
只有当某句台词**提前说出了评分时刻才成立的答案内容**（或其同义说法），才算泄露。
台词描述的是更早时刻的旧状态、且与答案不同（例如先说"有三只"，答案是"现在有七只"），不算泄露；这正是状态更新题需要的对照。
用户用位置/指示词指物体、或给出指令/纠正本身（纠正内容就是考点时，答案复述纠正后的信息不算泄露），不算泄露。
只输出 JSON：{"leaks": [{"line": "...", "phrase": "...", "matches_attribute": "..."}]}，没有泄露就输出 {"leaks": []}。
```

### verify（看帧核验：台词成立 / 要点成立 / 本类考点条件）

```text
你是严格的视觉核验员。给你一道题（题型 {cap}）和真实帧，帧按时刻分组标注（组名就是时刻名，M 是中间时刻）。题型要求：{rule}
逐项判断：
1. lines_ok：每句用户台词在它说出的时刻画面里成立（所指对象可见、指代能对上）；
2. 每个槽位的 key_facts 在该槽位时刻的帧里是否成立（逐条）；
3. 题型考点条件（逐条）：{checks}
只输出 JSON：{{"lines_ok": bool, "lines_note": "...", "facts": [{{"branch_id": "...", "slot_id": "...", "fact": "...", "holds": bool, "why": "..."}}],
"checks": {{{check_keys}}}}}
```

### t0_gate（10-13 v3b：temporal_update / selective_attention 的独立闸门，verify 通过后再查 t0−1s、t0、t0+1s 三帧）

```text
你是严格的视觉核验员。第一视角博物馆视频里，用户在 t0 时刻对模型说了一句话，让模型留意/只关注某个对象 X。
给你 t0−1s、t0、t0+1s 的真实帧。判断：用户在 t0 开口时，X 是否**已经清楚地出现在画面里**（在 t0−1s 和 t0 的帧里都能看到并辨认出它，不是尚未出现、刚入画一角、被遮挡或模糊到无法确认）。
台词里的描述（位置、衣着、外观）必须能在 t0 帧里对上。只输出 JSON：{"visible_at_t0": bool, "note": "一句话说明在哪帧看到/没看到什么"}
```

代码检查（det_check，不调模型）：feasible；proactive_clarify 的 ue0 不得匹配 `^\s*(这|那)(是|个|些|块|件|边|里)\s*(是|有|什么|啥|叫|在|用|几|多|长|属于|放|来自|做|为|的|怎么|哪)`；时刻间隔；分支/事件/槽位 id、时刻、op 与骨架一致；SPEAK 的 key_facts 2–4 条；SPEAK/CLARIFY 有 target_text；台词无“/”模板残留；same_text / diff_text / same_text_pairs；distinct 对的答案 2-gram Jaccard < 0.5。

## 2. 历史指代（视觉记忆）出题线 gen_hr.py
### author（{min_gap}/{max_gap} 默认 30/60）

```text
你在为第一视角博物馆流式交互 benchmark 改写"历史指代 / 视觉记忆"题。
目标：考模型能否记住并复述**过去看到的物体外观细节**，而不是复述名字或判断是否还在画面里。

硬性要求：
1. 绑定台词只能用位置、指示词或画面关系来指物体（如"中间柱子后面那件""左边第二个展柜里那个"），再给一个与外观无关的代号（如 A1、hidden_furniture、"那件东西"）。
   绑定台词和**所有干预台词**里都不得出现任何将要考的属性：颜色、材质、形状、物体类别名（床/镜子/雕像…）、上面的文字或图案。
   原题的干预台词通常已经泄露了答案（如"就是那张带帷幔的床"），必须重写，例如 keep："对，就是它，以后就叫它 X。"；rebind："不对，换成壁炉上方挂着的那个，以后它叫 X。"
2. 提问时间 query_time_s 必须满足：距离每个分支的指代对象最后一次被确立（keep 分支是 bind，rebind 分支是 rebind 那句）在 {min_gap}–{max_gap} 秒之间；
   并且根据逐秒摘要，提问时刻和此前 4 秒内该物体都已不在画面中（物体离开画面越早越好，早就离开了几十秒完全没问题）。两个分支的提问时间和问句完全相同。
3. 问句形如"<代号>是个什么？描述一下它的样子。"，不得给出任何外观提示。
4. 每个 SPEAK 槽位给 category（物体是什么，如"四柱床"）和 2–4 条 key_attributes，每条只写**一个**短属性（≤10 字，如"白色帷幔""盔顶鹰形装饰"），不要把多个细节塞进一条，也不要写数量、"垂下"这类难核对的细节，并给出依据的 window_id（只能用绑定/重绑时刻附近、你看过的标注窗口）。
   属性只能是外观（颜色、形状、部件、图案、文字内容），不能是位置（"位于壁炉上方"）——位置是用户指认时说的，不算记忆。
   属性必须能从给你的绑定时刻画面帧里清楚看到（画面昏暗、被遮挡、太远看不清的不要写；核验员会看同样的帧逐条核对），不得推测材质年代等看不出的东西；画面看不清的属性不要写。
5. target_text 是自然口语回答：先说是什么，再描述这些属性。forbidden_claims 写 2–3 条干扰描述：另一分支对象的真实属性、提问时刻画面里物体的属性。
6. 分支结构保持原题（keep/rebind/cancel 等 branch_id 不变）。cancel 分支：用户已取消代号，如果问句对指代仍不确定就 CLARIFY，target_text 写澄清句，不需要 key_attributes。
   rebind 分支的新对象必须是当时真实可见、可用位置唯一指出的另一个物体，同样只用位置描述；rebind 后代号指代是明确的，所以 rebind 分支必须 SPEAK，描述新对象的外观。
7. 镜头转去看别的展品之后再问以前的物体，正是这类题要考的，不能以"语境已切换"为由拒绝；物体在提问前已离开画面 30 秒以上也完全没问题，不算"超出范围"。
   指代对象可以是展品、展牌/说明牌（上面能读清的标题文字算外观属性）、也可以是画面里的人（衣着、手持物算外观属性），不能因为对象是人或展牌就拒绝。
   台词里可以用"那个""那件""那块""那位"这类量词指代，不必回避量词本身。
   如果在视频剩余部分找不到满足第 2 条的提问时刻，或物体外观在画面里看不清，就输出 feasible=false 并说明原因，不要硬凑。

只输出一个 JSON 对象：
{"feasible": true/false, "infeasible_reason": "",
 "bind": {"time_s": 数字, "text": "..."},
 "query": {"time_s": 数字, "text": "..."},
 "query_time_rationale": "依据哪些窗口摘要判断物体已离开画面",
 "branches": [{"branch_id": "...",
   "intervention": {"time_s": 数字, "text": "...", "state_op": {"op": "keep_alias|rebind_alias|cancel_alias", "id": "代号", "target": "指代对象的内部描述(可含外观，不给模型看)"}},
   "slot": {"expected_action": "SPEAK|CLARIFY", "category": "...", "target_text": "...",
            "key_attributes": [{"attribute": "...", "evidence_window_id": "..."}],
            "forbidden_claims": ["..."]}}],
 "bind_state_target": "绑定对象的内部描述(可含外观)"}
```

### leak（外观泄露检查）

```text
判断用户台词有没有透露物体外观。给你若干句用户台词和一个"答案属性清单"。
台词只允许用位置、指示词、代号来指物体；如果台词里出现了物体类别（如"床""盔甲""板状展品"）、颜色、材质、形状、纹样、上面的文字，
或与清单中任何属性同义/近义的描述（如"刻线"与"凹陷线条"、"孔洞"与"圆形大孔"），都算泄露。
只看**被指代物体本身**：用来指路的其他物体（如"黑柱子后面那件"里的黑柱子、"壁炉上方挂着的那个"里的壁炉）可以带外观，不算泄露。
"立着的""那件东西""家具""展品"这种很笼统、不对应清单属性的说法不算。
只输出 JSON：{"leaks": [{"line": "...", "phrase": "...", "matches_attribute": "..."}]}，没有泄露就输出 {"leaks": []}。
```

### verify（A 组 = 指认时刻帧 t、t+0.5、t+1；B 组 = 提问前 4s 到提问时刻共 5 帧）

```text
你是严格的视觉核验员。给你两组真实视频帧：
A 组：某物体被用户指认时的帧；B 组：后来用户提问时刻及之前 4 秒的帧。
对给定的物体描述和属性清单，逐条判断：
- 在 A 组帧里能否确认该物体，以及每条属性是否在 A 组帧里清楚可见（看不清、需要推测的一律 false）；
- 该物体在 B 组任意一帧里是否仍然可见（部分可见也算可见）。
只输出 JSON：{"referent_found_in_A": bool, "attributes": [{"attribute": "...", "supported": bool, "why": "..."}],
"visible_in_B": bool, "B_note": "..."}
```

代码检查（deterministic_check）：问句含外观词（`长什么样|样子|外观|模样|颜色|材质|描述|什么样|形状|图案|刻着|写着|写的|装饰`）；提问时刻 +3s 不超时长；分支 id 与种子一致；rebind 分支必须 SPEAK；bind < 干预 < 提问；指代确立到提问在 min–max gap；key_attributes 2–5 条；category 的 2-gram 不得出现在台词；属性不能是位置（`^(位于|在.*(上方|下方|旁边|左边|右边|左侧|右侧|后面|前面)|挂在|放在|摆在)`）；标准答案与台词 2-gram 重合 < 25%。

种子骨架（make_seeds）：每 45s 取一个有物体的秒，40s 后摘要与当下 2-gram 重合 < 35%；每视频先保证 1 个、最多 3 个；分支为 keep_alias + rebind（偶数种子）或 keep_alias + cancel（奇数种子）。

## 3. 主动引导（视角）出题线 gen_pg.py
候选（candidates，不调模型）：逐秒摘要在 t 秒匹配 BAD `看不清|无法辨认|不可读|难以辨认|模糊|被截断|截断|只露出|露出一角|部分遮挡|被遮挡|反光|过远|太远|画面边缘|半露` 且 TEXTY `文字|字样|字迹|牌|标签|展签|说明|标题|OCR`；t+3..t+15 内某秒匹配 READABLE `清晰可读|可以读出|可辨认|写着|字样|标题为|内容为|“` 且 TEXTY 且不匹配 BAD；同一视频候选间隔 ≥ 30s。

### screen（只看 t−2..t2+2 的逐秒摘要）

```text
你在为第一视角博物馆视频挑选"主动引导"出题时刻。给你一段逐秒摘要。
判断是否存在这样的时刻：在 t 秒，某个具体物体上的某条信息（展签/牌子上的文字、物体某一面的细节、被遮挡部分等）在画面里拿不到；
之后 3–20 秒内（t2），镜头**因为拍摄者的移动**（平移、靠近、绕行、抬高等）让同一物体的这条信息变得清楚可得。
必须是同一个物体；仅仅是换了一个场景不算。
只输出 JSON：{"ok": bool, "object": "...", "info": "拿不到的是什么信息", "t": 数字, "t2": 数字,
"camera_motion": "t 到 t2 镜头怎么动（如向右平移并靠近）", "reason": "..."}
```

### author（帧：t−1、t、t+0.5、t2、t2+0.5、t2+1）

```text
你在为第一视角博物馆流式交互 benchmark 出"主动引导视角"题。用户戴着摄像头参观，模型实时看画面。
题目考：用户问的信息当前画面里拿不到时，模型能否**主动、具体地引导用户调整视角**（往哪个方向、对准什么），并在视角调整后信息出现时主动给出答案。

给你：t 前后和 t2 前后的真实帧、逐秒标注（含 OCR）、初判结论。写两个分支，提问时间都是 t：
分支 need_guidance：
  - 用户在 t 问关于该物体的问题（自然口语，例如"这块牌子上写的是哪个朝代？"），问句不能包含答案；
  - 槽位 guide（窗口 t 到 t+3）：SPEAK，说明为什么现在看不到，并给出具体引导，方向必须与视频里 t→t2 镜头的真实移动一致（如"牌子右半边在画面外，往右挪一点再靠近些"）；
  - 槽位 answer（窗口 t2 到 t2+3，由画面触发，无新的用户话语）：SPEAK，信息出现后主动给出答案，答案必须是 t2 帧里能清楚看到/读到的内容，不得编造。
    问题要问 t2 帧里**确实会出现**的信息；如果 t2 帧里还是读不清或找不到，就 feasible=false，不要写"没找到"之类的答案。
分支 no_guidance：
  - 用户在 t 问另一个**当前画面里已经看得清**的物体（同一画面中的其他物体），问句不含答案；
  - 槽位 direct（窗口 t 到 t+3）：SPEAK，直接回答，不需要也不应该引导用户移动。
硬性要求：答案（包括引导里的方向）都要能从给你的帧里核对；看不清的不要写；OCR 文字只写你在帧里确实读得清的部分。
只输出 JSON：
{"feasible": bool, "infeasible_reason": "",
 "t": 数字, "t2": 数字, "guidance_direction": "如 右/靠近/右+靠近/上/绕到左侧",
 "need_guidance": {"query": "...", "guide_text": "...", "answer_text": "...", "answer_key_facts": ["可核对的答案要点1", "..."],
                   "forbidden_claims": ["编造的未读清内容", "错误方向"]},
 "no_guidance": {"query": "...", "control_object": "...", "direct_text": "...", "answer_key_facts": ["..."],
                 "forbidden_claims": ["要求用户移动视角"]}}
```

### verify（P 组 = t−1、t；Q 组 = t2、t2+0.5、t2+1）

```text
你是严格的视觉核验员。给你三组真实帧：
P 组：用户提问时刻 t 及之前 1 秒；Q 组：后来时刻 t2 附近；并给出题目信息。逐项判断：
1. info_hidden_at_t：在 P 组帧里，目标信息（answer_key_facts）是否确实看不到/读不清（看得清就是 false）；
2. answer_visible_at_t2：在 Q 组帧里，answer_key_facts 是否每条都能清楚看到/读到，逐条给出；
3. direction_matches：从 P 到 Q，镜头/拍摄者的实际移动是否与 guidance_direction 一致；
4. control_visible_at_t：对照物体及其答案要点在 P 组帧里是否清楚可见；
5. answers_question：Q 组帧里出现的信息是否真正回答了用户的问题（只是看到物体其他部分、没有出现被问的信息，则为 false）。
只输出 JSON：{"info_hidden_at_t": bool, "answer_facts": [{"fact": "...", "visible": bool, "why": "..."}],
"direction_matches": bool, "actual_motion": "...", "control_visible_at_t": bool, "control_note": "...", "same_object": bool,
"answers_question": bool}
```

代码检查（det_check）：3 ≤ t2−t ≤ 20；t2+3 不超时长；answer_text 不含 NEGATIVE `没有找到|没找到|看不清|无法(辨认|看清|读出|确认)|看不到|不清楚|并没有`；guide_text 含 DIRECTION `往|向|靠近|凑近|走近|左|右|上|下|侧面|背面|绕|蹲|抬|拉近|退后|后退|换个?角度|避开`；问句不含答案要点（2-gram ≥ 50% 即泄露）；no_guidance 的回答不含移动引导；answer_key_facts 非空。

## 4. 出题后的题型后处理 postprocess_v3b.py（10-13）
### VIS_SYSTEM（temporal_update / selective_attention：t0 开口时对象是否已清楚在画面；不成立的题撤下）

```text
你是严格的视觉核验员。第一视角博物馆视频里，用户在 t0 时刻对模型说了一句话，让模型留意/只关注某个对象 X。
给你 t0−1s、t0、t0+1s 的真实帧。判断：用户在 t0 开口时，X 是否**已经清楚地出现在画面里**（在 t0−1s 和 t0 的帧里都能看到并辨认出它，不是尚未出现、刚入画一角、被遮挡或模糊到无法确认）。
台词里的描述（位置、衣着、外观）必须能在 t0 帧里对上。只输出 JSON：{"visible_at_t0": bool, "note": "一句话说明在哪帧看到/没看到什么"}
```

### CLR_SYSTEM（proactive_clarify：问句须点出物品类别；不合格的由模型看 t0 帧改写两个分支的 ue0，再用同一 prompt 复核，改不好的撤下）

```text
你在审核第一视角博物馆流式交互 benchmark 的"主动澄清"题。规则：ambiguous 分支的问句必须**点出物品的类别名称**（如"这幅画是什么年代的""这里面的服饰是什么颜色的""这两个陶瓷哪个更大"），歧义只能出在"指的是哪一个"上；不能只说"这个是什么颜色的""这是什么"这类连物品是什么都没说的问法。specific 分支的问句要点明是哪一个并且也点出类别。
给你题目（两个分支的台词、澄清句、答案要点）和 t0 的真实帧。
1. 判断 ambiguous 分支 ue0 是否已经点出类别名称（names_category）。
2. 如果没有，改写两句 ue0：ambiguous 的新问句用"这个/那个/这件 + 类别词"的单数形式（如"这个陶碗是什么颜色的？""这幅画是什么年代的？""这里面的服饰是什么颜色的？"），点出类别但不说明是哪一个——画面里有 ≥2 个该类候选，所以模型仍需澄清；**不要**改成"这几件…分别…""这些…都…"这种问全部的说法。specific 的新问句点出类别并说清是哪一个（原句已经合格就原样保留）；问的内容（颜色/年代/文字/动作…）保持不变，不能透露答案，类别词要和画面里的东西相符、口语自然。
只输出 JSON：{"names_category": bool, "note": "...", "ambiguous_text": "新问句或原句", "specific_text": "新问句或原句"}
```

改写的代码闸门：新问句非空、不匹配 VAGUE_Q、与原句不同、不含“分别/都是/都有/各是/各自”。

## 5. eval（MiniCPM-o 4.5 视觉历史版 causal eval）eval_visual_history.py
### system

```text
你是博物馆实时讲解员和主动视觉助手。严格因果，只使用当前这一帧与同一分支此前历史（历史中附带的图片是当时实际看到的画面），绝不使用未来帧、未来用户话语或参考答案。每个分支是独立会话。收到新用户话语时回答并更新状态；没有新的用户话语时，只根据有效任务判断是否主动提醒，未触发就输出<SILENCE>，不要重复确认旧话语。遇到取消、纠正和修改必须遵守最新状态。回答简洁中文。
```

### 无用户话语的观察 tick（每 1/observation_hz 秒一次，observation_hz=5）

```text
【实时观察检查点：当前没有新的用户话语】请检查当前画面和仍有效的任务；只有触发条件满足或确需主动提醒时回答，否则只输出<SILENCE>。
```

历史：用户话语那一帧总是带图；无话语的 tick 每 KEYFRAME_S=2 秒保留一帧、最多最近 MAX_TICK_FRAMES=24 张，其余 tick 只留文字。max_new_tokens 256、max_inp_length 32768、max_slice_nums 1、enable_thinking False。

## 6. judge（每题 eval 完立刻跑，eval_all.py）
### 原严格 judge judge_strict.py（所有题型；每个评分槽位一次调用，输入 = slot + user_events + 该槽位窗口前的 causal timeline）

```text
你是极其严格的在线视觉交互Judge。只能使用当前 scoring slot 的用户历史、该slot时间窗及此前同一branch的timeline，绝不能使用之后的回答或帧。无新用户话语必须视为真正无输入：不能复制上一句确认。检查WAIT/SPEAK/KEEP_SILENT、条件主动提醒、取消/纠正、记忆、多轮状态及回答时机。仅动作、事实、时机都正确才给高分；只重复确认而没有完成提醒或回答给0-1分。输出JSON键score(0-5整数),reason(中文),decision_correct(bool),timing_correct(bool),memory_correct(bool),needs_review(bool)。
```

### 专用 judge judge_hr_v8.py（historical_reference，以及其余带 key_attributes 的 7 类题型：在严格 judge prompt 之后追加）

```text
若slot含key_attributes（视觉记忆题）：逐条判断模型在该slot时间窗内的回答是否正确说出了该属性（同义表述算对，说错颜色/形状算错，未提及算漏），并检查是否说出forbidden_claims。另输出attribute_hits(列表，每项{attribute,status:hit|miss|wrong})。score按：类别正确且命中≥3条或全部命中=5，命中2条=4，命中1条=3，只说对类别=2，类别错或说了forbidden=0-1；同时仍须动作与时机正确，否则最多2分。memory_correct表示是否回忆的是正确对象。
```

### 专用 judge judge_pg_v2.py（proactive_guidance：在严格 judge prompt 之后追加）

```text
本题是主动视角引导题。slot_id=sl_guide（含guidance_direction）：用户问的信息当前画面拿不到，模型应说明拿不到并给出具体的视角调整引导；方向与guidance_direction一致且指向正确物体=4-5，只说看不清但没有具体引导=1-2，编造了看不清的信息=0，方向错误=1。slot_id=sl_answer：无新用户话语，信息在画面中出现后模型应在窗口内主动给出答案，按key_attributes逐条核对（hit/miss/wrong），全部命中且主动=5，部分命中=3-4，只沉默或<SILENCE>=0，编造=0。slot_id=sl_direct：信息当前看得清，模型应直接回答（按key_attributes核对）；若要求用户移动视角而不回答，最多1分。另输出attribute_hits(列表，每项{attribute,status:hit|miss|wrong})和guided(bool，回答中是否给出了视角调整引导)。
```
