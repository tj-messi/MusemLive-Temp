# Point-based Judge prompts and deterministic scoring

The executable implementation is `../evaluators/judge_points.py`. The
frozen rubric used by the final450 snapshot has SHA-256
`3f998b202dc885c433315cf5bc1401177ea6600ebb0fcda76e27da00764d017b`
and 4,048 entries. The rubric remains an external controlled artifact; this
repository publishes the code, prompts, profile parameters, and validation
contract.

The LLM never emits the final numeric score. It labels actions and atomic
points; Python computes the score deterministically.

## Core/support classification prompt

~~~text
你在为流式视频交互 benchmark 的一个评分点标注得分点的类别。给你用户这一轮说的话、期望动作、参考答案和一组要点（出题时记录的事实）。
逐条标注：core = 回答用户这句话时必须表达的内容（参考答案里体现了它，或不说它就没回答上问题）；
support = 出题时的佐证/背景（例如画面里同时出现的字幕、镜头构图、转写原文出处），回答里不说也不算答错。至少标一条 core。
只输出 JSON：{"labels": ["core"|"support", ...]}（与要点一一对应）
~~~

## Rubric construction prompt

Used only when a slot has a reference answer but no authored
`key_attributes`.

~~~text
你在为第一视角博物馆流式交互 benchmark 制定评分细则。给你某个评分点的期望动作和参考答案（target_text），
把参考答案拆成 1–3 个"得分点"：每个得分点是一条可独立核对的事实或行为要求（例如"指出讲解员说的是秦人"、"反问用户指的是哪一件展品"），
不要把措辞、语气、礼貌用语当得分点；参考答案里的关键事实都要覆盖到。期望动作是 CLARIFY 时，得分点应描述澄清问句必须点出的内容。
只输出 JSON：{"points": ["...", "..."]}
~~~

## Match prompt

~~~text
你是流式视频交互 benchmark 的得分点匹配 Judge。只做匹配，不打总分。
输入：题型、期望动作（SPEAK=直接回答/提醒，CLARIFY=先反问澄清，WAIT=不该回答）、参考答案、得分点列表、禁止说法列表、
本轮用户语音的文字、模型本轮回答，以及此前对话（仅供理解上下文，评的只是"模型本轮回答"）。
trigger=visual_event 时用户此刻没有说话：这是画面/声音触发的时机，"模型本轮回答"是模型在该时间窗内主动说的话（空 = 保持沉默），
期望 SPEAK 表示模型应主动提醒/告知，期望 WAIT 表示模型应保持安静；此前对话里有用户先前的要求。

判断：
1. action：只看模型本轮回应的"类型"是否符合期望动作，不看内容对错（内容对错由 points 体现）。
   - SPEAK：针对用户这句话（或画面触发时针对用户先前的要求）给出了回答/提醒（哪怕答错了、对象搞错了）= correct；回答了但同时夹杂拒答或大段跑题 = partial；
     只说"好的/我会留意"之类确认、拒答（"视频里没有…"）、复述用户问题、或完全没回应用户这句话（答非所问、自顾自描述别的东西）= wrong。
   - CLARIFY：提出了针对性的澄清问句 = correct；有澄清意味但没形成问句或没点出要区分的对象 = partial；直接给出答案/猜测 = wrong。
   - WAIT：沉默或只做极简确认 = correct；说了实质内容 = wrong。
   observed 填模型实际动作：answer / clarify / ack_only / refuse / silent / other。
2. points：逐条判断得分点：
   - hit：回答表达出了该点的核心意思即可（同义、概括、换种说法、细节不全但关键对象/事实对上都算 hit，不要求逐字或面面俱到）；
   - partial：沾边但核心不完整（例如方向对但对象不具体、只说出一半关键信息、用了更笼统的说法但没有说错）；
   - miss：没有提到；
   - contradict：明确说了与该点相反或错误的内容（说错颜色/对象/数字、张冠李戴）；只是笼统或没提到不算 contradict。
   拿不准 hit 还是 partial 时判 hit；拿不准 partial 还是 miss 时判 partial。
   得分点里的 [音频]/[视觉] 只是来源标签，"对应转写"是证据，不要求模型逐字复述。
   人称不同（他/她/您/你/用户）不算错；只有把说话人或对象搞错、导致事实错误时才算 contradict。
   evidence 摘录模型回答中对应的原话（没有就留空）。
3. forbidden：逐条判断模型是否说出了该禁止说法（意思相同即算）。
只输出 JSON：{"action": "correct|partial|wrong", "observed": "...", "action_reason": "...",
"points": [{"id": "P1", "status": "hit|partial|miss|contradict", "evidence": "...", "why": "..."}],
"forbidden": [{"id": "F1", "violated": true|false, "evidence": "..."}]}
~~~

## Context-correction addendum

~~~text


本题型是上下文纠错（context_correction），输入里有 correction_check。另外判断 correction，这是本题最重要的评分维度：
- mode=must_correct：用户此前说过一个错误理解（user_claim），之后没有自己更正。参考答案会明确指出用户之前的理解不对并给出正确事实。
  判断模型本轮回答是否像参考答案一样做了纠正：
  corrected = 明确指出用户之前的说法/判断不对（如"其实不是X，而是Y""您之前说的有误"），并给出与参考答案一致的正确事实；
  partial = 纠正不完整：只是在回答里用了正确说法却没点明用户之前错了，或点明了不对但没给出正确事实、给出的事实与参考答案不符；
  not_corrected = 没有触及用户的错误理解（只回答了本轮问题）；
  reinforced = 附和、默认或沿用了用户的错误理解。
- mode=must_follow：用户已经自己更正（user_correction）。判断模型是否以更正后的理解为准：
  corrected = 回答明确以更正后的事实为准（确认了用户的更正，或回答里用的是更正后的说法），与参考答案一致；
  partial = 回答没有体现更正内容，但也没有与更正后的事实冲突；
  reinforced = 回到或沿用了用户更正前的错误理解。
JSON 里增加："correction": {"status": "corrected|partial|not_corrected|reinforced", "evidence": "模型回答原话", "why": "..."}
~~~

## Deterministic score

For non-WAIT, non-context-correction slots:

~~~text
score = clip_0_5(
    5 * (w_action * action_credit + w_points * point_coverage)
    - pen_forbid * forbidden_hits
    - pen_contra * contradictions
)
~~~

If support points exist, `point_coverage = 0.8 * core_coverage + 0.2 *
support_coverage`; otherwise it is core coverage. A required SPEAK/CLARIFY
slot with no response is exactly 0. WAIT is scored only from action
correctness. Context-correction uses its dedicated correction/action/point
weights and a reinforcement penalty.

Profiles:

| Parameter | strict_v2 | lenient_v3 |
|---|---:|---:|
| action weight | 0.30 | 0.40 |
| point weight | 0.70 | 0.60 |
| partial point credit | 0.70 | 0.80 |
| partial action credit | 0.50 | 0.70 |
| forbidden penalty | 1.00 | 0.75 |
| contradiction penalty | 0.25 | 0.25 |
| correction/action/points | 0.50/0.15/0.35 | 0.40/0.25/0.35 |
| wrong-belief reinforcement penalty | 0.50 | 0.50 |

The current 10 FPS schedule uses `lenient_v3`; the finalized 1 FPS snapshot
preserves its historical judge profile. Those raw scores are therefore not a
controlled FPS-only comparison until both tracks are rescored with the same
profile and frozen rubric.
