# QA post-processing prompts

These two executed 10-16 passes ran after multimodal candidate generation and
before per-batch pool selection. They correct text presentation only; task
times, IDs, facts, branches, and slot structure remain fixed. Original files
are backed up by the scripts before an accepted rewrite is written.

## First-person camera-wearer normalization

~~~text
你在修订第一视角博物馆流式交互 benchmark 的题目文本。视频是戴着摄像头的用户第一视角拍的，音频里标为“主播/博主/UP主/拍摄者”的声音就是**用户本人**。
规则：
1. user_events 里的 text 是用户（视频主角本人）对模型说的话，必须第一人称，并且口吻要像主角本人一边逛一边随口问助手（接着自己刚说的话/刚做的事的语境，用词风格和转写里主角的说话风格一致）：把“主播/博主/UP主/拍摄者”改成“我”（如“帮我记一下我接下来提到的展品名称”“刚才我说的那个年代是多少？”）。
2. slots 里的 target_text、key_attributes、forbidden_claims 是模型对用户说的话/评分要点：把“主播/博主/UP主/拍摄者（说/提到/念…）”改成“您（刚才说/提到…）”，如“您刚才说是太古代”“[音频] 您在258.3s提到了‘顺山集遗址’”。
3. 讲解员、游客、广播、旁白等**其他人**保持第三人称不动。如果某处“主播”明显指的不是用户本人而是画面里另一个人，保持不变并在 note 里说明。
4. 只改称呼和随之必需的语法，不改事实、时间、数字、引号里的转写原句（audio_refs 的 text 是转写原句，不动）、标签“[音频]/[视觉]”、id。
只输出 JSON：{"branches": [{"branch_id": "...", "user_events": [{"event_id": "...", "text": "..."}], "slots": [{"slot_id": "...", "target_text": "...", "key_attributes": ["..."], "forbidden_claims": ["..."]}]}], "note": ""}
~~~

## Scene-language normalization

~~~text
You are revising items of a first-person museum live-streaming benchmark. The video's spoken language is English (or nobody speaks), so every item text must be in natural English.
Rewrite the given JSON: translate user_events.text (the camera-wearer talking to the assistant, first person, casual spoken tone consistent with how the host actually talks), target_text (the assistant's spoken reply to the user, address the user as "you"), key_attributes and forbidden_claims into English.
Rules: keep the leading tags "[音频]" / "[视觉]" EXACTLY as they are (these Chinese tags are machine labels; never translate them to [Audio]/[Visual]); keep quoted transcript snippets (text inside quotes that was copied from the audio) in their original language; do not change ids, times, facts or numbers; do not add or remove items.
Output JSON only: {"branches": [{"branch_id": "...", "user_events": [{"event_id": "...", "text": "..."}], "slots": [{"slot_id": "...", "target_text": "...", "key_attributes": ["..."], "forbidden_claims": ["..."]}]}]}
~~~

The output of each pass is structurally validated against the existing branch,
event, and slot IDs before replacement. The full implementations are
`../generation/qa_gen/fix_first_person.py` and
`../generation/qa_gen/fix_language.py`.
