# Visual and audio annotation prompts

These are the prompts used by the executed preprocessing jobs. They are
published separately from generated annotations because raw media, transcripts,
and provider responses are controlled external artifacts.

## Visual evidence freeze

Each source is decoded with `ffmpeg -vf fps=25,scale=640:-2`. Only complete
25-frame seconds enter the manifest; a partial final second is excluded. The
annotator receives all 25 ordered frames for one second and may cite only frame
IDs from that second. The implementation is
`../generation/annotate_visual_seconds.py`.

Observed completed state for the 316-video batch: 316/316 videos annotated,
with no unresolved visual-window gaps after the recorded gap-fill pass.

### Exact visual system prompt

~~~text
你是视频逐秒标注员。输入是同一秒内按时间排序的25帧，每张图附有帧ID。
只根据当前窗口可见内容，用中文写出主体、状态、动作、镜头变化和可读文字。不要猜测身份、年代、材质、用途，不借用其他秒的信息。不要逐帧机械复述。
只输出JSON：{"summary":"这一秒的概述","events":[{"frame_start":"帧ID","frame_end":"帧ID","description":"可见事实","objects":[],"ocr":[],"uncertainty":"不确定处"}],"limitations":"局限"}。
即使没有变化，也输出一个描述稳定状态的事件；证据范围只能引用本次给出的帧。
~~~

The output validator requires a non-empty `summary`, at least one event,
valid `frame_start` and `frame_end` IDs, and a non-reversed evidence range.
Invalid or refused windows are retried and otherwise recorded as gaps; they are
not synthesized locally.

## Audio annotation and one-second merge

The executed audio job cuts source audio into ordered 15-second, 16 kHz mono
PCM chunks. Chunks are processed sequentially per video so the previous
chunk's last transcript lines can be supplied only as continuity context. All
returned relative times are clamped to the chunk and converted to absolute
source seconds. Speech and sound events are then intersected with each frozen
one-second visual window and merged into its `audio` field.

Recorded run evidence:

- legacy 50-video path: 2,364 expected chunks, 2,313 annotation files and
  3,064 speech segments in the captured progress record;
- 316-video path: 12,933 chunks, 12,868 annotation files, 65 explicit gaps and
  15,679 speech segments; completed files plus explicit gaps cover all chunks.

A provider refusal, empty result, parse failure, or absent audio is represented
as a gap/status value. The pipeline never fills an inaudible word by guessing.

### Exact audio system prompt

~~~text
你是博物馆直播视频的音频标注员。只根据给出的这一段音频如实标注，不要猜测画面内容，不要编造听不清的词。输出必须是一个 JSON 对象，不要加任何解释或代码块标记。
~~~

### Exact audio user-prompt template

~~~text
这是视频 {clip} 第 {start:.0f}–{end:.0f} 秒的音频（共 {dur:.1f} 秒）。{ctx}
请输出 JSON：{{"speech":[{{"start":相对本段开头的秒,"end":秒,"speaker":"讲解员/游客/主播/旁白/广播/其他","lang":"zh/en/其他","text":"逐字转写"}}],"events":[{{"start":秒,"end":秒,"type":"背景音乐/人群嘈杂/广播/脚步或走动/安静/其他","description":"简述"}}],"limitations":"听不清、重叠、不确定之处如实说明"}}。
要求：speech 按时间排序，一句一条，时间尽量准确；没有人说话就给空列表；听不清的词用「…」代替，不要补全；不要描述画面。
~~~

## Boundary

The annotations are generation evidence, not model input at 25 FPS. Evaluation
uses the same frozen source timeline but supplies either 1 or 10 ordered visual
frames per one-second decision tick. MOSS receives the merged ideal-ASR scene
transcript condition documented in the evaluation protocol; native-audio
models receive scene audio.
