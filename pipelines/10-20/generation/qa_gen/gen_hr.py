"""historical_reference 出题线（Omni 版 10-16：记忆属性分 [音频]（当时讲解/主播对它说了什么）和 [视觉]（外观），每题至少一条音频属性；其余沿用 hrv8 协议；316 批从零出题的种子骨架来自 run316_gen.cmd_hr）。

考点：记住并复述**过去看到的物体外观**。t0 用户只用位置/指示词指认一件物体并起代号（bind），t0+1 干预（keep_alias / rebind_alias / cancel_alias），
30–60s 后、物体已离开画面 ≥4s 时问"<代号>是个什么？描述一下它的样子"。SPEAK 槽位带 category + 2–4 条 key_attributes（只写外观，帧里看得清）。
每题流程：种子骨架（只有绑定时刻与分支结构）→ Qwen 看绑定时刻帧 + 逐秒标注改写台词/提问时刻/答案 → 确定性检查 → 泄露检查 → 看帧核验
（属性在指认帧里成立、提问前 4s 物体不在画面）→ 不过则带累积原因重写，最多 --max-attempts 次。
用法: python3 gen_hr.py --batch 50|316 --seeds N [--budget 5] [--workers 8] [--min-gap 30] [--max-gap 60]
     或 --tasks <job_root 列表文件>（每行一个含 candidate_task.json 的目录，用已有题当起点）
输出: runs/<batch>/runs_hr/<task_id>/{refresh_log.json, refreshed_task.json, a*_*.response.txt}；status ∈ pass / needs_revision / infeasible / error。
"""
import argparse
import json
import random
import re
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R  # noqa: E402
import config as CFG  # noqa: E402
from common import bigrams, APPEARANCE, GENERIC, POSITION, img_part  # noqa: E402

MODEL = R.MODEL
OUT = None  # runs/<batch>/runs_hr
WIN_G = {}  # 逐秒窗口（to_task 标记人声重叠用）

AUTHOR_SYSTEM = """你在为第一视角博物馆流式交互 benchmark 改写"历史指代 / 全模态记忆"题。
目标：考模型能否记住并复述**过去看到的物体外观细节**以及**当时听到的关于它的介绍**（讲解员/主播/游客说的名称、朝代、用途、数字、评价），而不是只复述名字或判断是否还在画面里。
""" + R.OMNI_RULE + """
属性标注规则：key_attributes 每条以“[音频]”或“[视觉]”开头；[音频] 属性必须是转写里在绑定时刻前后（绑定前 10 秒到干预后 15 秒内）说出的**关于该物体本身**的介绍（名称、年代、用途、来历、评价），逐字或明确同义；旁人无关的闲聊、语气词（“好的”“哇”）、与物体无关的广播不算。至少 1 条 [音频] 属性、至少 1 条 [视觉] 属性。
**选物体时优先选当时正被讲解员/主播介绍的那件**；如果绑定时刻附近没有人介绍画面里的任何物体，就 feasible=false，不要硬凑。
[音频] 属性的 evidence_window_id 写成 "audio@<start>-<end>"（转写里那句话的时间）。问句固定形如“<代号>是个什么？描述一下它的样子，当时是怎么介绍它的？”

硬性要求：
1. 绑定台词只能用位置、指示词或画面关系来指物体（如"中间柱子后面那件""左边第二个展柜里那个"），再给一个与外观无关的代号（如 A1、hidden_furniture、"那件东西"）。
   绑定台词和**所有干预台词**里都不得出现任何将要考的属性：颜色、材质、形状、物体类别名（床/镜子/雕像…）、上面的文字或图案。
   原题的干预台词通常已经泄露了答案（如"就是那张带帷幔的床"），必须重写，例如 keep："对，就是它，以后就叫它 X。"；rebind："不对，换成壁炉上方挂着的那个，以后它叫 X。"
2. 提问时间 query_time_s 必须满足：距离每个分支的指代对象最后一次被确立（keep 分支是 bind，rebind 分支是 rebind 那句）在 {min_gap}–{max_gap} 秒之间；
   并且根据逐秒摘要，提问时刻和此前 4 秒内该物体都已不在画面中（物体离开画面越早越好，早就离开了几十秒完全没问题）。两个分支的提问时间和问句完全相同。
3. 问句形如"<代号>是个什么？描述一下它的样子，当时是怎么介绍它的？"，不得给出任何外观或介绍内容的提示。
4. 每个 SPEAK 槽位给 category（物体是什么，如"四柱床"）和 2–5 条 key_attributes（含 [音频] 和 [视觉]），每条只写**一个**短属性（标签后 ≤12 字，如"[视觉] 白色帷幔""[音频] 讲解说是唐代的"），不要把多个细节塞进一条，也不要写数量、"垂下"这类难核对的细节，并给出依据的 window_id（只能用绑定/重绑时刻附近、你看过的标注窗口）。
   属性只能是外观（颜色、形状、部件、图案、文字内容），不能是位置（"位于壁炉上方"）——位置是用户指认时说的，不算记忆。
   属性必须能从给你的绑定时刻画面帧里清楚看到（画面昏暗、被遮挡、太远看不清的不要写；核验员会看同样的帧逐条核对），不得推测材质年代等看不出的东西；画面看不清的属性不要写。
5. target_text 是自然口语回答：先说是什么，再描述这些属性。forbidden_claims 写 2–3 条干扰描述：另一分支对象的真实属性、提问时刻画面里物体的属性。
6. 分支结构保持原题（keep/rebind/cancel 等 branch_id 不变）。cancel 分支：用户已取消代号，如果问句对指代仍不确定就 CLARIFY，target_text 写澄清句，不需要 key_attributes。
   rebind 分支的新对象必须是当时真实可见、可用位置唯一指出的另一个物体，同样只用位置描述；rebind 后代号指代是明确的，所以 rebind 分支必须 SPEAK，描述新对象的外观。
8. """ + R.TIMING_RULE + """
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
 "bind_state_target": "绑定对象的内部描述(可含外观)"}"""

LEAK_SYSTEM = """判断用户台词有没有透露物体外观。给你若干句用户台词和一个"答案属性清单"。
台词只允许用位置、指示词、代号来指物体；如果台词里出现了物体类别（如"床""盔甲""板状展品"）、颜色、材质、形状、纹样、上面的文字，
或与清单中任何属性同义/近义的描述（如"刻线"与"凹陷线条"、"孔洞"与"圆形大孔"），都算泄露。
只看**被指代物体本身**：用来指路的其他物体（如"黑柱子后面那件"里的黑柱子、"壁炉上方挂着的那个"里的壁炉）可以带外观，不算泄露。
"立着的""那件东西""家具""展品"这种很笼统、不对应清单属性的说法不算。
只输出 JSON：{"leaks": [{"line": "...", "phrase": "...", "matches_attribute": "..."}]}，没有泄露就输出 {"leaks": []}。"""

VERIFY_SYSTEM = """你是严格的视觉核验员。给你两组真实视频帧：
A 组：某物体被用户指认时的帧；B 组：后来用户提问时刻及之前 4 秒的帧。
对给定的物体描述和属性清单，逐条判断：
- 在 A 组帧里能否确认该物体，以及每条属性是否在 A 组帧里清楚可见（看不清、需要推测的一律 false）；
- 该物体在 B 组任意一帧里是否仍然可见（部分可见也算可见）。
只输出 JSON：{"referent_found_in_A": bool, "attributes": [{"attribute": "...", "supported": bool, "why": "..."}],
"visible_in_B": bool, "B_note": "..."}"""


def establishing_times(task):
    """各分支指代对象被确立的时刻：rebind 分支取 rebind，否则取 bind。"""
    b0 = task['branches'][0]['user_events']
    bind_t = next(e['time_s'] for e in b0 if (e.get('state_op') or {}).get('op') == 'bind')
    times = {bind_t}
    for b in task['branches']:
        for e in b['user_events']:
            if (e.get('state_op') or {}).get('op') == 'rebind_alias':
                times.add(e['time_s'])
    return sorted(times)


def query_range(task, duration, min_gap, max_gap):
    times = establishing_times(task)
    return max(times) + min_gap, min(min(times) + max_gap, duration - 3)


def build_author_messages(task, windows, zf, src, media, duration, min_gap, max_gap, feedback):
    q_lo, q_hi = query_range(task, duration, min_gap, max_gap)
    times = establishing_times(task)
    t0 = times[0]
    ann = [a for s in range(int(t0) - 1, int(times[-1]) + 2) if (a := R.annotation(zf, src, s))]
    later = [f"{w['window_id']} [{w['start_s']}-{w['end_s']}s] {w['summary']}" + (f"  🔊 {w['audio']}" if w.get('audio') else '')
             for s, w in sorted(windows.items()) if t0 - 2 <= s <= min(times[-1] + max_gap + 5, duration)]
    tr = R.transcript(windows, t0 - 10, times[-1] + 15)
    original = [{'branch_id': b['branch_id'], 'user_events': b['user_events'],
                 'slots': [{k: s.get(k) for k in ('expected_action', 'target_text', 'forbidden_claims')} for s in b['slots']]}
                for b in task['branches']]
    text = (R.lang_rule(R.video_lang(windows)) + f"\n源视频 {src}，时长 {duration:.0f}s。原题（需要改写）：\n{json.dumps(original, ensure_ascii=False, indent=1)}\n\n"
            f"绑定/重绑时刻附近的逐秒详细标注：\n{json.dumps(ann, ensure_ascii=False, indent=1)}\n\n"
            f"绑定时刻前后的音频转写（{t0 - 10:.0f}–{times[-1] + 15:.0f}s，[音频] 属性只能从这里取）：\n{tr}\n\n"
            f"之后的逐秒摘要（用于选提问时刻）：\n" + '\n'.join(later) +
            f"\n\n下面依次附上时刻 {', '.join(f'{t:.1f}s' for t in times)} 及其后 0.5s 的画面帧。")
    text += f"\n\n【提问时刻 query.time_s 只能在 {q_lo:.0f}–{q_hi:.0f}s 之间选】（已按每个分支的指代确立时刻算好），请在这个区间里挑一个物体已离开画面的秒数。"
    if feedback:
        text += '\n\n之前的改写没有通过检查，以下问题全部要避免（被否决的属性不要再写）：\n' + '\n'.join(f'- {x}' for x in feedback)
    content = [{'type': 'text', 'text': text}]
    for t in times:
        for dt in (0.0, 0.5):
            content.append({'type': 'text', 'text': f'[帧 {t + dt:.1f}s]'})
            content.append(img_part(R.frame(src, media, t + dt)))
    return [{'role': 'system', 'content': AUTHOR_SYSTEM.replace('{min_gap}', str(int(min_gap))).replace('{max_gap}', str(int(max_gap)))},
            {'role': 'user', 'content': content}]


# 不算泄露的泛用 2-gram
GENERIC = {'一个', '一张', '一面', '一只', '一尊', '上面', '上方', '下方', '左侧', '右侧', '有一', '形状', '颜色', '样子', '那个', '这个', '正中', '中间', '画面', '装饰', '模型', '东西', '展品', '作品', '物件', '左边', '右边', '后面', '前面', '旁边', '展柜', '展台', '墙上'}
POSITION = re.compile(r'^(位于|在.*(上方|下方|旁边|左边|右边|左侧|右侧|后面|前面)|挂在|放在|摆在)')


def deterministic_check(r, task, duration, min_gap, max_gap, windows=None):
    fails = []
    if not r.get('feasible'):
        return ['infeasible']
    q = r['query']
    if windows:
        lang = R.video_lang(windows)
        for txt in [r['bind']['text'], q['text']] + [b['intervention']['text'] for b in r['branches']] + [b['slot'].get('target_text', '') for b in r['branches']]:
            m = R.lang_bad(txt, lang)
            if m: fails.append('语言不符：' + m)
        times = {'bind': r['bind']['time_s'], 'query': q['time_s']}
        for b in r['branches']:
            times[f"{b['branch_id']}_intervention"] = b['intervention']['time_s']
        clash = R.speech_clash(windows, times)
        if clash:
            fails.append('用户开口时刻与人声重叠：' + clash)
    if not APPEARANCE.search(q['text']):
        fails.append(f"问句不是问外观: {q['text']}")
    if q['time_s'] + 3 > duration:
        fails.append('提问时刻超出视频长度')
    ids = {b['branch_id'] for b in task['branches']}
    got = {b['branch_id'] for b in r['branches']}
    if ids != got:
        fails.append(f'分支不一致: 原 {sorted(ids)} 新 {sorted(got)}')
    for b in r['branches']:
        iv, slot = b['intervention'], b['slot']
        ref_t = iv['time_s'] if iv['state_op']['op'] == 'rebind_alias' else r['bind']['time_s']
        if iv['state_op']['op'] == 'rebind_alias' and slot['expected_action'] != 'SPEAK':
            fails.append(f"{b['branch_id']}: 重绑后指代明确，应 SPEAK 描述新对象外观，不能 CLARIFY")
        if not (r['bind']['time_s'] < iv['time_s'] < q['time_s']):
            fails.append(f"{b['branch_id']}: 时间顺序应为 bind < 干预 < 提问")
        if not min_gap <= q['time_s'] - ref_t <= max_gap:
            fails.append(f"{b['branch_id']}: 指代对象确立({ref_t}s)到提问({q['time_s']}s)应在 {min_gap}–{max_gap}s 之间")
        lines = r['bind']['text'] + iv['text'] + q['text']
        line_bg = bigrams(lines, drop=(iv['state_op'].get('id', ''),))
        if slot['expected_action'] == 'SPEAK':
            attrs = slot.get('key_attributes') or []
            if not 2 <= len(attrs) <= 5:
                fails.append(f"{b['branch_id']}: key_attributes 需要 2–5 条，现有 {len(attrs)}")
            tags = [(a.get('attribute') or '').strip()[:4] for a in attrs]
            if any(t not in ('[音频]', '[视觉]') for t in tags):
                fails.append(f"{b['branch_id']}: 每条属性必须以 [音频] 或 [视觉] 开头")
            if '[音频]' not in tags:
                fails.append(f"{b['branch_id']}: 全模态要求至少 1 条 [音频] 属性（当时听到的关于它的介绍）")
            if '[视觉]' not in tags:
                fails.append(f"{b['branch_id']}: 至少 1 条 [视觉] 外观属性")
            for word in [slot.get('category', '')]:  # 属性的近义泄露交给 leak_check
                hit = sorted((bigrams(word) - GENERIC) & line_bg)
                if hit:
                    fails.append(f"{b['branch_id']}: “{word}”里的“{'/'.join(hit)}”已出现在用户台词里，台词需要改成只用位置/指示词")
            for a in attrs:
                if POSITION.search(strip_tag(a['attribute'])):
                    fails.append(f"{b['branch_id']}: 属性“{a['attribute']}”是位置不是外观")
            tb = bigrams(slot['target_text'], drop=(iv['state_op'].get('id', ''),))
            if tb and len(tb & line_bg) / len(tb) >= 0.25:
                fails.append(f"{b['branch_id']}: 标准答案与用户台词重合过多")
    return fails


def strip_tag(x):
    x = (x or '').strip()
    return x[4:].strip() if x[:4] in ('[音频]', '[视觉]') else x


AUDIO_CHECK_SYSTEM = """你是严格的音频核验员。给你一段带时间的音频转写和若干条“[音频] 属性”（声称当时讲解/主播/游客说过的关于某物体的内容）。
逐条判断：转写里是否确实说过这句话（逐字或明确同义），且时间落在给定区间内，并且这句话确实是在介绍/评价**这件物体**（名称、年代、用途、来历、评价）；只凭画面能知道、转写里没有、或与该物体无关的闲聊/语气词一律 false。
只输出 JSON：{"attributes": [{"attribute": "...", "supported": bool, "quote": "转写原句或空", "why": "..."}]}"""


def establishing_times_r(r):
    ts = {r['bind']['time_s']}
    for b in r['branches']:
        if b['intervention']['state_op']['op'] == 'rebind_alias':
            ts.add(b['intervention']['time_s'])
    return sorted(ts)


def audio_check(r, windows, log_dir, attempt, budget):
    fails, records = [], []
    times = establishing_times_r(r)
    lo, hi = times[0] - 10, times[-1] + 15
    tr = R.transcript(windows, lo, hi)
    for b in r['branches']:
        slot = b['slot']
        if slot['expected_action'] != 'SPEAK':
            continue
        attrs = [a['attribute'] for a in slot.get('key_attributes') or [] if (a.get('attribute') or '').strip().startswith('[音频]')]
        if not attrs:
            continue
        desc = b['intervention']['state_op']['target'] if b['intervention']['state_op']['op'] == 'rebind_alias' else r.get('bind_state_target', '')
        v = R.call([{'role': 'system', 'content': AUDIO_CHECK_SYSTEM},
                    {'role': 'user', 'content': f"物体：{desc}\n区间 {lo:.0f}–{hi:.0f}s 的转写：\n{tr}\n\n待核属性：{json.dumps(attrs, ensure_ascii=False)}"}],
                   f'a{attempt}_audiocheck_{b["branch_id"]}', log_dir, budget, max_tokens=1200)
        records.append({'branch_id': b['branch_id'], **v})
        for a in v.get('attributes', []):
            if not a.get('supported'):
                fails.append(f"{b['branch_id']}: 属性“{a.get('attribute')}”在音频转写里找不到（{a.get('why', '')}）")
    return fails, records


def leak_check(r, log_dir, attempt, budget):
    """所有分支的绑定/干预台词，对照所有 SPEAK 分支的类别和属性做近义泄露检查。"""
    attrs = []
    for b in r['branches']:
        if b['slot']['expected_action'] == 'SPEAK':
            attrs += [b['slot'].get('category', '')] + [a['attribute'] for a in b['slot'].get('key_attributes') or []]
    fails, records = [], []
    if not attrs:
        return fails, records
    for b in r['branches']:
        lines = [r['bind']['text'], b['intervention']['text']]  # 问句是固定模板，不查
        v = R.call([{'role': 'system', 'content': LEAK_SYSTEM},
                  {'role': 'user', 'content': f"用户台词：{json.dumps(lines, ensure_ascii=False)}\n答案属性清单：{json.dumps(attrs, ensure_ascii=False)}"}],
                 f'a{attempt}_leak_{b["branch_id"]}', log_dir, budget, max_tokens=800)
        records.append({'branch_id': b['branch_id'], **v})
        for x in v.get('leaks', []):
            fails.append(f"{b['branch_id']}: 台词“{x.get('line')}”中的“{x.get('phrase')}”透露了外观（对应“{x.get('matches_attribute')}”），要改成只用位置/指示词")
    return fails, records


def visual_check(r, src, media, log_dir, attempt, budget):
    fails, records = [], []
    q_t = r['query']['time_s']
    b_frames = [R.frame(src, media, q_t - d) for d in (4, 3, 2, 1, 0)]
    for b in r['branches']:
        slot, iv = b['slot'], b['intervention']
        if slot['expected_action'] != 'SPEAK':
            continue
        rebind = iv['state_op']['op'] == 'rebind_alias'
        ref_t = iv['time_s'] if rebind else r['bind']['time_s']
        desc = iv['state_op']['target'] if rebind else r.get('bind_state_target', '')
        locate = iv['text'] if rebind else r['bind']['text']
        attrs = [strip_tag(a['attribute']) for a in slot['key_attributes'] if (a.get('attribute') or '').strip().startswith('[视觉]')]
        content = [{'type': 'text', 'text': f"物体：{desc}\n用户指认它时说：{locate}\n属性清单（只核外观）：{json.dumps(attrs, ensure_ascii=False)}\n\nA 组帧："}]
        for dt in (0.0, 0.5, 1.0):
            content += [{'type': 'text', 'text': f'[A {ref_t + dt:.1f}s]'}, img_part(R.frame(src, media, ref_t + dt))]
        content.append({'type': 'text', 'text': 'B 组帧：'})
        for d, p in zip((4, 3, 2, 1, 0), b_frames):
            content += [{'type': 'text', 'text': f'[B {q_t - d:.1f}s]'}, img_part(p)]
        v = R.call([{'role': 'system', 'content': VERIFY_SYSTEM}, {'role': 'user', 'content': content}],
                 f'a{attempt}_verify_{b["branch_id"]}', log_dir, budget, max_tokens=1500)
        records.append({'branch_id': b['branch_id'], **v})
        if not v.get('referent_found_in_A'):
            fails.append(f"{b['branch_id']}: 核验员在指认时刻的帧里找不到该物体")
        for a in v.get('attributes', []):
            if not a.get('supported'):
                fails.append(f"{b['branch_id']}: 属性“{a.get('attribute')}”在指认时刻的帧里看不清（{a.get('why', '')}）")
        if v.get('visible_in_B'):
            fails.append(f"{b['branch_id']}: 提问时刻前 4 秒该物体仍在画面里（{v.get('B_note', '')}），需要换更晚的提问时刻")
    return fails, records


def to_task(task, r):
    """把改写结果装回 candidate_task 格式；原字段保留，新增 refresh 元信息。"""
    new = json.loads(json.dumps(task))
    q = r['query']
    alias = r['branches'][0]['intervention']['state_op'].get('id', 'alias')
    new['task_id'] = task['task_id'].replace('OP_HR', 'OM_HR') + '_omni'
    new['modality'] = 'omni'; new['language'] = R.video_lang(WIN_G.get(task['source_id'], {}))
    if task.get('capability') != 'historical_reference':  # 2026-10-11：从被替换的其他题型旧题借用绑定时刻和分支结构
        new['capability'] = 'historical_reference'
        new['interaction_family'] = 'recurrence_reference'
    new['status'] = 'refresh_candidate_pending_human_review'
    new['session'] = {**task['session'], 'start_s': min(task['session']['start_s'], r['bind']['time_s']),
                      'end_s': q['time_s'] + 3}
    new['shared_future_from_s'] = q['time_s']
    new['branches'] = []
    for b in r['branches']:
        iv, slot = b['intervention'], b['slot']
        ev = [{'event_id': 'ue_bind', 'time_s': r['bind']['time_s'], 'text': r['bind']['text'], 'speech_overlap': bool(R.speech_at(WIN_G.get(task['source_id'], {}), r['bind']['time_s'])),
               'state_op': {'op': 'bind', 'id': alias, 'target': r.get('bind_state_target', '')}},
              {'event_id': 'ue_intervention', 'time_s': iv['time_s'], 'text': iv['text'], 'state_op': iv['state_op'], 'speech_overlap': bool(R.speech_at(WIN_G.get(task['source_id'], {}), iv['time_s']))},
              {'event_id': 'ue_query', 'time_s': q['time_s'], 'text': q['text'], 'speech_overlap': bool(R.speech_at(WIN_G.get(task['source_id'], {}), q['time_s'])),
               'state_op': {'op': 'query', 'id': 'recall_appearance', 'target': alias}}]
        attrs = slot.get('key_attributes') or []
        new['branches'].append({'branch_id': b['branch_id'], 'user_events': ev, 'slots': [{
            'slot_id': 'sl_recall_appearance',
            'trigger': {'kind': 'user_query', 'event_id': 'ue_query'},
            'window': {'start_s': q['time_s'], 'end_s': q['time_s'] + 3},
            'depends_on': ['ue_bind', 'ue_intervention'],
            'evidence_ids': sorted({a['evidence_window_id'] for a in attrs}),
            'expected_action': slot['expected_action'], 'target_text': slot['target_text'],
            'key_attributes': [a['attribute'] for a in attrs],
            'modality_facts': {'audio': [a['attribute'] for a in attrs if (a['attribute'] or '').startswith('[音频]')],
                               'visual': [a['attribute'] for a in attrs if (a['attribute'] or '').startswith('[视觉]')]},
            'forbidden_claims': slot.get('forbidden_claims', [])}]})
    new['diagnostic_slot_ids'] = ['sl_recall_appearance']
    new['refresh'] = {'version': 'hr_omni_v1_20261016', 'source_task_id': task['task_id'],
                      'model': MODEL, 'query_time_rationale': r.get('query_time_rationale', '')}
    return new


def refresh_one(entry, windows, media_map, args):
    task = json.load(open(Path(entry['job_root']) / 'candidate_task.json'))
    src = task['source_id']
    media, duration = media_map[src]
    log_dir = OUT / task['task_id']
    log_dir.mkdir(parents=True, exist_ok=True)
    feedback, history = [], []
    q_lo, q_hi = query_range(task, duration, args.min_gap, args.max_gap)
    if q_lo > q_hi:  # 各分支的指代确立时刻相距太远或太靠近视频结尾，不调模型
        history.append({'attempt': 0, 'result': 'infeasible', 'reason': f'允许的提问区间为空: {q_lo}–{q_hi}'})
    with zipfile.ZipFile(R.ARCHIVE) as zf:
        for attempt in range(1, (args.max_attempts if q_lo <= q_hi else 0) + 1):
            rec = {'attempt': attempt, 'feedback_in': feedback}
            try:
                msgs = build_author_messages(task, windows.get(src, {}), zf, src, media, duration, args.min_gap, args.max_gap, feedback)
                r = R.call(msgs, f'a{attempt}_author', log_dir, args.budget)
                rec['author_output'] = r
                fails = deterministic_check(r, task, duration, args.min_gap, args.max_gap, windows.get(src, {}))
                if fails == ['infeasible']:
                    rec['result'] = 'infeasible'
                    history.append(rec)
                    break
                if not fails:
                    fails, rec['leak_check'] = leak_check(r, log_dir, attempt, args.budget)
                if not fails:
                    fails, rec['visual_check'] = visual_check(r, src, media, log_dir, attempt, args.budget)
                if not fails:
                    fails, rec['audio_check'] = audio_check(r, windows.get(src, {}), log_dir, attempt, args.budget)
            except Exception as e:  # provider 拒答/解析失败如实记录
                rec['error'] = f'{type(e).__name__}: {e}'
                history.append(rec)
                if 'HTTP 4' in str(e) or 'budget' in str(e):
                    break
                continue
            rec['fails'] = fails
            history.append(rec)
            if not fails:
                rec['result'] = 'pass'
                json.dump(to_task(task, r), open(log_dir / 'refreshed_task.json', 'w'), ensure_ascii=False, indent=1)
                break
            feedback = list(dict.fromkeys(feedback + fails))  # 累积，避免反复写回已否决的属性
    status = history[-1].get('result') or ('error' if history and 'error' in history[-1] else 'needs_revision')
    out = {'task_id': task['task_id'], 'source_id': src, 'status': status, 'attempts': history}
    json.dump(out, open(log_dir / 'refresh_log.json', 'w'), ensure_ascii=False, indent=1)
    print(f"[{status}] {task['task_id']} attempts={len(history)} spent=${R._spent['usd']:.3f}", flush=True)
    return {'task_id': task['task_id'], 'status': status, 'attempts': len(history)}



# ---------------------------------------------------------------- 种子（从零出题，10-11 316 批用）
def seed_task(src, t0, kind, dur):
    """视觉记忆题的种子骨架：只给绑定时刻和分支结构，台词/提问/答案全部交给 refresh_one 改写。"""
    second = 'rebind_alias' if kind == 'rebind' else 'cancel_alias'
    tid = f'OP_HR{R.BATCH}_{src}_{int(t0):05d}'

    def branch(bid, op):
        ev = [{'event_id': 'ue_bind', 'time_s': t0, 'text': '（待改写：用位置指代画面中的一件物体并起代号）',
               'state_op': {'op': 'bind', 'id': 'A1', 'target': '待定'}},
              {'event_id': 'ue_intervention', 'time_s': t0 + 1, 'text': '（待改写）', 'state_op': {'op': op, 'id': 'A1', 'target': '待定'}},
              {'event_id': 'ue_query', 'time_s': min(t0 + 40, dur - 4), 'text': 'A1是个什么？描述一下它的样子，当时是怎么介绍它的？',
               'state_op': {'op': 'query', 'id': 'q', 'target': 'A1'}}]
        act = 'CLARIFY' if op == 'cancel_alias' else 'SPEAK'
        return {'branch_id': bid, 'user_events': ev, 'slots': [{
            'slot_id': 'sl_recall', 'trigger': {'kind': 'user_query', 'event_id': 'ue_query'},
            'window': {'start_s': ev[2]['time_s'], 'end_s': ev[2]['time_s'] + 3}, 'expected_action': act,
            'target_text': '（待改写）', 'forbidden_claims': []}]}
    return {'source_id': src, 'capability': 'historical_reference', 'interaction_family': 'recurrence_reference',
            'task_id': tid, 'status': 'seed', 'session': {'start_s': t0, 'end_s': t0 + 45, 'observation_hz': 5, 'history_policy': 'causal_prefix'},
            'shared_future_from_s': t0, 'branches': [branch('keep_alias', 'keep_alias'), branch(kind, second)]}


def make_seeds(n, windows, media, seeds_dir, offset=0):
    """候选绑定时刻：每 45s 取一个有物体的秒，且 40s 后的摘要与当下重合 <35%（画面大概率已换、物体已离开）；每视频先保证 1 个、最多 3 个。"""
    random.seed(20261011)
    cands = []
    blocked = R._cfg.get('content_blocked', set())
    old = defaultdict(list)  # offset>0（10-16 补题）：已有种子的绑定时刻，新种子离它们 ≥15 秒
    if offset and seeds_dir.exists():
        for d in seeds_dir.iterdir():
            src0, t0 = d.name[len(f'OP_HR{R.BATCH}_'):].rsplit('_', 1)
            old[src0].append(int(t0))
    for src, w in sorted(windows.items()):
        if src in blocked or src not in media:
            continue
        dur = media[src][1]
        secs = sorted(s for s in w if 10 <= s <= dur - 70)
        with zipfile.ZipFile(R.ARCHIVE) as zf:
            for s in secs[offset::45]:
                if any(abs(s - b) < 15 for b in old.get(src, [])):
                    continue
                a = R.annotation(zf, src, s)
                if not a or not any(e.get('objects') for e in a['events']):
                    continue
                later = w.get(s + 40, {}).get('summary', '')
                overlap = len(bigrams(w[s]['summary']) & bigrams(later)) / max(len(bigrams(w[s]['summary'])), 1)
                if overlap < 0.35 and R.informative_speech(w, s - 8, s + 15, 60):  # Omni：绑定时刻附近要有有信息量的讲解
                    cands.append((src, float(s), dur))
    random.shuffle(cands)
    per, picked = {}, []
    for src, s, dur in cands:
        if src not in per:
            per[src] = 1
            picked.append((src, s, dur))
    for src, s, dur in cands:
        if len(picked) >= n:
            break
        if (src, s, dur) not in picked and per[src] < 6:
            per[src] += 1
            picked.append((src, s, dur))
    entries = []
    for i, (src, s, dur) in enumerate(picked[:n]):
        t = seed_task(src, s, 'rebind' if i % 2 == 0 else 'cancel', dur)
        d = seeds_dir / t['task_id']
        d.mkdir(parents=True, exist_ok=True)
        (d / 'candidate_task.json').write_text(json.dumps(t, ensure_ascii=False, indent=1))
        entries.append({'job_root': str(d)})
    return entries


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, choices=list(CFG.BATCHES))
    ap.add_argument('--seeds', type=int, default=0, help='从零出题：生成多少个种子骨架')
    ap.add_argument('--tasks', default='', help='或：job_root 列表文件（每行一个含 candidate_task.json 的目录）')
    ap.add_argument('--min-gap', type=float, default=30)
    ap.add_argument('--max-gap', type=float, default=60)
    ap.add_argument('--max-attempts', type=int, default=4)
    ap.add_argument('--budget', type=float, default=5.0)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--seed-offset', type=int, default=0, help='10-16 补题：候选绑定时刻错开 N 秒取（secs[N::45]），避开已有种子')
    args = ap.parse_args()
    R.use_batch(args.batch)
    OUT = CFG.RUNS / args.batch / 'runs_hr'
    OUT.mkdir(parents=True, exist_ok=True)
    windows, media_map = R.load_windows(), R.media_paths()
    WIN_G.update(windows)
    if args.tasks:
        entries = [{'job_root': x.strip()} for x in open(args.tasks) if x.strip() and not x.startswith('#')]
    else:
        entries = make_seeds(args.seeds, windows, media_map, CFG.RUNS / args.batch / 'seeds_hr', args.seed_offset)
    entries = [e for e in entries if not (OUT / json.load(open(Path(e['job_root']) / 'candidate_task.json'))['task_id'] / 'refresh_log.json').exists()]
    print('to run', len(entries), flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda e: refresh_one(e, windows, media_map, args), entries))
    summary = {'counts': {s: sum(1 for r in results if r['status'] == s) for s in {r['status'] for r in results}},
               'spent_usd': round(R._spent['usd'], 4), 'calls': R._spent['calls'], 'min_gap_s': args.min_gap}
    json.dump(summary, open(CFG.RUNS / args.batch / 'summary_hr_last_run.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
