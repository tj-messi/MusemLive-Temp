#!/usr/bin/env python3
"""得分点匹配 judge（10-17，取代整体打分的 strict/special judge，缓解 0/5 两极分化）。

设计：
  1. 评分细则固定（rubric）：slot 有 key_attributes 就用它当得分点；只有 target_text 的，先让 LLM 拆成 1–3 个原子得分点并缓存
     （rubric.json，与被测模型无关，换模型复用）。
  2. LLM 只做匹配，不直接给总分：动作是否正确（correct/partial/wrong）、每个得分点 hit/partial/miss/contradict（附回答原话证据）、
     每条 forbidden_claims 是否说出。
  3. 分数由公式确定性计算（score()）：5 ×（0.25×动作 + 0.75×得分点覆盖率）− 1.0×违禁条数 − 0.5×说反条数，截断到 [0,5]，保留 1 位。
     沉默/没回答 = 0。
  4. 画面触发（visual_event）的 slot：半双工 eval 只在用户说完话时生成、模型没机会主动开口，改用全双工 eval（eval_duplex.py，
     duplex_eval/*gpu*/）的结果：取与 [window.start_s, window.end_s + SLOT_SLACK] 有交叠的模型发言当本 slot 的回答（没有 = 沉默）。
     该分支还没有全双工结果时记 scorable=false。用户提问（user_query）slot 仍用半双工结果。
  6. 10-17 strict_v2：partial 得分点记 0.7、动作权重 0.3、说反扣 0.25；匹配提示词采用"核心意思对上即 hit"。
  7. 10-20 lenient_v3：保留沉默=0、WAIT 误开口、违禁和说反惩罚；只提高有效作答中的动作权重与 partial credit。
     该 profile 用既有匹配结果离线校准，三模型 3750 个 slot 的合并均值由 0.92 调至约 1.03，不按模型单独调参。
  5. 时机不单独计分：半双工协议下回复时刻固定在用户语音结束。
  8. 10-17 上下文纠错（context_correction）加"纠正"维度并作为主分项：LLM 额外判断 correction
     （no_correction 分支 = 用户错误理解一直没自己改，模型须像参考答案一样明确纠正；correction 分支 = 用户已自我更正，模型须以更正后为准），
     得分 = 5 ×（0.5×纠正 + 0.15×动作 + 0.35×得分点覆盖）− 违禁 − 说反；附和/沿用错误理解再扣 0.5。

用法（10-18 加 --eval-glob：直接评某个模型的全双工结果，user_query 取时间线里对用户话语的回应、visual_event 取窗口内发言；
       评分细则先从 out_v2/rubric.json 复制，保证与 MiniCPM 同一套得分点；此模式不写 per_task）：
  python3 judge_points.py run --eval-glob '<dir>/shard*/eval/evaluation_results.json' --out <dir>/judge_points
  python3 judge_points.py rescore --source-out <dir>/judge_points --out <dir>/judge_points_lenient_v3 --profile lenient_v3
  python3 judge_points.py run [--tasks-glob 'OM_V3_TEMP*'] [--sample N] [--workers 24] [--out DIR]
  python3 judge_points.py report [--out DIR]
输出（--out，默认 judge_points/out）：rubric.json、raw/<judge_id>.json（LLM 原始匹配，可续跑）、results.jsonl、summary.json；
并写 per_task/<tid>/judge_points/{judge_results.json,summary.json}（与旧 judge 同 schema，score 为新分）。
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PER_TASK = Path(os.environ.get('MUSEUMLIVE_PER_TASK_ROOT', ROOT / 'per_task'))
KEY = os.environ['OPENROUTER_API_KEY']
MODEL = os.environ.get('JUDGE_MODEL', 'qwen/qwen3.8-omni-flash')
DUPLEX = Path(os.environ.get('MUSEUMLIVE_DUPLEX_ROOT', ROOT / 'duplex_eval'))
JUDGE_EXTRA = json.loads(os.environ.get('JUDGE_EXTRA', '{}'))  # 10-18 换 judge：额外请求字段（如 {"reasoning": {"enabled": false}}、{"max_tokens": 6000}）
SLOT_SLACK = 3.0  # 画面触发 slot：窗口结束后 3 秒内开口也算
JUDGE_PROFILES = {
    'strict_v2': {
        'w_action': 0.3, 'w_points': 0.7, 'pen_forbid': 1.0, 'pen_contra': 0.25,
        'point_partial': 0.7, 'action_partial': 0.5,
        'w_corr': 0.5, 'w_cc_action': 0.15, 'w_cc_points': 0.35, 'pen_reinforce': 0.5,
        'corr_partial_must_correct': 0.4, 'corr_partial_must_follow': 0.3,
    },
    # 只放宽"已作答但不完整"：沉默仍为 0，WAIT 仍按动作正确性计分，明确违禁/说反仍扣分。
    'lenient_v3': {
        'w_action': 0.4, 'w_points': 0.6, 'pen_forbid': 0.75, 'pen_contra': 0.25,
        'point_partial': 0.8, 'action_partial': 0.7,
        'w_corr': 0.4, 'w_cc_action': 0.25, 'w_cc_points': 0.35, 'pen_reinforce': 0.5,
        'corr_partial_must_correct': 0.6, 'corr_partial_must_follow': 0.5,
    },
}
ACTIVE_PROFILE = None
PROFILE = None
W_ACTION = W_POINTS = PEN_FORBID = PEN_CONTRA = None
W_CORR = W_CC_ACTION = W_CC_POINTS = PEN_REINFORCE = None
CREDIT = ACT = CORR_CREDIT = None


def set_profile(name):
    """激活统一计分 profile；匹配 prompt 与原始标注不随模型变化。"""
    global ACTIVE_PROFILE, PROFILE, W_ACTION, W_POINTS, PEN_FORBID, PEN_CONTRA
    global W_CORR, W_CC_ACTION, W_CC_POINTS, PEN_REINFORCE, CREDIT, ACT, CORR_CREDIT
    if name not in JUDGE_PROFILES:
        raise ValueError(f'unknown judge profile: {name}')
    p = JUDGE_PROFILES[name]
    ACTIVE_PROFILE, PROFILE = name, p
    W_ACTION, W_POINTS = p['w_action'], p['w_points']
    PEN_FORBID, PEN_CONTRA = p['pen_forbid'], p['pen_contra']
    CREDIT = {'hit': 1.0, 'partial': p['point_partial'], 'miss': 0.0, 'contradict': 0.0}
    ACT = {'correct': 1.0, 'partial': p['action_partial'], 'wrong': 0.0}
    W_CORR, W_CC_ACTION, W_CC_POINTS = p['w_corr'], p['w_cc_action'], p['w_cc_points']
    PEN_REINFORCE = p['pen_reinforce']
    CORR_CREDIT = {
        'must_correct': {'corrected': 1.0, 'partial': p['corr_partial_must_correct'], 'not_corrected': 0.0, 'reinforced': 0.0},
        'must_follow': {'corrected': 1.0, 'partial': p['corr_partial_must_follow'], 'not_corrected': 0.0, 'reinforced': 0.0},
    }


set_profile(os.environ.get('JUDGE_PROFILE', 'strict_v2'))
SPENT = {'usd': 0.0, 'calls': 0}
LOCK = threading.Lock()

CORE_SYSTEM = """你在为流式视频交互 benchmark 的一个评分点标注得分点的类别。给你用户这一轮说的话、期望动作、参考答案和一组要点（出题时记录的事实）。
逐条标注：core = 回答用户这句话时必须表达的内容（参考答案里体现了它，或不说它就没回答上问题）；
support = 出题时的佐证/背景（例如画面里同时出现的字幕、镜头构图、转写原文出处），回答里不说也不算答错。至少标一条 core。
只输出 JSON：{"labels": ["core"|"support", ...]}（与要点一一对应）"""

RUBRIC_SYSTEM = """你在为第一视角博物馆流式交互 benchmark 制定评分细则。给你某个评分点的期望动作和参考答案（target_text），
把参考答案拆成 1–3 个"得分点"：每个得分点是一条可独立核对的事实或行为要求（例如"指出讲解员说的是秦人"、"反问用户指的是哪一件展品"），
不要把措辞、语气、礼貌用语当得分点；参考答案里的关键事实都要覆盖到。期望动作是 CLARIFY 时，得分点应描述澄清问句必须点出的内容。
只输出 JSON：{"points": ["...", "..."]}"""

MATCH_SYSTEM = """你是流式视频交互 benchmark 的得分点匹配 Judge。只做匹配，不打总分。
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
"forbidden": [{"id": "F1", "violated": true|false, "evidence": "..."}]}"""

MATCH_COMPACT_ADDENDUM = """

如果自由文本解释可能破坏 JSON，请改用紧凑格式：禁止输出 action_reason、evidence、why
或其他解释字段。只保留 action、observed、points、forbidden；若输入包含
correction_check，再保留 correction。points 每项只含 id/status，forbidden 每项只含
id/violated，correction 只含 status。仍须覆盖输入中的每个 P/F 编号。只输出 JSON。
"""


CORR_ADDENDUM = """

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
JSON 里增加："correction": {"status": "corrected|partial|not_corrected|reinforced", "evidence": "模型回答原话", "why": "..."}"""


def correction_ctx(r, trig_event_id):
    """context_correction：从 user_events 的 state_op 取用户的错误理解、是否已自我更正。"""
    evs = sorted(r.get('user_events') or [], key=lambda e: e['time_s'])
    tq = next((e['time_s'] for e in evs if e['event_id'] == trig_event_id), 1e9)
    prev = [e for e in evs if e['time_s'] < tq]
    claim = [e for e in prev if (e.get('state_op') or {}).get('op') == 'claim']
    corr = [e for e in prev if (e.get('state_op') or {}).get('op') == 'correct']
    if not claim: return None
    c = {'mode': 'must_follow' if corr else 'must_correct',
         'user_claim': {'utterance': claim[-1]['text'], 'meaning': claim[-1]['state_op'].get('target', '')}}
    if corr: c['user_correction'] = {'utterance': corr[-1]['text'], 'meaning': corr[-1]['state_op'].get('target', '')}
    return c


def call(messages, max_tokens=1500):
    body = {'model': MODEL, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens, 'response_format': {'type': 'json_object'}} | JUDGE_EXTRA
    last = None
    for a in range(4):
        try:
            s = requests.Session(); s.trust_env = False
            r = s.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY}, json=body, timeout=(20, 240))
            if not r.ok:
                raise RuntimeError(f'HTTP {r.status_code}: {r.text[:500]}')
            j = r.json()
            u = j.get('usage') or {}
            with LOCK:
                SPENT['calls'] += 1; SPENT['usd'] += float(u.get('cost') or 0)
            txt = j['choices'][0]['message']['content']
            txt = re.sub(r'^```(?:json)?|```$', '', txt.strip()).strip()
            return json.loads(txt)
        except Exception as e:
            last = e; time.sleep(2 * (a + 1))
    raise RuntimeError(f'judge call failed: {str(last)[:300]}')


def strip_ref(p):
    """key_attribute 里的"(对应转写: ...)"保留给 judge 当证据，但展示用短文本。"""
    return re.sub(r'\s*[（(]对应转写[:：].*?[)）]\s*$', '', p).strip()


# ---------------------------------------------------------------- 收集 slot
def load_duplex():
    """branch_key → 全双工 eval 结果（duplex_eval/*gpu*/eval/evaluation_results.json）。"""
    out = {}
    for f in sorted(DUPLEX.glob('*gpu*/eval/evaluation_results.json')):
        try:
            for r in json.load(open(f))['results']: out[r['branch_key']] = r
        except Exception:
            pass
    return out


def duplex_response(dr, sl):
    lo, hi = float(sl['window']['start_s']), float(sl['window']['end_s']) + SLOT_SLACK
    turns = [x for x in dr.get('duplex_turns') or [] if x['start_s'] <= hi and x['end_s'] >= lo]
    hist = [{'t': x['time_s'], 'user': x.get('utterance', ''), 'model': x.get('response', '')}
            for x in dr['timeline'] if x.get('type') == 'user_event' and float(x['time_s']) < lo][-4:]
    return ' / '.join(x['text'] for x in turns), [{'start_s': x['start_s'], 'end_s': x['end_s']} for x in turns], hist


def load_rows(tasks_glob='*'):
    rows = []
    dup = load_duplex()
    for f in sorted(PER_TASK.glob(f'{tasks_glob}/eval/evaluation_results.json')):
        for r in json.load(open(f))['results']:
            ev = {e.get('event_id'): e for e in r['timeline'] if e.get('type') == 'user_event'}
            proactive = bool((r.get('omni') or {}).get('tick_generate'))
            for s in r['slots']:
                sl = s['slot']
                trig = sl['trigger']
                jid = f"{r['branch_key']}::{sl['slot_id']}"
                row = {'judge_id': jid, 'task_id': r['task_id'], 'branch_id': r['branch_id'], 'capability': r['capability'],
                       'slot_id': sl['slot_id'], 'expected_action': sl['expected_action'], 'trigger_kind': trig['kind'],
                       'target_text': sl.get('target_text') or '', 'key_attributes': sl.get('key_attributes') or [],
                       'forbidden_claims': sl.get('forbidden_claims') or [], 'guidance_direction': sl.get('guidance_direction')}
                if r['capability'] == 'context_correction':
                    row['correction_check'] = correction_ctx(r, trig.get('event_id'))
                if trig['kind'] != 'user_query' and not proactive:
                    dr = dup.get(r['branch_key'])
                    if dr is None:
                        row['scorable'] = False; row['unscorable_reason'] = 'visual_event_duplex_pending'
                    else:
                        row['scorable'] = True; row['protocol'] = 'duplex'
                        row['response'], row['duplex_turns'], row['history'] = duplex_response(dr, sl)
                        row['utterance'] = ''
                else:
                    e = ev.get(trig.get('event_id'))
                    row['scorable'] = True
                    row['utterance'] = (e or {}).get('utterance', '')
                    row['response'] = (e or {}).get('response')
                    row['history'] = [{'t': x['time_s'], 'user': x.get('utterance', ''), 'model': x.get('response', '')}
                                      for x in s.get('timeline_prefix') or [] if x.get('type') == 'user_event' and x.get('event_id') != trig.get('event_id')][-4:]
                rows.append(row)
    return rows


def load_rows_duplex(pattern):
    """--eval-glob：全部 slot 都来自同一份全双工结果（Venus / Gander / MOSS 等对比模型）。"""
    import glob as _g
    rows = []
    for f in sorted(_g.glob(pattern)):
        for r in json.load(open(f))['results']:
            ev = {e.get('event_id'): e for e in r['timeline'] if e.get('type') == 'user_event'}
            for s in r['slots']:
                sl = s['slot']; trig = sl['trigger']
                row = {'judge_id': f"{r['branch_key']}::{sl['slot_id']}", 'task_id': r['task_id'], 'branch_id': r['branch_id'], 'capability': r['capability'],
                       'slot_id': sl['slot_id'], 'expected_action': sl['expected_action'], 'trigger_kind': trig['kind'],
                       'target_text': sl.get('target_text') or '', 'key_attributes': sl.get('key_attributes') or [],
                       'forbidden_claims': sl.get('forbidden_claims') or [], 'guidance_direction': sl.get('guidance_direction'),
                       'scorable': True, 'protocol': 'duplex'}
                if r['capability'] == 'context_correction':
                    row['correction_check'] = correction_ctx(r, trig.get('event_id'))
                if trig['kind'] != 'user_query':
                    row['response'], row['duplex_turns'], row['history'] = duplex_response(r, sl); row['utterance'] = ''
                else:
                    e = ev.get(trig.get('event_id'))
                    row['utterance'] = (e or {}).get('utterance', ''); row['response'] = (e or {}).get('response')
                    row['history'] = [{'t': x['time_s'], 'user': x.get('utterance', ''), 'model': x.get('response', '')}
                                      for x in s.get('timeline_prefix') or [] if x.get('type') == 'user_event' and x.get('event_id') != trig.get('event_id')][-4:]
                rows.append(row)
    return rows


# ---------------------------------------------------------------- 评分细则
def rubric_key(row):
    return hashlib.md5(json.dumps([row['expected_action'], row['target_text']], ensure_ascii=False).encode()).hexdigest()[:16]


def core_key(row):
    return hashlib.md5(json.dumps([row['utterance'], row['expected_action'], row['target_text'], row['key_attributes']], ensure_ascii=False).encode()).hexdigest()[:16]


def build_rubrics(rows, out, workers):
    path = out / 'rubric.json'
    rub = json.load(open(path)) if path.exists() else {}
    todo = {}
    ctodo = {}
    for r in rows:
        if r['scorable'] and not r['key_attributes'] and r['target_text'] and r['expected_action'] in ('SPEAK', 'CLARIFY'):
            k = rubric_key(r)
            if k not in rub: todo[k] = r
        if r['scorable'] and len(r['key_attributes']) >= 2 and r['expected_action'] in ('SPEAK', 'CLARIFY'):
            k = 'core_' + core_key(r)
            if k not in rub: ctodo[k] = r

    def one_core(item):
        k, r = item
        try:
            j = call([{'role': 'system', 'content': CORE_SYSTEM},
                      {'role': 'user', 'content': json.dumps({'user_utterance': r['utterance'], 'expected_action': r['expected_action'],
                                                               'reference_answer': r['target_text'],
                                                               'points': [{'id': f'P{i + 1}', 'text': p} for i, p in enumerate(r['key_attributes'])]},
                                                              ensure_ascii=False)}], 400)
            lab = [str(x).lower() for x in j.get('labels', [])]
            if len(lab) != len(r['key_attributes']) or 'core' not in lab: return k, None
            return k, lab
        except Exception:
            return k, None
    with ThreadPoolExecutor(workers) as ex:
        for k, lab in ex.map(one_core, ctodo.items()):
            if lab: rub[k] = lab
    print(f'core labels: {len(ctodo)} new', flush=True)

    def one(item):
        k, r = item
        try:
            j = call([{'role': 'system', 'content': RUBRIC_SYSTEM},
                      {'role': 'user', 'content': json.dumps({'capability': r['capability'], 'expected_action': r['expected_action'],
                                                               'target_text': r['target_text']}, ensure_ascii=False)}], 600)
            pts = [p for p in j.get('points', []) if isinstance(p, str) and p.strip()][:3]
            return k, pts or [r['target_text']]
        except Exception:
            return k, None
    with ThreadPoolExecutor(workers) as ex:
        for k, pts in ex.map(one, todo.items()):
            if pts: rub[k] = pts
    json.dump(rub, open(path, 'w'), ensure_ascii=False, indent=1)
    print(f'rubric: {len(rub)} cached, {len(todo)} new', flush=True)
    return rub


def core_of(row, rub, pts):
    """每个得分点是否核心；key_attributes 只有 1 条、或标注失败时全部当核心。"""
    if row['key_attributes'] and len(row['key_attributes']) >= 2:
        lab = rub.get('core_' + core_key(row))
        if lab and len(lab) == len(pts): return [x == 'core' for x in lab]
    return [True] * len(pts)


def points_of(row, rub):
    if row['key_attributes']:
        return list(row['key_attributes'])
    if row['expected_action'] in ('SPEAK', 'CLARIFY') and row['target_text']:
        return rub.get(rubric_key(row)) or [row['target_text']]
    return []


# ---------------------------------------------------------------- 匹配 + 计分
def valid_match(m, row, pts):
    if not isinstance(m, dict):
        return False
    if m.get('action') not in {'correct', 'partial', 'wrong'}:
        return False
    if m.get('observed') not in {'answer', 'clarify', 'ack_only', 'refuse', 'silent', 'other'}:
        return False
    point_rows = m.get('points') or []
    forbidden_rows = m.get('forbidden') or []
    if not all(isinstance(x, dict) for x in point_rows + forbidden_rows):
        return False
    if {x.get('id') for x in point_rows} != {f'P{i + 1}' for i in range(len(pts))}:
        return False
    if any(x.get('status') not in {'hit', 'partial', 'miss', 'contradict'} for x in point_rows):
        return False
    if {x.get('id') for x in forbidden_rows} != {
        f'F{i + 1}' for i in range(len(row['forbidden_claims']))
    }:
        return False
    if row.get('correction_check') and (
        (m.get('correction') or {}).get('status')
        not in {'corrected', 'partial', 'not_corrected', 'reinforced'}
    ):
        return False
    return True


def match(row, pts):
    payload = {'capability': row['capability'], 'slot_id': row['slot_id'], 'expected_action': row['expected_action'], 'trigger': row['trigger_kind'],
               'reference_answer': row['target_text'],
               'points': [{'id': f'P{i + 1}', 'text': p} for i, p in enumerate(pts)],
               'forbidden': [{'id': f'F{i + 1}', 'text': p} for i, p in enumerate(row['forbidden_claims'])],
               'previous_turns': row['history'], 'user_utterance_this_turn': row['utterance'], 'model_response_this_turn': row['response']}
    if row.get('guidance_direction'):
        payload['guidance_direction'] = row['guidance_direction']
    system = MATCH_SYSTEM
    if row.get('correction_check'):
        payload['correction_check'] = row['correction_check']; system = MATCH_SYSTEM + CORR_ADDENDUM
    msgs = [{'role': 'system', 'content': system}, {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}]
    last = None
    for _ in range(3):
        try:
            m = call(msgs)
            if valid_match(m, row, pts):
                return m
            last = RuntimeError('judge returned an incomplete match schema')
        except Exception as exc:
            last = exc

    # Some providers truncate free-text reason/evidence fields while still
    # returning HTTP 200. Retry the same Judge with the same rubric and labels,
    # but a compact JSON schema containing only fields used by deterministic
    # scoring. This changes neither the evaluated response nor the score rule.
    compact_msgs = [
        {'role': 'system', 'content': system + MATCH_COMPACT_ADDENDUM},
        {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)},
    ]
    for _ in range(3):
        try:
            m = call(compact_msgs)
            if valid_match(m, row, pts):
                m['_judge_fallback'] = 'compact_json'
                return m
            last = RuntimeError('judge returned an incomplete compact match schema')
        except Exception as exc:
            last = exc
    raise RuntimeError(f'judge returned malformed structure after compact fallback: {last}')


def silent(resp):
    return resp is None or not str(resp).strip() or 'SILENCE' in str(resp).upper()


def score(row, pts, m, core=None):
    """确定性计分，返回 (score, detail)。覆盖率：有佐证点时 = 0.8×核心覆盖 + 0.2×佐证覆盖，否则 = 核心覆盖。"""
    exp = row['expected_action']
    if exp != 'WAIT' and silent(row['response']):
        return 0.0, {'action': 'wrong', 'observed': 'silent', 'coverage': 0.0, 'n_points': len(pts), 'statuses': ['miss'] * len(pts),
                     'core': core or [True] * len(pts), 'forbidden_hits': 0, 'contradict': 0, 'note': 'silent'}
    act = ACT.get((m.get('action') or '').lower(), 0.0)
    st = {p.get('id'): (p.get('status') or 'miss').lower() for p in m.get('points') or []}
    statuses = [st.get(f'P{i + 1}', 'miss') for i in range(len(pts))]
    core = core or [True] * len(statuses)
    cs = [CREDIT.get(x, 0.0) for x, c in zip(statuses, core) if c]
    ss = [CREDIT.get(x, 0.0) for x, c in zip(statuses, core) if not c]
    if not statuses:
        cov = None
    elif ss:
        cov = 0.8 * (sum(cs) / len(cs)) + 0.2 * (sum(ss) / len(ss))
    else:
        cov = sum(cs) / len(cs)
    n_contra = sum(s == 'contradict' for s in statuses)
    n_forbid = sum(bool(f.get('violated')) for f in m.get('forbidden') or [])
    cc = row.get('correction_check'); corr = None; n_reinf = 0
    if cc and exp != 'WAIT':
        cst = ((m.get('correction') or {}).get('status') or 'not_corrected').lower()
        corr = CORR_CREDIT[cc['mode']].get(cst, 0.0); n_reinf = int(cst == 'reinforced')
    if corr is not None:
        base = 5.0 * (W_CORR * corr + W_CC_ACTION * act + W_CC_POINTS * (cov if cov is not None else act)) - PEN_REINFORCE * n_reinf
    elif exp == 'WAIT':
        base = 5.0 * act
    elif cov is None:
        base = 5.0 * act
    else:
        base = 5.0 * (W_ACTION * act + W_POINTS * cov)
    sc = max(0.0, min(5.0, base - PEN_FORBID * n_forbid - PEN_CONTRA * n_contra))
    return round(sc, 1), {'action': m.get('action'), 'observed': m.get('observed'), 'coverage': None if cov is None else round(cov, 3),
                          'n_points': len(pts), 'statuses': statuses, 'core': core, 'forbidden_hits': n_forbid, 'contradict': n_contra} | \
        ({'correction_mode': cc['mode'], 'correction': (m.get('correction') or {}).get('status'), 'correction_credit': corr} if corr is not None else {})


def judge_row(row, rub, raw_dir):
    if not row['scorable']:
        return {**slim(row), 'scorable': False, 'reason': row['unscorable_reason']}
    pts = points_of(row, rub)
    f = raw_dir / (hashlib.md5(row['judge_id'].encode()).hexdigest() + '.json')
    if f.exists():
        m = json.load(open(f))['match']
    elif row['expected_action'] != 'WAIT' and silent(row['response']):
        m = {'action': 'wrong', 'observed': 'silent', 'points': [], 'forbidden': []}
    elif row['expected_action'] == 'WAIT' and silent(row['response']):
        m = {'action': 'correct', 'observed': 'silent', 'points': [], 'forbidden': []}
    else:
        try:
            m = match(row, pts)
        except Exception as e:
            return {**slim(row), 'scorable': True, 'status': 'api_failed', 'error': str(e)[:300]}
        json.dump({'judge_id': row['judge_id'], 'points': pts, 'match': m}, open(f, 'w'), ensure_ascii=False)
    sc, d = score(row, pts, m, core_of(row, rub, pts))
    return {**slim(row), 'scorable': True, 'status': 'ok', 'judge_profile': ACTIVE_PROFILE,
            'score': sc, **d, 'points': [strip_ref(p) for p in pts], 'match': m}


def slim(row):
    return {k: row[k] for k in ('judge_id', 'task_id', 'branch_id', 'capability', 'slot_id', 'expected_action', 'trigger_kind')} | \
           ({'response': row.get('response'), 'utterance': row.get('utterance'), 'protocol': row.get('protocol', 'half_duplex')} if row.get('scorable') else {}) | \
           ({'duplex_turns': row['duplex_turns']} if row.get('duplex_turns') is not None else {})


# ---------------------------------------------------------------- 主流程
def write_per_task(results):
    by = defaultdict(list)
    for r in results: by[r['task_id']].append(r)
    for tid, rs in by.items():
        d = PER_TASK / tid / 'judge_points'; d.mkdir(parents=True, exist_ok=True)
        out = []
        for r in rs:
            if not r['scorable']:
                out.append({'judge_id': r['judge_id'], 'status': 'not_scorable', 'reason': r['reason']}); continue
            if r.get('status') != 'ok':
                out.append({'judge_id': r['judge_id'], 'status': 'api_failed', 'error': r.get('error')}); continue
            reason = '；'.join(f"{s}:{p[:40]}" for s, p in zip(r['statuses'], r['points']))
            out.append({'judge_id': r['judge_id'], 'status': 'ok', 'judgment': {
                'score': r['score'], 'reason': f"动作 {r['action']}（{r.get('observed')}）；覆盖率 {r['coverage']}；违禁 {r['forbidden_hits']}；说反 {r['contradict']}｜{reason}",
                'action': r['action'], 'coverage': r['coverage'], 'point_hits': [{'point': p, 'status': s} for p, s in zip(r['points'], r['statuses'])],
                'forbidden_hits': r['forbidden_hits'], 'core': r.get('core'), 'match': r['match']} | \
                ({'correction': {'mode': r['correction_mode'], 'status': r.get('correction'), 'credit': r.get('correction_credit'),
                                 'evidence': (r['match'].get('correction') or {}).get('evidence'), 'why': (r['match'].get('correction') or {}).get('why')}}
                 if r.get('correction_mode') else {})})
        ok = [x['judgment']['score'] for x in out if x['status'] == 'ok']
        json.dump({'judge': f'omni_points_judge_{ACTIVE_PROFILE}', 'profile': ACTIVE_PROFILE, 'results': out},
                  open(d / 'judge_results.json', 'w'), ensure_ascii=False, indent=1)
        json.dump({'status': 'complete', 'completed': len(ok), 'not_scorable': sum(x['status'] == 'not_scorable' for x in out),
                   'api_failed': sum(x['status'] == 'api_failed' for x in out), 'mean_score': sum(ok) / len(ok) if ok else None},
                  open(d / 'summary.json', 'w'), ensure_ascii=False, indent=1)


def report(out):
    res = [json.loads(l) for l in open(out / 'results.jsonl')]
    ok = [r for r in res if r['scorable'] and r.get('status') == 'ok']
    hist = Counter(min(int(r['score']), 4) for r in ok)  # 0-1,1-2,2-3,3-4,4-5
    per = defaultdict(list)
    for r in ok: per[r['capability']].append(r)
    profiles = sorted({r.get('judge_profile') for r in ok if r.get('judge_profile')})
    summ = {'judge_profile': profiles[0] if len(profiles) == 1 else ACTIVE_PROFILE,
            'scoring': PROFILE,
            'slots': len(res), 'scorable_ok': len(ok), 'not_scorable': sum(not r['scorable'] for r in res),
            'api_failed': sum(r['scorable'] and r.get('status') != 'ok' for r in res),
            'mean': round(sum(r['score'] for r in ok) / len(ok), 2) if ok else None,
            'hist': {f'[{k},{k + 1}{")" if k < 4 else "]"}': hist[k] for k in range(5)},
            'exact_0': sum(r['score'] == 0 for r in ok), 'exact_5': sum(r['score'] == 5 for r in ok),
            'per_cap': {c: {'n': len(v), 'mean': round(sum(r['score'] for r in v) / len(v), 2),
                            'point_hit_rate': round(sum(s == 'hit' for r in v for s in r['statuses']) / max(1, sum(len(r['statuses']) for r in v)), 3),
                            'core_hit_rate': round(sum(s == 'hit' for r in v for s, c in zip(r['statuses'], r.get('core') or []) if c) / max(1, sum(sum(r.get('core') or []) for r in v)), 3),
                            'action_correct': round(sum(r['action'] == 'correct' for r in v) / len(v), 3),
                            'not_scorable': sum(1 for x in res if x['capability'] == c and not x['scorable'])}
                        for c, v in sorted(per.items())},
            'spent_usd': round(SPENT['usd'], 4), 'calls': SPENT['calls'],
            'compact_json_fallbacks': sum(
                (r.get('match') or {}).get('_judge_fallback') == 'compact_json' for r in ok
            )}
    json.dump(summ, open(out / 'summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    return summ


def rescore(source, out):
    """复用已有 LLM 匹配标注，只重算确定性分数；不发 API 请求。"""
    src = [json.loads(l) for l in open(source / 'results.jsonl') if l.strip()]
    dst = []
    for r in src:
        if not r.get('scorable') or r.get('status') != 'ok':
            dst.append({**r, 'judge_profile': ACTIVE_PROFILE})
            continue
        row = {'expected_action': r['expected_action'], 'response': r.get('response')}
        if r.get('correction_mode'):
            row['correction_check'] = {'mode': r['correction_mode']}
        pts = r.get('points') or []
        sc, detail = score(row, pts, r.get('match') or {}, r.get('core'))
        keep = {k: v for k, v in r.items() if k not in {
            'score', 'action', 'observed', 'coverage', 'n_points', 'statuses', 'core',
            'forbidden_hits', 'contradict', 'correction_mode', 'correction', 'correction_credit',
            'judge_profile', 'note',
        }}
        dst.append({**keep, 'judge_profile': ACTIVE_PROFILE, 'score': sc, **detail})
    with open(out / 'results.jsonl', 'w') as f:
        for r in dst:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    metadata = {'judge_profile': ACTIVE_PROFILE, 'scoring': PROFILE, 'source_out': str(source.resolve()),
                'source_results_sha256': hashlib.sha256((source / 'results.jsonl').read_bytes()).hexdigest(),
                'api_calls': 0}
    json.dump(metadata, open(out / 'rescore_metadata.json', 'w'), ensure_ascii=False, indent=1)
    report(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['run', 'report', 'write', 'rescore'])
    ap.add_argument('--tasks-glob', default='*')
    ap.add_argument('--sample', type=int, default=0, help='按题型分层随机抽 N 个可评 slot（试跑）')
    ap.add_argument('--workers', type=int, default=24)
    ap.add_argument('--out', default=str(HERE / 'out_v2'))
    ap.add_argument('--no-per-task', action='store_true')
    ap.add_argument('--ids', default='', help='只跑这些 judge_id（每行一个）')
    ap.add_argument('--eval-glob', default='', help='评对比模型的全双工结果（所有 slot 都用它）；不写 per_task')
    ap.add_argument('--profile', choices=sorted(JUDGE_PROFILES), default=os.environ.get('JUDGE_PROFILE', 'strict_v2'))
    ap.add_argument('--source-out', default='', help='rescore 的旧 judge 输出目录；只复用匹配结果，不调用 API')
    a = ap.parse_args()
    set_profile(a.profile)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True); raw = out / 'raw'; raw.mkdir(exist_ok=True)
    if a.cmd == 'rescore':
        if not a.source_out:
            ap.error('rescore requires --source-out')
        rescore(Path(a.source_out), out); return
    if a.cmd == 'report':
        report(out); return
    if a.cmd == 'write':  # 从 results.jsonl 重写 per_task/<tid>/judge_points/
        write_per_task([json.loads(l) for l in open(out / 'results.jsonl')]); print('per_task written'); return
    if a.eval_glob:
        a.no_per_task = True
        if not (out / 'rubric.json').exists():
            import shutil
            source = Path(os.environ['MUSEUMLIVE_RUBRIC_FILE']).expanduser().resolve()
            shutil.copy(source, out / 'rubric.json')
        rows = load_rows_duplex(a.eval_glob)
    else:
        rows = load_rows(a.tasks_glob)
    if a.ids:
        want = {x.strip() for x in open(a.ids) if x.strip()}
        rows = [r for r in rows if r['judge_id'] in want]
    if a.sample:
        random.seed(20261017)
        by = defaultdict(list)
        for r in rows:
            if r['scorable']: by[r['capability']].append(r)
        k = max(1, a.sample // len(by))
        rows = [r for v in by.values() for r in random.sample(v, min(k, len(v)))]
    print(f'slots {len(rows)} (scorable {sum(r["scorable"] for r in rows)})', flush=True)
    rub = build_rubrics(rows, out, a.workers)
    t0 = time.time(); res = []
    with ThreadPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(lambda r: judge_row(r, rub, raw), rows), 1):
            res.append(r)
            if i % 200 == 0: print(f'{i}/{len(rows)} {time.time() - t0:.0f}s ${SPENT["usd"]:.3f}', flush=True)
    with open(out / 'results.jsonl', 'w') as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    if not a.sample and not a.ids and not a.no_per_task:
        write_per_task(res)
    report(out)


if __name__ == '__main__':
    main()
