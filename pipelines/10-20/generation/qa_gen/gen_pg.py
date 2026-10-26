"""proactive_guidance 出题线（Omni 版 10-16：提问必须由刚听到的内容引出（audio_cue），要点分 [音频]/[视觉]；其余沿用 pgv2 协议）：主动引导用户调整视角以获取当前看不到的信息。

用法: python3 gen_pg.py --batch 50|316 [--videos N] [--per-video 1] [--target 10] [--budget 3]
输出: runs/<batch>/runs_pg/<src>_<t>/{log.json, task.json, candidate_task.json}
流程（每个候选时刻）:
  1. 粗筛: 逐秒摘要里"看不清/截断/反光/遮挡"之后 3-15s 出现"清晰/完整/靠近"
  2. Qwen 只看摘要做初判（同一物体？是否因镜头移动变清楚？）
  3. Qwen 看真实帧出题（分支 A 需要引导 + 信息出现后主动回答；分支 B 问当前看得清的物体，直接回答）
  4. 代码检查 + Qwen 看真实帧核验（t 时看不到、t2 时看得到且答案正确、镜头移动方向与引导一致、对照物 t 时看得清）
  5. 不过则带累积原因重写，最多 3 次
"""
import argparse
import json
import re
import sys
import threading
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R  # noqa: E402  call / frame / img_part / annotation / load_windows / media_paths / bigrams / GENERIC
import config as CFG  # noqa: E402

OUT = None  # runs/<batch>/runs_pg，由 main() 按 --batch 设定
windows_g = {}  # 逐秒窗口（det_check 核对 audio_cue 用）
BAD = re.compile(r'看不清|无法辨认|不可读|难以辨认|模糊|被截断|截断|只露出|露出一角|部分遮挡|被遮挡|反光|过远|太远|画面边缘|半露')
GOOD = re.compile(r'清晰可读|清晰可见|完整(显示|呈现|入画)|可以读出|可辨认|居中|拉近|靠近|特写|正对')
TEXTY = re.compile(r'文字|字样|字迹|牌|标签|展签|说明|标题|OCR')
READABLE = re.compile(r'清晰可读|可以读出|可辨认|写着|字样|标题为|内容为|“')
NEGATIVE = re.compile(r'没有找到|没找到|看不清|无法(辨认|看清|读出|确认)|看不到|不清楚|并没有')
DIRECTION = re.compile(r'往|向|靠近|凑近|走近|左|右|上|下|侧面|背面|绕|蹲|抬|拉近|退后|后退|换个?角度|避开')

SCREEN_SYSTEM = """你在为第一视角博物馆视频挑选"主动引导"出题时刻。给你一段逐秒摘要（含 🔊 音频）和音频转写。
全模态要求：t 附近（t 前 15 秒内）必须有人说到这个物体或这条信息（讲解员/主播提到它、广播提到它），用户的提问要由这句话引出；转写里没有相关的话就 ok=false。
判断是否存在这样的时刻：在 t 秒，某个具体物体上的某条信息（展签/牌子上的文字、物体某一面的细节、被遮挡部分等）在画面里拿不到；
之后 3–20 秒内（t2），镜头**因为拍摄者的移动**（平移、靠近、绕行、抬高等）让同一物体的这条信息变得清楚可得。
必须是同一个物体；仅仅是换了一个场景不算。
只输出 JSON：{"ok": bool, "object": "...", "info": "拿不到的是什么信息", "t": 数字, "t2": 数字,
"camera_motion": "t 到 t2 镜头怎么动（如向右平移并靠近）", "audio_cue": {"start": 秒, "end": 秒, "text": "转写里引出提问的那句原话"}, "reason": "..."}"""

AUTHOR_SYSTEM = """你在为第一视角博物馆流式交互 benchmark 出"主动引导视角"题。用户戴着摄像头参观，模型实时看画面。
题目考：用户问的信息当前画面里拿不到时，模型能否**主动、具体地引导用户调整视角**（往哪个方向、对准什么），并在视角调整后信息出现时主动给出答案。

""" + R.OMNI_RULE + """
本题型的音频用法：need_guidance 分支的提问必须由**刚听到的一句话**引出（audio_cue，转写原句，说出时间在 t 前 15 秒内），问句里要体现“她刚说的/广播里提的”这种引用，但不能说出答案；answer_key_facts 每条以 [视觉] 或 [音频] 开头，至少一条 [音频]（可以是 audio_cue 里关于该物体的内容）。
""" + R.TIMING_RULE + """（t 可以在初判的基础上微调到间隙里，t2 相应保持与画面一致）
给你：t 前后和 t2 前后的真实帧、逐秒标注（含 OCR）、音频转写、初判结论。写两个分支，提问时间都是 t：
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
 "audio_cue": {"start": 秒, "end": 秒, "text": "转写原句"},
 "need_guidance": {"query": "...", "guide_text": "...", "answer_text": "...", "answer_key_facts": ["[视觉] 可核对的答案要点1", "[音频] ..."],
                   "forbidden_claims": ["编造的未读清内容", "错误方向"]},
 "no_guidance": {"query": "...", "control_object": "...", "direct_text": "...", "answer_key_facts": ["..."],
                 "forbidden_claims": ["要求用户移动视角"]}}"""

VERIFY_SYSTEM = """你是严格的视觉核验员。给你三组真实帧：
P 组：用户提问时刻 t 及之前 1 秒；Q 组：后来时刻 t2 附近；并给出题目信息。逐项判断：
1. info_hidden_at_t：在 P 组帧里，目标信息（answer_key_facts）是否确实看不到/读不清（看得清就是 false）；
2. answer_visible_at_t2：在 Q 组帧里，answer_key_facts 是否每条都能清楚看到/读到，逐条给出；
3. direction_matches：从 P 到 Q，镜头/拍摄者的实际移动是否与 guidance_direction 一致；
4. control_visible_at_t：对照物体及其答案要点在 P 组帧里是否清楚可见；
5. answers_question：Q 组帧里出现的信息是否真正回答了用户的问题（只是看到物体其他部分、没有出现被问的信息，则为 false）。
只输出 JSON：{"info_hidden_at_t": bool, "answer_facts": [{"fact": "...", "visible": bool, "why": "..."}],
"direction_matches": bool, "actual_motion": "...", "control_visible_at_t": bool, "control_note": "...", "same_object": bool,
"answers_question": bool}"""


BAD_X = re.compile(BAD.pattern + r'|侧面|斜角|角度|逆光|局部|太小|较小|远处|背面')
READABLE_X = re.compile(READABLE.pattern + r'|清晰|完整|全貌|正面|特写|细节')
ANY = re.compile('.')


def _tried_pg():
    tried, passed = defaultdict(list), defaultdict(list)
    if OUT is None or not OUT.exists():
        return tried, passed
    for d in OUT.iterdir():
        f = d / 'log.json'
        if not f.exists():
            continue
        src, t = d.name.rsplit('_', 1)
        tried[src].append(int(t))
        try:
            if json.load(open(f)).get('status') == 'pass':
                passed[src].append(int(t))
        except Exception:
            pass
    return tried, passed


def candidates(windows, per_video, max_videos, expand=0):
    """expand=True（10-16 补题）：间隔 30→20 秒、t2 搜索 3–15→3–25 秒、遮挡关键词加宽；跳过离已试时刻 <20 秒、离已通过时刻 <30 秒的；筛选/出题/核验规则不变。"""
    out = []
    tried, passed = _tried_pg() if expand else ({}, {})
    bad, gap, dmax = (BAD_X, 20, 26) if expand else (BAD, 30, 16)
    # expand>=2：不再要求是文字类信息（被遮挡的物体细节也算，screen 提示本就允许），揭示帧只要清晰/完整/正面/特写
    texty, readable = (ANY, READABLE_X) if expand >= 2 else (TEXTY, READABLE)
    for src in sorted(windows):
        if src in R._cfg.get('content_blocked', set()):  # 内容审查固定拦截的视频，跳过避免白花钱
            continue
        w, got, last = windows[src], 0, -99
        for s in sorted(w):
            if got >= per_video:
                break
            if s < last + gap or not (bad.search(w[s]['summary']) and texty.search(w[s]['summary'])):
                continue
            if expand and (any(abs(s - b) < 20 for b in tried.get(src, [])) or any(abs(s - b) < 30 for b in passed.get(src, []))):
                continue
            if not R.informative_speech(w, s - 15, s + 1, 30):  # Omni：提问前 15 秒内要有有信息量的人声
                continue
            for d in range(3, dmax):
                nxt = w.get(s + d, {}).get('summary', '')
                if nxt and readable.search(nxt) and texty.search(nxt) and not bad.search(nxt):
                    out.append({'source_id': src, 't': s, 't2': s + d})
                    got, last = got + 1, s
                    break
        if max_videos and len({c['source_id'] for c in out}) >= max_videos:
            break
    return out


def screen(c, windows, log_dir, budget):
    w = windows[c['source_id']]
    text = '\n'.join(R.line(w, s) for s in range(c['t'] - 2, c['t2'] + 3) if s in w)
    text += f"\n\n音频转写（{c['t'] - 15}–{c['t2'] + 3}s）：\n" + R.transcript(w, c['t'] - 15, c['t2'] + 3)
    text = R.lang_rule(R.video_lang(w)) + '\n' + text
    return R.call([{'role': 'system', 'content': SCREEN_SYSTEM}, {'role': 'user', 'content': text}],
                  'screen', log_dir, budget, max_tokens=800)


def author_messages(c, sc, zf, media, feedback, windows):
    src, t, t2 = c['source_id'], sc['t'], sc['t2']
    ann = [a for s in sorted({int(t) - 1, int(t), int(t2), int(t2) + 1}) if (a := R.annotation(zf, src, s))]
    tr = R.transcript(windows[src], t - 15, t2 + 3)
    content = [{'type': 'text', 'text': R.lang_rule(R.video_lang(windows[src])) + f"\n初判：{json.dumps(sc, ensure_ascii=False)}\n逐秒标注：{json.dumps(ann, ensure_ascii=False)}\n音频转写（{t - 15:.0f}–{t2 + 3:.0f}s）：\n{tr}"}]
    for tt in (t - 1, t, t + 0.5, t2, t2 + 0.5, t2 + 1):
        content += [{'type': 'text', 'text': f'[帧 {tt:.1f}s]'}, R.img_part(R.frame(src, media, tt))]
    if feedback:
        content.append({'type': 'text', 'text': '之前的版本没有通过检查，以下问题全部要避免：\n' + '\n'.join(f'- {x}' for x in feedback)})
    return [{'role': 'system', 'content': AUTHOR_SYSTEM}, {'role': 'user', 'content': content}]


def det_check(r, duration):
    if not r.get('feasible'):
        return ['infeasible']
    fails = []
    t, t2 = r['t'], r['t2']
    if not 3 <= t2 - t <= 20:
        fails.append(f't2-t 应在 3–20s，现为 {t2 - t}')
    if t2 + 3 > duration:
        fails.append('t2 超出视频长度')
    ng, nog = r['need_guidance'], r['no_guidance']
    if NEGATIVE.search(ng['answer_text']):
        fails.append('answer_text 没有真正回答问题（出现“没找到/看不清”）：只能选 t2 时信息确实出现的时刻，否则 feasible=false')
    if not DIRECTION.search(ng['guide_text']):
        fails.append('guide_text 没有具体方向/动作')
    for fact in ng['answer_key_facts']:
        fb = R.bigrams(fact) - R.GENERIC
        if fb and len(fb & R.bigrams(ng['query'])) / len(fb) >= 0.5:
            fails.append(f'问句里已经包含答案要点“{fact}”')
    if re.search(r'挪|移动|靠近|往左|往右|换个?角度', nog['direct_text']):
        fails.append('no_guidance 分支的回答不应引导用户移动')
    if not ng['answer_key_facts'] or not nog['answer_key_facts']:
        fails.append('answer_key_facts 不能为空')
    if windows_g:
        lang = R.video_lang(windows_g.get(r.get('_src', ''), {}))
        for txt in (ng['query'], nog['query'], ng['guide_text'], ng['answer_text'], nog['direct_text']):
            m = R.lang_bad(txt, lang)
            if m: fails.append('语言不符：' + m)
        clash = R.speech_clash(windows_g.get(r.get('_src', ''), {}), {'t': t})
        if clash:
            fails.append('用户开口时刻与人声重叠：' + clash)
    cue = r.get('audio_cue') or {}
    if not (cue.get('text') or '').strip():
        fails.append('缺少 audio_cue：提问必须由转写里的一句话引出')
    else:
        try:
            if not (t - 15 <= float(cue.get('start', -1)) <= t + 0.5):
                fails.append(f"audio_cue 的时间 {cue.get('start')} 应在提问前 15 秒内")
        except (TypeError, ValueError):
            fails.append('audio_cue 缺少 start/end 时间')
        if windows_g:
            tr = R.transcript(windows_g.get(r.get('_src', ''), {}), t - 20, t + 2)
            cb = R.bigrams(cue['text'])
            if cb and len(cb & R.bigrams(tr)) / len(cb) < 0.5:
                fails.append('audio_cue 的文字在转写里对不上，只能引用转写原句')
    for fact in ng['answer_key_facts'] + nog['answer_key_facts']:
        if not (fact or '').strip().startswith(('[音频]', '[视觉]')):
            fails.append(f'要点“{fact}”必须以 [音频] 或 [视觉] 开头')
    if not any((f or '').strip().startswith('[音频]') for f in ng['answer_key_facts']):
        fails.append('need_guidance 的 answer_key_facts 至少要有一条 [音频] 要点')
    return fails


def visual_check(c, r, media, log_dir, attempt, budget):
    src, t, t2 = c['source_id'], r['t'], r['t2']
    vis = lambda fs: [f[4:].strip() for f in fs if (f or '').strip().startswith('[视觉]')]
    info = {'guidance_direction': r['guidance_direction'], 'question': r['need_guidance']['query'],
            'answer_key_facts': vis(r['need_guidance']['answer_key_facts']) or r['need_guidance']['answer_key_facts'],
            'control_object': r['no_guidance']['control_object'], 'control_facts': vis(r['no_guidance']['answer_key_facts']) or r['no_guidance']['answer_key_facts']}
    content = [{'type': 'text', 'text': '题目信息：' + json.dumps(info, ensure_ascii=False) + '\nP 组：'}]
    for tt in (t - 1, t):
        content += [{'type': 'text', 'text': f'[P {tt:.1f}s]'}, R.img_part(R.frame(src, media, tt))]
    content.append({'type': 'text', 'text': 'Q 组：'})
    for tt in (t2, t2 + 0.5, t2 + 1):
        content += [{'type': 'text', 'text': f'[Q {tt:.1f}s]'}, R.img_part(R.frame(src, media, tt))]
    v = R.call([{'role': 'system', 'content': VERIFY_SYSTEM}, {'role': 'user', 'content': content}],
               f'a{attempt}_verify', log_dir, budget, max_tokens=1500)
    fails = []
    if not v.get('same_object', True):
        fails.append('核验员认为 t 与 t2 不是同一个物体')
    if not v.get('answers_question'):
        fails.append('t2 画面里出现的信息并没有回答用户的问题')
    if not v.get('info_hidden_at_t'):
        fails.append('提问时刻目标信息其实已经看得清，不需要引导')
    for f in v.get('answer_facts', []):
        if not f.get('visible'):
            fails.append(f"答案要点“{f.get('fact')}”在 t2 帧里看不清（{f.get('why', '')}），只写确实读得清的内容")
    if not v.get('direction_matches'):
        fails.append(f"引导方向与镜头实际移动不符（实际：{v.get('actual_motion', '')}）")
    if not v.get('control_visible_at_t'):
        fails.append(f"对照物在提问时刻不够清楚（{v.get('control_note', '')}），换一个当时看得清的物体")
    return fails, v


def to_task(c, r):
    t, t2 = float(r['t']), float(r['t2'])
    ng, nog = r['need_guidance'], r['no_guidance']
    task_id = f"OM_PGV2_{c['source_id']}_{int(t):05d}"
    q = lambda text: {'event_id': 'ue_query', 'time_s': t, 'text': text, 'speech_overlap': bool(R.speech_at(windows_g.get(c['source_id'], {}), t)),
                      'state_op': {'op': 'query', 'id': 'info_request', 'target': 'requested_info'}}
    return {
        'source_id': c['source_id'], 'capability': 'proactive_guidance', 'interaction_family': 'viewpoint_guidance',
        'task_id': task_id, 'status': 'omni_candidate_pending_human_review', 'modality': 'omni', 'language': R.video_lang(windows_g.get(c['source_id'], {})), 'audio_cue': r.get('audio_cue'),
        'session': {'start_s': max(0.0, t - 5), 'end_s': t2 + 3, 'observation_hz': 5, 'history_policy': 'causal_prefix'},
        'shared_future_from_s': t, 'diagnostic_slot_ids': ['sl_guide', 'sl_answer', 'sl_direct'],
        'branches': [
            {'branch_id': 'need_guidance', 'user_events': [q(ng['query'])], 'slots': [
                {'slot_id': 'sl_guide', 'trigger': {'kind': 'user_query', 'event_id': 'ue_query'},
                 'window': {'start_s': t, 'end_s': t + 3}, 'expected_action': 'SPEAK', 'target_text': ng['guide_text'],
                 'guidance_direction': r['guidance_direction'],
                 'forbidden_claims': ng.get('forbidden_claims', []) + ['直接编造当前看不清的信息']},
                {'slot_id': 'sl_answer', 'trigger': {'kind': 'visual_event', 'evidence_ids': [f"{c['source_id']}_S{int(t2):06d}"]},
                 'window': {'start_s': t2, 'end_s': t2 + 3}, 'expected_action': 'SPEAK', 'target_text': ng['answer_text'],
                 'key_attributes': ng['answer_key_facts'], 'forbidden_claims': ng.get('forbidden_claims', [])}]},
            {'branch_id': 'no_guidance', 'user_events': [q(nog['query'])], 'slots': [
                {'slot_id': 'sl_direct', 'trigger': {'kind': 'user_query', 'event_id': 'ue_query'},
                 'window': {'start_s': t, 'end_s': t + 3}, 'expected_action': 'SPEAK', 'target_text': nog['direct_text'],
                 'key_attributes': nog['answer_key_facts'],
                 'forbidden_claims': nog.get('forbidden_claims', []) + ['要求用户移动或调整视角']}]}],
        'pgv2': {'version': 'pg_omni_v1_20261016', 'model': R.MODEL, 'control_object': nog['control_object']},
    }


def run_one(c, windows, media_map, args, stop):
    if stop.is_set():
        return None
    src = c['source_id']
    media, duration = media_map[src]
    log_dir = OUT / f"{src}_{c['t']:05d}"
    log_dir.mkdir(parents=True, exist_ok=True)
    rec = {'candidate': c, 'attempts': []}
    status = 'screen_rejected'
    try:
        sc = screen(c, windows, log_dir, args.budget)
        rec['screen'] = sc
        if sc.get('ok'):
            status, feedback = 'needs_revision', []
            with zipfile.ZipFile(R.ARCHIVE) as zf:
                for attempt in range(1, args.max_attempts + 1):
                    a = {'attempt': attempt, 'feedback_in': feedback}
                    r = R.call(author_messages(c, sc, zf, media, feedback, windows), f'a{attempt}_author', log_dir, args.budget)
                    a['author_output'] = r
                    r['_src'] = c['source_id']
                    fails = det_check(r, duration)
                    if fails == ['infeasible']:
                        status = 'infeasible'
                        rec['attempts'].append(a)
                        break
                    if not fails:
                        fails, a['visual_check'] = visual_check(c, r, media, log_dir, attempt, args.budget)
                    a['fails'] = fails
                    rec['attempts'].append(a)
                    if not fails:
                        status = 'pass'
                        task = to_task(c, r)
                        json.dump(task, open(log_dir / 'task.json', 'w'), ensure_ascii=False, indent=1)
                        # qa_viewer / eval 按 job_root/candidate_task.json 读取
                        json.dump(task, open(log_dir / 'candidate_task.json', 'w'), ensure_ascii=False, indent=1)
                        break
                    feedback = list(dict.fromkeys(feedback + fails))
    except Exception as e:
        rec['error'] = f'{type(e).__name__}: {e}'
        status = 'error'
    rec['status'] = status
    json.dump(rec, open(log_dir / 'log.json', 'w'), ensure_ascii=False, indent=1)
    print(f"[{status}] {src}@{c['t']} spent=${R._spent['usd']:.3f}", flush=True)
    if status == 'pass':
        with R._lock:
            args.passed += 1
            if args.passed >= args.target:
                stop.set()
    return status


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, choices=list(CFG.BATCHES))
    ap.add_argument('--videos', type=int, default=0, help='最多处理多少个视频（0=全部）')
    ap.add_argument('--per-video', type=int, default=1)
    ap.add_argument('--target', type=int, default=10, help='通过多少题后停止')
    ap.add_argument('--max-attempts', type=int, default=3)
    ap.add_argument('--budget', type=float, default=3.0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--expand', action='count', default=0, help='10-16 补题：放宽候选，可叠加 --expand --expand（见 candidates）')
    args = ap.parse_args()
    args.passed = 0
    R.use_batch(args.batch)
    OUT = CFG.RUNS / args.batch / 'runs_pg'
    windows, media_map = R.load_windows(), R.media_paths()
    windows_g.update(windows)
    cands = [c for c in candidates(windows, args.per_video, args.videos, args.expand)
             if not (OUT / f"{c['source_id']}_{c['t']:05d}" / 'log.json').exists()]
    print('candidates', len(cands), flush=True)
    stop = threading.Event()
    with ThreadPoolExecutor(args.workers) as ex:
        results = list(ex.map(lambda c: run_one(c, windows, media_map, args, stop), cands))
    counts = defaultdict(int)
    for s in results:
        counts[s or 'skipped'] += 1
    summary = {'counts': dict(counts), 'spent_usd': round(R._spent['usd'], 4), 'calls': R._spent['calls']}
    json.dump(summary, open(CFG.RUNS / args.batch / 'summary_pg_last_run.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
