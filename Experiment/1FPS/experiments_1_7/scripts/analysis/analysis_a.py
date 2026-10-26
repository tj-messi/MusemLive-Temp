#!/usr/bin/env python3
"""借鉴实验 1–4（10-18）：只用现成的全双工输出 + 得分点 judge，不占 GPU。
  1 回应率 × 条件质量 = 联合分（SocialOmni）
  2 时间指标：用户提问时延、抢话（用户还没说完就开口）、被打断后溢出时长、画面触发时延 / 提前报、slot 外主动发言；
    事件召回 × 发言时间精度 → F1，附"一直说 / 一直沉默"琐碎基线（OmniInteract IA-QTF1、Omni-DuplexEval）
  3 分层：证据距离（音频证据结束 → 提问）、session 长度、视频内位置、50/316 视频、语言、提问前用户轮数、得分点模态（[音频]/[视觉]）
  4 换 judge：4 个不同家族 judge（judges/<judge>/<model>/results.jsonl）与主 judge（qwen3.8-omni-flash）比较：均分 / 排名 / Kendall τ /
    逐 slot 相关、自家偏袒（混元 hy3 × Gander）；按题聚类 bootstrap 95% CI 与两两可分离度
输出：analysis_a.md、analysis_a.json
"""
import glob, json, math, random, re, statistics as st
from collections import Counter, defaultdict
from pathlib import Path

D = Path(__file__).resolve().parent
MC = D.parent / 'model_compare_20261018'
W = D.parent
MODELS = [('minicpm', 'MiniCPM-o 4.5'), ('venus', 'Venus'), ('gander', 'Gander'), ('moss', 'MOSS*')]
JUDGES = [('qwen', 'qwen3.8-omni-flash（主）'), ('tencent_hy3', '混元 hy3'), ('deepseek_deepseek-v4-pro', 'DeepSeek-v4-pro'),
          ('z-ai_glm-5.2', 'GLM-5.2'), ('moonshotai_kimi-k2.6', 'Kimi-k2.6')]
CN = {'temporal_update': '时间更新', 'task_resume': '任务恢复', 'context_correction': '上下文纠正', 'proactive_clarify': '主动澄清',
      'constraint_update': '约束更新', 'historical_reference': '历史指代', 'monitor_cancel': '监测与取消', 'selective_attention': '选择性注意',
      'proactive_guidance': '主动引导'}
REPLY_WAIT, SLACK = 6.0, 3.0
FINAL = json.load(open(MC / 'tasks_final450.json'))
WANT = {f"{t['task_id']}::{b['branch_id']}::{s['slot_id']}" for t in FINAL for b in t['branches'] for s in b['slots']}
TASK = {t['task_id']: t for t in FINAL}


def load_durations():
    """评测实际使用的 clip 时长，而不是原始平台长视频的页面时长。"""
    out = {}
    p50 = W / 'museumlive_50_qa_10_5_qwen_20261007/media_preflight.json'
    p316 = W / 'museumlive_316_qa_20261011/media_manifest.json'
    for row in json.load(open(p50)).get('sources', []):
        if row.get('duration_s') is not None:
            out[row['source_id']] = float(row['duration_s'])
    for row in json.load(open(p316)):
        if row.get('duration_s') is not None:
            out[row['source_id']] = float(row['duration_s'])
    return out


DURATIONS = load_durations()


def mean(xs):
    xs = list(xs); return round(sum(xs) / len(xs), 2) if xs else None


def pct(x):
    return '—' if x is None else f'{x:.0%}'


def q(xs, p):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else None


def load_judge(path):
    return {r['judge_id']: r for r in map(json.loads, open(path)) if r['judge_id'] in WANT and r.get('status') == 'ok'}


def load_eval(m):
    out = {}
    for f in glob.glob(str(MC / m / '*/eval/evaluation_results.json')):
        for r in json.load(open(f))['results']: out[r['branch_key']] = r
    return out


def kind(r):
    return 'user_query' if r['trigger_kind'] == 'user_query' else f"visual_{r['expected_action']}"


# ---------------------------------------------------------------- 1 回应率 / 条件质量
def pre_onset(E):
    """user_query slot 的 judge_id → 被算作回答的那段发言是否在用户说完（tts_end − 0.5 s）之前就开始了。"""
    out = {}
    for bk, r in E.items():
        sp = {x['event_id']: x for x in r['omni']['tts_spans']}
        ev = {e['event_id']: e for e in r['timeline'] if e.get('type') == 'user_event'}
        for s in r['slots']:
            sl = s['slot']; e = ev.get(sl['trigger'].get('event_id'))
            if sl['trigger']['kind'] == 'user_query' and e and e.get('response_time_s') is not None:
                out[f"{bk}::{sl['slot_id']}"] = e['response_time_s'] < sp[e['event_id']]['tts_end'] - 0.5
    return out


def exp1(J, E):
    out = {}
    for m, _ in MODELS:
        rows = list(J[m].values()); d = {}; pre = pre_onset(E[m])
        uq = [r for r in rows if r['trigger_kind'] == 'user_query']
        pr = [r for r in uq if pre.get(r['judge_id'])]
        d['pre_onset'] = {'share_of_responses': len(pr) / max(1, sum(r.get('observed') != 'silent' for r in uq)), 'mean_pre': mean(r['score'] for r in pr),
                          'strict_resp_rate': sum(r.get('observed') != 'silent' and not pre.get(r['judge_id']) for r in uq) / len(uq),
                          'strict_uq_joint': mean(0 if pre.get(r['judge_id']) else r['score'] for r in uq),
                          'strict_total': mean(0 if pre.get(r['judge_id']) else r['score'] for r in rows), 'total': mean(r['score'] for r in rows)}
        for k in ['all_speak', 'user_query', 'visual_SPEAK']:
            rs = [r for r in rows if r['expected_action'] != 'WAIT' and (k == 'all_speak' or kind(r) == k)]
            resp = [r for r in rs if r.get('observed') != 'silent']
            d[k] = {'n': len(rs), 'resp_rate': len(resp) / len(rs), 'cond': mean(r['score'] for r in resp), 'joint': mean(r['score'] for r in rs),
                    'cond_action_ok': sum(r.get('action') == 'correct' for r in resp) / max(1, len(resp))}
        w = [r for r in rows if r['expected_action'] == 'WAIT']
        d['visual_WAIT'] = {'n': len(w), 'false_alarm': sum(r.get('observed') != 'silent' for r in w) / len(w), 'joint': mean(r['score'] for r in w)}
        d['by_cap'] = {}
        for c in CN:
            rs = [r for r in rows if r['capability'] == c and r['expected_action'] != 'WAIT']
            resp = [r for r in rs if r.get('observed') != 'silent']
            d['by_cap'][c] = {'resp_rate': len(resp) / max(1, len(rs)), 'cond': mean(r['score'] for r in resp)}
        out[m] = d
    return out


# ---------------------------------------------------------------- 2 时间指标
def exp2(E, J):
    out = {}
    for m, _ in MODELS:
        lat_uq, lat_vis, barge, over, early, unsol, n_turns, stream_s = [], [], 0, [], 0, 0, 0, 0.0
        n_events = 0; n_interrupted = 0
        pos_s = spk_pos = spk_all = spk_dc = 0; hit = tot = 0; total_s = 0; dc_total = 0
        for bk, r in E[m].items():
            if f"{bk}::{r['slots'][0]['slot']['slot_id']}" not in WANT: continue
            spans = {x['event_id']: x for x in r['omni']['tts_spans']}
            turns = r['duplex_turns']; chunks = r['duplex_chunks']
            T0 = chunks[0]['t'] - 1 if chunks else 0
            stream_s += len(chunks); n_turns += len(turns)
            # 抢话 / 被打断溢出：以每句用户话语为单位
            for sp in spans.values():
                n_events += 1
                if any(sp['tts_start'] < t['start_s'] <= sp['tts_end'] - 0.5 for t in turns): barge += 1  # 开口时刻（该秒末）早于用户说完
                ong = [t for t in turns if t['start_s'] - 1 < sp['tts_start'] <= t['end_s']]  # 用户开口时模型正在说
                if ong:
                    n_interrupted += 1; over.append(max(t['end_s'] for t in ong) - sp['tts_start'])
            # 用户提问时延
            for s in r['slots']:
                sl = s['slot']
                if sl['trigger']['kind'] == 'user_query':
                    sp = spans.get(sl['trigger'].get('event_id'))
                    ev = next((x for x in r['timeline'] if x.get('type') == 'user_event' and x.get('event_id') == sl['trigger'].get('event_id')), None)
                    if sp and ev and ev.get('response_time_s') is not None and ev['response_time_s'] >= sp['tts_end'] - 0.5:
                        lat_uq.append(ev['response_time_s'] - sp['tts_end'])  # 只算说完后才开口的
                elif sl['expected_action'] == 'SPEAK':
                    lo, hi = float(sl['window']['start_s']), float(sl['window']['end_s']) + SLACK
                    h = [t for t in turns if t['start_s'] <= hi and t['end_s'] >= lo]
                    if h: lat_vis.append(h[0]['start_s'] - lo)
                    # 提前报：最后一句早于窗口的用户话语回应期结束后、窗口开始前就开口
                    prev = [x['tts_end'] for x in spans.values() if x['tts_end'] < lo]
                    if prev and any(max(prev) + REPLY_WAIT < t['start_s'] < lo for t in turns): early += 1
            # slot 外发言：既不在任何用户话语回应期，也不在任何画面触发 slot 窗口内
            ok_win = [(x['tts_start'], x['tts_end'] + REPLY_WAIT) for x in spans.values()] + \
                     [(float(s['slot']['window']['start_s']), float(s['slot']['window']['end_s']) + SLACK) for s in r['slots'] if s['slot']['trigger']['kind'] != 'user_query']
            unsol += sum(not any(a <= t['start_s'] <= b + 1 for a, b in ok_win) for t in turns)
            # 秒级：正例 = SPEAK slot 窗口；不计 = 其他用户话语回应期；其余为负例
            pos, dc = set(), set()
            for s in r['slots']:
                sl = s['slot']
                if sl['expected_action'] == 'WAIT': continue
                if sl['trigger']['kind'] == 'user_query':
                    sp = spans.get(sl['trigger'].get('event_id'))
                    if not sp: continue
                    lo, hi = sp['tts_end'], sp['tts_end'] + REPLY_WAIT
                else:
                    lo, hi = float(sl['window']['start_s']), float(sl['window']['end_s']) + SLACK
                secs = {c['t'] for c in chunks if lo < c['t'] <= hi + 1}
                pos |= secs; tot += 1
                if any(not c['listen'] for c in chunks if c['t'] in secs): hit += 1
            slot_ev = {s['slot']['trigger'].get('event_id') for s in r['slots'] if s['slot']['trigger']['kind'] == 'user_query'}
            for eid, sp in spans.items():
                if eid in slot_ev: continue
                dc |= {c['t'] for c in chunks if sp['tts_start'] < c['t'] <= sp['tts_end'] + REPLY_WAIT + 1}
            dc -= pos
            for c in chunks:
                total_s += 1; pos_s += c['t'] in pos; dc_total += c['t'] in dc
                if not c['listen']:
                    if c['t'] in dc: spk_dc += 1; continue
                    spk_all += 1; spk_pos += c['t'] in pos
        prec = spk_pos / max(1, spk_all); rec = hit / max(1, tot)
        base_prec = pos_s / max(1, total_s - dc_total)
        out[m] = {'uq_latency_med': q(lat_uq, .5), 'uq_latency_p90': q(lat_uq, .9), 'uq_within2s': sum(x <= 2 for x in lat_uq) / max(1, len(lat_uq)), 'uq_n': len(lat_uq),
                  'barge_in_rate': barge / max(1, n_events), 'interrupted_rate': n_interrupted / max(1, n_events),
                  'overflow_med_s': q(over, .5), 'overflow_mean_s': mean(over), 'n_interrupted': n_interrupted,
                  'vis_latency_med': q(lat_vis, .5), 'vis_hit': len(lat_vis), 'vis_early': early,
                  'turns': n_turns, 'unsolicited_turns': unsol, 'unsolicited_frac': unsol / max(1, n_turns), 'unsolicited_per_min': unsol / max(1e-9, stream_s / 60),
                  'speak_frac': (spk_all + spk_dc) / max(1, total_s),
                  'event_recall': rec, 'time_precision': prec, 'f1': 2 * prec * rec / max(1e-9, prec + rec),
                  'baseline_always_speak': {'recall': 1.0, 'precision': base_prec, 'f1': 2 * base_prec / (1 + base_prec)}}
    return out


# ---------------------------------------------------------------- 3 分层
def strata(row, t, sl, branch):
    out = {}
    q_t = float(sl['window']['start_s'])
    refs = [a for a in sl.get('audio_refs') or [] if isinstance(a, dict) and 'end' in a and float(a['end']) <= q_t]
    if refs:
        # 距离定义为最近一条相关音频证据结束到提问/触发的间隔。
        g = q_t - max(float(a['end']) for a in refs)
        out['evidence_gap'] = '<5s' if g < 5 else '5–15s' if g < 15 else '15–30s' if g < 30 else '≥30s'
    L = t['session']['end_s'] - t['session']['start_s']
    out['session_len'] = '<15s' if L < 15 else '15–30s' if L < 30 else '≥30s'
    p = t['session']['start_s']
    out['video_pos'] = '<5min' if p < 300 else '5–20min' if p < 1200 else '≥20min'
    duration = DURATIONS.get(t['source_id'])
    if duration is not None:
        out['video_duration'] = '<5min' if duration < 300 else '5–10min' if duration < 600 else '10–15min' if duration < 900 else '≥15min'
    out['video_set'] = '50' if any((W / 'media50_backup_20261015' / (t['source_id'] + e)).exists() for e in ('.mp4', '.mkv')) else '316'
    out['language'] = t['language']
    n_prev = sum(e['time_s'] < q_t - 0.01 for e in branch['user_events'])
    out['prior_user_turns'] = str(min(n_prev, 3)) + ('+' if n_prev >= 3 else '')
    return out


def exp3(J):
    slot = {}
    for t in FINAL:
        for b in t['branches']:
            for s in b['slots']: slot[f"{t['task_id']}::{b['branch_id']}::{s['slot_id']}"] = (t, s, b)
    out = defaultdict(lambda: defaultdict(dict))
    for m, _ in MODELS:
        groups = defaultdict(list)
        for jid, r in J[m].items():
            if r['expected_action'] == 'WAIT': continue
            t, s, b = slot[jid]
            for k, v in strata(r, t, s, b).items(): groups[(k, v)].append(r['score'])
        for (k, v), xs in groups.items(): out[k][v][m] = {'n': len(xs), 'mean': mean(xs)}
        # 得分点模态：按 [音频]/[视觉] 标签统计 hit / partial 率（只算有回答的 slot）
        mod = defaultdict(lambda: [0, 0.0])
        for r in J[m].values():
            if r.get('observed') == 'silent' or r['expected_action'] == 'WAIT': continue
            for p, stt in zip(r.get('points') or [], r.get('statuses') or []):
                tag = '音频' if p.startswith('[音频]') else '视觉' if p.startswith('[视觉]') else '其他'
                mod[tag][0] += 1; mod[tag][1] += {'hit': 1, 'partial': .7}.get(stt, 0)
        for tag, (n, c) in mod.items():
            if tag != '其他': out['point_modality'][tag][m] = {'n': n, 'mean': round(c / n, 3)}
    return out


# ---------------------------------------------------------------- 4 换 judge + bootstrap
def kendall(a, b):
    ks = list(a); c = d = 0
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            s = (a[ks[i]] - a[ks[j]]) * (b[ks[i]] - b[ks[j]]); c += s > 0; d += s < 0
    return (c - d) / max(1, c + d)


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y); sx = math.sqrt(sum((a - mx) ** 2 for a in x)); sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy) if sx and sy else None


def bootstrap(J, B=2000):
    random.seed(20261018)
    tids = sorted(TASK); by = {m: defaultdict(list) for m, _ in MODELS}
    for m, _ in MODELS:
        for r in J[m].values(): by[m][r['task_id']].append(r['score'])
    means = {m: [] for m, _ in MODELS}
    for _ in range(B):
        samp = [random.choice(tids) for _ in tids]
        for m, _ in MODELS:
            xs = [x for t in samp for x in by[m][t]]; means[m].append(sum(xs) / len(xs))
    ci = {m: (round(q(v, .025), 2), round(q(v, .975), 2)) for m, v in means.items()}
    sep = {f'{a}>{b}': round(sum(x > y for x, y in zip(means[a], means[b])) / B, 3) for a, _ in MODELS for b, _ in MODELS if a != b}
    return ci, sep


def exp4(J):
    JJ = {'qwen': J}
    for jk, _ in JUDGES[1:]:
        d = {}
        for m, _ in MODELS:
            f = D / 'judges' / jk / m / 'results.jsonl'
            if f.exists(): d[m] = load_judge(f)
        if len(d) == len(MODELS): JJ[jk] = d
    res = {'judges': list(JJ), 'mean': {}, 'rank': {}, 'kendall_vs_qwen': {}, 'slot_r_vs_qwen_answered': {}, 'coverage': {}}
    for jk, d in JJ.items():
        common = set.intersection(*(set(d[m]) & set(J[m]) for m, _ in MODELS)) if jk != 'qwen' else None
        mm = {}
        for m, _ in MODELS:
            ids = [i for i in d[m] if common is None or i in set(d[m]) & set(J[m])]
            mm[m] = mean(d[m][i]['score'] for i in ids); res['coverage'].setdefault(jk, {})[m] = len(ids)
        res['mean'][jk] = mm; res['rank'][jk] = sorted(mm, key=lambda m: -mm[m])
        if jk != 'qwen':
            res['kendall_vs_qwen'][jk] = round(kendall(mm, res['mean']['qwen']), 2)
            xs, ys = [], []
            for m, _ in MODELS:
                for i, r in d[m].items():
                    if i in J[m] and J[m][i].get('observed') != 'silent': xs.append(r['score']); ys.append(J[m][i]['score'])
            res['slot_r_vs_qwen_answered'][jk] = round(pearson(xs, ys), 3) if xs else None
    # 自家偏袒：hy3 相对主 judge 给 Gander 的加分 − 给其他模型的平均加分（只算有回答的 slot，配对）
    if 'tencent_hy3' in JJ:
        delta = {}
        for m, _ in MODELS:
            pr = [(JJ['tencent_hy3'][m][i]['score'], J[m][i]['score']) for i in JJ['tencent_hy3'][m] if i in J[m] and J[m][i].get('observed') != 'silent']
            delta[m] = round(st.mean(a - b for a, b in pr), 3) if pr else None
        others = [delta[m] for m in delta if m != 'gander' and delta[m] is not None]
        res['self_pref_hy3'] = {'delta_vs_qwen': delta, 'gander_minus_others': round(delta['gander'] - st.mean(others), 3) if others else None}
    # 各 judge 下 bootstrap
    res['bootstrap'] = {}
    for jk, d in JJ.items():
        ci, sep = bootstrap(d); res['bootstrap'][jk] = {'ci95': ci, 'separability': sep}
    return res


# ---------------------------------------------------------------- 输出
def main():
    J = {m: load_judge(MC / m / 'judge_points/results.jsonl') for m, _ in MODELS}
    E = {m: load_eval(m) for m, _ in MODELS}
    r1, r2, r3, r4 = exp1(J, E), exp2(E, J), exp3(J), exp4(J)
    json.dump({'exp1': r1, 'exp2': r2, 'exp3': r3, 'exp4': r4}, open(D / 'analysis_a.json', 'w'), ensure_ascii=False, indent=1)
    H = ' | '.join(n for _, n in MODELS)
    L = ['# 借鉴实验 1–4：现成全双工输出分析（10-18）', '', '最终 450 题 / 1250 slot（1000 用户提问 + 100 画面触发·该说 + 150 画面触发·该安静）；分数 = 得分点 judge v2（0–5）。'
         '*MOSS 无音频输入（用户提问为文字、现场声音为理想 ASR 转写）。', '',
         '## 1 回应率 × 条件质量 = 联合分', '', '条件质量 = 只在模型开口的 slot 上的均分；联合分 = 全部该开口 slot 的均分（沉默记 0）。', '',
         f'| | {H} |', '|---|' + '---|' * len(MODELS)]
    for k, lab in [('user_query', '用户提问'), ('visual_SPEAK', '画面触发·该说'), ('all_speak', '全部该开口')]:
        L.append(f'| {lab}·回应率 | ' + ' | '.join(pct(r1[m][k]['resp_rate']) for m, _ in MODELS) + ' |')
        L.append(f'| {lab}·条件质量 | ' + ' | '.join(str(r1[m][k]['cond']) for m, _ in MODELS) + ' |')
        L.append(f'| {lab}·联合分 | ' + ' | '.join(str(r1[m][k]['joint']) for m, _ in MODELS) + ' |')
    L.append('| 回应后动作正确率 | ' + ' | '.join(pct(r1[m]['all_speak']['cond_action_ok']) for m, _ in MODELS) + ' |')
    L.append('| 画面触发·该安静·误报率 | ' + ' | '.join(pct(r1[m]['visual_WAIT']['false_alarm']) for m, _ in MODELS) + ' |')
    L += ['', '**抢答**：被算作用户提问回答的发言，在用户说完（TTS 结束 − 0.5 s）之前就已开始。MOSS 的问题文字在说完那一秒才送入，这类发言的开头不可能是在回答本题。'
          '严格口径 = 抢答记 0。', '', f'| | {H} |', '|---|' + '---|' * len(MODELS),
          '| 抢答占回答的比例 | ' + ' | '.join(pct(r1[m]['pre_onset']['share_of_responses']) for m, _ in MODELS) + ' |',
          '| 抢答那部分的均分 | ' + ' | '.join(str(r1[m]['pre_onset']['mean_pre']) for m, _ in MODELS) + ' |',
          '| 用户提问·严格回应率 | ' + ' | '.join(pct(r1[m]['pre_onset']['strict_resp_rate']) for m, _ in MODELS) + ' |',
          '| 用户提问·严格联合分 | ' + ' | '.join(str(r1[m]['pre_onset']['strict_uq_joint']) for m, _ in MODELS) + ' |',
          '| 总均分（原 → 严格） | ' + ' | '.join(f"{r1[m]['pre_onset']['total']} → {r1[m]['pre_onset']['strict_total']}" for m, _ in MODELS) + ' |']
    L += ['', '分题型（回应率 / 条件质量）：', '', f'| 题型 | {H} |', '|---|' + '---|' * len(MODELS)]
    for c, cn in CN.items():
        L.append(f'| {cn} | ' + ' | '.join(f"{pct(r1[m]['by_cap'][c]['resp_rate'])} / {r1[m]['by_cap'][c]['cond']}" for m, _ in MODELS) + ' |')
    n_wait = sum(1 for t in FINAL for b in t['branches'] for s in b['slots'] if s['expected_action'] == 'WAIT')
    L += ['', f'琐碎基线：**一直沉默** 总均分 = {5 * n_wait / len(WANT):.2f}（只拿该安静 slot 的 5 分）。', '',
          '## 2 时间指标', '', '时间单位秒；模型每秒决定一次，"开口时刻"= 第一段发言所在秒的末尾。', '',
          f'| | {H} |', '|---|' + '---|' * len(MODELS)]
    rows2 = [('用户提问·时延中位（说完→开口，不含抢答）', 'uq_latency_med'), ('用户提问·时延 P90', 'uq_latency_p90'), ('用户提问·2 s 内开口占比', 'uq_within2s'),
             ('抢话率（用户没说完就开口）', 'barge_in_rate'), ('用户开口时模型正在说的比例', 'interrupted_rate'), ('被打断后溢出·中位', 'overflow_med_s'),
             ('被打断后溢出·均值', 'overflow_mean_s'), ('画面触发·时延中位（窗口起点→开口）', 'vis_latency_med'), ('画面触发·提前报（条）', 'vis_early'),
             ('发言段数', 'turns'), ('slot 外发言段占比', 'unsolicited_frac'), ('slot 外发言·每分钟', 'unsolicited_per_min'), ('说话秒数占比', 'speak_frac'),
             ('事件召回', 'event_recall'), ('发言时间精度', 'time_precision'), ('**F1**', 'f1')]
    for lab, k in rows2:
        f = (lambda v: pct(v)) if k in ('uq_within2s', 'barge_in_rate', 'interrupted_rate', 'unsolicited_frac', 'speak_frac', 'event_recall', 'time_precision') else \
            (lambda v: f'**{v:.2f}**') if k == 'f1' else (lambda v: '—' if v is None else (f'{v:.1f}' if isinstance(v, float) else str(v)))
        L.append(f'| {lab} | ' + ' | '.join(f(r2[m][k]) for m, _ in MODELS) + ' |')
    b = r2[MODELS[0][0]]['baseline_always_speak']
    L += ['', f'琐碎基线：**一直说** 召回 100%、精度 {b["precision"]:.0%}、F1 {b["f1"]:.2f}；**一直沉默** F1 = 0。',
          '事件召回 = 该开口 slot 窗口内有发言的比例；发言时间精度 = 说话的秒中落在该开口窗口内的比例（非 slot 的用户话语回应期不计）。', '',
          '## 3 分层（只算该开口的 slot）', '']
    names = {'evidence_gap': '最近音频证据结束 → 提问的距离', 'session_len': 'session 长度', 'video_duration': '评测 clip 的真实时长',
             'video_pos': 'session 在视频里的位置', 'video_set': '视频来源',
             'language': '语言', 'prior_user_turns': '提问前用户已说的轮数', 'point_modality': '得分点模态（有回答时的得分点命中率，hit=1 / partial=0.7）'}
    order = {'evidence_gap': ['<5s', '5–15s', '15–30s', '≥30s'], 'session_len': ['<15s', '15–30s', '≥30s'],
             'video_duration': ['<5min', '5–10min', '10–15min', '≥15min'], 'video_pos': ['<5min', '5–20min', '≥20min'],
             'prior_user_turns': ['0', '1', '2', '3+']}
    for k, nm in names.items():
        L += [f'**{nm}**', '', f'| | n | {H} |', '|---|---|' + '---|' * len(MODELS)]
        for v in order.get(k, sorted(r3[k])):
            if v not in r3[k]: continue
            n = r3[k][v][MODELS[0][0]]['n']
            L.append(f'| {v} | {n} | ' + ' | '.join(str(r3[k][v].get(m, {}).get('mean')) for m, _ in MODELS) + ' |')
        L.append('')
    L += ['## 4 换 judge 的稳定性 + bootstrap', '']
    if len(r4['judges']) > 1:
        jn = dict(JUDGES)
        L += [f'| judge | {H} | 排名 | Kendall τ（vs 主） | 逐 slot r（有回答，vs 主） |', '|---|' + '---|' * len(MODELS) + '---|---|---|']
        for jk in r4['judges']:
            L.append(f'| {jn[jk]} | ' + ' | '.join(str(r4['mean'][jk][m]) for m, _ in MODELS) + ' | ' + ' > '.join(dict(MODELS)[m] for m in r4['rank'][jk]) +
                     f" | {r4['kendall_vs_qwen'].get(jk, '—')} | {r4['slot_r_vs_qwen_answered'].get(jk, '—')} |")
        if 'self_pref_hy3' in r4:
            sp = r4['self_pref_hy3']
            L += ['', '自家偏袒（混元 hy3 判 Gander）：hy3 相对主 judge 的平均分差（有回答的 slot）：' +
                  '，'.join(f"{dict(MODELS)[m]} {v:+.2f}" for m, v in sp['delta_vs_qwen'].items()) + f"；Gander 比其他模型多拿 **{sp['gander_minus_others']:+.2f}**。"]
    L += ['', '按题聚类 bootstrap（2000 次）95% CI 与两两可分离度 P(A > B)：', '']
    for jk, bs in r4['bootstrap'].items():
        L.append(f"- {dict(JUDGES)[jk]}：" + '，'.join(f"{dict(MODELS)[m]} [{a}, {b}]" for m, (a, b) in bs['ci95'].items()))
    bs = r4['bootstrap']['qwen']['separability']
    L += ['', f'| P(行 > 列)（主 judge） | {H} |', '|---|' + '---|' * len(MODELS)]
    for a, an in MODELS:
        L.append(f'| {an} | ' + ' | '.join('—' if a == b else f"{bs[f'{a}>{b}']:.3f}" for b, _ in MODELS) + ' |')
    (D / 'analysis_a.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
