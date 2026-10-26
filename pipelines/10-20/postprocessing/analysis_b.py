#!/usr/bin/env python3
"""10 FPS 实验 5–7 汇总：读取 JUDGE_DIR_NAME 下的统一判分并与主运行配对。
  5 强制作答（force_*）：原本沉默的用户提问 slot 被逼开口后能拿几分 →"不会"vs"不说"
  6 模态消融（mute_minicpm / black_minicpm / notext_moss / black_moss）：总分、分题型、[音频]/[视觉] 得分点命中率
  7 盲基线（blind_*，只喂提问时刻）+ 离线上界（offline_qwen_omni）：用户提问 slot
没跑完的 run 自动跳过。输出 analysis_b.md / analysis_b.json
"""
import json
import os
import random
from collections import defaultdict
from pathlib import Path

D_SOURCE = Path(__file__).resolve().parent
D = Path(os.environ['EXPERIMENT_OUTPUT_ROOT']).resolve()
EVAL_ROOT = Path(os.environ.get('MAIN25_ROOT', D / 'runs')).resolve()
JUDGE_DIR = os.environ.get('JUDGE_DIR_NAME', 'judge_points_lenient_v3')
FINAL = json.load(open(os.environ['MUSEUMLIVE_TASKS_FILE']))
WANT = {f"{t['task_id']}::{b['branch_id']}::{s['slot_id']}" for t in FINAL for b in t['branches'] for s in b['slots']}
NAME = {'minicpm': 'MiniCPM-o 4.5', 'venus': 'Venus', 'gander': 'Gander', 'moss': 'MOSS*'}
CN = {'temporal_update': '时间更新', 'task_resume': '任务恢复', 'context_correction': '上下文纠正', 'proactive_clarify': '主动澄清',
      'constraint_update': '约束更新', 'historical_reference': '历史指代', 'monitor_cancel': '监测与取消', 'selective_attention': '选择性注意',
      'proactive_guidance': '主动引导'}


def load(f):
    f = Path(f)
    if not f.exists(): return None
    return {r['judge_id']: r for r in map(json.loads, open(f)) if r['judge_id'] in WANT and r.get('status') == 'ok'}


def base(m):
    return load(EVAL_ROOT / f'main_{m}' / JUDGE_DIR / 'results.jsonl')


def run(name):
    return load(D / 'runs' / name / JUDGE_DIR / 'results.jsonl')


def mean(xs):
    xs = list(xs); return round(sum(xs) / len(xs), 2) if xs else None


def pct(x):
    return '—' if x is None else f'{x:.0%}'


def paired_bootstrap(a, b, predicate=lambda r: True, rounds=2000):
    """按 task 聚类的配对 bootstrap：返回共同 slot 数、均值差和 95% CI。"""
    ids = [i for i in a if i in b and predicate(a[i]) and predicate(b[i])]
    by_task = defaultdict(list)
    for i in ids:
        by_task[a[i]['task_id']].append(b[i]['score'] - a[i]['score'])
    tasks = sorted(by_task)
    if not tasks:
        return {'n': 0, 'delta': None, 'ci95': [None, None]}
    delta = sum(x for xs in by_task.values() for x in xs) / sum(len(xs) for xs in by_task.values())
    random.seed(20261019)
    draws = []
    for _ in range(rounds):
        sampled = [random.choice(tasks) for _ in tasks]
        xs = [x for t in sampled for x in by_task[t]]
        draws.append(sum(xs) / len(xs))
    draws.sort()
    return {'n': len(ids), 'delta': round(delta, 3),
            'ci95': [round(draws[int(.025 * rounds)], 3), round(draws[min(rounds - 1, int(.975 * rounds))], 3)]}


def pt_rate(rows, tag):
    n = c = 0
    for r in rows:
        if r.get('observed') == 'silent' or r['expected_action'] == 'WAIT': continue
        for p, s in zip(r.get('points') or [], r.get('statuses') or []):
            if p.startswith(f'[{tag}]'): n += 1; c += {'hit': 1, 'partial': .7}.get(s, 0)
    return round(c / n, 3) if n else None


def main():
    L = ['# 实验 5–7：10 FPS GPU 重跑 + 离线上界', '', '与正式全双工结果按 slot 配对比较；分数 = 得分点 Judge lenient_v3（0–5，qwen3.8-omni-flash）。', '']
    res = {}
    # ---------------- 5 强制作答
    L += ['## 5 强制作答：分开"不会"和"不说"', '',
          '用户提问说完的那一秒，如果模型还没开口，就屏蔽 <|listen|> 逼它开口（之后照常）。只看用户提问 slot。', '',
          '| 模型 | 原沉默 slot | 逼开口后均分 | 其中 ≥2 分（其实会） | 其中 0 分 | 原本就开口的 slot：原 → 强制 run | 用户提问联合分：原 → 强制 |', '|---|---|---|---|---|---|---|']
    for m in ['venus', 'gander', 'minicpm']:
        b, f = base(m), run(f'force_{m}')
        if not f: L.append(f'| {NAME[m]} | （未跑完） |||||| '); continue
        ids = [i for i in f if i in b and b[i]['trigger_kind'] == 'user_query']
        sil = [i for i in ids if b[i].get('observed') == 'silent']
        spk = [i for i in ids if b[i].get('observed') != 'silent']
        fs = [f[i]['score'] for i in sil]
        d = {'n_silent': len(sil), 'forced_mean_on_silent': mean(fs), 'ge2': sum(x >= 2 for x in fs) / max(1, len(fs)), 'zero': sum(x == 0 for x in fs) / max(1, len(fs)),
             'still_silent': sum(f[i].get('observed') == 'silent' for i in sil) / max(1, len(sil)),
             'spoken_orig': mean(b[i]['score'] for i in spk), 'spoken_forcedrun': mean(f[i]['score'] for i in spk),
             'uq_orig': mean(b[i]['score'] for i in ids), 'uq_forced': mean(f[i]['score'] for i in ids),
             'paired': paired_bootstrap(b, f, lambda r: r['trigger_kind'] == 'user_query'),
             'by_cap': {c: (mean(b[i]['score'] for i in ids if b[i]['capability'] == c), mean(f[i]['score'] for i in ids if f[i]['capability'] == c)) for c in CN}}
        res[f'force_{m}'] = d
        L.append(f"| {NAME[m]} | {d['n_silent']} | {d['forced_mean_on_silent']} | {pct(d['ge2'])} | {pct(d['zero'])} | {d['spoken_orig']} → {d['spoken_forcedrun']} | "
                 f"{d['uq_orig']} → **{d['uq_forced']}**（Δ {d['paired']['delta']:+.2f}，95% CI [{d['paired']['ci95'][0]:+.2f}, {d['paired']['ci95'][1]:+.2f}]） |")
    if any(k.startswith('force_') for k in res):
        L += ['', '分题型（用户提问联合分，原 → 强制）：', '', '| 题型 | ' + ' | '.join(NAME[m] for m in ['venus', 'gander', 'minicpm'] if f'force_{m}' in res) + ' |',
              '|---|' + '---|' * sum(f'force_{m}' in res for m in ['venus', 'gander', 'minicpm'])]
        for c, cn in CN.items():
            L.append(f'| {cn} | ' + ' | '.join(f"{res[f'force_{m}']['by_cap'][c][0]} → {res[f'force_{m}']['by_cap'][c][1]}" for m in ['venus', 'gander', 'minicpm'] if f'force_{m}' in res) + ' |')
    fallback = D / 'runs' / 'force_gander' / JUDGE_DIR / 'fallback_provenance.json'
    if fallback.exists():
        fb = json.load(open(fallback))
        L += ['', f"评分完整性：Gander 强制作答有 1/1250 slot 被主 Judge 的内容审查 HTTP 400 拒绝；该 slot 使用 {fb['fallback_model']}、"
              f"同一固定 rubric 与确定性计分公式补齐，得分 {fb['fallback_score']}。详细 provenance 已单独落盘。"]
    # ---------------- 6 模态消融
    L += ['', '## 6 模态消融', '',
          'MiniCPM：静音（现场原声置零，保留用户语音）/ 黑屏（帧换成同尺寸纯黑）。MOSS（本来就没有音频）：去掉现场转写 / 黑屏。', '',
          '| 条件 | 总均分 | 用户提问 | 画面触发·该说 | 画面触发·该安静 | [音频] 得分点命中 | [视觉] 得分点命中 |', '|---|---|---|---|---|---|---|']
    for m, variants in [('minicpm', [('完整', None), ('静音', 'mute_minicpm'), ('黑屏', 'black_minicpm')]),
                        ('moss', [('完整（画面+转写）', None), ('无现场转写', 'notext_moss'), ('仅转写文字（黑屏）', 'black_moss')])]:
        b = base(m)
        for lab, rn in variants:
            d = b if rn is None else run(rn)
            if d is None: L.append(f'| {NAME[m]}·{lab} | （未跑完） ||||||'); continue
            ids = [i for i in d if i in b]; rows = [d[i] for i in ids]
            k = lambda r: 'uq' if r['trigger_kind'] == 'user_query' else r['expected_action']
            x = {'total': mean(r['score'] for r in rows), 'uq': mean(r['score'] for r in rows if k(r) == 'uq'), 'SPEAK': mean(r['score'] for r in rows if k(r) == 'SPEAK'),
                 'WAIT': mean(r['score'] for r in rows if k(r) == 'WAIT'), 'audio_pt': pt_rate(rows, '音频'), 'visual_pt': pt_rate(rows, '视觉'),
                 'by_cap': {c: mean(r['score'] for r in rows if r['capability'] == c) for c in CN},
                 'paired_vs_full': None if rn is None else paired_bootstrap(b, d)}
            res[f'{m}_{lab}'] = x
            L.append(f"| {NAME[m]}·{lab} | **{x['total']}** | {x['uq']} | {x['SPEAK']} | {x['WAIT']} | {x['audio_pt']} | {x['visual_pt']} |")
            if rn is not None:
                p = x['paired_vs_full']
                L.append(f"| ↳ 配对变化 vs 完整 | Δ {p['delta']:+.2f}，95% CI [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] | | | | | |")
    cols = [k for k in res if k.startswith(('minicpm_', 'moss_'))]
    if cols:
        L += ['', '分题型：', '', '| 题型 | ' + ' | '.join(NAME[k.split('_')[0]] + '·' + k.split('_')[1] for k in cols) + ' |', '|---|' + '---|' * len(cols)]
        for c, cn in CN.items(): L.append(f'| {cn} | ' + ' | '.join(str(res[k]['by_cap'][c]) for k in cols) + ' |')
    # ---------------- 7 盲基线 + 离线上界
    L += ['', '## 7 最近 30 秒 / 3 秒盲基线与离线上界', '',
          '最近 30 秒基线：每个用户提问 slot 单独起流，只保留提问前最近 30 秒内的画面、声音和用户事件。'
          '3 秒盲基线：只从提问前 3 s 开始喂，之前的用户指令、画面、声音都不给。'
          '离线上界：qwen3.8-omni-flash 一次拿到 session 开始到用户说完的全部帧（≤32）+ 音轨 + 用户各轮文字后作答（无"何时开口"压力；它也是出题模型，偏乐观）。只看用户提问 slot。', '']
    conds = [('MiniCPM 全双工', base('minicpm')), ('MiniCPM 最近30秒', run('recent30_minicpm')), ('MiniCPM 3秒盲基线', run('blind_minicpm')),
             ('MOSS 全双工', base('moss')), ('MOSS 最近30秒', run('recent30_moss')), ('MOSS 3秒盲基线', run('blind_moss')),
             ('Venus 全双工', base('venus')), ('Gander 全双工', base('gander')), ('qwen-omni 离线上界', run('offline_qwen_omni'))]
    conds = [(n, d) for n, d in conds if d]
    uq = sorted(i for i in WANT if i.split('::')[2] and any(True for _ in [0]))
    L += ['| 题型 | ' + ' | '.join(n for n, _ in conds) + ' |', '|---|' + '---|' * len(conds)]
    get = lambda d, c=None: mean(r['score'] for r in d.values() if r['trigger_kind'] == 'user_query' and (c is None or r['capability'] == c))
    L.append('| **全部用户提问** | ' + ' | '.join(f'**{get(d)}**' for _, d in conds) + ' |')
    for c, cn in CN.items(): L.append(f'| {cn} | ' + ' | '.join(str(get(d, c)) for _, d in conds) + ' |')
    res['exp7'] = {n: {'all': get(d), **{c: get(d, c) for c in CN}} for n, d in conds}
    for m in ('minicpm', 'moss'):
        b, r30, r3 = base(m), run(f'recent30_{m}'), run(f'blind_{m}')
        if r30:
            res['exp7'][f'{NAME[m]} 最近30秒配对'] = paired_bootstrap(b, r30, lambda r: r['trigger_kind'] == 'user_query')
        if r3:
            res['exp7'][f'{NAME[m]} 3秒配对'] = paired_bootstrap(b, r3, lambda r: r['trigger_kind'] == 'user_query')
    offline_provenance = D / 'runs' / 'offline_qwen_omni' / 'reuse_provenance.json'
    if offline_provenance.exists():
        provenance = json.load(open(offline_provenance))
        public_provenance = dict(provenance)
        public_provenance['source_run'] = 'external finalized 1 FPS experiment artifact'
        res['offline_reuse_provenance'] = public_provenance
        L += ['', f"离线上界完整性：复用既有 1000 个生成缓存；其中 {provenance['successful_timing_pairs_checked']} 个成功回答与当前 10 FPS TTS 时间线逐事件一致，"
              f"{provenance['cached_generation_errors']} 个原始 API 失败按空回答保留，没有外部重试。"]
    json.dump(res, open(D / 'analysis_b.json', 'w'), ensure_ascii=False, indent=1)
    (D / 'analysis_b.md').write_text('\n'.join(L) + '\n')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
