#!/usr/bin/env python3
"""多批次题库合并后每个题型挑 N 题作为最终内容（10-13 用户计划：每类 100 题）。
用法: python3 select_final.py --pool 50=<runs/50/final 或 targeted_coverage_live.json> --pool 316=<runs/316/final> [--cap 100]
                              [--judge <glob>]...   （judge_special/judge_results.json 的 glob，只用于在清单里附均分）
  → runs/final100/tasks_<cap>.json（选中）、alternates_<cap>.json（备选，按轮转顺序）、all_tasks.json、summary.json
  → runs/final100/review/<cap>.md 与 review_final100.html：逐题 QA 清单（用户台词 + 期望回复 + 要点），供人工只看 QA。
规则：各批合并后按视频轮转（视频顺序按 source_id 的 md5 固定，视频内按时间先后），不按分数筛；结果可复现。
"""
import argparse, glob, hashlib, html, json, pathlib, statistics, sys
from collections import defaultdict, Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import config as CFG  # noqa: E402
HERE = CFG.RUNS / 'final100'
POOLS = {}
JUDGE_GLOBS = []
CAPS = ['temporal_update', 'selective_attention', 'constraint_update', 'context_correction', 'task_resume',
        'monitor_cancel', 'proactive_clarify', 'historical_reference', 'proactive_guidance']
CN = {'temporal_update': '时间更新', 'selective_attention': '选择性注意', 'constraint_update': '约束更新',
      'context_correction': '上下文纠正', 'task_resume': '任务恢复', 'monitor_cancel': '监测/取消',
      'proactive_clarify': '主动澄清', 'historical_reference': '历史指代', 'proactive_guidance': '主动引导'}


def collect():
    """POOLS: {batch: path}。path 是目录（含 tasks_<cap>.json，select_pool 的输出）或 targeted_coverage_live.json（50 主线格式）。"""
    rows = []
    for batch, path in POOLS.items():
        path = pathlib.Path(path)
        if path.is_file():
            for r in json.load(open(path))['accepted']:
                t = json.load(open(pathlib.Path(r['job_root']) / 'candidate_task.json'))
                t['_batch'] = batch; t['_source'] = r['source']; t['_path'] = str(pathlib.Path(r['job_root']) / 'candidate_task.json')
                rows.append(t)
        else:
            for cap in CAPS:
                f = path / f'tasks_{cap}.json'
                if not f.exists():
                    continue
                for t in json.load(open(f)):
                    t['_batch'] = batch; t['_source'] = f'{batch}_pool'; t['_path'] = str(f)
                    rows.append(t)
    return rows


def judge_scores(globs=JUDGE_GLOBS):
    by_task = defaultdict(list)
    for g in globs:
        for f in glob.glob(str(g)):
            for r in json.load(open(f))['results']:
                if r.get('status') != 'ok':
                    continue
                tid = r['judge_id'].split('::')[0]
                j = r.get('judgment', {})
                by_task[tid].append((j.get('score'), bool(j.get('needs_review'))))
    out = {}
    for tid, v in by_task.items():
        sc = [s for s, _ in v if isinstance(s, (int, float))]
        out[tid] = {'judge_mean': round(statistics.mean(sc), 2) if sc else None, 'judged_slots': len(v),
                    'needs_review': sum(nr for _, nr in v)}
    return out


def balanced(rows, cap):
    by_src = defaultdict(list)
    for r in sorted(rows, key=lambda r: r['session']['start_s']):
        by_src[r['source_id']].append(r)
    order = sorted(by_src, key=lambda s: hashlib.md5(s.encode()).hexdigest())
    seq, k = [], 0
    while True:
        layer = [by_src[s][k] for s in order if k < len(by_src[s])]
        if not layer:
            break
        seq += layer; k += 1
    return seq[:cap], seq[cap:]


def render_task(i, t):
    s = t['session']
    j = t.get('_judge') or {}
    jtxt = f"judge 均分 {j['judge_mean']}/5（{j['judged_slots']} 槽位）" if j.get('judge_mean') is not None else '未 eval'
    lines = [f"### {i}. {t['task_id']}",
             f"- 批次 {t['_batch']} ｜ 视频 {t['source_id']} ｜ 片段 {s['start_s']:.0f}–{s['end_s']:.0f}s ｜ {jtxt}"]
    for b in t['branches']:
        lines.append(f"- **分支 {b['branch_id']}**")
        items = []
        for e in b['user_events']:
            items.append((e['time_s'], 0, f"  - [{e['time_s']:.0f}s] 用户：{e['text']}"))
        for sl in b['slots']:
            w = sl.get('window', {}); t0 = w.get('start_s', 0)
            act = sl.get('expected_action', '')
            tgt = sl.get('target_text') or ''
            txt = f"  - [{t0:.0f}–{w.get('end_s', 0):.0f}s] 期望 {act}（{sl['slot_id']}）：{tgt}" if tgt else f"  - [{t0:.0f}–{w.get('end_s', 0):.0f}s] 期望 {act}（{sl['slot_id']}）"
            ka = sl.get('key_attributes') or []
            if ka:
                txt += "\n    - 要点：" + '；'.join(ka)
            items.append((t0, 1, txt))
        lines += [x[2] for x in sorted(items, key=lambda x: (x[0], x[1]))]
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cap', type=int, default=100)
    ap.add_argument('--pool', action='append', required=True, help='batch=path，可多次')
    ap.add_argument('--judge', action='append', default=[], help='judge_special/judge_results.json 的 glob，可多次')
    a = ap.parse_args()
    for x in a.pool:
        b, p = x.split('=', 1); POOLS[b] = p
    JUDGE_GLOBS[:] = a.judge
    out = HERE; rev = HERE / 'review'
    out.mkdir(parents=True, exist_ok=True); rev.mkdir(exist_ok=True)
    rows = collect(); js = judge_scores()
    for t in rows:
        t['_judge'] = js.get(t['task_id'])
    summary, all_tasks, html_parts = {}, [], []
    for cap in CAPS:
        pool = [t for t in rows if t['capability'] == cap]
        picked, alt = balanced(pool, a.cap)
        json.dump(picked, open(out / f'tasks_{cap}.json', 'w'), ensure_ascii=False, indent=1)
        json.dump(alt, open(out / f'alternates_{cap}.json', 'w'), ensure_ascii=False, indent=1)
        all_tasks += picked
        judged = [t for t in picked if t.get('_judge') and t['_judge']['judge_mean'] is not None]
        summary[cap] = {'pool': len(pool), 'selected': len(picked), 'alternates': len(alt),
                        'batch': dict(Counter(t['_batch'] for t in picked)),
                        'videos': len({t['source_id'] for t in picked}),
                        'judged': len(judged),
                        'judge_mean': round(statistics.mean(t['_judge']['judge_mean'] for t in judged), 2) if judged else None}
        md = [f"# {CN[cap]} {cap} — 选中 {len(picked)} 题（题库 {len(pool)}，备选 {len(alt)}）\n"]
        md += [render_task(i, t) for i, t in enumerate(picked, 1)]
        md.append(f"\n\n# 备选（按轮转顺序，换题时从前往后取）\n")
        md += [render_task(f'备{i}', t) for i, t in enumerate(alt, 1)]
        (rev / f'{cap}.md').write_text('\n\n'.join(md), encoding='utf-8')
        html_parts.append(f"<details open><summary><h2 style='display:inline'>{CN[cap]} {cap}（{len(picked)} 题）</h2></summary><pre>" +
                          html.escape('\n\n'.join(md)) + "</pre></details>")
    summary['_total'] = len(all_tasks)
    json.dump(all_tasks, open(out / 'all_tasks.json', 'w'), ensure_ascii=False, indent=1)
    json.dump(summary, open(out / 'summary.json', 'w'), ensure_ascii=False, indent=2)
    (out / 'review_final100.html').write_text(
        "<meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1100px;margin:auto}pre{white-space:pre-wrap;font-size:13px}</style>"
        f"<h1>366 视频最终题库候选：每类 {a.cap} 题</h1><p>{html.escape(json.dumps(summary, ensure_ascii=False))}</p>" + '\n'.join(html_parts), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
