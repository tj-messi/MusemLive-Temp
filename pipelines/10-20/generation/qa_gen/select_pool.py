"""汇总一个批次三条出题线通过的题（runs_v3 / runs_hr / runs_pg），过滤缺口秒，超出 --cap 的题型按视频轮转均衡截取。

用法: python3 select_pool.py --batch 50|316 [--cap 175]
  → runs/<batch>/final/tasks_<capability>.json、all_tasks.json、summary.json
规则：同一视频内按时间先后轮流取；视频轮转顺序固定（按 source_id 的 md5），结果可复现。
被 v3b 撤下的题（candidate_task.rejected_v3b.json）不在内；改写过的题带 task['v3b']。
"""
import argparse
import glob
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R  # noqa: E402
import config as CFG  # noqa: E402


def load(p):
    return json.load(open(p))


def collect(root):
    """root = runs/<batch>；返回 (来源, task) 列表。hr 看 refresh_log.json 的 status，v3/pg 看 task.json 是否存在。"""
    tasks = []
    for log in sorted(glob.glob(str(root / 'runs_hr/*/refresh_log.json'))):
        if load(log).get('status') == 'pass' and (Path(log).parent / 'refreshed_task.json').exists():
            tasks.append(('hr', load(Path(log).parent / 'refreshed_task.json')))
    for name in ('runs_pg', 'runs_v3'):
        for p in sorted(glob.glob(str(root / name / '*/task.json'))):
            tasks.append((name[5:], load(p)))
    return tasks


def hits_gap(task, gaps):
    s, e = task['session']['start_s'], task['session']['end_s']
    return any(s - 1 <= g <= e + 1 for g in gaps.get(task['source_id'], ()))


class KeepSet:
    """旧题库里的题：按 task_id + session 认（同 id 的新题不算旧题）。支持 `task in keep` 与 `task_id in keep` 两种用法前者更严。"""
    def __init__(self, tasks):
        self.m = {t['task_id']: json.dumps(t['session'], sort_keys=True) for t in tasks}

    def __contains__(self, r):
        if isinstance(r, dict):
            return self.m.get(r['task_id']) == json.dumps(r['session'], sort_keys=True)
        return r in self.m


def t_first(t):
    return min(e['time_s'] for b in t['branches'] for e in b['user_events'])


def dedup(rows, keep, sec):
    """同题型同视频、首个用户事件相差 < sec 秒的题只留一道：keep 里的优先，其次按收集顺序；keep 之间不互删。同 task_id 的也在此去重。"""
    out, seen = [], defaultdict(list)
    for r in sorted(rows, key=lambda r: r not in keep):
        k, t = r['source_id'], t_first(r)
        if r not in keep and any(abs(t - x) < sec or i == r['task_id'] for x, i in seen[k]):
            continue
        seen[k].append((t, r['task_id'])); out.append(r)
    return out


def balanced(rows, cap, keep=frozenset()):
    """keep 里的题必选（10-16 补题：已入库的旧题不被新题挤掉），剩余名额再按视频轮转补。"""
    kept = [r for r in rows if r in keep]
    rows = [r for r in rows if r not in keep]
    cap = max(cap - len(kept), 0)
    by_src = defaultdict(list)
    for r in sorted(rows, key=lambda r: r['session']['start_s']):
        by_src[r['source_id']].append(r)
    order = sorted(by_src, key=lambda s: hashlib.md5(s.encode()).hexdigest())
    picked, k = [], 0
    while len(picked) < cap:
        layer = [by_src[s][k] for s in order if k < len(by_src[s])]
        if not layer:
            break
        picked += layer[:cap - len(picked)]
        k += 1
    return kept + picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--batch', required=True, choices=list(CFG.BATCHES))
    ap.add_argument('--cap', type=int, default=175)
    ap.add_argument('--keep-from', default='', help='已有题库 json（如 final/all_tasks.before_*.json），其中的题必选')
    ap.add_argument('--dedup-s', type=float, default=0, help='>0 时同题型同视频首事件相差小于该秒数的新题去重')
    args = ap.parse_args()
    keep = KeepSet(load(args.keep_from)) if args.keep_from else frozenset()
    R.use_batch(args.batch)
    root = CFG.RUNS / args.batch
    OUT = root / 'final'
    gaps = R.gap_seconds()
    by_cap, origin, dropped_gap = defaultdict(list), {}, Counter()
    for src_name, t in collect(root):
        if hits_gap(t, gaps):
            dropped_gap[t['capability']] += 1
            continue
        origin[t['task_id']] = src_name
        by_cap[t['capability']].append(t)
    OUT.mkdir(exist_ok=True)
    summary, everything = {}, []
    for cap, rows in sorted(by_cap.items()):
        n0 = len(rows)
        if args.dedup_s > 0:
            rows = dedup(rows, keep, args.dedup_s)
        picked = balanced(rows, args.cap, keep) if len(rows) > args.cap else rows
        json.dump(picked, open(OUT / f'tasks_{cap}.json', 'w'), ensure_ascii=False, indent=1)
        everything += picked
        summary[cap] = {'passed': len(rows), 'dropped_gap': dropped_gap[cap], 'selected': len(picked), 'dedup_dropped': n0 - len(rows),
                        'videos': len({t['source_id'] for t in picked}), 'origin': dict(Counter(origin[t['task_id']] for t in picked))}
    json.dump(everything, open(OUT / 'all_tasks.json', 'w'), ensure_ascii=False, indent=1)
    summary['_total'] = len(everything)
    json.dump(summary, open(OUT / 'summary.json', 'w'), ensure_ascii=False, indent=1)
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
