#!/usr/bin/env python3
"""实验 7：构造“仅保留每个用户提问前最近 30 秒”的捷径基线。

每个 user_query slot 单独成为一个伪分支：
- session.start_s = max(原 session.start_s, 触发话语时刻 - 30s)
- 丢弃截断点以前的用户事件
- 只保留当前评分 slot

restore 会把 eval 结果的 branch_id/user_events 恢复成正式题目口径，确保
judge_id 与正式全双工结果一致，可做逐 slot 配对比较。
"""
import copy
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
LEAD = 30.0
TASKS = D / 'tasks_recent30.json'


def make():
    final = json.load(open(D.parent / 'model_compare_20261018/tasks_final450.json'))
    out = []
    for t in final:
        for b in t['branches']:
            events = {e['event_id']: e for e in b['user_events']}
            for slot in b['slots']:
                if slot['trigger']['kind'] != 'user_query':
                    continue
                trigger = events[slot['trigger']['event_id']]
                t0 = max(float(t['session']['start_s']), float(trigger['time_s']) - LEAD)
                nt = {k: copy.deepcopy(v) for k, v in t.items() if k != 'branches'}
                nt['session'] = dict(t['session'], start_s=t0)
                nt['recent30'] = {
                    'orig_branch_id': b['branch_id'],
                    'orig_user_events': b['user_events'],
                    'lead_s': LEAD,
                    'orig_session_start_s': t['session']['start_s'],
                }
                nt['branches'] = [{
                    'branch_id': f"{b['branch_id']}@@{slot['slot_id']}",
                    'user_events': [copy.deepcopy(e) for e in b['user_events'] if float(e['time_s']) >= t0],
                    'slots': [copy.deepcopy(slot)],
                }]
                out.append(nt)
    TASKS.write_text(json.dumps(out, ensure_ascii=False))
    print(len(out), 'recent-30s pseudo-tasks')


def restore(files):
    tasks = json.load(open(TASKS))
    orig = {(t['task_id'], t['recent30']['orig_branch_id']): t['recent30']['orig_user_events'] for t in tasks}
    for name in files:
        f = Path(name)
        payload = json.load(open(f))
        for row in payload['results']:
            branch_id = row['branch_id'].split('@@')[0]
            row['recent30_user_events'] = row['user_events']
            row['branch_id'] = branch_id
            row['branch_key'] = f"{row['task_id']}::{branch_id}"
            row['user_events'] = orig[(row['task_id'], branch_id)]
            row.setdefault('causal_audit', {})['recent_context_s'] = LEAD
        target = f.parent / 'evaluation_results.restored.json'
        target.write_text(json.dumps(payload, ensure_ascii=False))
        print(f, len(payload['results']))


if __name__ == '__main__':
    if len(sys.argv) < 2 or sys.argv[1] not in ('make', 'restore'):
        raise SystemExit('usage: recent30_tasks.py make | restore <evaluation_results.json>...')
    make() if sys.argv[1] == 'make' else restore(sys.argv[2:])
