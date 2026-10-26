#!/usr/bin/env python3
"""盲基线（10-18，实验 7）：每个 user_query slot 单独成一段流，从触发话语前 3 秒才开始喂（之前的指令 / 画面 / 声音全不给），
检验题目是否真的需要历史与状态跟踪。visual_event slot 不做（触发它的用户指令都在更早的时候）。
  make    tasks_final450.json → tasks_blind.json（每个 slot 一个伪分支：branch_id = <原分支>@@<slot_id>，session.start_s = 触发 − 3 s）
  restore <eval_results.json>... → 同目录 evaluation_results.restored.json：branch_key / branch_id / user_events 还原成原分支
          （judge 的 judge_id 与正式结果一致；上下文纠错的 correction_check 仍按原分支的 claim/correct 判）。
"""
import copy, json, sys
from pathlib import Path

D = Path(__file__).resolve().parent
LEAD = 3.0


def make():
    final = json.load(open(D.parent / 'model_compare_20261018/tasks_final450.json'))
    out = []
    for t in final:
        for b in t['branches']:
            ev = {e['event_id']: e for e in b['user_events']}
            for sl in b['slots']:
                if sl['trigger']['kind'] != 'user_query': continue
                t0 = max(t['session']['start_s'], ev[sl['trigger']['event_id']]['time_s'] - LEAD)
                nt = {k: v for k, v in t.items() if k != 'branches'}
                nt['session'] = dict(t['session'], start_s=t0)
                nt['blind'] = {'orig_branch_id': b['branch_id'], 'orig_user_events': b['user_events'], 'lead_s': LEAD}
                nt['branches'] = [{'branch_id': f"{b['branch_id']}@@{sl['slot_id']}",
                                   'user_events': [copy.deepcopy(e) for e in b['user_events'] if e['time_s'] >= t0], 'slots': [sl]}]
                out.append(nt)
    json.dump(out, open(D / 'tasks_blind.json', 'w'), ensure_ascii=False)
    print(len(out), 'blind pseudo-tasks')


def restore(files):
    orig = {(t['task_id'], t['blind']['orig_branch_id']): t['blind']['orig_user_events'] for t in json.load(open(D / 'tasks_blind.json'))}
    for f in files:
        f = Path(f); j = json.load(open(f))
        for r in j['results']:
            bid = r['branch_id'].split('@@')[0]
            r['blind_user_events'] = r['user_events']
            r['branch_id'] = bid; r['branch_key'] = f"{r['task_id']}::{bid}"; r['user_events'] = orig[(r['task_id'], bid)]
        (f.parent / 'evaluation_results.restored.json').write_text(json.dumps(j, ensure_ascii=False))
        print(f, len(j['results']))


if __name__ == '__main__':
    make() if sys.argv[1] == 'make' else restore(sys.argv[2:])
