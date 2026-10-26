#!/usr/bin/env python3
"""出题后的题型后处理（v3b 审计，10-13 用户反馈）：对 runs/<batch>/runs_v3 里三类题做看帧审计 / 改写：
  temporal_update / selective_attention：有些题 t0 提问时目标还没出现 → 看帧审计，不成立的题撤下（候选文件改名），再用改过的 gen_v3 补齐；
  proactive_clarify：问句太含糊（"这个是什么颜色的"）→ 至少点出物品类别；不合格的题由模型看帧改写问句并复核，改不好的撤下。
用法: python3 postprocess_v3b.py audit  --batch 50|316 [--caps ...] [--limit N] [--budget 10] [--workers 12] [--dry]
      python3 postprocess_v3b.py status --batch 50|316      # 各类当前有效题数 / 撤下数 / 改写数
注意：gen_v3 现已把 visible_at_t0 闸门和 names_category 规则内置，新出的题一般都能通过本审计；本脚本用于复查/对老题整改。
审计结果写在每个 run 目录的 v3b_audit.json；撤下 = candidate_task.json→candidate_task.rejected_v3b.json、task.json→task.rejected_v3b.json。
改写 = 覆盖 candidate_task.json / task.json（原件备份为 *.before_v3b.json），task['v3b'] 记录改动。
"""
import argparse, json, re, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R  # noqa: E402
import config as CFG  # noqa: E402
import gen_v3 as V  # noqa: E402  vague_question
CAPS3 = ['temporal_update', 'selective_attention', 'proactive_clarify']

VIS_SYSTEM = """你是严格的视觉核验员。第一视角博物馆视频里，用户在 t0 时刻对模型说了一句话，让模型留意/只关注某个对象 X。
给你 t0−1s、t0、t0+1s 的真实帧。判断：用户在 t0 开口时，X 是否**已经清楚地出现在画面里**（在 t0−1s 和 t0 的帧里都能看到并辨认出它，不是尚未出现、刚入画一角、被遮挡或模糊到无法确认）。
台词里的描述（位置、衣着、外观）必须能在 t0 帧里对上。只输出 JSON：{"visible_at_t0": bool, "note": "一句话说明在哪帧看到/没看到什么"}"""

CLR_SYSTEM = """你在审核第一视角博物馆流式交互 benchmark 的"主动澄清"题。规则：ambiguous 分支的问句必须**点出物品的类别名称**（如"这幅画是什么年代的""这里面的服饰是什么颜色的""这两个陶瓷哪个更大"），歧义只能出在"指的是哪一个"上；不能只说"这个是什么颜色的""这是什么"这类连物品是什么都没说的问法。specific 分支的问句要点明是哪一个并且也点出类别。
给你题目（两个分支的台词、澄清句、答案要点）和 t0 的真实帧。
1. 判断 ambiguous 分支 ue0 是否已经点出类别名称（names_category）。
2. 如果没有，改写两句 ue0：ambiguous 的新问句用"这个/那个/这件 + 类别词"的单数形式（如"这个陶碗是什么颜色的？""这幅画是什么年代的？""这里面的服饰是什么颜色的？"），点出类别但不说明是哪一个——画面里有 ≥2 个该类候选，所以模型仍需澄清；**不要**改成"这几件…分别…""这些…都…"这种问全部的说法。specific 的新问句点出类别并说清是哪一个（原句已经合格就原样保留）；问的内容（颜色/年代/文字/动作…）保持不变，不能透露答案，类别词要和画面里的东西相符、口语自然。
只输出 JSON：{"names_category": bool, "note": "...", "ambiguous_text": "新问句或原句", "specific_text": "新问句或原句"}"""


def setup(batch):
    R.use_batch(batch)
    return R, V, CFG.RUNS / batch / 'runs_v3'


def task_dirs(runs, caps):
    for cap in caps:
        for p in sorted(runs.glob(f'{cap}_*/candidate_task.json')):
            yield cap, p.parent


def ue(task, bid, eid):
    for b in task['branches']:
        if b['branch_id'] == bid:
            for e in b['user_events']:
                if e['event_id'] == eid:
                    return e
    return None


def reject(d):
    for name in ('candidate_task.json', 'task.json'):
        p = d / name
        if p.exists():
            p.rename(d / name.replace('.json', '.rejected_v3b.json'))


def audit_visibility(R, task, d, media, budget):
    b0 = task['branches'][0]
    e0 = b0['user_events'][0]
    t0 = float(e0['time_s'])
    content = [{'type': 'text', 'text': f"t0={t0}s 用户说：“{e0['text']}”\n对象 X 的内部描述：{json.dumps(e0.get('state_op', {}).get('target', ''), ensure_ascii=False)}"}]
    for lab, tt in (('t0−1s', max(0.0, t0 - 1)), ('t0', t0), ('t0+1s', t0 + 1)):
        content += [{'type': 'text', 'text': f'[{lab} 帧 {tt:.1f}s]'}, R.img_part(R.frame(task['source_id'], media, tt))]
    v = R.call([{'role': 'system', 'content': VIS_SYSTEM}, {'role': 'user', 'content': content}], 'v3b_visibility', d, budget, max_tokens=400)
    return {'check': 'visible_at_t0', 'ok': bool(v.get('visible_at_t0')), 'note': v.get('note', '')}


def audit_clarify(R, V, task, d, media, budget):
    amb, spc = ue(task, 'ambiguous', 'ue0'), ue(task, 'specific', 'ue0')
    if not amb or not spc:
        return {'check': 'names_category', 'ok': False, 'note': '缺少 ue0'}
    t0 = float(amb['time_s'])
    brief = {'ambiguous_ue0': amb['text'], 'specific_ue0': spc['text'],
             'clarify_text': next((s['target_text'] for b in task['branches'] if b['branch_id'] == 'ambiguous' for s in b['slots'] if s['slot_id'] == 'sl_clarify'), ''),
             'answer_key': next((s['key_attributes'] for b in task['branches'] if b['branch_id'] == 'specific' for s in b['slots']), [])}
    content = [{'type': 'text', 'text': '题目：' + json.dumps(brief, ensure_ascii=False)}]
    for tt in (t0, t0 + 1):
        content += [{'type': 'text', 'text': f'[t0 帧 {tt:.1f}s]'}, R.img_part(R.frame(task['source_id'], media, tt))]
    v = R.call([{'role': 'system', 'content': CLR_SYSTEM}, {'role': 'user', 'content': content}], 'v3b_clarify', d, budget, max_tokens=600)
    ok_before = bool(v.get('names_category')) and not V.vague_question(amb['text'])
    if ok_before:
        return {'check': 'names_category', 'ok': True, 'note': v.get('note', '')}
    new_a, new_s = (v.get('ambiguous_text') or '').strip(), (v.get('specific_text') or '').strip()
    if not new_a or V.vague_question(new_a) or new_a == amb['text'] or not new_s or V.vague_question(new_s) or re.search(r'分别|都是|都有|各是|各自', new_a):
        return {'check': 'names_category', 'ok': False, 'note': '改写失败：' + v.get('note', ''), 'rewrite': v}
    # 复核改写后的问句（文字 + 帧）
    brief2 = dict(brief, ambiguous_ue0=new_a, specific_ue0=new_s)
    content2 = [{'type': 'text', 'text': '题目：' + json.dumps(brief2, ensure_ascii=False)}] + content[1:]
    v2 = R.call([{'role': 'system', 'content': CLR_SYSTEM}, {'role': 'user', 'content': content2}], 'v3b_clarify_recheck', d, budget, max_tokens=400)
    if not v2.get('names_category'):
        return {'check': 'names_category', 'ok': False, 'note': '改写后复核仍不通过：' + v2.get('note', ''), 'rewrite': v}
    return {'check': 'names_category', 'ok': True, 'note': v.get('note', ''), 'rewrite': {'ambiguous_ue0': [amb['text'], new_a], 'specific_ue0': [spc['text'], new_s]}}


def apply_rewrite(d, task, rw):
    for name in ('candidate_task.json', 'task.json'):
        p = d / name
        if p.exists():
            bak = d / name.replace('.json', '.before_v3b.json')
            if not bak.exists():
                p.rename(bak)
    for bid, key in (('ambiguous', 'ambiguous_ue0'), ('specific', 'specific_ue0')):
        e = ue(task, bid, 'ue0')
        e['text'] = rw[key][1]
    task['v3b'] = {'fix': 'clarify_names_category', 'rewrite': rw, 'date': time.strftime('%F')}
    for name in ('candidate_task.json', 'task.json'):
        if (d / name.replace('.json', '.before_v3b.json')).exists():
            json.dump(task, open(d / name, 'w'), ensure_ascii=False, indent=1)


def one(args_, R, V, media_map, cap, d, budget):
    out = d / 'v3b_audit.json'
    if out.exists() and not args_.redo:
        return cap, json.load(open(out)).get('result', 'done')
    task = json.load(open(d / 'candidate_task.json'))
    media = media_map[task['source_id']][0]
    try:
        res = audit_clarify(R, V, task, d, media, budget) if cap == 'proactive_clarify' else audit_visibility(R, task, d, media, budget)
        if res['ok'] and res.get('rewrite'):
            if not args_.dry:
                apply_rewrite(d, task, res['rewrite'])
            result = 'rewritten'
        elif res['ok']:
            result = 'keep'
        else:
            if not args_.dry:
                reject(d)
            result = 'rejected'
        json.dump({'task_id': task['task_id'], 'cap': cap, 'result': result, **res}, open(out, 'w'), ensure_ascii=False, indent=1)
    except Exception as e:
        result = 'error'
        json.dump({'task_id': task['task_id'], 'cap': cap, 'result': 'error', 'error': f'{type(e).__name__}: {e}'}, open(out, 'w'), ensure_ascii=False, indent=1)
    print(f'[{result}] {cap} {d.name} spent=${R._spent["usd"]:.3f}', flush=True)
    return cap, result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['audit', 'status'])
    ap.add_argument('--batch', required=True, choices=list(CFG.BATCHES))
    ap.add_argument('--caps', default=','.join(CAPS3))
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--budget', type=float, default=10.0)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--dry', action='store_true', help='只审计不改文件')
    ap.add_argument('--redo', action='store_true')
    a = ap.parse_args()
    R, V, runs = setup(a.batch)
    caps = a.caps.split(',')
    if a.cmd == 'status':
        st = {}
        for cap in caps:
            dirs = list(runs.glob(f'{cap}_*/'))
            c = Counter()
            for d in dirs:
                if (d / 'candidate_task.json').exists():
                    c['active'] += 1
                    if (d / 'candidate_task.before_v3b.json').exists():
                        c['rewritten'] += 1
                    if (d / 'v3b_audit.json').exists():
                        c['audited'] += 1
                elif (d / 'candidate_task.rejected_v3b.json').exists():
                    c['rejected'] += 1
            st[cap] = dict(c)
        print(json.dumps(st, ensure_ascii=False, indent=1)); return
    media_map = R.media_paths()
    jobs = list(task_dirs(runs, caps))
    if a.limit:
        per = Counter(); jobs = [j for j in jobs if per.update([j[0]]) or per[j[0]] <= a.limit]
    print('tasks', len(jobs), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda j: one(a, R, V, media_map, j[0], j[1], a.budget), jobs))
    summ = {}
    for cap, r in res:
        summ.setdefault(cap, Counter())[r] += 1
    print(json.dumps({k: dict(v) for k, v in summ.items()}, ensure_ascii=False), f'spent=${R._spent["usd"]:.2f}')


if __name__ == '__main__':
    main()
