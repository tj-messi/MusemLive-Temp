#!/usr/bin/env python3
"""10-16 用户：QA 里用户是第一人称和模型对话；转写里的“主播/博主”就是用户本人，不能写成“主播说了 xxx”。
把 runs/<batch>/{runs_v3,runs_pg}/*/task.json(+candidate_task.json)、runs_hr/*/refreshed_task.json 里涉及的文本用 Qwen 改写：
用户台词 → 第一人称（我），模型答案/要点/禁止项 → 用“您”称呼用户；其他结构不动。原文件备份 *.before_fp_20261016.json，task 加 first_person_fix。
用法: python3 fix_first_person.py --batch 50a [--workers 12]
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R; import config as CFG
PAT = re.compile(r'主播|博主|UP主|up主|拍摄者')
SYSTEM = """你在修订第一视角博物馆流式交互 benchmark 的题目文本。视频是戴着摄像头的用户第一视角拍的，音频里标为“主播/博主/UP主/拍摄者”的声音就是**用户本人**。
规则：
1. user_events 里的 text 是用户（视频主角本人）对模型说的话，必须第一人称，并且口吻要像主角本人一边逛一边随口问助手（接着自己刚说的话/刚做的事的语境，用词风格和转写里主角的说话风格一致）：把“主播/博主/UP主/拍摄者”改成“我”（如“帮我记一下我接下来提到的展品名称”“刚才我说的那个年代是多少？”）。
2. slots 里的 target_text、key_attributes、forbidden_claims 是模型对用户说的话/评分要点：把“主播/博主/UP主/拍摄者（说/提到/念…）”改成“您（刚才说/提到…）”，如“您刚才说是太古代”“[音频] 您在258.3s提到了‘顺山集遗址’”。
3. 讲解员、游客、广播、旁白等**其他人**保持第三人称不动。如果某处“主播”明显指的不是用户本人而是画面里另一个人，保持不变并在 note 里说明。
4. 只改称呼和随之必需的语法，不改事实、时间、数字、引号里的转写原句（audio_refs 的 text 是转写原句，不动）、标签“[音频]/[视觉]”、id。
只输出 JSON：{"branches": [{"branch_id": "...", "user_events": [{"event_id": "...", "text": "..."}], "slots": [{"slot_id": "...", "target_text": "...", "key_attributes": ["..."], "forbidden_claims": ["..."]}]}], "note": ""}"""


def affected(t):
    for b in t['branches']:
        for e in b['user_events']:
            if PAT.search(e.get('text', '')): return True
        for s in b['slots']:
            if PAT.search(s.get('target_text', '') or '') or any(PAT.search(k or '') for k in (s.get('key_attributes') or []) + (s.get('forbidden_claims') or [])): return True
    return False


def fix_one(path, log_dir, budget):
    t = json.load(open(path))
    if t.get('first_person_fix') or not affected(t):
        return 'skip'
    brief = {'branches': [{'branch_id': b['branch_id'], 'user_events': [{'event_id': e['event_id'], 'text': e['text']} for e in b['user_events']],
                           'slots': [{'slot_id': s['slot_id'], 'target_text': s.get('target_text', ''), 'key_attributes': s.get('key_attributes') or [], 'forbidden_claims': s.get('forbidden_claims') or []} for s in b['slots']]} for b in t['branches']]}
    r = R.call([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(brief, ensure_ascii=False)}], 'fp_' + t['task_id'][:60], log_dir, budget, max_tokens=4000)
    nb = {b['branch_id']: b for b in r.get('branches', [])}
    if set(nb) != {b['branch_id'] for b in t['branches']}:
        return 'bad_branches'
    before = json.loads(json.dumps(brief)); changed = 0
    for b in t['branches']:
        x = nb[b['branch_id']]
        ev = {e['event_id']: e['text'] for e in x.get('user_events', [])}
        for e in b['user_events']:
            if e['event_id'] in ev and ev[e['event_id']] and ev[e['event_id']] != e['text']:
                e['text'] = ev[e['event_id']]; changed += 1
        sl = {s['slot_id']: s for s in x.get('slots', [])}
        for s in b['slots']:
            y = sl.get(s['slot_id'])
            if not y: continue
            if y.get('target_text') and y['target_text'] != s.get('target_text'): s['target_text'] = y['target_text']; changed += 1
            ka = y.get('key_attributes')
            if isinstance(ka, list) and len(ka) == len(s.get('key_attributes') or []) and ka != s.get('key_attributes'):
                s['key_attributes'] = ka; changed += 1
                if 'modality_facts' in s:
                    s['modality_facts'] = {'audio': [f for f in ka if f.startswith('[音频]')], 'visual': [f for f in ka if f.startswith('[视觉]')]}
            fc = y.get('forbidden_claims')
            if isinstance(fc, list) and len(fc) == len(s.get('forbidden_claims') or []) and fc != s.get('forbidden_claims'):
                s['forbidden_claims'] = fc; changed += 1
    t['first_person_fix'] = {'version': 'fp_20261016', 'changed_fields': changed, 'note': r.get('note', ''), 'before': before}
    bak = path.with_name(path.stem + '.before_fp_20261016.json')
    if not bak.exists(): bak.write_text(json.dumps(json.load(open(path)), ensure_ascii=False, indent=1))
    path.write_text(json.dumps(t, ensure_ascii=False, indent=1))
    return f'fixed({changed})'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch', required=True); ap.add_argument('--workers', type=int, default=12); ap.add_argument('--budget', type=float, default=5.0)
    a = ap.parse_args(); R.use_batch(a.batch); root = CFG.RUNS / a.batch
    paths = list(root.glob('runs_v3/*/task.json')) + list(root.glob('runs_pg/*/task.json')) + list(root.glob('runs_hr/*/refreshed_task.json'))
    log_dir = root / 'fp_fix_logs'; log_dir.mkdir(exist_ok=True)
    print('tasks', len(paths), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda p: (p, fix_one(p, log_dir, a.budget)), paths))
    import collections; print(collections.Counter(r.split('(')[0] for _, r in res))
    for p, r in res:  # candidate_task.json 与 task.json 同步
        if r.startswith('fixed') and p.name == 'task.json' and (p.parent / 'candidate_task.json').exists():
            (p.parent / 'candidate_task.json').write_text(p.read_text())
    print('spent', R._spent)


if __name__ == '__main__':
    main()
