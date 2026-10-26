#!/usr/bin/env python3
"""10-16 用户：QA 语言要和场景一致（英文场景/全程无人说话 → 英文；中文场景 → 中文）。
给每道题打 language，英文视频上的中文题用 Qwen 改写成英文（用户台词仍是主角本人口吻、第一人称；模型回答称 you；[音频]/[视觉] 标签保留；转写引用原文不动）。
备份 *.before_lang_20261016.json，task 加 language_fix。用法: python3 fix_language.py --batch 50a
"""
import argparse, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import common as R; import config as CFG
SYSTEM = """You are revising items of a first-person museum live-streaming benchmark. The video's spoken language is English (or nobody speaks), so every item text must be in natural English.
Rewrite the given JSON: translate user_events.text (the camera-wearer talking to the assistant, first person, casual spoken tone consistent with how the host actually talks), target_text (the assistant's spoken reply to the user, address the user as "you"), key_attributes and forbidden_claims into English.
Rules: keep the leading tags "[音频]" / "[视觉]" EXACTLY as they are (these Chinese tags are machine labels; never translate them to [Audio]/[Visual]); keep quoted transcript snippets (text inside quotes that was copied from the audio) in their original language; do not change ids, times, facts or numbers; do not add or remove items.
Output JSON only: {"branches": [{"branch_id": "...", "user_events": [{"event_id": "...", "text": "..."}], "slots": [{"slot_id": "...", "target_text": "...", "key_attributes": ["..."], "forbidden_claims": ["..."]}]}]}"""


def fix_one(path, lang_of, log_dir, budget):
    t = json.load(open(path)); lang = lang_of.get(t['source_id'], 'en')
    changed = False
    if t.get('language') != lang:
        t['language'] = lang; changed = True
    if lang == 'en' and not t.get('language_fix'):
        bad = any(R.lang_bad(e['text'], 'en') for b in t['branches'] for e in b['user_events']) or any(R.lang_bad(s.get('target_text', ''), 'en') for b in t['branches'] for s in b['slots'])
        if bad:
            brief = {'branches': [{'branch_id': b['branch_id'], 'user_events': [{'event_id': e['event_id'], 'text': e['text']} for e in b['user_events']],
                                   'slots': [{'slot_id': s['slot_id'], 'target_text': s.get('target_text', ''), 'key_attributes': s.get('key_attributes') or [], 'forbidden_claims': s.get('forbidden_claims') or []} for s in b['slots']]} for b in t['branches']]}
            r = R.call([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(brief, ensure_ascii=False)}], 'lang_' + t['task_id'][:60], log_dir, budget, max_tokens=4000)
            nb = {b['branch_id']: b for b in r.get('branches', [])}
            if set(nb) != {b['branch_id'] for b in t['branches']}: return 'bad_branches'
            bak = path.with_name(path.stem + '.before_lang_20261016.json')
            if not bak.exists(): bak.write_text(path.read_text())
            n = 0
            for b in t['branches']:
                x = nb[b['branch_id']]; ev = {e['event_id']: e['text'] for e in x.get('user_events', [])}
                for e in b['user_events']:
                    if ev.get(e['event_id']): e['text'] = ev[e['event_id']]; n += 1
                sl = {s['slot_id']: s for s in x.get('slots', [])}
                for s in b['slots']:
                    y = sl.get(s['slot_id'])
                    if not y: continue
                    if y.get('target_text'): s['target_text'] = y['target_text']; n += 1
                    ka = y.get('key_attributes')
                    if isinstance(ka, list) and len(ka) == len(s.get('key_attributes') or []):
                        s['key_attributes'] = ka
                        if 'modality_facts' in s: s['modality_facts'] = {'audio': [f for f in ka if f.startswith('[音频]')], 'visual': [f for f in ka if f.startswith('[视觉]')]}
                    fc = y.get('forbidden_claims')
                    if isinstance(fc, list) and len(fc) == len(s.get('forbidden_claims') or []): s['forbidden_claims'] = fc
            t['language_fix'] = {'version': 'lang_20261016', 'to': 'en', 'changed_fields': n, 'before': brief}
            changed = True
    if changed:
        path.write_text(json.dumps(t, ensure_ascii=False, indent=1))
        if path.name == 'task.json' and (path.parent / 'candidate_task.json').exists(): (path.parent / 'candidate_task.json').write_text(path.read_text())
    return 'rewritten' if t.get('language_fix') and changed else ('tagged' if changed else 'skip')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--batch', required=True); ap.add_argument('--workers', type=int, default=12); ap.add_argument('--budget', type=float, default=5.0)
    a = ap.parse_args(); R.use_batch(a.batch); root = CFG.RUNS / a.batch
    w = R.load_windows(); lang_of = {s: R.video_lang(w[s]) for s in w}
    paths = list(root.glob('runs_v3/*/task.json')) + list(root.glob('runs_pg/*/task.json')) + list(root.glob('runs_hr/*/refreshed_task.json'))
    log_dir = root / 'lang_fix_logs'; log_dir.mkdir(exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        res = list(ex.map(lambda p: fix_one(p, lang_of, log_dir, a.budget), paths))
    import collections; print(collections.Counter(res), 'spent', R._spent)


if __name__ == '__main__':
    main()
