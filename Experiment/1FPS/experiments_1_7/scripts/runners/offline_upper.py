#!/usr/bin/env python3
"""离线上界（10-18，实验 7）：每个 user_query slot，把 session 开始到用户说完这一段的因果前缀（每秒 1 帧 + 混音音轨，与全双工 eval 同一份 mix）
一次性交给 API 全模态模型，再附上用户各轮话语文字，让它离线作答 —— 没有"何时开口"的压力、音频不必实时听，看在线 vs 离线差多少。
可选模型受地区限制：Gemini / GPT / Claude 在本机 OpenRouter 403，只能用 qwen/qwen3.8-omni-flash（注意：它也是出题模型，上界偏乐观）。
输出与全双工 eval 同 schema（timeline 的 user_event.response = 离线回答；duplex_turns 为空 → 画面触发 slot 记沉默，分析时只看 user_query），
judge 直接 judge_points.py run --eval-glob。
  /usr/bin/python3 offline_upper.py [--workers 16] [--limit N]
"""
import argparse, base64, io, json, os, re, sys, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

D = Path(__file__).resolve().parent
OUT = D / 'runs/offline_qwen_omni'
os.environ['EVAL_OUT'] = str(OUT / '_frames')
sys.path.insert(0, str(D.parent / 'omni_eval_20261016'))
import eval_omni as eo  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

MC = D.parent / 'model_compare_20261018'
KEY = (D.parent / 'museumlive_316_qa_20261011/main/.openrouter_key').read_text().strip()
MODEL = os.environ.get('OFFLINE_MODEL', 'qwen/qwen3.8-omni-flash')
MAX_FRAMES = 32
SYSTEM = ('你是一个戴在用户身上的实时博物馆助手。下面给你用户从这一段开始到刚刚为止看到的画面（每秒一帧，按时间顺序）和听到的全部声音'
          '（讲解员、广播、人群，以及用户本人说的话；用户对你说的话也会以文字列出）。请回答用户最后一句话。'
          '规则：回答要基于看到和听到的内容；记住用户之前说过的话和听到的讲解内容；遇到用户取消、纠正或修改要求时，以最新的要求为准；'
          '用简洁的口语回答，用户说什么语言就用什么语言。')
SPENT = {'usd': 0.0}; LOCK = threading.Lock()


def jpg_b64(p, w=448):
    im = Image.open(p).convert('RGB'); im = im.resize((w, int(im.height * w / im.width)))
    b = io.BytesIO(); im.save(b, 'JPEG', quality=80); return base64.b64encode(b.getvalue()).decode()


def call(content):
    body = {'model': MODEL, 'temperature': 0, 'max_tokens': 400, 'messages': [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': content}]}
    last = None
    for a in range(4):
        try:
            s = requests.Session(); s.trust_env = False
            r = s.post('https://openrouter.ai/api/v1/chat/completions', headers={'Authorization': 'Bearer ' + KEY}, json=body, timeout=(20, 300))
            r.raise_for_status(); j = r.json()
            with LOCK: SPENT['usd'] += float((j.get('usage') or {}).get('cost') or 0)
            return (j['choices'][0]['message']['content'] or '').strip()
        except Exception as e:
            last = e; time.sleep(3 * (a + 1))
    raise RuntimeError(str(last)[:300])


def answer(task, branch, ev_id, mix_wav, T0, spans):
    """返回 (回答, 说完时刻)。说完时刻取全双工 MiniCPM eval 记下的 tts_spans（同一份 mix）。"""
    events = sorted(branch['user_events'], key=lambda e: e['time_s'])
    mix, sr = sf.read(mix_wav, dtype='float32')
    tend = {x['event_id']: x['tts_end'] for x in spans}[ev_id]
    seg = mix[:int((tend - T0) * sr)]
    b = io.BytesIO(); sf.write(b, seg, sr, format='WAV', subtype='PCM_16')
    frames = [f for f in eo.extract_frames(eo.media_of(task['source_id']), task['source_id'], T0, float(np.ceil(tend))) if f.exists()]
    if len(frames) > MAX_FRAMES: frames = [frames[int(i * len(frames) / MAX_FRAMES)] for i in range(MAX_FRAMES)]
    said = [e for e in events if e['time_s'] <= tend - 0.01]
    content = [{'type': 'text', 'text': f'画面（{len(frames)} 帧，时间从早到晚）：'}]
    content += [{'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + jpg_b64(f)}} for f in frames]
    content += [{'type': 'text', 'text': '现场声音（这一段的完整音轨）：'},
                {'type': 'input_audio', 'input_audio': {'data': base64.b64encode(b.getvalue()).decode(), 'format': 'wav'}},
                {'type': 'text', 'text': '用户对你说过的话（按时间）：\n' + '\n'.join(f'[{e["time_s"] - T0:.0f}s] {e["text"]}' for e in said) +
                 f'\n\n请回答用户最后一句话：{said[-1]["text"]}'}]
    return call(content), tend


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=16); ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    final = json.load(open(MC / 'tasks_final450.json'))
    mix_of = {}
    for f in MC.glob('minicpm/*/eval/evaluation_results.json'):
        for r in json.load(open(f))['results']: mix_of[r['branch_key']] = (r['omni']['mix_wav'], r['omni']['window_s'][0], r['omni']['tts_spans'])
    (OUT / 'eval').mkdir(parents=True, exist_ok=True)
    cache_f = OUT / 'answers.jsonl'
    cache = {}
    if cache_f.exists():
        for l in open(cache_f): x = json.loads(l); cache[x['key']] = x
    jobs = []
    for t in final:
        for b in t['branches']:
            bk = f"{t['task_id']}::{b['branch_id']}"
            for sl in b['slots']:
                if sl['trigger']['kind'] != 'user_query': continue
                k = f"{bk}::{sl['trigger']['event_id']}"
                if k not in cache and all(j[0] != k for j in jobs): jobs.append((k, t, b, sl['trigger']['event_id']))
    if a.limit: jobs = jobs[:a.limit]
    print(f'{len(jobs)} calls to make ({len(cache)} cached)', flush=True)

    def run(j):
        k, t, b, eid = j; bk = f"{t['task_id']}::{b['branch_id']}"
        try:
            resp, tend = answer(t, b, eid, *mix_of[bk])
            x = {'key': k, 'response': resp, 'tts_end': tend}
        except Exception as e:
            x = {'key': k, 'error': str(e)[:300]}
        with LOCK:
            open(cache_f, 'a').write(json.dumps(x, ensure_ascii=False) + '\n')
        return x
    t0 = time.time()
    with ThreadPoolExecutor(a.workers) as ex:
        for i, x in enumerate(ex.map(run, jobs), 1):
            cache[x['key']] = x
            if i % 50 == 0 or i == len(jobs): print(f'{i}/{len(jobs)} {time.time() - t0:.0f}s ${SPENT["usd"]:.3f}', flush=True)
    # 写成全双工 eval schema
    results = []
    for t in final:
        for b in t['branches']:
            bk = f"{t['task_id']}::{b['branch_id']}"; tl = []
            for e in sorted(b['user_events'], key=lambda e: e['time_s']):
                x = cache.get(f'{bk}::{e["event_id"]}') or {}
                tl.append({'type': 'user_event', 'time_s': e['time_s'], 'event_id': e['event_id'], 'utterance': e['text'], 'response': x.get('response', ''),
                           'response_time_s': x.get('tts_end')})
            slots = [{'slot': sl, 'timeline_prefix': [y for y in tl if y['time_s'] <= float(sl['window']['end_s']) + 1e-6]} for sl in b['slots']]
            results.append({'branch_key': bk, 'task_id': t['task_id'], 'source_id': t['source_id'], 'capability': t['capability'], 'branch_id': b['branch_id'],
                            'session': t['session'], 'user_events': b['user_events'], 'slots': slots, 'timeline': tl, 'duplex_turns': [],
                            'omni': {'offline': True, 'model': MODEL, 'max_frames': MAX_FRAMES}})
    json.dump({'schema_version': 'museumlive_offline_upper_v1', 'model': MODEL, 'results': results}, open(OUT / 'eval/evaluation_results.json', 'w'), ensure_ascii=False)
    print('errors', sum('error' in x for x in cache.values()), 'spent', round(SPENT['usd'], 3))


if __name__ == '__main__':
    main()
