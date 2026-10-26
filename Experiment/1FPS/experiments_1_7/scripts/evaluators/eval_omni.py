#!/usr/bin/env python3
"""
Omni eval（10-16 试验版）：MiniCPM-o 4.5 半双工流式 —— 每秒喂 1 帧画面 + 1 秒混合音轨（视频原声 + 把用户台词 TTS 成语音插进去），
用户语音说完后让模型回答（纯音频触发，不给文字提问）。

与旧 eval_visual_history 的区别：
  - 模型 init_audio=True，走 streaming_prefill(omni_mode=True) / streaming_generate，文字输出；
  - 用户台词 → edge-tts（zh-CN 声音）→ 16k wav，按 user_events.time_s 叠进原音轨；TTS 期间原声压到 DUCK 倍；
  - 回答时刻 = 用户语音结束所在秒的末尾（记在 response_time_s；timeline 的 time_s 仍写 user_events.time_s，便于沿用 judge）。
用法：
  EVAL_TASKS=<tasks.json> EVAL_TASK_ID=<task_id> EVAL_OUT=<dir> MODEL_DEVICE=4 python3 eval_omni.py
可选环境变量：TTS_VOICE(zh-CN-XiaoxiaoNeural) DUCK(0.3) TICK_GENERATE(0：只在用户说完后生成；1：每秒都让模型决定说不说) USER_MARK(0/1：音频里插用户语音的那一秒附带文字提示“【用户正在对你说话】”)
产物：EVAL_OUT/eval/evaluation_results.json（与旧 schema 兼容 + omni 字段）、EVAL_OUT/mix/<branch>.wav、EVAL_OUT/frames/、evaluation.log
"""
import asyncio, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'pylib'))
import edge_tts  # noqa: E402
import librosa  # noqa: E402
import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

W = Path(os.environ.get('MUSEUMLIVE_WORK_ROOT', '.')).resolve()
MODEL_PATH = os.environ.get('MINICPM_MODEL_PATH', 'MiniCPM-o-4_5')
MEDIA50_ROOT = Path(os.environ.get('MUSEUMLIVE_MEDIA50_ROOT', W / 'media50'))
OUT = Path(os.environ['EVAL_OUT']); (OUT / 'eval').mkdir(parents=True, exist_ok=True); (OUT / 'mix').mkdir(exist_ok=True); (OUT / 'frames').mkdir(exist_ok=True)
LOG = OUT / 'evaluation.log'
VOICES = {'zh': os.environ.get('TTS_VOICE_ZH', 'zh-CN-XiaoxiaoNeural'), 'en': os.environ.get('TTS_VOICE_EN', 'en-US-JennyNeural')}
TTS_VOICE = VOICES['zh']; TTS_BACKEND = os.environ.get('TTS_BACKEND', 'edge')  # edge | qwen（qwen 需先用 tts_prepare.py 预合成）
DUCK = float(os.environ.get('DUCK', '0.3'))
TICK_GENERATE = os.environ.get('TICK_GENERATE', '0') == '1'; USER_MARK = os.environ.get('USER_MARK', '0') == '1'
USE_TTS_TEMPLATE = os.environ.get('USE_TTS_TEMPLATE', '1') == '1'  # 10-16 变体试验：必须为 1，否则多轮输出 <think> 乱码
DO_SAMPLE = os.environ.get('DO_SAMPLE', '0') == '1'  # greedy 更准
ASSIST_PREFILL = os.environ.get('ASSIST_PREFILL', '0') == '1'
SR = 16000
EXP_ABLATE = os.environ.get('EXP_ABLATE', '')  # 10-18 模态消融：mute_ambient（现场原声置零、保留用户语音）| black_video（帧换成同尺寸纯黑）| no_transcript（MOSS：不给现场转写）
TTS_CACHE = HERE / 'tts_cache'; TTS_CACHE.mkdir(exist_ok=True)

SYSTEM = ('你是一个戴在用户身上的实时博物馆助手。用户正戴着摄像头在博物馆里逛，你同时看到用户看到的画面、听到现场的全部声音'
          '（讲解员、广播、人群，以及用户本人说的话）。用户会直接用语音对你提问或下指令。'
          '规则：只有当用户对你说话时才回答，用简洁的中文文字回答，回答要基于你刚才看到和听到的内容；'
          '记住用户之前说过的话和你听到的讲解内容；遇到用户取消、纠正或修改要求时，以最新的要求为准；'
          '没有人对你说话时，只输出<SILENCE>。用户说什么语言，你就用什么语言回答（用户说英语就用英语回答）。')


def log(s):
    q = time.strftime('[%F %T] ') + s; print(q, flush=True); LOG.open('a').write(q + '\n')


FR316 = W / 'museumlive_450_frames_20261007/frames'; AUD316 = W / 'audio_annotations_20261015/media316_audio'


def media_of(src):
    """50 视频：原片路径；316 视频：('316', 25fps 帧目录, 找回的 wav)。"""
    for ext in ('.mp4', '.mkv'):
        p = MEDIA50_ROOT / (src + ext)
        if p.exists(): return p
    if (AUD316 / (src + '.wav')).exists() and (FR316 / src).is_dir(): return ('316', FR316 / src, AUD316 / (src + '.wav'))
    raise FileNotFoundError(src)


QWEN_INDEX = {}
for _f in sorted(HERE.glob('tts_index_qwen*.json')):
    QWEN_INDEX.update(json.load(open(_f)))


def tts(text, voice=None, task_id=None, event_id=None):
    voice = voice or TTS_VOICE
    if TTS_BACKEND == 'qwen':
        hit = (QWEN_INDEX.get(task_id) or {}).get(event_id)
        if hit and Path(hit['wav']).exists():
            y, _ = librosa.load(hit['wav'], sr=SR, mono=True); return y.astype(np.float32)
        log(f'qwen tts missing for {task_id}/{event_id}, fallback to edge')
    key = hashlib.md5((voice + '|' + text).encode()).hexdigest()[:16]
    wav = TTS_CACHE / f'{key}.wav'
    if not wav.exists():
        mp3 = TTS_CACHE / f'{key}.mp3'
        async def run():
            await edge_tts.Communicate(text, voice).save(str(mp3))
        asyncio.run(run())
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(mp3), '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_s16le', str(wav)], check=True)
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    return y.astype(np.float32)


def build_mix(media, T0, T1, events, voice, task_id=None):
    """返回 (mixed[T0:T1] 16k float32, 每个事件的 {tts_start, tts_end})。"""
    if isinstance(media, tuple):  # 316：直接从找回的 wav 截
        y, _ = librosa.load(str(media[2]), sr=SR, mono=True, offset=T0, duration=T1 - T0)
    else:
        tmp = OUT / 'mix' / f'_orig_{media.stem}_{T0:.0f}_{T1:.0f}.wav'
        if not tmp.exists():
            subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{T0:.3f}', '-t', f'{T1 - T0:.3f}', '-i', str(media), '-vn', '-ac', '1', '-ar', str(SR), '-c:a', 'pcm_s16le', str(tmp)], check=True)
        y, _ = librosa.load(str(tmp), sr=SR, mono=True)
    n = int((T1 - T0) * SR)
    if len(y) < n: y = np.concatenate([y, np.zeros(n - len(y), dtype=np.float32)])
    y = y[:n].astype(np.float32)
    if EXP_ABLATE == 'mute_ambient': y = np.zeros_like(y)
    gain = np.ones(n, dtype=np.float32); add = np.zeros(n, dtype=np.float32); spans = []
    for e in events:
        v = tts(e['text'], voice, task_id, e['event_id']); a = int((e['time_s'] - T0) * SR); b = min(n, a + len(v))
        gain[a:b] = np.minimum(gain[a:b], DUCK); add[a:b] += v[:b - a] * 0.9
        spans.append({'event_id': e['event_id'], 'tts_start': round(e['time_s'], 3), 'tts_end': round(T0 + b / SR, 3), 'tts_duration_s': round((b - a) / SR, 3)})
    mix = np.clip(y * gain + add, -1.0, 1.0)
    return mix, spans


def extract_frames(media, src, T0, T1):
    fr = _extract_frames(media, src, T0, T1)
    if EXP_ABLATE != 'black_video': return fr
    d = OUT / 'frames_black'; d.mkdir(exist_ok=True); out = []
    for f in fr:
        if not f.exists(): out.append(f); continue
        sz = Image.open(f).size; b = d / f'{sz[0]}x{sz[1]}.jpg'
        if not b.exists(): Image.new('RGB', sz).save(b)
        out.append(b)
    return out


def _extract_frames(media, src, T0, T1):
    if isinstance(media, tuple):  # 316：第 k 秒取 F{(T0+k)*25+13:06d}（秒中间那帧），越界取最后一帧
        fs = sorted(media[1].glob('F*.jpg')); out = []
        for k in range(int(T1 - T0)):
            idx = int((T0 + k) * 25) + 13; out.append(fs[min(max(idx, 1), len(fs)) - 1])
        return out
    d = OUT / 'frames' / src; d.mkdir(parents=True, exist_ok=True)
    if not (d / f'{T0:.0f}_000.jpg').exists():
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{T0:.3f}', '-t', f'{T1 - T0:.3f}', '-i', str(media), '-vf', 'fps=1', '-q:v', '3',
                        str(d / f'{T0:.0f}_%03d.jpg')], check=True)
    return [d / f'{T0:.0f}_{k + 1:03d}.jpg' for k in range(int(T1 - T0))]


def patch_cache():
    """transformers 4.57 的 DynamicCache 用 layers[i].keys/values，MiniCPM-o 的 utils.py 仍按 cache.key_cache[i] 读写；加一个可读写的 list 视图。"""
    from transformers.cache_utils import DynamicCache
    if hasattr(DynamicCache(), 'key_cache'):
        return  # transformers<=4.5x 自带 key_cache，不用垫
    from transformers.cache_utils import DynamicLayer

    class _KV:
        __slots__ = ('c', 'a')
        def __init__(self, c, a): self.c = c; self.a = a
        def _ls(self): return [l for l in self.c.layers if getattr(l, 'is_initialized', False)]
        def __len__(self): return len(self._ls())
        def __getitem__(self, i): return getattr(self._ls()[i], self.a)
        def __setitem__(self, i, v): setattr(self._ls()[i], self.a, v)
        def __iter__(self): return iter([getattr(l, self.a) for l in self._ls()])
        def __bool__(self): return len(self._ls()) > 0

    def setter(attr):
        def f(self, lst):
            lst = list(lst)
            while len(self.layers) < len(lst): self.layers.append(DynamicLayer())
            for l, t in zip(self.layers, lst):
                if not getattr(l, 'is_initialized', False): l.lazy_initialization(t)
                setattr(l, attr, t)
        return f
    DynamicCache.key_cache = property(lambda self: _KV(self, 'keys'), setter('keys'))
    DynamicCache.value_cache = property(lambda self: _KV(self, 'values'), setter('values'))
    if not hasattr(DynamicCache, 'seen_tokens'): DynamicCache.seen_tokens = property(lambda x: x.get_seq_length())


def main():
    import torch
    from transformers import AutoModel
    patch_cache()
    tasks = json.loads(Path(os.environ['EVAL_TASKS']).read_text())
    tid = os.environ.get('EVAL_TASK_ID'); tasks = [t for t in tasks if not tid or t['task_id'] == tid]
    assert tasks, 'no task'
    device = int(os.environ.get('MODEL_DEVICE', '0'))
    model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16,
                                      init_vision=True, init_audio=True, init_tts=False)
    model.eval().cuda(device); log(f'MiniCPM-o loaded (vision+audio) on cuda:{device}')
    results = []
    for task in tasks:
        src = task['source_id']; media = media_of(src); s = task['session']
        for branch in task['branches']:
            bk = f"{task['task_id']}::{branch['branch_id']}"; events = sorted(branch['user_events'], key=lambda e: e['time_s'])
            T0 = float(np.floor(s['start_s'])); T1 = float(np.ceil(max(s['end_s'], max(e['time_s'] for e in events) + 4)))
            lang = task.get('language') or ('zh' if any('\u4e00' <= ch <= '\u9fff' for e in events for ch in e['text']) else 'en'); voice = VOICES[lang]
            mix, spans = build_mix(media, T0, T1, events, voice, task['task_id']); sf.write(str(OUT / 'mix' / f'{bk.replace("::", "__")}.wav'), mix, SR)
            frames = extract_frames(media, src, T0, T1)
            span_of = {x['event_id']: x for x in spans}
            # 每个事件：用户语音结束所在的秒索引（相对 T0）→ 在该秒喂完后 generate
            gen_at = {}
            for e in events:
                k = int(np.ceil(span_of[e['event_id']]['tts_end'] - T0)) - 1; gen_at.setdefault(k, []).append(e)
            sid = bk; model.reset_session()
            model.streaming_prefill(session_id=sid, msgs=[{'role': 'system', 'content': [SYSTEM]}], omni_mode=True, use_tts_template=USE_TTS_TEMPLATE)
            timeline = []; t_branch = time.time()
            for k in range(int(T1 - T0)):
                t_abs = T0 + k
                content = []
                if frames[k].exists(): content.append(Image.open(frames[k]).convert('RGB'))
                if USER_MARK and any(x['tts_start'] <= t_abs + 1 and x['tts_end'] > t_abs for x in spans): content.append('【用户正在对你说话】')
                chunk = mix[k * SR:(k + 1) * SR]
                if len(chunk) < SR: chunk = np.concatenate([chunk, np.zeros(SR - len(chunk), dtype=np.float32)])
                last = k in gen_at or (TICK_GENERATE)
                for c in content:
                    model.streaming_prefill(session_id=sid, msgs=[{'role': 'user', 'content': [c]}], omni_mode=True, use_tts_template=USE_TTS_TEMPLATE, max_slice_nums=1)
                model.streaming_prefill(session_id=sid, msgs=[{'role': 'user', 'content': [chunk]}], omni_mode=True, use_tts_template=USE_TTS_TEMPLATE, is_last_chunk=last)
                if last:
                    text = ''
                    for piece in model.streaming_generate(session_id=sid, generate_audio=False, use_tts_template=USE_TTS_TEMPLATE, enable_thinking=False, do_sample=DO_SAMPLE, max_new_tokens=200):
                        text += piece[0] if isinstance(piece, (tuple, list)) else str(piece)
                    text = text.strip()
                    if ASSIST_PREFILL:
                        model.streaming_prefill(session_id=sid, msgs=[{'role': 'assistant', 'content': [text or '<SILENCE>']}], omni_mode=True, use_tts_template=USE_TTS_TEMPLATE)
                    if k in gen_at:
                        for e in gen_at[k]:
                            sp = span_of[e['event_id']]
                            timeline.append({'type': 'user_event', 'time_s': e['time_s'], 'response_time_s': round(t_abs + 1, 3), 'tts_duration_s': sp['tts_duration_s'],
                                             'frame_path': str(frames[k]), 'input_mode': 'audio_user_speech', 'event_id': e['event_id'], 'utterance': e['text'], 'response': text})
                        log(f'{bk} @{t_abs + 1:.0f}s user[{" / ".join(e["text"] for e in gen_at[k])}] -> {text[:120]!r}')
                    else:
                        timeline.append({'type': 'observation_tick', 'time_s': round(t_abs + 1, 3), 'frame_path': str(frames[k]), 'input_mode': 'no_new_user_input', 'utterance': '【当前没有新的用户话语】', 'response': text})
            slots = []
            for slot in branch.get('slots', []):
                hi = float(slot['window']['end_s']); slots.append({'slot': slot, 'timeline_prefix': [x for x in timeline if float(x['time_s']) <= hi + 1e-6]})
            results.append({'branch_key': bk, 'task_id': task['task_id'], 'source_id': src, 'capability': task['capability'], 'branch_id': branch['branch_id'], 'session': s,
                            'user_events': branch['user_events'], 'slots': slots, 'timeline': timeline,
                            'omni': {'mix_wav': str(OUT / 'mix' / f'{bk.replace("::", "__")}.wav'), 'tts_voice': voice, 'tts_backend': TTS_BACKEND, 'language': lang, 'duck': DUCK, 'tts_spans': spans, 'chunk_s': 1, 'fps': 1,
                                     'tick_generate': TICK_GENERATE, 'user_mark': USER_MARK, 'use_tts_template': USE_TTS_TEMPLATE, 'do_sample': DO_SAMPLE, 'assist_prefill': ASSIST_PREFILL, 'window_s': [T0, T1], 'wall_s': round(time.time() - t_branch, 1)},
                            'causal_audit': {'independent_branch': True, 'future_frames_sent': False, 'future_events_sent': False, 'audio_input': True, 'text_prompt_for_user_events': False}})
            out = {'schema_version': 'museumlive_minicpm_o45_omni_streaming_v1', 'model': 'MiniCPM-o-4_5', 'batch': 'omni_eval_20261016', 'results': results}
            (OUT / 'eval/evaluation_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
            log(f'branch done {bk} ({len(timeline)} entries, {time.time() - t_branch:.0f}s)')
    log('eval complete')


if __name__ == '__main__':
    main()
