#!/usr/bin/env python3
"""
Omni eval 全双工版：MiniCPM-o 4.5 原生 duplex（model.as_duplex()）—— 每秒喂 VISUAL_FPS 帧 + 1 秒混合音轨，
每秒由模型自己决定 listen / speak（官方 1Hz 主动交互机制），文字输出（generate_audio=False）。

用于画面触发（visual_event）slot：半双工 eval_omni.py 只在用户说完话时生成，模型没机会主动开口，那些 slot 不可评。
半双工下打开 TICK_GENERATE=1（每秒强制生成）试过：模型每秒复读同一句、从不沉默，不可用。

混音、取帧、TTS 复用 eval_omni.py（同一套 Qwen TTS 语音、DUCK、帧率）。
输出与 eval_omni 同 schema，另加：
  - duplex_turns：模型每段发言 {start_s, end_s, text}（start_s = 开口那一秒的末尾，绝对秒）；
  - timeline 里 user_event 的 response = 用户开口后、下一句用户话语之前、语音结束后 REPLY_WAIT 秒内开始的第一段发言（没有则为空 = 沉默）；
  - omni.duplex = true。judge_points.py 对 visual_event slot 取与 [window.start_s, window.end_s + SLOT_SLACK] 有交叠的发言。
用法：EVAL_TASKS=<tasks.json> EVAL_OUT=<dir> MODEL_DEVICE=4 TTS_BACKEND=qwen python3 eval_duplex.py
可选：DUPLEX_PROMPT=default|museum（系统提示：官方默认 "Streaming Omni Conversation." 或带博物馆助手说明）、REPLY_WAIT(6)、DECODE(greedy|sampling)
10-18：EXP_FORCE=1 强制作答：用户提问（user_query slot 的触发话语）说完的那一秒，若模型自 tts_start 起还没开过口，
      屏蔽 <|listen|>（logit=-inf）逼它开口；之后照常（用于区分"不会"和"不说"）。EXP_ABLATE 见 eval_omni.py。
10-18：DUPLEX_MODEL=minicpm|venus（Realtime-Venus-Omni 与 MiniCPM-o 4.5 同一套 as_duplex 接口，协议/混音/取帧完全相同）
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_omni as eo  # noqa: E402  （导入即按 EVAL_OUT 建目录、加载 Qwen TTS 索引）
import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

OUT = eo.OUT; SR = eo.SR; log = eo.log
PROMPT_MODE = os.environ.get('DUPLEX_PROMPT', 'museum')
REPLY_WAIT = float(os.environ.get('REPLY_WAIT', '6'))
DECODE = os.environ.get('DECODE', 'sampling')  # 官方 duplex 用 sampling；每个分支按 branch_key 固定种子
EXTRA_MAX = 10
DUPLEX_MODEL = os.environ.get('DUPLEX_MODEL', 'minicpm')
MODELS = {'minicpm': (eo.MODEL_PATH, 'MiniCPM-o-4_5'),
          'venus': (os.environ['VENUS_MODEL_PATH'], 'Realtime-Venus-Omni')}
MODEL_PATH, MODEL_NAME = MODELS[DUPLEX_MODEL]
MUSEUM_PROMPT = ('Streaming Omni Conversation. 你是戴在用户身上的实时博物馆助手，同时看到用户看到的画面、听到现场全部声音（讲解员、广播、人群和用户本人）。'
                 '用户对你说话时要回答；用户让你留意某样东西时，等它真的出现在画面或声音里再主动提醒；用户取消或修改要求时以最新要求为准；'
                 '其余时间保持安静。回答简洁，用户说什么语言就用什么语言。')

EXP_FORCE = os.environ.get('EXP_FORCE', '0') == '1'
FORCE = {'on': False, 'n': 0}


def install_force(duplex):
    """包住 duplex.decoder.decode：FORCE['on'] 时把 <|listen|> 的 logit 置 -inf（MiniCPM / Venus / Gander 同一套 decoder）。"""
    dec, lid = duplex.decoder, duplex.listen_token_id
    orig = dec.decode

    def decode(*a, **kw):
        if FORCE['on']:
            lg = kw['logits'].clone(); lg[..., lid] = float('-inf'); kw['logits'] = lg
        return orig(*a, **kw)
    dec.decode = decode


def force_points(branch, span_of, T0):
    """{chunk 序号 k: tts_start}：user_query slot 的触发话语结束所在的那一秒（chunk k 覆盖 [T0+k, T0+k+1)）。"""
    out = {}
    for sl in branch.get('slots', []):
        if sl['trigger']['kind'] == 'user_query' and sl['trigger'].get('event_id') in span_of:
            sp = span_of[sl['trigger']['event_id']]
            out[max(0, int(np.ceil(sp['tts_end'] - T0)) - 1)] = sp['tts_start']
    return out


def set_force(k, fpts, chunks):
    """第 k 秒是否强制开口：是强制点，且 tts_start 之后模型还没说过话。"""
    FORCE['on'] = EXP_FORCE and k in fpts and not any(not l and t > fpts[k] for t, l, _, _ in chunks)
    FORCE['n'] += FORCE['on']
    return FORCE['on']


def turns_of(chunks):
    """chunks: [(t_end_abs, is_listen, text, end_of_turn)] → 发言段列表。"""
    turns, cur = [], None
    for t, listen, text, eot in chunks:
        if not listen:
            if cur is None: cur = {'start_s': t, 'end_s': t, 'text': ''}
            cur['text'] += text; cur['end_s'] = t
            if eot: turns.append(cur); cur = None
        elif cur is not None:
            turns.append(cur); cur = None
    if cur is not None: turns.append(cur)
    for x in turns: x['text'] = x['text'].strip()
    return [x for x in turns if x['text']]


def main():
    import torch
    from transformers import AutoModel
    eo.patch_cache()
    tasks = json.loads(Path(os.environ['EVAL_TASKS']).read_text())
    tid = os.environ.get('EVAL_TASK_ID'); tasks = [t for t in tasks if not tid or t['task_id'] == tid]
    assert tasks, 'no task'
    device = int(os.environ.get('MODEL_DEVICE', '0'))
    base = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, attn_implementation='sdpa', torch_dtype=torch.bfloat16,
                                     init_vision=True, init_audio=True, init_tts=True)  # as_duplex 无论是否出音频都要 tts 模块
    base.eval().cuda(device)
    base.init_tts = lambda *a, **k: None  # 只出文字：跳过 Token2wav 声码器（需 minicpmo-utils，本机未装）
    model = base.as_duplex(device=f'cuda:{device}', generate_audio=False)
    if EXP_FORCE: install_force(model)
    log(f'{MODEL_NAME} duplex loaded on cuda:{device} prompt={PROMPT_MODE} decode={DECODE}')
    prompt = MUSEUM_PROMPT if PROMPT_MODE == 'museum' else 'Streaming Omni Conversation.'
    rf = OUT / 'eval/evaluation_results.json'
    results = json.loads(rf.read_text())['results'] if rf.exists() else []  # 续跑：已完成的分支跳过
    done = {r['branch_key'] for r in results}
    for task in tasks:
        src = task['source_id']; media = eo.media_of(src); s = task['session']
        for branch in task['branches']:
            bk = f"{task['task_id']}::{branch['branch_id']}"
            if bk in done: continue
            events = sorted(branch['user_events'], key=lambda e: e['time_s'])
            T0 = float(np.floor(s['start_s'])); T1 = float(np.ceil(max(s['end_s'], max(e['time_s'] for e in events) + 4)))
            lang = task.get('language') or ('zh' if any('一' <= ch <= '鿿' for e in events for ch in e['text']) else 'en'); voice = eo.VOICES[lang]
            _, spans = eo.build_mix(media, T0, T1, events, voice, task['task_id'])  # 先取用户语音时长，定下 T1
            span_of = {x['event_id']: x for x in spans}
            # 多喂几秒：最后一句用户话后留 REPLY_WAIT 秒回应、visual_event 窗口后留 3 秒；到点时模型还在说就续到说完（最多 +EXTRA_MAX 秒）
            T1 = max(T1, float(np.ceil(max(x['tts_end'] for x in spans) + REPLY_WAIT + 1)),
                     float(np.ceil(max([float(sl['window']['end_s']) + 3 for sl in branch.get('slots', [])] + [T1]))))
            T_MAX = T1 + EXTRA_MAX
            mix, spans = eo.build_mix(media, T0, T_MAX, events, voice, task['task_id']); sf.write(str(OUT / 'mix' / f'{bk.replace("::", "__")}.wav'), mix, SR)
            frame_windows = eo.extract_frame_windows_25fps(src, T0, T_MAX)
            torch.manual_seed(int(hashlib.md5(bk.encode()).hexdigest()[:8], 16))
            model.prepare(prefix_system_prompt=prompt)
            chunks = []; t_branch = time.time(); fpts = force_points(branch, span_of, T0); forced = []
            for k in range(int(T_MAX - T0)):
                if T0 + k >= T1 and (not chunks or chunks[-1][1] or chunks[-1][3]): break
                chunk = mix[k * SR:(k + 1) * SR]
                if len(chunk) < SR: chunk = np.concatenate([chunk, np.zeros(SR - len(chunk), dtype=np.float32)])
                fl = eo.load_rgb_frames(frame_windows[k])
                model.streaming_prefill(audio_waveform=chunk, frame_list=fl, max_slice_nums=1)
                if set_force(k, fpts, chunks): forced.append(T0 + k + 1)
                r = model.streaming_generate(decode_mode=DECODE, max_new_speak_tokens_per_chunk=20)
                chunks.append((T0 + k + 1, bool(r['is_listen']), r.get('text') or '', bool(r.get('end_of_turn'))))
            turns = turns_of(chunks)
            timeline = []
            for i, e in enumerate(events):
                sp = span_of[e['event_id']]; nxt = span_of[events[i + 1]['event_id']]['tts_start'] if i + 1 < len(events) else 1e9
                hit = [x for x in turns if sp['tts_start'] <= x['start_s'] <= sp['tts_end'] + REPLY_WAIT and x['start_s'] < nxt + 1]
                resp = hit[0]['text'] if hit else ''
                timeline.append({'type': 'user_event', 'time_s': e['time_s'], 'response_time_s': hit[0]['start_s'] if hit else None, 'tts_duration_s': sp['tts_duration_s'],
                                 'input_mode': 'audio_user_speech', 'event_id': e['event_id'], 'utterance': e['text'], 'response': resp})
                log(f'{bk} user@{e["time_s"]:.0f}s [{e["text"][:40]}] -> {resp[:100]!r}')
            for x in turns:
                timeline.append({'type': 'duplex_turn', 'time_s': x['start_s'], 'end_s': x['end_s'], 'input_mode': 'duplex', 'utterance': '', 'response': x['text']})
            timeline.sort(key=lambda x: x['time_s'])
            slots = []
            for slot in branch.get('slots', []):
                hi = float(slot['window']['end_s']); slots.append({'slot': slot, 'timeline_prefix': [x for x in timeline if float(x['time_s']) <= hi + 1e-6]})
            results.append({'branch_key': bk, 'task_id': task['task_id'], 'source_id': src, 'capability': task['capability'], 'branch_id': branch['branch_id'], 'session': s,
                            'user_events': branch['user_events'], 'slots': slots, 'timeline': timeline, 'duplex_turns': turns,
                            'duplex_chunks': [{'t': t, 'listen': l, 'text': x, 'eot': eot} for t, l, x, eot in chunks],
                            'omni': {'mix_wav': str(OUT / 'mix' / f'{bk.replace("::", "__")}.wav'), 'tts_voice': voice, 'tts_backend': eo.TTS_BACKEND, 'language': lang, 'duck': eo.DUCK, 'tts_spans': spans,
                                     'chunk_s': 1, 'visual_fps': eo.VISUAL_FPS, 'frames_per_chunk': eo.VISUAL_FPS, 'decision_hz': 1,
                                     'duplex': True, 'tick_generate': True, 'duplex_prompt': PROMPT_MODE, 'decode': DECODE, 'reply_wait_s': REPLY_WAIT, 'exp_force': EXP_FORCE, 'forced_at': forced, 'exp_ablate': eo.EXP_ABLATE,
                                     'window_s': [T0, chunks[-1][0]], 'wall_s': round(time.time() - t_branch, 1)},
                            'causal_audit': {'independent_branch': True, 'future_frames_sent': False, 'future_events_sent': False, 'audio_input': True, 'text_prompt_for_user_events': False}})
            out = {'schema_version': 'museumlive_minicpm_o45_omni_duplex_v1', 'model': MODEL_NAME, 'batch': 'omni_eval_20261016', 'results': results}
            (OUT / 'eval/evaluation_results.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
            log(f'branch done {bk} ({len(turns)} turns, {time.time() - t_branch:.0f}s)')
    log('eval complete')


if __name__ == '__main__':
    main()
