#!/usr/bin/env python3
"""
Gander（混元，MiniCPM-o 4.5 微调）全双工 eval（10-18）。
用官方 mcpmft 推理核心（third_party/Omni-Interaction-Agent/minicpm_ft，GitHub API tarball 拉下）：
  load_for_infer（MiniCPM-o 4.5 底座 + Gander thinker 叠加 + 专用 token）→ OnlineRunner（as_duplex + Gander 控制 token / 工具语法）。
每秒照官方 DuplexLiveSession._step：runner.duplex.streaming_prefill(audio_waveform=1 s 混音, frame_list=[1 帧], max_slice_nums=1)
→ runner.streaming_generate(...)；全双工参数取官方 configs/infer.yaml（= serve 配置：每单元 8 字 / 最多 11 token、sampling top_k 20、context_slate）。
系统提示 = 官方 GANDER_DUPLEX_SYSTEM_PROMPT（不给工具）+ 与 MiniCPM/Venus 相同的博物馆助手说明。
混音 / 取帧 / 时间轴 / 输出 schema 全部与 eval_duplex.py 相同（judge_points.py 直接读）。
用法：CUDA_VISIBLE_DEVICES=<gpu> EVAL_TASKS=<tasks.json> EVAL_OUT=<dir> [EVAL_TASK_ID=..] python3 eval_gander.py
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

E = Path(os.environ.get('MUSEUMLIVE_EVAL_PYLIB', Path(__file__).resolve().parent / 'pylib')).resolve()
G = Path(os.environ.get('GANDER_CODE_ROOT', 'Omni-Interaction-Agent/minicpm_ft')).resolve()
sys.path.insert(0, str(E))
os.environ.setdefault('TTS_BACKEND', 'qwen')
import eval_omni as eo  # noqa: E402  （pylib：transformers 4.51 = Gander 要求的版本）
from eval_duplex import turns_of, REPLY_WAIT, EXTRA_MAX, EXP_FORCE, install_force, force_points, set_force  # noqa: E402
sys.path.insert(1, str(G))
import soundfile as sf  # noqa: E402
import yaml  # noqa: E402
from PIL import Image  # noqa: E402

OUT = eo.OUT; SR = eo.SR; log = eo.log
BASE = os.environ.get('MINICPM_MODEL_PATH', 'MiniCPM-o-4_5')
THINKER = os.environ.get('GANDER_THINKER_PATH', 'Gander/thinker')
MUSEUM = ('\n\n# 当前场景\n\n你是戴在用户身上的实时博物馆助手，同时看到用户看到的画面、听到现场全部声音（讲解员、广播、人群和用户本人）。'
          '用户对你说话时要回答；用户让你留意某样东西时，等它真的出现在画面或声音里再主动提醒；用户取消或修改要求时以最新要求为准；'
          '其余时间保持安静。回答简洁，用户说什么语言就用什么语言。')


def main():
    import torch
    from mcpmft.args import ModelArguments
    from mcpmft.infer import cli
    from mcpmft.infer.common import load_for_infer
    from mcpmft.infer.online import OnlineRunner
    from mcpmft.prompts import GANDER_DUPLEX_SYSTEM_PROMPT
    eo.patch_cache()
    cfg = yaml.safe_load(open(G / 'configs/infer.yaml'))
    margs = cli._typed(ModelArguments, {**cfg['model'], 'model_name_or_path': BASE, 'processor_name_or_path': BASE}, 'model')
    dcfg = cli._typed(cli.DuplexConfig, cfg['inference']['duplex'], 'inference.duplex')
    params = cli._duplex_params(dcfg, False)
    tasks = json.loads(Path(os.environ['EVAL_TASKS']).read_text())
    tid = os.environ.get('EVAL_TASK_ID'); tasks = [t for t in tasks if not tid or t['task_id'] == tid]
    bundle = load_for_infer(margs, checkpoint=THINKER, init_token2wav=False, load_processor=False)
    runner = OnlineRunner(bundle, params)
    if EXP_FORCE: install_force(runner.duplex)  # 10-18 强制作答，见 eval_duplex.py
    prompt = GANDER_DUPLEX_SYSTEM_PROMPT + MUSEUM
    gen_kw = dict(max_new_speak_tokens_per_chunk=params.max_new_speak_tokens_per_chunk, decode_mode=params.decode_mode,
                  temperature=params.temperature, top_p=params.top_p, top_k=params.top_k)
    log(f'Gander loaded (cuda {os.environ.get("CUDA_VISIBLE_DEVICES")}) params={gen_kw} window={params.sliding_window_mode}')
    rf = OUT / 'eval/evaluation_results.json'
    results = json.loads(rf.read_text())['results'] if rf.exists() else []
    done = {r['branch_key'] for r in results}
    snap = None
    for task in tasks:
        src = task['source_id']; media = eo.media_of(src); s = task['session']
        for branch in task['branches']:
            bk = f"{task['task_id']}::{branch['branch_id']}"
            if bk in done: continue
            events = sorted(branch['user_events'], key=lambda e: e['time_s'])
            T0 = float(np.floor(s['start_s'])); T1 = float(np.ceil(max(s['end_s'], max(e['time_s'] for e in events) + 4)))
            lang = task.get('language') or ('zh' if any('一' <= ch <= '鿿' for e in events for ch in e['text']) else 'en'); voice = eo.VOICES[lang]
            _, spans = eo.build_mix(media, T0, T1, events, voice, task['task_id'])
            span_of = {x['event_id']: x for x in spans}
            T1 = max(T1, float(np.ceil(max(x['tts_end'] for x in spans) + REPLY_WAIT + 1)),
                     float(np.ceil(max([float(sl['window']['end_s']) + 3 for sl in branch.get('slots', [])] + [T1]))))
            T_MAX = T1 + EXTRA_MAX
            mix, spans = eo.build_mix(media, T0, T_MAX, events, voice, task['task_id']); sf.write(str(OUT / 'mix' / f'{bk.replace("::", "__")}.wav'), mix, SR)
            frames = eo.extract_frames(media, src, T0, T_MAX)
            torch.manual_seed(int(hashlib.md5(bk.encode()).hexdigest()[:8], 16))
            t_prep = time.time()
            if snap is None:  # 与官方 serve 相同：系统提示只预填一次，之后每个分支从快照恢复（会话隔离）
                runner.prepare(system_prompt=prompt); snap = runner.capture_prefix_snapshot()
            else:
                runner.restore_prefix_snapshot(snap)
            t_prep = time.time() - t_prep
            chunks = []; raw = []; t_branch = time.time(); fpts = force_points(branch, span_of, T0); forced = []
            for k in range(int(T_MAX - T0)):
                if T0 + k >= T1 and (not chunks or chunks[-1][1] or chunks[-1][3]): break
                chunk = mix[k * SR:(k + 1) * SR]
                if len(chunk) < SR: chunk = np.concatenate([chunk, np.zeros(SR - len(chunk), dtype=np.float32)])
                kw = {'audio_waveform': chunk}
                if frames[k].exists(): kw.update(frame_list=[Image.open(frames[k]).convert('RGB')], max_slice_nums=1, batch_vision_feed=False)
                runner.duplex.streaming_prefill(**kw)
                if set_force(k, fpts, chunks): forced.append(T0 + k + 1)
                r = runner.streaming_generate(**gen_kw)
                chunks.append((T0 + k + 1, bool(r.get('is_listen', True)), r.get('text') or '', bool(r.get('end_of_turn'))))
                if r.get('is_interrupt') or r.get('tool_calls'): raw.append({'t': T0 + k + 1, 'interrupt': bool(r.get('is_interrupt')), 'tool_calls': r.get('tool_calls')})
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
            slots = [{'slot': sl, 'timeline_prefix': [x for x in timeline if float(x['time_s']) <= float(sl['window']['end_s']) + 1e-6]} for sl in branch.get('slots', [])]
            results.append({'branch_key': bk, 'task_id': task['task_id'], 'source_id': src, 'capability': task['capability'], 'branch_id': branch['branch_id'], 'session': s,
                            'user_events': branch['user_events'], 'slots': slots, 'timeline': timeline, 'duplex_turns': turns,
                            'duplex_chunks': [{'t': t, 'listen': l, 'text': x, 'eot': eot} for t, l, x, eot in chunks], 'gander_events': raw,
                            'omni': {'mix_wav': str(OUT / 'mix' / f'{bk.replace("::", "__")}.wav'), 'tts_voice': voice, 'tts_backend': eo.TTS_BACKEND, 'language': lang, 'duck': eo.DUCK, 'tts_spans': spans,
                                     'chunk_s': 1, 'fps': 1, 'duplex': True, 'decode': params.decode_mode, 'gen': gen_kw, 'window_mode': params.sliding_window_mode,
                                     'reply_wait_s': REPLY_WAIT, 'exp_force': EXP_FORCE, 'forced_at': forced, 'exp_ablate': eo.EXP_ABLATE, 'window_s': [T0, chunks[-1][0]], 'wall_s': round(time.time() - t_branch, 1)},
                            'causal_audit': {'independent_branch': True, 'future_frames_sent': False, 'future_events_sent': False, 'audio_input': True, 'text_prompt_for_user_events': False}})
            out = {'schema_version': 'museumlive_omni_duplex_v1', 'model': 'Gander', 'batch': 'model_compare_20261018', 'results': results}
            rf.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            log(f'branch done {bk} ({len(turns)} turns, {time.time() - t_branch:.0f}s, prepare {t_prep:.1f}s)')
    log('eval complete')


if __name__ == '__main__':
    main()
