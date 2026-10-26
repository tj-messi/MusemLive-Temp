#!/usr/bin/env python3
"""
MOSS-VL-Realtime 全双工 eval（10-18）。MOSS 没有音频输入，用户选方案 B：
  - 用户台词：在用户语音（同一份 Qwen TTS 时长）结束的那一秒，以文字 push 成一个用户回合，前缀【用户对你说】；
  - 现场原声：音频标注（annotations_audio{,_316}，qwen3.8-omni-flash 转写，绝对时间戳）里每段话在其结束的那一秒以文字 push，
    前缀【现场声音·说话人】；主播/博主/UP主/拍摄者 = 戴摄像头的用户本人，标成"用户本人（不是对你说）"。
  - 画面：每秒 1 帧（与 eval_duplex 同一套取帧），时间戳 = 相对秒。
驱动方式：替换模型的 _real_time_sample，按"模拟时钟"同步解码 —— 每秒先把这一秒的帧/文字放进队列，再逐 token 解码，
直到吐出 <|silence|>（模型自己停下等输入）或这一秒已出 TOKENS_PER_SEC 个 token（与 MiniCPM/Venus 每块 20 token 对齐），
然后推进到下一秒；说话中途到来的新帧照官方逻辑在 token 之间拼进上下文。不用官方的线程 session，结果可复现。
输出与 eval_duplex 同 schema（duplex_turns / duplex_chunks / timeline / slots），judge_points.py 可直接读。
用法：EVAL_TASKS=<tasks.json> EVAL_OUT=<dir> MODEL_DEVICE=4 [EVAL_TASK_ID=..] python3 eval_moss.py
"""
import glob, json, os, queue, sys, time
from pathlib import Path
import numpy as np

E = Path(os.environ.get('MUSEUMLIVE_EVAL_PYLIB', Path(__file__).resolve().parent / 'pylib')).resolve()
sys.path.insert(0, str(E))
os.environ.setdefault('TTS_BACKEND', 'qwen')
import eval_omni as eo  # noqa: E402  （取帧 / TTS 时长 / 媒体路径；它会把 pylib(transformers 4.51) 插到 sys.path 最前）
from eval_duplex import turns_of, REPLY_WAIT, EXTRA_MAX  # noqa: E402
sys.path = [p for p in sys.path if not p.rstrip('/').endswith('pylib')]  # MOSS 要 transformers 4.57（系统版）
assert 'transformers' not in sys.modules, 'transformers 已被 pylib 版本导入'
sys.path.insert(0, str(Path(__file__).resolve().parent / 'pylib_moss'))  # torchvision / torchcodec
from PIL import Image  # noqa: E402

OUT = eo.OUT; log = eo.log
MODEL_PATH = os.environ.get('MOSS_MODEL_PATH', 'MOSS-VL-Realtime')
TOKENS_PER_SEC = int(os.environ.get('TOKENS_PER_SEC', '20'))
ANN = [Path(p) for p in os.environ.get('MOSS_ANNOTATION_ROOTS', '').split(os.pathsep) if p]
SELF = {'主播', '博主', 'UP主', '拍摄者'}
MUSEUM = ('\n\n你是戴在用户身上的实时博物馆助手，看到用户看到的画面。你听不到声音，现场声音会以文字转写的形式按时间实时给你：'
          '以【现场声音·说话人】开头的消息是环境里的话（讲解员、广播、人群、用户本人自言自语或和同伴说话），不是对你说的，不需要回应，但要记住内容；'
          '以【用户对你说】开头的消息才是用户直接对你说的话，要回答。用户让你留意某样东西时，等它真的出现在画面或现场声音里再主动提醒；'
          '用户取消或修改要求时以最新要求为准；其余时间输出 `<|silence|>` 保持安静。回答简洁，用户说什么语言就用什么语言。')


def transcript(src, t0, t1):
    """[(end_s, text)]：src 在 [t0, t1) 内结束的转写句子（绝对秒）。10-18 EXP_ABLATE=no_transcript 时不给。"""
    out = []
    if eo.EXP_ABLATE == 'no_transcript': return out
    for d in ANN:
        for f in sorted(glob.glob(str(d / src / '*.json'))):
            for s in json.load(open(f)).get('speech') or []:
                if t0 <= float(s['end']) < t1 and (s.get('text') or '').strip():
                    who = '用户本人（不是对你说）' if s.get('speaker') in SELF else (s.get('speaker') or '未知')
                    out.append((float(s['end']), f'【现场声音·{who}】{s["text"].strip()}'))
    return sorted(set(out))


class Driver:
    """替换 model._real_time_sample：按 schedule（每秒的帧 + 文字）同步推进，记录每秒输出。"""

    def __init__(self, model, n_sec, n_min, frame_at, prompts_at):
        self.m, self.n_sec, self.n_min, self.frame_at, self.prompts_at = model, n_sec, n_min, frame_at, prompts_at
        self.chunks = []  # [(k, listen, text, eot)]

    def feed(self, k, fq, pq):
        for p in self.prompts_at.get(k, []): pq.put(p)
        img = self.frame_at(k)
        if img is not None: fq.put((img, float(k)))

    def __call__(self, input_ids, new_video_frames, new_prompts, output_text_queue, processor, logits_processor,
                 stopping_criteria, generation_config, streamer=None, max_tokens_per_turn=86400, **model_kwargs):
        import torch
        import torch.nn.functional as F
        m, tok = self.m, processor.tokenizer
        sil = tok.convert_tokens_to_ids('<|silence|>')
        ell = (tok.convert_tokens_to_ids(['<|...|>']) or [-1])[0]; ell = -1 if ell is None else ell
        model_kwargs = m._get_initial_cache_position(input_ids.shape[1], input_ids.device, model_kwargs)
        k, ntok, buf, text, first = 0, 0, [], '', True
        self.feed(0, new_video_frames, new_prompts)
        while True:
            outputs = m(**m.prepare_inputs_for_real_time_generation(input_ids, **model_kwargs), return_dict=True)
            scores = logits_processor(input_ids, outputs.logits[:, -1, :].clone().float().to(input_ids.device))
            if ell >= 0: scores[0, ell] = float('-inf')
            probs = F.softmax(scores, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(1) if generation_config.do_sample else torch.argmax(probs, dim=-1)
            t = int(nxt.item()); is_sil = t == sil
            if not is_sil:
                buf.append(t); d = tok.decode(buf, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if d and d[-1] != '�': text += d; buf = []
                ntok += 1
            input_ids = torch.cat([input_ids, nxt[:, None]], dim=-1)
            if first:  # 预填充后的第一个 token 还没看到第 0 秒的帧：不算一秒，直接让它取第 0 秒输入
                first = False
            elif is_sil or ntok >= TOKENS_PER_SEC:  # 这一秒结束：记录，推进时钟
                self.chunks.append((k, not text.strip(), text, is_sil))
                k += 1; ntok = 0; text = ''
                speaking = not is_sil
                if k >= self.n_sec or (k >= self.n_min and not speaking): break
                self.feed(k, new_video_frames, new_prompts)
            input_ids, model_kwargs = m._update_model_kwargs_for_real_time_generation(
                outputs=outputs, input_ids=input_ids, model_kwargs=model_kwargs, should_wait_for_new_input=is_sil,
                new_video_frames=new_video_frames, new_prompts=new_prompts, output_text_queue=output_text_queue,
                token_buffer=__import__('collections').deque(), processor=processor)
            del outputs
        return input_ids


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    tasks = json.loads(Path(os.environ['EVAL_TASKS']).read_text())
    tid = os.environ.get('EVAL_TASK_ID'); tasks = [t for t in tasks if not tid or t['task_id'] == tid]
    device = int(os.environ.get('MODEL_DEVICE', '0'))
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True, frame_extract_num_threads=1)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
                                                 attn_implementation='sdpa', device_map={'': device}).eval()
    mod = sys.modules[type(model).__module__]
    log(f'MOSS-VL-Realtime loaded on cuda:{device} tokens/sec={TOKENS_PER_SEC}')
    # 官方 real_time_generate 的系统提示（原样）+ 博物馆助手与文字转写说明
    import inspect
    src_txt = inspect.getsource(type(model).real_time_generate)
    official = eval(src_txt[src_txt.index('system_prompt = (') + len('system_prompt = '):src_txt.index('# TODO(realtime-sft)')].strip())
    system_prompt = official + MUSEUM
    ids = processor.tokenizer.apply_chat_template([{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': ''}],
                                                  add_generation_prompt=True, tokenize=True, return_tensors='pt').to(model.device)
    rf = OUT / 'eval/evaluation_results.json'
    results = json.loads(rf.read_text())['results'] if rf.exists() else []
    done = {r['branch_key'] for r in results}
    for task in tasks:
        src = task['source_id']; media = eo.media_of(src); s = task['session']
        for branch in task['branches']:
            bk = f"{task['task_id']}::{branch['branch_id']}"
            if bk in done: continue
            events = sorted(branch['user_events'], key=lambda e: e['time_s'])
            T0 = float(np.floor(s['start_s'])); T1 = float(np.ceil(max(s['end_s'], max(e['time_s'] for e in events) + 4)))
            lang = task.get('language') or ('zh' if any('一' <= ch <= '鿿' for e in events for ch in e['text']) else 'en'); voice = eo.VOICES[lang]
            _, spans = eo.build_mix(media, T0, T1, events, voice, task['task_id'])  # 只用来取用户语音时长（与音频模型同一时刻"说完"）
            span_of = {x['event_id']: x for x in spans}
            T1 = max(T1, float(np.ceil(max(x['tts_end'] for x in spans) + REPLY_WAIT + 1)),
                     float(np.ceil(max([float(sl['window']['end_s']) + 3 for sl in branch.get('slots', [])] + [T1]))))
            T_MAX = T1 + EXTRA_MAX
            frames = eo.extract_frames(media, src, T0, T_MAX)
            prompts_at, fed = {}, []
            for e in events:  # 用户语音结束所在的那一秒
                k = max(0, int(np.ceil(span_of[e['event_id']]['tts_end'] - T0)) - 1)
                prompts_at.setdefault(k, []).append((span_of[e['event_id']]['tts_end'], f'【用户对你说】{e["text"]}'))
            for end, txt in transcript(src, T0, T_MAX):
                k = max(0, int(np.ceil(end - T0)) - 1); prompts_at.setdefault(k, []).append((end, txt))
            prompts_at = {k: [p for _, p in sorted(v)] for k, v in prompts_at.items()}
            for k, v in sorted(prompts_at.items()): fed += [{'t': T0 + k + 1, 'text': p} for p in v]
            drv = Driver(model, int(T_MAX - T0), int(T1 - T0),
                         lambda k: Image.open(frames[k]).convert('RGB') if k < len(frames) and frames[k].exists() else None, prompts_at)
            model._real_time_sample = drv
            torch.manual_seed(0); t_branch = time.time()
            model.start_real_time_generate()
            try:
                model._real_time_generate(new_video_frames=queue.Queue(), new_prompts=queue.Queue(), output_text_queue=queue.Queue(),
                                          processor=processor, inputs=ids, attention_mask=torch.ones_like(ids),
                                          position_ids=torch.arange(ids.shape[1], device=model.device).view(1, 1, -1).expand(3, 1, -1).contiguous(),
                                          realtime_next_position=ids.shape[1], full_vision_token_info=None, do_sample=False, max_new_tokens=100000)
            finally:
                model.stop_real_time_generate()
            chunks = [(T0 + k + 1, listen, text, eot) for k, listen, text, eot in drv.chunks]
            turns = turns_of(chunks)
            timeline = []
            for i, e in enumerate(events):
                sp = span_of[e['event_id']]; nxt = span_of[events[i + 1]['event_id']]['tts_start'] if i + 1 < len(events) else 1e9
                hit = [x for x in turns if sp['tts_start'] <= x['start_s'] <= sp['tts_end'] + REPLY_WAIT and x['start_s'] < nxt + 1]
                resp = hit[0]['text'] if hit else ''
                timeline.append({'type': 'user_event', 'time_s': e['time_s'], 'response_time_s': hit[0]['start_s'] if hit else None, 'tts_duration_s': sp['tts_duration_s'],
                                 'input_mode': 'text_user_prompt', 'event_id': e['event_id'], 'utterance': e['text'], 'response': resp})
                log(f'{bk} user@{e["time_s"]:.0f}s [{e["text"][:40]}] -> {resp[:100]!r}')
            for x in turns:
                timeline.append({'type': 'duplex_turn', 'time_s': x['start_s'], 'end_s': x['end_s'], 'input_mode': 'duplex', 'utterance': '', 'response': x['text']})
            timeline.sort(key=lambda x: x['time_s'])
            slots = [{'slot': sl, 'timeline_prefix': [x for x in timeline if float(x['time_s']) <= float(sl['window']['end_s']) + 1e-6]} for sl in branch.get('slots', [])]
            results.append({'branch_key': bk, 'task_id': task['task_id'], 'source_id': src, 'capability': task['capability'], 'branch_id': branch['branch_id'], 'session': s,
                            'user_events': branch['user_events'], 'slots': slots, 'timeline': timeline, 'duplex_turns': turns,
                            'duplex_chunks': [{'t': t, 'listen': l, 'text': x, 'eot': eot} for t, l, x, eot in chunks], 'text_inputs': fed,
                            'omni': {'tts_spans': spans, 'language': lang, 'chunk_s': 1, 'fps': 1, 'duplex': True, 'audio_input': False, 'asr_text_input': True,
                                     'tokens_per_sec': TOKENS_PER_SEC, 'decode': 'greedy', 'reply_wait_s': REPLY_WAIT, 'window_s': [T0, chunks[-1][0] if chunks else T0],
                                     'wall_s': round(time.time() - t_branch, 1)},
                            'causal_audit': {'independent_branch': True, 'future_frames_sent': False, 'future_events_sent': False, 'audio_input': False,
                                             'text_prompt_for_user_events': True, 'asr_transcript_as_text': True}})
            out = {'schema_version': 'museumlive_omni_duplex_v1', 'model': 'MOSS-VL-Realtime', 'batch': 'model_compare_20261018', 'results': results}
            rf.write_text(json.dumps(out, ensure_ascii=False, indent=1))
            log(f'branch done {bk} ({len(turns)} turns, {len(fed)} text inputs, {time.time() - t_branch:.0f}s)')
    log('eval complete')


if __name__ == '__main__':
    main()
