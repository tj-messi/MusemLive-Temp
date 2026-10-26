"""三条出题线共用的底层：模型调用、取帧、逐秒摘要/标注读取、泄露检查用的 2-gram，以及两批数据入口的切换。

来源：work/historical_reference_refresh_20261010/refresh_hr.py（call/frame/img_part/load_windows/annotation/media_paths/GENERIC）、
      check_hr.py（bigrams/APPEARANCE）、spacing_refresh_20261011/refresh_spacing.py（LEAK_SYSTEM_FACTS）、
      museumlive_316_qa_20261011/ctx316.py（316 目录版数据入口）。
用法：import common as R; R.use_batch('50' 或 '316')，之后 R.load_windows()/R.annotation()/R.frame()/R.media_paths() 自动对应该批次。
"""
import base64
import json
import os
import re
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as CFG  # noqa: E402

MODEL = CFG.MODEL
KEY = os.environ.get('OPENROUTER_API_KEY', '')
BATCH = None
ARCHIVE = None      # 50：标注 zip；316：空 zip（保持 with zipfile.ZipFile(R.ARCHIVE) 的调用方式不变）
FRAMES = None       # 50：ffmpeg 抽帧缓存目录；316：直接读源帧，不用缓存
OUT = None          # gen_hr 的输出目录（由 run_gen 设定）
_cfg = {}

_lock = threading.Lock()
_spent = {'usd': 0.0, 'calls': 0}

PUNCT = r'[\s，。、‘’“”？！,.\'"（）()：:；;…\-_/]'
APPEARANCE = re.compile(r'长什么样|样子|外观|模样|颜色|材质|描述|什么样|形状|图案|刻着|写着|写的|装饰')
# 不算泄露的泛用 2-gram
GENERIC = {'一个', '一张', '一面', '一只', '一尊', '上面', '上方', '下方', '左侧', '右侧', '有一', '形状', '颜色', '样子', '那个', '这个', '正中', '中间', '画面', '装饰', '模型', '东西', '展品', '作品', '物件', '左边', '右边', '后面', '前面', '旁边', '展柜', '展台', '墙上'}
POSITION = re.compile(r'^(位于|在.*(上方|下方|旁边|左边|右边|左侧|右侧|后面|前面)|挂在|放在|摆在)')

# 台词泄露检查（gen_v3 用）：只有提前说出评分时刻才成立的答案才算泄露
LEAK_SYSTEM_FACTS = """判断用户台词有没有提前透露评分时刻的答案。给你若干句用户台词（带时间）和"评分时刻答案要点"。
只有当某句台词**提前说出了评分时刻才成立的答案内容**（或其同义说法），才算泄露。
台词描述的是更早时刻的旧状态、且与答案不同（例如先说"有三只"，答案是"现在有七只"），不算泄露；这正是状态更新题需要的对照。
用户用位置/指示词指物体、或给出指令/纠正本身（纠正内容就是考点时，答案复述纠正后的信息不算泄露），不算泄露。
只输出 JSON：{"leaks": [{"line": "...", "phrase": "...", "matches_attribute": "..."}]}，没有泄露就输出 {"leaks": []}。"""


def bigrams(text, drop=()):
    s = re.sub(PUNCT, '', text or '')
    for d in drop:
        s = s.replace(re.sub(PUNCT, '', d), '')
    return {s[i:i + 2] for i in range(len(s) - 1)}


# ---------------------------------------------------------------- 模型调用
def call(messages, name, log_dir, budget, max_tokens=4000):
    """OpenRouter 调用，要求 JSON 对象输出；每次响应原文存到 log_dir/<name>.response.txt；按 usage.cost 累计花费，超过 budget 抛错。"""
    if not KEY:
        raise RuntimeError('OPENROUTER_API_KEY is required')
    with _lock:
        if _spent['usd'] >= budget:
            raise RuntimeError('budget reached')
    body = {'model': MODEL, 'messages': messages, 'max_tokens': max_tokens,
            'reasoning': {'effort': 'none'}, 'temperature': 0.3, 'response_format': {'type': 'json_object'}}
    last = None
    for wait in (0, 15, 30):
        time.sleep(wait)
        try:
            r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                              headers={'Authorization': 'Bearer ' + KEY}, json=body, timeout=(30, 240))
        except requests.RequestException as e:
            last = repr(e)
            continue
        text = r.text.replace(KEY, '[REDACTED]')
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / f'{name}.response.txt').write_text(text, encoding='utf-8')
        if r.status_code == 429 or r.status_code >= 500:
            last = f'HTTP {r.status_code}'
            continue
        if r.status_code != 200:  # 400 data_inspection_failed（内容审查）等：不重试，如实记 error
            raise RuntimeError(f'HTTP {r.status_code}: {text[:300]}')
        raw = r.json()
        cost = (raw.get('usage') or {}).get('cost') or 0
        with _lock:
            _spent['usd'] += cost
            _spent['calls'] += 1
        content = raw['choices'][0]['message'].get('content') or ''
        m = re.search(r'\{.*\}', content, re.S)
        try:
            out = json.loads(m.group(0)) if m else json.loads(content)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                out = out[0]
            if not isinstance(out, dict):
                raise json.JSONDecodeError('not an object', content, 0)
            return out
        except json.JSONDecodeError:
            last = 'invalid JSON in model output'
            continue
    raise RuntimeError(f'provider failed: {last}')


def img_part(p):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + base64.b64encode(Path(p).read_bytes()).decode()}}


# ---------------------------------------------------------------- 50 视频入口（zip + mp4）
def _load_windows_50():
    w = {}
    with open(_cfg['windows']) as f:
        for line in f:
            r = json.loads(line)
            w.setdefault(r['source_id'], {})[int(r['start_s'])] = r
    return w


def _annotation_50(zf, src, sec):
    name = f'basic_understanding_1s25/annotations/{src}/{src}_S{sec:06d}.json'
    try:
        a = json.loads(zf.read(name))
    except KeyError:
        return None
    return {'window_id': a['window_id'], 'summary': a['summary'],
            'events': [{k: e.get(k) for k in ('description', 'objects', 'ocr', 'uncertainty')} for e in a['events']]}


def _media_paths_50():
    return {x['source_id']: (x['media_path'], x['duration_s'])
            for x in json.load(open(_cfg['media_preflight']))['sources']}


def _frame_50(src, media, t):
    FRAMES.mkdir(parents=True, exist_ok=True)
    p = FRAMES / f'{src}_{t:.1f}.jpg'
    if not p.exists():
        subprocess.run(['/usr/bin/ffmpeg', '-loglevel', 'error', '-y', '-ss', f'{max(t, 0):.2f}', '-i', media,
                        '-frames:v', '1', '-vf', 'scale=768:-2', '-q:v', '4', str(p)], check=True, timeout=120)
    return p


# ---------------------------------------------------------------- 316 视频入口（目录标注 + 25fps 帧）
def _read_316(src, sec):
    p = _cfg['src'] / 'annotations' / src / f'{src}_S{sec:06d}.json'
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _load_windows_316():
    cache = CFG.RUNS / '316/windows_316.jsonl'
    if not cache.exists():  # 首次从标注目录生成逐秒摘要缓存（316 视频约 21 万秒，几分钟）
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open('w') as f:
            for d in sorted((_cfg['src'] / 'annotations').iterdir()):
                for p in sorted(d.glob('*.json')):
                    a = json.loads(p.read_text())
                    f.write(json.dumps({'source_id': d.name, 'window_id': a['window_id'], 'start_s': a['start_s'],
                                        'end_s': a['end_s'], 'summary': a['summary']}, ensure_ascii=False) + '\n')
    w = {}
    with cache.open() as f:
        for line in f:
            r = json.loads(line)
            w.setdefault(r['source_id'], {})[int(r['start_s'])] = r
    return w


def _annotation_316(zf, src, sec):  # zf 忽略，保持签名一致
    a = _read_316(src, sec)
    if a is None:
        return None
    return {'window_id': a['window_id'], 'summary': a['summary'],
            'events': [{k: e.get(k) for k in ('description', 'objects', 'ocr', 'uncertainty')} for e in a['events']]}


def _media_paths_316():
    """生成阶段不需要 mp4；media_path 取 manifest 里的路径（eval 用），时长按帧数算。"""
    manifest = {m['source_id']: m['media_path'] for m in json.loads(_cfg['media_manifest'].read_text())}
    out = {}
    for d in sorted((_cfg['src'] / 'annotations').iterdir()):
        n = len(list((_cfg['src'] / 'frames' / d.name).glob('F*.jpg')))
        out[d.name] = (manifest.get(d.name, ''), n / 25)
    return out


def _frame_316(src, media, t):
    """t 秒对应 F{round(t*25)+1:06d}.jpg（越界取最后一帧）。"""
    d = _cfg['src'] / 'frames' / src
    idx = max(1, int(round(float(t) * 25)) + 1)
    p = d / f'F{idx:06d}.jpg'
    if not p.exists():
        frames = sorted(d.glob('F*.jpg'))
        p = frames[min(idx, len(frames)) - 1]
    return p


# ---------------------------------------------------------------- 带音频的目录入口（50a：annotations_with_audio/<src>/<src>_S000123.json，有 audio 字段）
SELF_SPK = ('主播', '博主', 'UP主', 'up主', '拍摄者')


def spk(x):
    """转写说话人标签 → 给出题模型看的称呼：主播/博主 = 戴摄像头的用户本人。"""
    x = (x or '人声').strip()
    return '您（用户本人）' if x in SELF_SPK else x


def _audio_text(a):
    """一秒窗口的 audio 字段 → 一行紧凑文本（给逐秒摘要行用）；没有人声/事件返回空串。"""
    au = (a or {}).get('audio') or {}
    parts = [f"{spk(s.get('speaker'))}：{s.get('text', '')}" for s in au.get('speech', []) if s.get('text')]
    ev = [e.get('type') for e in au.get('events', []) if e.get('type') and e.get('type') not in ('安静',)]
    if ev:
        parts.append('[' + '/'.join(dict.fromkeys(ev)) + ']')
    if au.get('status') == 'gap':
        parts.append('[无音轨]')
    return ' ｜ '.join(parts)


def _read_dir(src, sec):
    p = _cfg['src'] / src / f'{src}_S{sec:06d}.json'
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def _load_windows_dir_audio():
    cache = CFG.RUNS / BATCH / f'windows_{BATCH}.jsonl'
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open('w') as f:
            for d in sorted(_cfg['src'].iterdir()):
                if not d.is_dir():
                    continue
                for p in sorted(d.glob('*.json')):
                    a = json.loads(p.read_text())
                    au = a.get('audio') or {}
                    f.write(json.dumps({'source_id': d.name, 'window_id': a['window_id'], 'start_s': a['start_s'], 'end_s': a['end_s'],
                                        'summary': a['summary'], 'audio': _audio_text(a), 'speech': bool(au.get('speech')),
                                        'speech_segs': [{k: x.get(k) for k in ('start', 'end', 'speaker', 'text', 'lang')} for x in au.get('speech', [])],
                                        'sound_events': [{k: x.get(k) for k in ('start', 'end', 'type', 'description')} for x in au.get('events', [])]},
                                       ensure_ascii=False) + '\n')
    w = {}
    with cache.open() as f:
        for line in f:
            r = json.loads(line)
            w.setdefault(r['source_id'], {})[int(r['start_s'])] = r
    return w


def _annotation_dir_audio(zf, src, sec):  # zf 忽略
    a = _read_dir(src, sec)
    if a is None:
        return None
    au = a.get('audio') or {}
    return {'window_id': a['window_id'], 'summary': a['summary'],
            'events': [{k: e.get(k) for k in ('description', 'objects', 'ocr', 'uncertainty')} for e in a['events']],
            'audio': {'speech': [{k: x.get(k) for k in ('start', 'end', 'speaker', 'text')} for x in au.get('speech', [])],
                      'sound_events': [{k: x.get(k) for k in ('start', 'end', 'type', 'description')} for x in au.get('events', [])],
                      'status': au.get('status', 'missing')}}


def _media_paths_dir_audio():
    out = {}
    for x in json.load(open(_cfg['media_preflight']))['sources']:
        sid = x['source_id']
        for ext in ('.mp4', '.mkv'):
            p = _cfg['media_dir'] / (sid + ext)
            if p.exists():
                out[sid] = (str(p), x['duration_s'])
                break
    return out


def _media_paths_dir_audio_316():
    manifest = {m['source_id']: m['media_path'] for m in json.loads(_cfg['media_manifest'].read_text())}
    out = {}
    for d in sorted(_cfg['src'].iterdir()):
        if d.is_dir():
            n = len(list((_cfg['frames'] / d.name).glob('F*.jpg')))
            out[d.name] = (manifest.get(d.name, ''), n / 25)
    return out


def _frame_dir_316(src, media, t):
    d = _cfg['frames'] / src
    idx = max(1, int(round(float(t) * 25)) + 1)
    p = d / f'F{idx:06d}.jpg'
    if not p.exists():
        frames = sorted(d.glob('F*.jpg'))
        p = frames[min(idx, len(frames)) - 1]
    return p


def transcript(w, lo, hi):
    """[lo, hi) 秒内的音频转写块：句子按 (start, text) 去重、按时间排序；附声音事件。w 是某视频的逐秒窗口 dict。"""
    segs, evs = {}, {}
    for s in range(int(lo), int(hi) + 1):
        r = w.get(s)
        if not r:
            continue
        for x in r.get('speech_segs', []):
            segs[(x.get('start'), x.get('text'))] = x
        for x in r.get('sound_events', []):
            evs[(x.get('start'), x.get('type'))] = x
    lines = [f"[{x['start']:.1f}–{x['end']:.1f}s] {spk(x.get('speaker'))}：{x['text']}" for x in sorted(segs.values(), key=lambda x: (x['start'] or 0))
             if x.get('start') is not None]
    elines = [f"[{x['start']:.0f}–{x['end']:.0f}s 声音] {x['type']}：{x.get('description', '')}" for x in sorted(evs.values(), key=lambda x: (x['start'] or 0))
              if x.get('start') is not None]
    return '\n'.join(lines + elines) if (lines or elines) else '（这段没有可辨识的人声或声音事件）'


def speech_at(w, t):
    """t 秒是否有人在说话：返回覆盖 t 的转写句子（dict）或 None。"""
    for s in range(int(t) - 1, int(t) + 2):
        for x in w.get(s, {}).get('speech_segs', []):
            if x.get('start') is not None and x.get('end') is not None and x['start'] - 0.2 <= t <= x['end'] + 0.2:
                return x
    return None


def nearest_gaps(w, t, radius=6.0, min_gap=1.0):
    """t±radius 内没有人说话的时刻（0.5s 网格，且前后各 min_gap/2 内都无人声），按离 t 的距离排序。"""
    out = []
    k = int(radius * 2)
    for i in range(-k, k + 1):
        tt = round(t + i * 0.5, 1)
        if tt < 0:
            continue
        if speech_at(w, tt) is None and speech_at(w, tt - min_gap / 2) is None and speech_at(w, tt + min_gap / 2) is None:
            out.append(tt)
    return sorted(out, key=lambda x: abs(x - t))


def speech_clash(w, times, radius=6.0):
    """用户开口时刻和转写里的人声重叠时返回说明（并给出附近的间隙建议）；附近没有间隙则不算冲突（连续讲解的视频允许重叠）。"""
    notes = []
    for name, t in times.items():
        seg = speech_at(w, float(t))
        if seg is None:
            continue
        gaps = nearest_gaps(w, float(t), radius)
        if gaps:
            notes.append(f"{name}={t}s 时有人在说话（{spk(seg.get('speaker'))}：{seg.get('text', '')[:30]}…），用户开口要选在无人说话的间隙，附近可选：{', '.join(f'{g}s' for g in gaps[:4])}")
    return '；'.join(notes)


TIMING_RULE = '用户开口的时刻要选在转写里没有人说话的间隙（相邻两句之间，前后各留 0.5 秒以上），不要和讲解员/主播同时说话；只有当附近 6 秒内完全没有间隙时才允许重叠。'


CJK = re.compile(r'[\u4e00-\u9fff]')


def video_lang(w):
    """某视频的主要语言 → 'zh' 或 'en'：按转写字数统计，中文占多数才是 zh；无人说话/其他语言一律 en（用户 10-16 规则）。"""
    zh = en = other = 0
    seen = set()
    for r in w.values():
        for x in r.get('speech_segs', []):
            k = (x.get('start'), x.get('text'))
            if k in seen: continue
            seen.add(k); t = x.get('text') or ''; l = (x.get('lang') or '').lower()
            if l.startswith('zh') or l in ('中文', '汉语'): zh += len(t)
            elif l.startswith('en') or l in ('英文', '英语'): en += len(t)
            else: other += len(t)
    if zh + en + other < 40: return 'en'
    return 'zh' if zh >= max(en, other) else 'en'


def lang_rule(lang):
    return ('【语言】本视频主要语言是英语（或全程无人说话）：用户台词、target_text、key_facts/forbidden_claims 的内容一律用**英文**写（标签 [音频]/[视觉] 保留，引用的转写原句保持原文）。'
            if lang == 'en' else '【语言】本视频主要语言是中文：用户台词、target_text、key_facts/forbidden_claims 一律用中文写（引用的转写原句保持原文）。')


def lang_bad(text, lang):
    """文本语言与视频语言不符时返回说明。"""
    t = re.sub(r'\[(音频|视觉)\]', '', text or '')
    n = len(CJK.findall(t))
    if lang == 'en' and n > 2: return f'“{t[:30]}…”应为英文（视频是英文/无人声场景）'
    if lang == 'zh' and n == 0 and re.search(r'[A-Za-z]{3,}', t): return f'“{t[:30]}…”应为中文（视频是中文场景）'
    return ''


def line(w, s):
    """逐秒摘要行：[s] 画面摘要 ｜ 🔊 音频。"""
    r = w[s]
    au = r.get('audio') or ''
    return f"[{s}s] {r['summary']}" + (f"  🔊 {au}" if au else '')


def informative_speech(w, lo, hi, min_chars=80, min_sentence=12):
    """[lo,hi) 内是否有“有信息量”的人声：去重后的句子总字数 ≥ min_chars 且至少一句 ≥ min_sentence 字（排除纯语气词/闲聊的窗口）。"""
    segs = {}
    for s in range(int(lo), int(hi)):
        for x in w.get(s, {}).get('speech_segs', []):
            segs[(x.get('start'), x.get('text'))] = x.get('text') or ''
    texts = [t.replace('…', '').strip() for t in segs.values()]
    return sum(len(t) for t in texts) >= min_chars and any(len(t) >= min_sentence for t in texts)


def has_speech(w, lo, hi, min_secs=5):
    return sum(1 for s in range(int(lo), int(hi)) if w.get(s, {}).get('speech')) >= min_secs


OMNI_RULE = ('【全模态要求】这是 Omni benchmark：模型同时**听视频音轨**（讲解员/主播/游客说的话、广播、音乐和环境声）和**看画面**。'
             '题目必须把音频内容作为考点的一部分：至少一个评分槽位的 key_facts 里要有**只有听了音频才能知道**的事实'
             '（讲解员/主播/游客说出的名称、朝代、数字、用途、评价、指令，或广播内容、谁在说话、音乐/声音事件），这类要点以“[音频]”开头，并且内容必须和给你的音频转写逐字对得上、时间在该槽位之前；'
             '只看画面就能核对的要点以“[视觉]”开头。用户台词可以引用听到的内容（如“她刚才说的那个朝代”“广播里提的那个展厅”），但不能把答案说出来。'
             '[音频] 要点必须和题目问的对象/任务直接相关（关于它的名称、年代、用途、来历、评价、指令或与它有关的声音事件），不能拿无关的闲聊、语气词或随口一句“好的”凑数。'
             '不要把画面里没有、转写里也没有的内容写进要点。'
             '【第一人称】视频是戴着摄像头的用户第一视角，转写里标为“主播/博主/UP主”的声音就是**用户本人**在说话（转写里已标成“您（用户本人）”）。'
             '用户台词一律第一人称（“我刚才说的那个年代”“帮我记一下我接下来提到的名称”），模型的 target_text/key_facts/forbidden_claims 里提到用户说过的话一律用“您”（“您刚才说是太古代”），绝不能写“主播说/博主说/UP主说”。讲解员、游客、广播等其他人仍用第三人称。'
             '用户台词要**模仿视频主角本人的口吻**来提问：接着他/她自己刚说的话或正在做的事自然地问（例如主角刚念完展签、刚吐槽完人多，就顺着这个语境问模型），语气、用词和主角在转写里的说话风格一致，像是主角一边逛一边随口问助手，而不是旁观者的书面提问。')


def load_windows():
    raise RuntimeError('先调用 common.use_batch()')


annotation = media_paths = frame = load_windows


def use_batch(batch):
    """切换数据入口；返回该批次配置。"""
    global BATCH, ARCHIVE, FRAMES, load_windows, annotation, media_paths, frame
    BATCH = str(batch)
    _cfg.clear(); _cfg.update(CFG.BATCHES[BATCH])
    (CFG.RUNS / BATCH).mkdir(parents=True, exist_ok=True)
    if _cfg['kind'] == 'zip':
        ARCHIVE = _cfg['archive']; FRAMES = CFG.RUNS / BATCH / 'frame_cache'
        load_windows, annotation, media_paths, frame = _load_windows_50, _annotation_50, _media_paths_50, _frame_50
    elif _cfg['kind'] == 'dir_audio_316':
        ARCHIVE = CFG.RUNS / BATCH / 'empty.zip'
        if not ARCHIVE.exists():
            zipfile.ZipFile(ARCHIVE, 'w').close()
        FRAMES = None
        load_windows, annotation, media_paths, frame = _load_windows_dir_audio, _annotation_dir_audio, _media_paths_dir_audio_316, _frame_dir_316
    elif _cfg['kind'] == 'dir_audio':
        ARCHIVE = CFG.RUNS / BATCH / 'empty.zip'
        if not ARCHIVE.exists():
            zipfile.ZipFile(ARCHIVE, 'w').close()
        FRAMES = CFG.RUNS / BATCH / 'frame_cache'
        load_windows, annotation, media_paths, frame = _load_windows_dir_audio, _annotation_dir_audio, _media_paths_dir_audio, _frame_50
    else:
        ARCHIVE = CFG.RUNS / BATCH / 'empty.zip'
        if not ARCHIVE.exists():
            zipfile.ZipFile(ARCHIVE, 'w').close()
        FRAMES = None
        load_windows, annotation, media_paths, frame = _load_windows_316, _annotation_316, _media_paths_316, _frame_316
    return _cfg


def gap_seconds():
    p = _cfg.get('gap_seconds')
    return json.load(open(p)) if p and Path(p).exists() else {}
