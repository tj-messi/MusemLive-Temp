#!/usr/bin/env python3
"""
借鉴实验 5–7 的 GPU 调度（10-18）：GPU 4–7（用户授权），按显存余量、按优先级排队起分片，挂了自动续跑，某个 run 分片全齐后自动跑得分点 judge。
  nohup python3 orchestrate.py > orchestrate.log 2>&1 &
  python3 orchestrate.py status
run 目录 runs/<name>/shard<i>/（tasks.json + eval/evaluation_results.json + run.log）。eval 脚本复用 omni_eval_20261016/eval_duplex.py
与 model_compare_20261018/eval_{gander,moss}.py，实验由环境变量切换：EXP_FORCE=1（强制作答）、EXP_ABLATE=mute_ambient|black_video|no_transcript。
盲基线 run 用 tasks_blind.json，judge 前先 blind_tasks.py restore。
"""
import glob, json, os, subprocess, sys, time
from pathlib import Path

D = Path(__file__).resolve().parent
E = D.parent / 'omni_eval_20261016'
MC = D.parent / 'model_compare_20261018'
GPUS = [4, 5, 6, 7]
NEED_GB = {'moss': 27, 'venus': 21, 'gander': 21, 'minicpm': 21}
FINAL = MC / 'tasks_final450.json'
# 按优先级排列：name → (model, env, n_shards, tasks)
RUNS = {
    'force_venus':   ('venus',   {'EXP_FORCE': '1'}, 4, FINAL),
    'force_gander':  ('gander',  {'EXP_FORCE': '1'}, 4, FINAL),
    'force_minicpm': ('minicpm', {'EXP_FORCE': '1'}, 4, FINAL),
    'blind_minicpm': ('minicpm', {}, 2, D / 'tasks_blind.json'),
    'blind_moss':    ('moss',    {}, 2, D / 'tasks_blind.json'),
    'mute_minicpm':  ('minicpm', {'EXP_ABLATE': 'mute_ambient'}, 4, FINAL),
    'black_minicpm': ('minicpm', {'EXP_ABLATE': 'black_video'}, 4, FINAL),
    'notext_moss':   ('moss',    {'EXP_ABLATE': 'no_transcript'}, 4, FINAL),
    'black_moss':    ('moss',    {'EXP_ABLATE': 'black_video'}, 4, FINAL),
}
PY = '/usr/bin/python3'


def log(s):
    print(time.strftime('[%F %T] ') + s, flush=True)


def shard_dirs(name):
    _, _, n, tf = RUNS[name]
    a = json.load(open(tf)); out = []
    for i in range(n):
        d = D / 'runs' / name / f'shard{i}'; d.mkdir(parents=True, exist_ok=True)
        if not (d / 'tasks.json').exists(): json.dump(a[i::n], open(d / 'tasks.json', 'w'), ensure_ascii=False)
        out.append(d)
    return out


def progress(d):
    total = sum(len(t['branches']) for t in json.load(open(d / 'tasks.json')))
    f = d / 'eval/evaluation_results.json'
    try:
        done = len(json.load(open(f))['results']) if f.exists() else 0
    except Exception:
        done = -1  # 正在写
    return done, total


def running_outs():
    """正在跑的 eval 进程 → {EVAL_OUT: (pid, gpu)}（从 /proc/<pid>/environ 读）。"""
    out = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit(): continue
        try:
            cmd = (p / 'cmdline').read_bytes().split(b'\0')
            if not any(x.endswith((b'eval_duplex.py', b'eval_moss.py', b'eval_gander.py')) for x in cmd): continue
            env = dict(x.split('=', 1) for x in (p / 'environ').read_bytes().decode(errors='ignore').split('\0') if '=' in x)
            g = env.get('MODEL_DEVICE') if 'gander' not in str(cmd) else env.get('CUDA_VISIBLE_DEVICES')
            out[str(Path(env.get('EVAL_OUT', '')).resolve())] = (int(p.name), int(g) if g and g.isdigit() else None)
        except Exception:
            pass
    return out


def free_gb():
    q = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True).stdout
    return {int(i): (int(t) - int(u)) / 1024 for i, u, t in (l.split(', ') for l in q.strip().splitlines())}


def launch(name, d, g):
    m, xenv, _, _ = RUNS[name]
    env = dict(os.environ, EVAL_TASKS=str(d / 'tasks.json'), EVAL_OUT=str(d), TTS_BACKEND='qwen', **xenv)
    if m in ('venus', 'minicpm'):
        env.update(MODEL_DEVICE=str(g), DUPLEX_MODEL=m); cmd, cwd = [PY, '-u', str(E / 'eval_duplex.py')], E
    elif m == 'moss':
        env.update(MODEL_DEVICE=str(g)); cmd, cwd = [PY, '-u', str(MC / 'eval_moss.py')], MC
    else:
        env.update(CUDA_VISIBLE_DEVICES=str(g), PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'); cmd, cwd = [PY, '-u', str(MC / 'eval_gander.py')], MC
    subprocess.Popen(cmd, cwd=cwd, env=env, stdout=open(d / 'run.log', 'a'), stderr=subprocess.STDOUT, start_new_session=True)
    log(f'launch {name} {d.name} on GPU {g}')


def start_judge(name):
    out = D / 'runs' / name / 'judge_points'; out.mkdir(exist_ok=True)
    pat = str(D / 'runs' / name / 'shard*/eval/evaluation_results.json')
    if name.startswith('blind_'):
        subprocess.run([PY, str(D / 'blind_tasks.py'), 'restore', *sorted(glob.glob(pat))], check=True)
        pat = pat.replace('evaluation_results.json', 'evaluation_results.restored.json')
    log(f'{name} eval complete → judge started')
    return subprocess.Popen([PY, '-u', str(E / 'judge_points/judge_points.py'), 'run', '--eval-glob', pat, '--out', str(out), '--workers', '16'],
                            cwd=E / 'judge_points', stdout=open(out / 'judge.log', 'a'), stderr=subprocess.STDOUT)


def status():
    run = running_outs()
    for name in RUNS:
        for d in shard_dirs(name):
            done, total = progress(d); r = run.get(str(d.resolve()))
            print(f'{name:14s} {d.name}: {done}/{total}' + (f'  running pid {r[0]} GPU {r[1]}' if r else ''))
        if (D / 'runs' / name / 'judge_points/summary.json').exists(): print(f'{name:14s} judge done')


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        status(); return
    only = sys.argv[1].split(',') if len(sys.argv) > 1 else list(RUNS)
    retries, judged, judging, last_launch = {}, set(), {}, 0
    while True:
        run = running_outs(); pending = []
        for name in only:
            for d in shard_dirs(name):
                done, total = progress(d)
                if done != total and str(d.resolve()) not in run: pending.append((name, d))
        # 起排队分片（每 2 分钟最多起一个，等显存稳定）
        if pending and time.time() - last_launch > 120:
            fg = free_gb()
            for name, d in pending:
                k = str(d); m = RUNS[name][0]
                if retries.get(k, 0) >= 4: continue
                g = max((g for g in GPUS if fg.get(g, 0) >= NEED_GB[m]), key=lambda g: fg[g], default=None)
                if g is None: continue
                if retries.get(k, 0): log(f'{name} {d.name} not running and incomplete → resume (retry {retries[k]})')
                retries[k] = retries.get(k, 0) + 1; launch(name, d, g); last_launch = time.time(); break
        for name in only:
            if name in judged or name in judging: continue
            if all(progress(d)[0] == progress(d)[1] for d in shard_dirs(name)):
                if (D / 'runs' / name / 'judge_points/summary.json').exists(): judged.add(name); continue
                judging[name] = start_judge(name)
        for name, p in list(judging.items()):
            if p.poll() is not None:
                log(f'{name} judge finished rc={p.returncode}'); judged.add(name); judging.pop(name)
        if len(judged) == len(only):
            log('all done: ' + ', '.join(only)); return
        time.sleep(60)


if __name__ == '__main__':
    main()
