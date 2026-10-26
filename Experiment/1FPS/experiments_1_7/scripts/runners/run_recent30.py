#!/usr/bin/env python3
"""实验 7 最近 30 秒基线的优化可复现 runner。

1000 个 user_query slot 中，若最近 30 秒截断点等于正式 session 起点，则模型输入
与正式全双工完全相同，直接复用正式输出；只对真正截断的 slot 做 GPU 推理。
最后合并为完整的 1000-slot eval，恢复正式 judge_id，并运行同一得分点 Judge。
"""
import copy
import glob
import json
import os
import subprocess
import time
from pathlib import Path

D = Path(__file__).resolve().parent
W = D.parent
MC = W / 'model_compare_20261018'
E = W / 'omni_eval_20261016'
PY = '/usr/bin/python3'
TASKS = D / 'tasks_recent30.json'
OPT = D / 'runs/recent30_optimized'
LOG = D / 'recent30_optimized.log'
MODELS = {
    'minicpm': {'gpus': [4, 5], 'script': E / 'eval_duplex.py'},
    'moss': {'gpus': [6, 7], 'script': MC / 'eval_moss.py'},
}


def log(msg):
    line = time.strftime('[%F %T] ') + msg
    print(line, flush=True)
    with LOG.open('a') as f:
        f.write(line + '\n')


def load_results(path):
    try:
        return json.load(open(path))['results']
    except Exception:
        return []


def truncated_tasks():
    tasks = json.load(open(TASKS))
    return [t for t in tasks if float(t['session']['start_s']) > float(t['recent30']['orig_session_start_s']) + 1e-6]


def prepare():
    changed = truncated_tasks()
    OPT.mkdir(parents=True, exist_ok=True)
    for model, cfg in MODELS.items():
        for i, _ in enumerate(cfg['gpus']):
            d = OPT / model / f'shard{i}'
            d.mkdir(parents=True, exist_ok=True)
            (d / 'tasks.json').write_text(json.dumps(changed[i::len(cfg['gpus'])], ensure_ascii=False))
    log(f'prepared {len(changed)} truncated pseudo-tasks; {1000-len(changed)} unchanged slots will reuse formal outputs')


def progress(d):
    total = len(json.load(open(d / 'tasks.json')))
    done = len(load_results(d / 'eval/evaluation_results.json'))
    return done, total


def launch():
    running = []
    for model, cfg in MODELS.items():
        for i, gpu in enumerate(cfg['gpus']):
            d = OPT / model / f'shard{i}'
            done, total = progress(d)
            if done == total:
                log(f'{model} shard{i} already complete {done}/{total}')
                continue
            env = dict(os.environ, EVAL_TASKS=str(d / 'tasks.json'), EVAL_OUT=str(d), MODEL_DEVICE=str(gpu), TTS_BACKEND='qwen')
            if model == 'minicpm':
                env['DUPLEX_MODEL'] = 'minicpm'
                cwd = E
            else:
                cwd = MC
            handle = open(d / 'run.log', 'a')
            p = subprocess.Popen([PY, '-u', str(cfg['script'])], cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((model, i, p, handle))
            log(f'launch {model} shard{i} on GPU {gpu}: resume {done}/{total}')
    while running:
        left = []
        for model, i, p, handle in running:
            rc = p.poll()
            if rc is None:
                left.append((model, i, p, handle))
            else:
                handle.close()
                done, total = progress(OPT / model / f'shard{i}')
                if rc != 0 or done != total:
                    raise RuntimeError(f'{model} shard{i} failed rc={rc} progress={done}/{total}')
                log(f'{model} shard{i} complete {done}/{total}')
        running = left
        if running:
            status = []
            for model, i, _, _ in running:
                done, total = progress(OPT / model / f'shard{i}')
                status.append(f'{model}{i}={done}/{total}')
            log('running ' + ' '.join(status))
            time.sleep(60)


def base_rows(model):
    pattern = str(MC / ('minicpm/*/eval/evaluation_results.json' if model == 'minicpm' else 'moss/shard*/eval/evaluation_results.json'))
    out = {}
    for f in glob.glob(pattern):
        for row in load_results(Path(f)):
            out[row['branch_key']] = row
    return out


def fresh_rows(model):
    out = {}
    for f in glob.glob(str(OPT / model / 'shard*/eval/evaluation_results.json')):
        for row in load_results(Path(f)):
            out[row['branch_key']] = row
    return out


def assemble(model):
    tasks = json.load(open(TASKS))
    base, fresh = base_rows(model), fresh_rows(model)
    rows = []
    reused = rerun = 0
    for task in tasks:
        branch = task['branches'][0]
        pseudo_id = branch['branch_id']
        orig_id = task['recent30']['orig_branch_id']
        pseudo_key = f"{task['task_id']}::{pseudo_id}"
        changed = float(task['session']['start_s']) > float(task['recent30']['orig_session_start_s']) + 1e-6
        source = fresh.get(pseudo_key) if changed else base.get(f"{task['task_id']}::{orig_id}")
        if source is None:
            raise KeyError(f'missing {model} row {pseudo_key} changed={changed}')
        row = copy.deepcopy(source)
        wanted_slot = branch['slots'][0]['slot_id']
        matched = [s for s in row['slots'] if s['slot']['slot_id'] == wanted_slot]
        if len(matched) != 1:
            raise ValueError(f'{model} {pseudo_key}: slot {wanted_slot} found {len(matched)} times')
        row['branch_id'] = pseudo_id
        row['branch_key'] = pseudo_key
        row['session'] = copy.deepcopy(task['session'])
        row['user_events'] = copy.deepcopy(branch['user_events'])
        row['slots'] = matched
        row.setdefault('causal_audit', {})['recent_context_s'] = 30.0
        row['causal_audit']['reused_formal_output'] = not changed
        rows.append(row)
        rerun += int(changed); reused += int(not changed)
    out = D / 'runs' / f'recent30_{model}' / 'combined/eval'
    out.mkdir(parents=True, exist_ok=True)
    f = out / 'evaluation_results.json'
    f.write_text(json.dumps({'schema_version': f'museumlive_recent30_{model}_v1', 'model': model, 'results': rows}, ensure_ascii=False, indent=1))
    subprocess.run([PY, str(D / 'recent30_tasks.py'), 'restore', str(f)], check=True)
    log(f'assembled {model}: {len(rows)} slots ({reused} reused, {rerun} rerun)')
    return f.parent / 'evaluation_results.restored.json'


def judge(model, restored):
    out = D / 'runs' / f'recent30_{model}' / 'judge_points'
    summary_path = out / 'summary.json'
    if summary_path.exists():
        summary = json.load(open(summary_path))
        if summary.get('slots') == 1000 and summary.get('scorable_ok') == 1000 and summary.get('api_failed') == 0:
            log(f'{model} judge already complete mean={summary["mean"]}')
            return

    payload = json.load(open(restored))
    changed_ids = []
    wanted_ids = []
    for row in payload['results']:
        if len(row['slots']) != 1:
            raise ValueError(f'{model} recent30 row must contain exactly one slot: {row["branch_key"]}')
        jid = f"{row['branch_key']}::{row['slots'][0]['slot']['slot_id']}"
        wanted_ids.append(jid)
        if not row.get('causal_audit', {}).get('reused_formal_output'):
            changed_ids.append(jid)
    if len(wanted_ids) != 1000 or len(set(wanted_ids)) != 1000 or len(changed_ids) != 190:
        raise ValueError(f'{model} judge id counts invalid: wanted={len(wanted_ids)} unique={len(set(wanted_ids))} changed={len(changed_ids)}')

    # 810 个未截断 slot 的输入与模型输出和正式评测逐字相同，安全复用既有判分；
    # 只让 LLM Judge 处理真正因 30 秒截断而重新推理的 190 个 slot。
    ids_path = out.parent / 'changed_judge_ids.txt'
    ids_path.write_text('\n'.join(changed_ids) + '\n')
    cmd = [PY, '-u', str(E / 'judge_points/judge_points.py'), 'run', '--eval-glob', str(restored),
           '--ids', str(ids_path), '--out', str(out), '--workers', '16']
    with (out.parent / 'judge.log').open('a') as handle:
        rc = subprocess.run(cmd, cwd=E / 'judge_points', stdout=handle, stderr=subprocess.STDOUT).returncode
    changed_summary = json.load(open(summary_path))
    if rc != 0 or changed_summary.get('slots') != 190 or changed_summary.get('scorable_ok') != 190 or changed_summary.get('api_failed') != 0:
        raise RuntimeError(f'{model} changed judge incomplete rc={rc}: {changed_summary}')

    changed = {r['judge_id']: r for r in map(json.loads, open(out / 'results.jsonl'))}
    base_file = MC / model / 'judge_points/results.jsonl'
    base = {r['judge_id']: r for r in map(json.loads, open(base_file))}
    merged = []
    changed_set = set(changed_ids)
    for jid in wanted_ids:
        source = changed.get(jid) if jid in changed_set else base.get(jid)
        if source is None:
            raise KeyError(f'{model} missing merged judgment {jid}')
        merged.append(source)
    with (out / 'results.jsonl').open('w') as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    subprocess.run([PY, str(E / 'judge_points/judge_points.py'), 'report', '--out', str(out)],
                   cwd=E / 'judge_points', check=True, stdout=subprocess.DEVNULL)
    summary = json.load(open(summary_path))
    summary['spent_usd'] = changed_summary.get('spent_usd', 0)
    summary['calls'] = changed_summary.get('calls', 0)
    summary['reused_formal_judgments'] = 810
    summary['rerun_judgments'] = 190
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    (out / 'reuse_provenance.json').write_text(json.dumps({
        'formal_results': str(base_file),
        'reused_judge_ids': 810,
        'rerun_judge_ids': 190,
        'reason': 'Recent-30s truncation changed only 190/1000 slots; the other 810 have identical input, output, judge_id and rubric.',
    }, ensure_ascii=False, indent=1))
    if summary.get('slots') != 1000 or summary.get('scorable_ok') != 1000 or summary.get('api_failed') != 0:
        raise RuntimeError(f'{model} merged judge incomplete: {summary}')
    log(f'{model} judge complete mean={summary["mean"]}')


def main():
    prepare()
    launch()
    for model in MODELS:
        restored = assemble(model)
        judge(model, restored)
    log('all recent-30s experiments complete')


if __name__ == '__main__':
    main()
