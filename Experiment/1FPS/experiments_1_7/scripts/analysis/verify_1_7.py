#!/usr/bin/env python3
"""实验 1–7 完整性检查；任何缺失、未评分或报告未更新都会返回非零。"""
import json
import sys
from pathlib import Path

D = Path(__file__).resolve().parent
errors = []


def check_summary(path, slots):
    if not path.exists():
        errors.append(f'missing {path.relative_to(D)}')
        return
    s = json.load(open(path))
    if s.get('slots') != slots:
        errors.append(f'{path.relative_to(D)} slots={s.get("slots")} expected={slots}')
    if s.get('scorable_ok') != slots or s.get('api_failed') != 0:
        errors.append(f'{path.relative_to(D)} scorable_ok={s.get("scorable_ok")} api_failed={s.get("api_failed")}')


for run, slots in {
    'force_venus':1250, 'force_gander':1250, 'force_minicpm':1250,
    'blind_minicpm':1000, 'blind_moss':1000,
    'recent30_minicpm':1000, 'recent30_moss':1000,
    'mute_minicpm':1250, 'black_minicpm':1250,
    'notext_moss':1250, 'black_moss':1250,
    'offline_qwen_omni':1250,
}.items():
    check_summary(D / 'runs' / run / 'judge_points/summary.json', slots)

for model in ['minicpm', 'moss']:
    path = D / 'runs' / f'recent30_{model}' / 'judge_points/summary.json'
    if path.exists():
        summary = json.load(open(path))
        if summary.get('reused_formal_judgments') != 810 or summary.get('rerun_judgments') != 190:
            errors.append(f'{path.relative_to(D)} missing 810/190 judgment provenance')

for judge in ['tencent_hy3','deepseek_deepseek-v4-pro','z-ai_glm-5.2','moonshotai_kimi-k2.6']:
    for model in ['minicpm','venus','gander','moss']:
        check_summary(D / 'judges' / judge / model / 'summary.json', 1250)

for f in ['analysis_a.json','analysis_a.md','analysis_b.json','analysis_b.md','report_1_7.json','report_1_7.md']:
    if not (D / f).exists():
        errors.append(f'missing {f}')
for f in ['response_decomposition.svg','timing_f1.svg','judge_stability.svg','modality_ablation.svg','context_baselines.svg','evidence_distance.svg']:
    if not (D / 'figures_1_7' / f).exists():
        errors.append(f'missing figures_1_7/{f}')

if (D / 'analysis_b.json').exists():
    b = json.load(open(D / 'analysis_b.json'))
    for key in ['MiniCPM 最近30秒','MOSS 最近30秒','MiniCPM 3秒盲基线','MOSS 3秒盲基线','qwen-omni 离线上界']:
        if key not in b.get('exp7', {}):
            errors.append(f'analysis_b missing {key}')

if (D / 'report_1_7.md').exists():
    report = (D / 'report_1_7.md').read_text()
    if '补跑中' in report or '未跑完' in report:
        errors.append('report_1_7.md still contains incomplete markers')
    if '| 7 最近30秒/3秒捷径基线与离线上界 | 完成 |' not in report:
        errors.append('report_1_7.md does not mark experiment 7 complete')
    for value in ['1.31→1.27→1.18', '2.01→1.93→1.79']:
        if value not in report:
            errors.append(f'report_1_7.md missing experiment 7 result {value}')

if errors:
    print('INCOMPLETE')
    print('\n'.join('- ' + x for x in errors))
    sys.exit(1)
print('COMPLETE: experiments 1–7, all required slots, analyses, reports and figures are present')
