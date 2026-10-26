#!/usr/bin/env python3
"""把实验 1–7 的分析结果整理成论文可用总报告、SVG 图和机器可读汇总。"""
import html
import json
from pathlib import Path

D = Path(__file__).resolve().parent
FIG = D / 'figures_1_7'
FIG.mkdir(exist_ok=True)
MODELS = [('minicpm', 'MiniCPM-o 4.5'), ('venus', 'Venus'), ('gander', 'Gander'), ('moss', 'MOSS*')]
COLORS = ['#4472C4', '#ED7D31', '#70AD47', '#A64D79', '#5B9BD5', '#FFC000']


def esc(x):
    return html.escape(str(x))


def grouped_bar(path, title, categories, series, ymax, ylabel='', note=''):
    width, height = 1100, 620
    left, right, top, bottom = 90, 30, 85, 125
    pw, ph = width - left - right, height - top - bottom
    ncat, nser = len(categories), len(series)
    group_w = pw / max(1, ncat)
    bar_w = min(42, group_w * .75 / max(1, nser))
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.bar{stroke:#fff;stroke-width:1}</style>',
           f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-weight="700">{esc(title)}</text>']
    for i in range(6):
        val = ymax * i / 5
        y = top + ph - ph * i / 5
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}"/>',
                f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="14">{val:.1f}</text>']
    out += [f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top+ph}"/>',
            f'<line class="axis" x1="{left}" y1="{top+ph}" x2="{left+pw}" y2="{top+ph}"/>']
    if ylabel:
        out.append(f'<text transform="translate(24 {top+ph/2}) rotate(-90)" text-anchor="middle" font-size="16">{esc(ylabel)}</text>')
    for ci, cat in enumerate(categories):
        cx = left + group_w * (ci + .5)
        out.append(f'<text x="{cx}" y="{top+ph+28}" text-anchor="middle" font-size="14">{esc(cat)}</text>')
        start = cx - nser * bar_w / 2
        for si, (name, values) in enumerate(series):
            value = values[ci]
            if value is None:
                continue
            h = max(0, min(ph, ph * value / ymax))
            x, y = start + si * bar_w, top + ph - h
            out += [f'<rect class="bar" x="{x}" y="{y}" width="{bar_w-2}" height="{h}" fill="{COLORS[si % len(COLORS)]}"/>',
                    f'<text x="{x+(bar_w-2)/2}" y="{max(top+12,y-5)}" text-anchor="middle" font-size="12">{value:.2f}</text>']
    lx = left
    for si, (name, _) in enumerate(series):
        out += [f'<rect x="{lx}" y="{height-55}" width="16" height="16" fill="{COLORS[si % len(COLORS)]}"/>',
                f'<text x="{lx+22}" y="{height-42}" font-size="14">{esc(name)}</text>']
        lx += 22 + max(95, len(name) * 14)
    if note:
        out.append(f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="13" fill="#555">{esc(note)}</text>')
    out.append('</svg>')
    path.write_text('\n'.join(out))


def line_chart(path, title, xlabels, series, ymax=2.5):
    width, height = 1100, 620
    left, right, top, bottom = 90, 40, 85, 120
    pw, ph = width-left-right, height-top-bottom
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<style>text{font-family:Arial,"Noto Sans CJK SC",sans-serif;fill:#222}.grid{stroke:#ddd}.axis{stroke:#333}</style>',
           f'<text x="{width/2}" y="35" text-anchor="middle" font-size="24" font-weight="700">{esc(title)}</text>']
    for i in range(6):
        value = ymax*i/5; y = top+ph-ph*i/5
        out += [f'<line class="grid" x1="{left}" y1="{y}" x2="{left+pw}" y2="{y}"/>',
                f'<text x="{left-12}" y="{y+5}" text-anchor="end" font-size="14">{value:.1f}</text>']
    xs = [left + pw*i/max(1, len(xlabels)-1) for i in range(len(xlabels))]
    for x, label in zip(xs, xlabels):
        out.append(f'<text x="{x}" y="{top+ph+30}" text-anchor="middle" font-size="13">{esc(label)}</text>')
    for si, (name, values) in enumerate(series):
        pts = [(x, top+ph-ph*v/ymax) for x, v in zip(xs, values)]
        color = COLORS[si]
        out.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x,y in pts)}" fill="none" stroke="{color}" stroke-width="3"/>')
        for x, y in pts:
            out += [f'<circle cx="{x}" cy="{y}" r="5" fill="{color}"/>',
                    f'<text x="{x}" y="{y-10}" text-anchor="middle" font-size="12">{values[pts.index((x,y))]:.2f}</text>']
    lx = left
    for si, (name, _) in enumerate(series):
        out += [f'<line x1="{lx}" y1="{height-42}" x2="{lx+24}" y2="{height-42}" stroke="{COLORS[si]}" stroke-width="4"/>',
                f'<text x="{lx+30}" y="{height-37}" font-size="14">{esc(name)}</text>']
        lx += 30 + max(100, len(name)*14)
    out.append('</svg>')
    path.write_text('\n'.join(out))


def load_summary(path):
    return json.load(open(path)) if path.exists() else None


def main():
    a = json.load(open(D / 'analysis_a.json'))
    b = json.load(open(D / 'analysis_b.json'))

    # 图 1：回应率 / 条件质量 / 联合分
    cats = [name for _, name in MODELS]
    grouped_bar(FIG / 'response_decomposition.svg', '回应行为拆分（用户提问 slot）', cats,
                [('回应率×5', [a['exp1'][m]['user_query']['resp_rate']*5 for m,_ in MODELS]),
                 ('回应后质量', [a['exp1'][m]['user_query']['cond'] for m,_ in MODELS]),
                 ('联合分', [a['exp1'][m]['user_query']['joint'] for m,_ in MODELS])], 5, '0–5', '回应率乘 5 仅用于与质量分共轴展示')
    grouped_bar(FIG / 'timing_f1.svg', '在线发言时间质量', cats,
                [('事件召回', [a['exp2'][m]['event_recall'] for m,_ in MODELS]),
                 ('时间精度', [a['exp2'][m]['time_precision'] for m,_ in MODELS]),
                 ('F1', [a['exp2'][m]['f1'] for m,_ in MODELS])], 1, '比例')

    judges = a['exp4']['judges']
    judge_names = {'qwen':'Qwen', 'tencent_hy3':'HY3', 'deepseek_deepseek-v4-pro':'DeepSeek', 'z-ai_glm-5.2':'GLM', 'moonshotai_kimi-k2.6':'Kimi'}
    line_chart(FIG / 'judge_stability.svg', '不同 Judge 下的模型均分', [judge_names[x] for x in judges],
               [(name, [a['exp4']['mean'][j][m] for j in judges]) for m,name in MODELS])

    ab_keys = [('MiniCPM完整','minicpm_完整'), ('MiniCPM静音','minicpm_静音'), ('MiniCPM黑屏','minicpm_黑屏'),
               ('MOSS完整','moss_完整（画面+转写）'), ('MOSS无转写','moss_无现场转写'), ('MOSS仅转写','moss_仅转写文字（黑屏）')]
    grouped_bar(FIG / 'modality_ablation.svg', '模态消融总分', [x[0] for x in ab_keys],
                [('总均分', [b[k]['total'] for _,k in ab_keys])], 5, '0–5')

    exp7 = b['exp7']
    context_keys = [('MiniCPM完整','MiniCPM 全双工'), ('MiniCPM近30秒','MiniCPM 最近30秒'), ('MiniCPM近3秒','MiniCPM 3秒盲基线'),
                    ('MOSS完整','MOSS 全双工'), ('MOSS近30秒','MOSS 最近30秒'), ('MOSS近3秒','MOSS 3秒盲基线'),
                    ('离线上界','qwen-omni 离线上界')]
    present = [(lab,key) for lab,key in context_keys if key in exp7]
    grouped_bar(FIG / 'context_baselines.svg', '上下文捷径基线与离线上界（用户提问）', [x[0] for x in present],
                [('均分', [exp7[k]['all'] for _,k in present])], 5, '0–5')

    gap_order = ['<5s','5–15s','15–30s','≥30s']
    grouped_bar(FIG / 'evidence_distance.svg', '最近音频证据距离分层', gap_order,
                [(name, [a['exp3']['evidence_gap'][g].get(m,{}).get('mean') for g in gap_order]) for m,name in MODELS], 5, '0–5',
                '≥30s 仅 2 个 slot，不能据此下稳定结论')

    checks = {}
    for run in ['force_venus','force_gander','force_minicpm','blind_minicpm','blind_moss','recent30_minicpm','recent30_moss',
                'mute_minicpm','black_minicpm','notext_moss','black_moss','offline_qwen_omni']:
        s = load_summary(D / 'runs' / run / 'judge_points/summary.json')
        checks[run] = None if s is None else {k:s.get(k) for k in ('slots','scorable_ok','api_failed','mean')}
    judges_check = {}
    for j in judges[1:]:
        judges_check[j] = {}
        for m,_ in MODELS:
            s = load_summary(D / 'judges' / j / m / 'summary.json')
            judges_check[j][m] = None if s is None else {k:s.get(k) for k in ('slots','scorable_ok','api_failed','mean')}

    payload = {'analysis_a': a, 'analysis_b': b, 'checks': checks, 'alternate_judges': judges_check,
               'figures': sorted(p.name for p in FIG.glob('*.svg'))}
    (D / 'report_1_7.json').write_text(json.dumps(payload, ensure_ascii=False, indent=1))

    r1, r2, r4 = a['exp1'], a['exp2'], a['exp4']
    recent_ready = 'MiniCPM 最近30秒' in exp7 and 'MOSS 最近30秒' in exp7
    lines = ['# OmniBench 实验 1–7 完整报告', '',
             '范围：最终人工选择的 450 题、1250 个评分 slot；四个实时模型使用同一得分点评分协议。MOSS 不接收原始音频，现场声音通过理想 ASR 转写文字输入。', '',
             '## 完成状态', '',
             '| 实验 | 状态 | 核心产物 |', '|---|---|---|',
             '| 1 回应率/回应后质量/联合分 | 完成 | response_decomposition.svg |',
             '| 2 时间指标与琐碎基线 | 完成 | timing_f1.svg |',
             '| 3 证据距离与视频时长分层 | 完成 | evidence_distance.svg；包含真实评测 clip 时长 |',
             '| 4 多 Judge 与 bootstrap | 完成 | 5 个 Judge、每个 1250 slot、按题聚类 bootstrap |',
             '| 5 强制作答 | 完成 | Venus/Gander/MiniCPM 配对分析与 95% CI |',
             '| 6 模态消融 | 完成 | 静音、黑屏、无转写、仅转写文字 |',
             f'| 7 最近30秒/3秒捷径基线与离线上界 | {"完成" if recent_ready else "最近30秒补跑中"} | context_baselines.svg |', '',
             '## 核心结论', '',
             f"1. Venus/Gander 的用户提问回应率分别为 {r1['venus']['user_query']['resp_rate']:.0%}/{r1['gander']['user_query']['resp_rate']:.0%}；回应后的条件质量为 {r1['venus']['user_query']['cond']}/{r1['gander']['user_query']['cond']}，证明低分同时包含“不说”和“说了也不够好”两个因素。",
             f"2. 时间 F1：MiniCPM {r2['minicpm']['f1']:.2f}、Venus {r2['venus']['f1']:.2f}、Gander {r2['gander']['f1']:.2f}、MOSS {r2['moss']['f1']:.2f}；永远说话基线为 {r2['minicpm']['baseline_always_speak']['f1']:.2f}。",
             '3. 评测实际使用的 clip 绝大多数为 10–15 分钟；模型每题实际接收的 session 通常只有几十秒。因此当前实验不能表述为模型处理了小时级连续前缀。',
             '4. Qwen、HY3、DeepSeek、GLM、Kimi 五个 Judge 都给出 MOSS > MiniCPM > Gander > Venus，排名 Kendall τ=1。',
             f"5. 强制作答的用户提问联合分变化：Venus {b['force_venus']['uq_orig']}→{b['force_venus']['uq_forced']}，Gander {b['force_gander']['uq_orig']}→{b['force_gander']['uq_forced']}，MiniCPM {b['force_minicpm']['uq_orig']}→{b['force_minicpm']['uq_forced']}。",
             '   Gander 强制作答中 1/1250 slot 因主 Judge 内容审查 HTTP 400，使用 GLM-5.2、同一 rubric 和确定性计分公式补齐；provenance 已保存。',
             f"6. MiniCPM 静音后 {b['minicpm_完整']['total']}→{b['minicpm_静音']['total']}，说明现场音频存在明显干扰；黑屏降至 {b['minicpm_黑屏']['total']}。MOSS 去转写降至 {b['moss_无现场转写']['total']}，仅转写文字为 {b['moss_仅转写文字（黑屏）']['total']}。",
             f"7. 用户提问均分从完整上下文到最近 30 秒再到 3 秒：MiniCPM "
             f"{exp7['MiniCPM 全双工']['all']}→{exp7['MiniCPM 最近30秒']['all']}→{exp7['MiniCPM 3秒盲基线']['all']}，MOSS "
             f"{exp7['MOSS 全双工']['all']}→{exp7['MOSS 最近30秒']['all']}→{exp7['MOSS 3秒盲基线']['all']}；"
             f"历史指代对截断最敏感。离线 Qwen-Omni 上界为 {exp7['qwen-omni 离线上界']['all']}。", '',
             '## 图表', '',
             '- [回应行为拆分](figures_1_7/response_decomposition.svg)',
             '- [时间 Precision/Recall/F1](figures_1_7/timing_f1.svg)',
             '- [多 Judge 稳定性](figures_1_7/judge_stability.svg)',
             '- [模态消融](figures_1_7/modality_ablation.svg)',
             '- [上下文基线与离线上界](figures_1_7/context_baselines.svg)',
             '- [证据距离分层](figures_1_7/evidence_distance.svg)', '',
             '## 重建分析与报告', '', '```bash',
             'cd "$MUSEUMLIVE_EXPERIMENT_ROOT"',
             '/usr/bin/python3 analysis_a.py', '/usr/bin/python3 analysis_b.py', '/usr/bin/python3 build_report_1_7.py', '```', '',
             '详细逐项表格见 `analysis_a.md`、`analysis_b.md`；完整机器可读结果见 `report_1_7.json`。']
    (D / 'report_1_7.md').write_text('\n'.join(lines) + '\n')
    print('written report_1_7.md, report_1_7.json and', len(payload['figures']), 'figures')


if __name__ == '__main__':
    main()
