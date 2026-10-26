#!/usr/bin/env python3
"""10-15 QA Pipeline 统一入口（出题 → 后处理 → 截取）。

用法: python3 run_gen.py <batch> <stage> [该阶段的参数...]
  batch ∈ 50 | 316（见 ../config.py BATCHES）
  stage:
    v3      7 类题型出题            → runs/<batch>/runs_v3/      例: run_gen.py 316 v3 --per-video 2 --budget 15 --workers 20
    hr      历史指代（视觉记忆）出题  → runs/<batch>/runs_hr/      例: run_gen.py 316 hr --seeds 380 --budget 5
    pg      主动引导（视角）出题      → runs/<batch>/runs_pg/      例: run_gen.py 316 pg --per-video 3 --target 190 --budget 5
    v3b     题型后处理审计/改写       → 改 runs_v3 里的题            例: run_gen.py 316 v3b audit [--dry]
    pool    汇总截取为题库            → runs/<batch>/final/        例: run_gen.py 316 pool --cap 175
  跨批次的最终挑题用 select_final.py，eval/judge 用 ../eval/eval_all.py。
"""
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = {'v3': 'gen_v3.py', 'hr': 'gen_hr.py', 'pg': 'gen_pg.py', 'v3b': 'postprocess_v3b.py', 'pool': 'select_pool.py'}

if __name__ == '__main__':
    if len(sys.argv) < 3 or sys.argv[2] not in STAGES:
        print(__doc__); sys.exit(1)
    batch, stage, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
    script = HERE / STAGES[stage]
    sys.argv = [str(script)] + (rest[:1] if stage == 'v3b' else []) + ['--batch', batch] + (rest[1:] if stage == 'v3b' else rest)
    runpy.run_path(str(script), run_name='__main__')
