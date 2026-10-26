#!/usr/bin/env python3
"""Validate the external 10 FPS Experiment 1-7 run before publication."""
from __future__ import annotations

import json
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path(os.environ["EXPERIMENT_OUTPUT_ROOT"]).expanduser().resolve()
JUDGE_DIR = os.environ.get("JUDGE_DIR_NAME", "judge_points_lenient_v3")
ERRORS: list[str] = []


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def check_summary(path: Path, slots: int) -> dict | None:
    if not path.exists():
        ERRORS.append(f"missing {path.relative_to(ROOT)}")
        return None
    summary = load(path)
    if summary.get("slots") != slots:
        ERRORS.append(f"{path.relative_to(ROOT)} slots={summary.get('slots')} expected={slots}")
    if summary.get("scorable_ok") != slots or summary.get("api_failed") != 0:
        ERRORS.append(
            f"{path.relative_to(ROOT)} scorable_ok={summary.get('scorable_ok')} "
            f"api_failed={summary.get('api_failed')}"
        )
    return summary


RUNS = {
    "main_minicpm": 1250,
    "main_venus": 1250,
    "main_gander": 1250,
    "main_moss": 1250,
    "force_venus": 1250,
    "force_gander": 1250,
    "force_minicpm": 1250,
    "mute_minicpm": 1250,
    "black_minicpm": 1250,
    "notext_moss": 1250,
    "black_moss": 1250,
    "blind_minicpm": 1000,
    "blind_moss": 1000,
    "recent30_minicpm": 1000,
    "recent30_moss": 1000,
    "offline_qwen_omni": 1250,
}
for run, slots in RUNS.items():
    summary = check_summary(ROOT / "runs" / run / JUDGE_DIR / "summary.json", slots)
    if run.startswith("recent30_") and summary:
        if summary.get("reused_formal_judgments") != 810 or summary.get("rerun_judgments") != 190:
            ERRORS.append(f"{run} missing 810/190 judgment provenance")

fallback = ROOT / "runs/force_gander" / JUDGE_DIR / "fallback_provenance.json"
if not fallback.exists():
    ERRORS.append("missing force_gander provider-block fallback provenance")
else:
    data = load(fallback)
    if data.get("raw_evaluation_unchanged") is not True or data.get("external_retry_authorized") is not False:
        ERRORS.append("force_gander fallback provenance is inconsistent")

offline = ROOT / "runs/offline_qwen_omni/reuse_provenance.json"
if not offline.exists():
    ERRORS.append("missing offline upper-bound reuse provenance")
else:
    data = load(offline)
    if data.get("timing_mismatches") != 0 or data.get("successful_timing_pairs_checked") != 999:
        ERRORS.append("offline upper-bound timing provenance is inconsistent")

alternate_fallback = ROOT / "alternate_judge_fallback_provenance.json"
if not alternate_fallback.exists():
    ERRORS.append("missing alternate Judge fallback provenance")
else:
    data = load(alternate_fallback)
    if (
        data.get("affected_slots") != 22
        or data.get("provider_reroute_slots") != 20
        or data.get("compact_json_slots") != 17
        or data.get("raw_evaluation_unchanged") is not True
        or data.get("rubric_unchanged") is not True
        or data.get("scoring_unchanged") is not True
    ):
        ERRORS.append("alternate Judge fallback provenance is inconsistent")

judges = ["tencent_hy3", "deepseek_deepseek-v4-pro", "z-ai_glm-5.2", "moonshotai_kimi-k2.6"]
for judge in judges:
    for model in ["minicpm", "venus", "gander", "moss"]:
        check_summary(ROOT / "judges" / judge / model / "summary.json", 1250)

for name in ["analysis_a.json", "analysis_a.md", "analysis_b.json", "analysis_b.md", "report_1_7.json", "report_1_7.md"]:
    if not (ROOT / name).exists():
        ERRORS.append(f"missing {name}")
for name in ["response_decomposition.svg", "timing_f1.svg", "judge_stability.svg", "modality_ablation.svg", "context_baselines.svg", "evidence_distance.svg"]:
    path = ROOT / "figures_1_7" / name
    if not path.exists():
        ERRORS.append(f"missing figures_1_7/{name}")
    else:
        try:
            ET.parse(path)
        except ET.ParseError as exc:
            ERRORS.append(f"invalid figures_1_7/{name}: {exc}")

analysis_a = ROOT / "analysis_a.json"
if analysis_a.exists():
    names = load(analysis_a).get("exp4", {}).get("judges", [])
    if len(names) != 5:
        ERRORS.append(f"analysis_a exp4 has {len(names)} judges, expected 5")
analysis_b = ROOT / "analysis_b.json"
if analysis_b.exists():
    exp7 = load(analysis_b).get("exp7", {})
    for key in ["MiniCPM 最近30秒", "MOSS 最近30秒", "MiniCPM 3秒盲基线", "MOSS 3秒盲基线", "qwen-omni 离线上界"]:
        if key not in exp7:
            ERRORS.append(f"analysis_b missing {key}")
report = ROOT / "report_1_7.md"
if report.exists():
    text = report.read_text()
    if "补跑中" in text or "未跑完" in text:
        ERRORS.append("report_1_7.md still contains incomplete markers")
    if "| 7 最近30秒/3秒捷径基线与离线上界 | 完成 |" not in text:
        ERRORS.append("report_1_7.md does not mark Experiment 7 complete")
    if "provider" not in text.lower() or "离线上界复用" not in text:
        ERRORS.append("report_1_7.md is missing integrity disclosures")

if ERRORS:
    print("INCOMPLETE")
    print("\n".join(f"- {error}" for error in ERRORS))
    sys.exit(1)
print("COMPLETE: 16 run summaries, 16 alternate-judge panels, analyses, reports, figures and provenance are valid")
