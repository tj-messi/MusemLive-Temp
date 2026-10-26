"""Build an offline presentation with original frames, QA and 0/5-score cases."""
from __future__ import annotations

import argparse
import base64
import hashlib
from html import escape
import json
from pathlib import Path
import re

SELECTION = [
    ("N004_K002_sparse_P03_T1", "问配色，准确回答蓝白相间"),
    ("N005_K004_full_P05_T1", "人物站立，却答成骑在动物上"),
    ("N006_K003_sparse_P03_T1", "两幅画的上下关系判断正确"),
    ("N006_K001_full_P04_T1", "镂空躯干，被判断为完整封闭"),
    ("N004_K006_sparse_P03_T1", "正确读出展板英文标题"),
    ("N004_K006_sparse_P02_T2", "把地图与隔断的先后顺序颠倒"),
    ("N004_K001_full_P01_T2", "判断室内布景，并给出依据"),
    ("N006_K002_full_P06_T2", "回忆途中陈列时答错对象"),
]

KIND_NAMES = {"current_view": "画面理解", "follow_up": "追问与比较", "temporal_order": "先后顺序",
              "ocr": "文字识别", "long_memory": "跨时段记忆", "revisit": "场景重访"}


def text(value):
    return escape(str(value), quote=True)


def frame_time(seconds):
    minutes, tenths = divmod(round(float(seconds) * 10), 600)
    return f"{minutes:02d}:{tenths / 10:04.1f}"


def render_case(case, index, images):
    """All essential content is real HTML, including when scripts are stripped."""
    score = case["effective"]["score"]
    evidence = case["evidence"]
    frames = []
    for n, e in enumerate(evidence):
        stamp = frame_time(e["clip_s"])
        caption = text(f"{case['clip_id']} · 片段时间 {stamp}")
        frames.append(
            f'<figure class="frame"><div class="image-button" data-image="{e["sha256"]}" '
            f'data-caption="{caption}"><img src="{text(images[e["sha256"]])}" '
            f'alt="{text(case["title"])}，原始评测截图，片段时间 {stamp}" loading="eager"></div>'
            f'<figcaption><span>{str(n + 1).zfill(2) + " · " if len(evidence) > 1 else ""}'
            f'片段 {stamp}</span><span class="zoom-hint" hidden>点击放大 ↗</span></figcaption></figure>')
    context = ''.join(
        f'<div class="context"><strong>对话背景</strong>　上一问：{text(p["question"])}<br>'
        f'模型上一答：{text(p["answer"])}</div>' for p in case["previous"])
    evidence_class = 'two' if len(evidence) == 2 else 'multi' if len(evidence) > 2 else ''
    return (
        f'<article class="case {"zero" if score == 0 else "five"}" data-id="{text(case["id"])}" '
        f'data-score="{score}"><header class="case-top"><div><div class="case-kicker">'
        f'CASE {index + 1:02d} / {text(KIND_NAMES.get(case["kind"], case["kind"]))}</div>'
        f'<h3 class="case-title">{text(case["title"])}</h3></div>'
        f'<div class="score" aria-label="得分 {score} 分，满分 5 分"><strong>{score}</strong><small>/ 5</small>'
        f'</div></header><div class="evidence {evidence_class}">{"".join(frames)}</div>'
        f'<div class="case-body">{context}<h4 class="question"><small>QUESTION / 问题</small>'
        f'{text(case["question"])}</h4><div class="answer-box"><div class="answer-label">模型回答</div>'
        f'<div class="answer-text">{text(case["candidate_answer"])}</div></div>'
        f'<div class="reference"><strong>参考答案</strong>{text(case["reference_answer"])}</div>'
        f'<div class="reason"><strong>为什么是 {score} 分</strong>{text(case["judgment"]["reason"])}</div>'
        f'</div><footer class="case-footer"><span>{text(case["id"])}</span>'
        f'<span>提问于 {frame_time(case["question_time_s"])}'
        f'{" · 已确认人工复核" if case["human_review"] else ""}</span></footer></article>')


def render_document(template, data):
    stats = data["summary"]["with_human_review"]
    n = stats["scored_items"]
    bands = [("0–1", "#b45b42"), ("2–3", "#c39134"), ("4–5", "#287c68")]
    values = {
        "MEAN": f'{stats["mean_score"]:.3f}', "NORMALIZED": f'折合 {stats["normalized_score"] * 100:.2f} / 100',
        "SCORED": str(n), "PENDING": str(stats["pending_qa_review"]),
        "FIVE_COUNT": str(stats["score_distribution"]["5"]), "ZERO_COUNT": str(stats["score_distribution"]["0"]),
        "FIVE_PERCENT": f'占有效评分 {stats["score_distribution"]["5"] / n * 100:.2f}%',
        "ZERO_PERCENT": f'占有效评分 {stats["score_distribution"]["0"] / n * 100:.2f}%',
        "STACK_LABEL": '，'.join(f'{b} 分 {stats["score_bands"][b]} 条' for b, _ in bands),
        "STACK": ''.join(f'<span style="width:{stats["score_bands"][b] / n * 100:.2f}%;background:{c}"></span>' for b, c in bands),
        "LEGEND": ''.join(f'<span><i class="dot" style="background:{c}"></i>{b} 分 <strong>{stats["score_bands"][b]}</strong> 条</span>' for b, c in bands),
        "EXACT_DIST": ''.join(f'<div><span>{i} 分</span><strong>{stats["score_distribution"][str(i)]}</strong></div>' for i in range(6)),
        "GALLERY": ''.join(render_case(c, i, data["images"]) for i, c in enumerate(data["cases"])),
        "RESULTS_SHA256": text(data["metadata"]["results_sha256"]),
    }
    assert set(re.findall(r'__([A-Z0-9_]+)__', template)) == set(values)
    # Substitute once, so literal placeholder-looking text in an answer is preserved.
    return re.sub(r'__([A-Z0-9_]+)__', lambda m: values[m[1]], template)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(run_dir, input_root, destination):
    manifest = read(run_dir / "manifest.json")
    verified = read(run_dir / "统计核验.json")
    assert verified["passed"] is True
    for name in ("sample", "results", "summary"):
        assert sha(run_dir / (name + ".json")) == manifest[name + "_sha256"]
    assert verified["results_sha256"] == manifest["results_sha256"]
    assert verified["summary_sha256"] == manifest["summary_sha256"]
    samples = read(run_dir / "sample.json")
    results = read(run_dir / "results.json")
    summary = read(run_dir / "summary.json")
    result_map = {r["id"]: r for r in results}
    assert len(result_map) == len(samples) == len(results) == 240
    assert {s["id"] for s in samples} == set(result_map)
    items = []
    for s in samples:
        r = result_map[s["id"]]
        items.append({k: s[k] for k in ("id", "clip_id", "method", "kind", "question",
                    "reference_answer", "candidate_answer")} | {
                    "status": r["status"], "judgment": r["judgment"],
                    "human_review": r["human_review"], "effective": r["effective"]})
    sample_map = {s["id"]: s for s in samples}
    item_map = {s["id"]: s for s in items}
    cases, images, selection = [], {}, []
    for tid, title in SELECTION:
        item, sample = item_map[tid], sample_map[tid]
        assert item["effective"]["score"] in (0, 5)
        evidence = []
        for e in sample["evidence"]:
            image_path = (input_root / e["path"]).resolve()
            assert image_path.is_relative_to(input_root.resolve())
            assert sha(image_path) == e["sha256"]
            assert e["clip_s"] <= sample["evidence_cutoff_s"] <= sample["question_time_s"]
            assert e["original_model_input_frame"] is True
            raw = image_path.read_bytes()
            mime = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}[image_path.suffix.lower()]
            images[e["sha256"]] = "data:" + mime + ";base64," + base64.b64encode(raw).decode()
            evidence.append({k: e[k] for k in ("frame_id", "clip_s", "path", "sha256")})
        previous = sample["prior_dialogue_for_human_only"][-1:] if tid.endswith("_T2") else []
        cases.append(dict(item, title=title, evidence=evidence, previous=previous,
                          question_time_s=sample["question_time_s"]))
        selection.append({"id": tid, "title": title, "score": item["effective"]["score"], "evidence": evidence})
    assert [c["effective"]["score"] for c in cases].count(0) == 4
    assert [c["effective"]["score"] for c in cases].count(5) == 4
    data = {"schema_version": "museumlive_judge_showcase_v2_static", "cases": cases, "images": images, "summary": summary,
            "metadata": {"clips": manifest["clips"], "items": manifest["items"],
                "judge_model": manifest["model"], "candidate_model": "MiniCPM-o-4.5",
                "protocol": manifest["version"], "input_sha256": manifest["input_sha256"],
                "results_sha256": manifest["results_sha256"], "summary_sha256": manifest["summary_sha256"],
                "reference_status": "machine_authored_pending_human_review",
                "scope": "current ten clips and 240 saved answers"}}
    template = Path(__file__).with_name("holistic_showcase_template.html")
    html = render_document(template.read_text(encoding="utf-8"), data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(html.encode("utf-8"))
    assert "sk-or-v1-" not in html
    report = {"html": destination.name, "html_sha256": sha(destination),
              "builder_sha256": sha(Path(__file__)), "template_sha256": sha(template),
              "results_sha256": manifest["results_sha256"], "summary_sha256": manifest["summary_sha256"],
              "items": len(items), "display_cases": len(cases), "zero_score_cases": 4, "five_score_cases": 4,
              "original_evidence_images": len(images), "external_dependencies": [], "original_scores_modified": False,
              "rendering": "static_html_with_optional_controls", "javascript_required_for_content": False}
    (destination.parent / "visualization_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination.parent / "showcase_selection.json").write_text(json.dumps({
        "selection": "Four reviewed zero-score and four full-score examples selected for illustration, not a representative sample.",
        "results_sha256": manifest["results_sha256"], "cases": selection}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or args.run_dir / "Judge_展示版_截图与QA.html"
    print(json.dumps(build(args.run_dir, args.input_root, target), ensure_ascii=False))


if __name__ == "__main__":
    main()
