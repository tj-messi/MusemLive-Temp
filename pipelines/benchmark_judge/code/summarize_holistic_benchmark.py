"""Independently verify full-run scores and write Chinese statistics, without HTML."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def independent_stats(rows, reviewed):
    counts = {str(n): 0 for n in range(6)}
    pending = failed = 0
    for row in rows:
        score = row["effective"]["score"] if reviewed else (row["judgment"] or {}).get("score")
        if score is None:
            status = row["effective"]["status"] if reviewed else row["status"]
            if status == "api_failed":
                failed += 1
            else:
                pending += 1
        else:
            assert type(score) is int and 0 <= score <= 5
            counts[str(score)] += 1
    n = sum(counts.values())
    total = sum(int(k) * v for k, v in counts.items())
    assert n + pending + failed == len(rows)
    return {"total_items": len(rows), "scored_items": n, "pending_qa_review": pending,
            "api_failed": failed, "score_sum": total, "maximum_score": 5*n,
            "mean_score": total/n if n else None, "normalized_score": total/(5*n) if n else None,
            "coverage": n/len(rows) if rows else None, "score_distribution": counts,
            "score_bands": {"0–1": counts['0']+counts['1'], "2–3": counts['2']+counts['3'],
                            "4–5": counts['4']+counts['5']}}


def verify_subset(rows, summary):
    for field, reviewed in (("raw_judge", False), ("with_human_review", True)):
        expected = independent_stats(rows, reviewed)
        assert all(summary[field][key] == value for key, value in expected.items()), field


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    args = ap.parse_args()
    out = args.output_dir
    source = args.input_root / "omni_review.json"
    document = read(source)
    config = read(out / "configuration.json")
    manifest = read(out / "manifest.json")
    items = read(out / "sample.json")
    results = read(out / "results.json")
    summary = read(out / "summary.json")
    reviews = read(out / "human_review.json")
    assert digest(source) == config["input_sha256"] == manifest["input_sha256"]
    for name in ("sample", "results", "summary"):
        assert digest(out / (name + ".json")) == manifest[name + "_sha256"]
    for name, expected in config["code_sha256"].items():
        assert digest(out / "reproducibility" / name) == expected
    assert digest(out / "human_review.json") == config["feedback_sha256"]
    system = (out / "system_prompt.txt").read_text(encoding="utf-8")
    template = (out / "user_prompt_template.txt").read_text(encoding="utf-8")
    # Prompt hashes describe the LF-normalized text sent in JSON, including on Windows.
    assert hashlib.sha256(system.encode()).hexdigest() == config["system_prompt_sha256"]
    assert hashlib.sha256(template.encode()).hexdigest() == config["user_template_sha256"]
    original = {}
    for video in document["videos"]:
        for method, points in video["methods"].items():
            for point in points:
                for turn in point["turns"]:
                    assert turn["id"] not in original
                    original[turn["id"]] = (video["clip_id"], method, point["id"], turn)
    item_map = {item["id"]: item for item in items}
    result_map = {row["id"]: row for row in results}
    assert set(original) == set(item_map) == set(result_map)
    assert len(original) == len(items) == len(results) == 240
    assert len({item['clip_id'] for item in items}) == 10
    assert Counter(Counter(i['point_id'] for i in items).values()) == {2: 120}
    responses = 0
    for tid, item in item_map.items():
        clip, method, point, turn = original[tid]
        assert (item['clip_id'], item['method'], item['point_id']) == (clip, method, point)
        assert item['question'] == turn['question']
        assert item['reference_answer'] == turn['reference_answer']
        assert item['candidate_answer'] == turn['omni']['response']
        assert item['evidence_cutoff_s'] <= item['question_time_s']
        assert all(e['clip_s'] <= item['evidence_cutoff_s'] for e in item['evidence'])
        row = result_map[tid]
        human = reviews.get(tid)
        assert row['human_review'] == human
        if human is not None:
            assert row['effective']['score'] == human['score']
            assert row['effective']['source'] == 'human_review'
            assert row['effective']['status'] == human['status']
        elif row['status'] == 'ok':
            assert row['effective']['score'] == row['judgment']['score']
            assert row['effective']['source'] == 'judge_v2'
            expected_status = 'pending_qa_review' if row['judgment']['needs_qa_review'] else 'scored'
            assert row['effective']['status'] == expected_status
        else:
            assert row['status'] == row['effective']['status'] == 'api_failed'
            assert row['effective']['score'] is None
        request_path = out / 'requests' / (tid + '.request.json')
        if tid in config['known_pending_ids']:
            assert row['status'] == 'excluded_by_human' and row['judgment']['score'] is None
            assert not request_path.exists()
            continue
        request = read(request_path)
        expected_request = {'model': config['model'], 'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': template.format(question=turn['question'],
                answer=turn['reference_answer'], pred=turn['omni']['response'])}],
            'temperature': 0, 'max_tokens': 900, 'response_format': {'type': 'json_object'}, 'stream': False}
        assert request == expected_request
        for attempt in (out/'requests').glob(tid + '.attempt_*.json'):
            assert read(attempt)['request_sha256'] == digest(request_path)
        if row['status'] == 'api_failed':
            assert row['judgment'] is None
            continue
        assert row['status'] == 'ok'
        response_path = out / row['response_path']
        assert response_path.resolve().parent == (out/'requests').resolve()
        response = read(response_path)
        assert response['http_status'] == 200
        assert response['body']['model'] == row['returned_model'] == config['model']
        assert response['request_sha256'] == row['request_sha256'] == digest(request_path)
        choice = response['body']['choices'][0]
        assert choice['finish_reason'] == 'stop'
        j = json.loads(choice['message']['content'])
        assert j == row['judgment']
        assert set(j) == {'pred', 'score', 'reason', 'needs_qa_review', 'qa_review_reason'}
        assert isinstance(j['reason'], str) and j['reason'].strip()
        assert type(j['needs_qa_review']) is bool
        if j['needs_qa_review']:
            assert j['score'] is None and j['pred'] is None and j['qa_review_reason'].strip()
        else:
            assert type(j['score']) is int and 0 <= j['score'] <= 5
            assert j['pred'] in ('yes', 'no') and j['qa_review_reason'] == ''
        responses += 1
    assert responses == manifest['api_completed']
    assert sum(r['status'] == 'api_failed' for r in results) == manifest['api_failed']
    assert responses + manifest['api_failed'] + len(config['known_pending_ids']) == 240
    verify_subset(results, summary)
    assert manifest['summary'] == summary
    scored = [r for r in results if (r['judgment'] or {}).get('score') is not None]
    yes = sum(r['judgment']['pred'] == 'yes' for r in scored)
    assert summary['raw_judge']['yes'] == yes
    assert summary['raw_judge']['no'] == len(scored) - yes
    assert summary['raw_judge']['yes_rate'] == (yes/len(scored) if scored else None)
    human_scored = [r for r in results if r['human_review'] and r['human_review']['score'] is not None]
    comparable = [r for r in human_scored if (r['judgment'] or {}).get('score') is not None]
    disagreement = [{'id': r['id'], 'judge_score': r['judgment']['score'], 'human_score': r['human_review']['score']}
                    for r in comparable if r['judgment']['score'] != r['human_review']['score']]
    assert summary['calibration']['human_scored_items'] == len(human_scored)
    assert summary['calibration']['comparable_items'] == len(comparable)
    assert summary['calibration']['exact_agreement'] == len(comparable) - len(disagreement)
    assert summary['calibration']['differences'] == disagreement
    for method, stats in summary['by_method'].items():
        verify_subset([r for r in results if item_map[r['id']]['method'] == method], stats)
    for kind, stats in summary['by_kind'].items():
        assert independent_stats([r for r in results if item_map[r['id']]['kind'] == kind], True) == stats
    groups = defaultdict(list)
    for row in results:
        groups[item_map[row['id']]['point_id']].append(row)
    eligible = [g for g in groups.values() if all((r['judgment'] or {}).get('pred') is not None for r in g)]
    yes_groups = sum(all(r['judgment']['pred'] == 'yes' for r in g) for g in eligible)
    assert summary['multi_turn_raw_judge']['eligible_groups'] == len(eligible)
    assert summary['multi_turn_raw_judge']['all_yes_groups'] == yes_groups
    assert summary['multi_turn_raw_judge']['all_yes_rate'] == (yes_groups/len(eligible) if eligible else None)
    # Never output a matching credential; a failure identifies only its file.
    scanned = 0
    for path in out.rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.txt', '.md', '.py'):
            assert not re.search(rb'sk-or-v1-[a-f0-9]{64}', path.read_bytes()), 'Credential in ' + path.name
            scanned += 1
    main_stats = summary['with_human_review']
    raw_stats = summary['raw_judge']
    pending = [dict(item_map[r['id']], review_reason=r['effective']['reason'],
                    automatic_judgment=r['judgment'], human_review=r['human_review'])
               for r in results if r['effective']['status'] == 'pending_qa_review']
    differences = summary['calibration']['differences']
    lines = [
        '# MuseumLive 当前全量评分统计', '',
        '范围：现有 10 段视频、240 条 QA；被评测模型为 MiniCPM-o-4.5。商业 Judge 为 x-ai/grok-4.6，按 0–5 分整体评分。', '',
        '以下主统计纳入你已经确认的 17 条人工复核评分；其余沿用本次自动评分。', '',
        f"- 有效评分：{main_stats['scored_items']} / 240 条。",
        f"- 待复核题目：{main_stats['pending_qa_review']} 条；请求失败：{main_stats['api_failed']} 条。二者均不记为 0 分，也不进入均分分母。",
        f"- 总分：{main_stats['score_sum']} / {main_stats['maximum_score']}。",
        f"- 均分：{main_stats['mean_score']:.4f} / 5；按满分归一化为 {main_stats['normalized_score']:.6f}（{main_stats['normalized_score']:.2%}）。这不是答对率。", '',
        '| 分数段 | 数量 | 占有效评分 |', '|---|---:|---:|',
    ]
    for band, count in main_stats['score_bands'].items():
        lines.append(f"| {band} 分 | {count} | {count/main_stats['scored_items']:.2%} |")
    lines += ['', '| 分数 | 0 | 1 | 2 | 3 | 4 | 5 |', '|---|---:|---:|---:|---:|---:|---:|',
              '| 数量 | ' + ' | '.join(str(main_stats['score_distribution'][str(k)]) for k in range(6)) + ' |', '',
              '## 自动评分留档', '',
              f"未叠加人工分数的统一自动评分：有效 {raw_stats['scored_items']} 条，均分 {raw_stats['mean_score']:.4f}/5，"
              f"总分 {raw_stats['score_sum']}/{raw_stats['maximum_score']}，待复核 {raw_stats['pending_qa_review']} 条，请求失败 {raw_stats['api_failed']} 条。", '',
              '自动评分分数段：' + '；'.join(f'{band} 分 {count} 条' for band, count in raw_stats['score_bands'].items()) + '。', '',
              f"17 条已有人审分数中，可比 {summary['calibration']['comparable_items']} 条；"
              f"本次自动分数与人工分数相同 {summary['calibration']['exact_agreement']} 条，不同 {len(differences)} 条。"
              '这些题用于评分提示词的开发校准，不能当作独立验证集。', '',
              '## 分组统计（纳入人工复核）', '',
              '| 分组 | 有效条数 | 待复核 | 均分 / 5 | 0–1 | 2–3 | 4–5 |', '|---|---:|---:|---:|---:|---:|---:|']
    for method, row in summary['by_method'].items():
        s = row['with_human_review']
        lines.append(f"| {method} | {s['scored_items']} | {s['pending_qa_review']} | {s['mean_score']:.4f} | "
                     + ' | '.join(str(s['score_bands'][b]) for b in ('0–1', '2–3', '4–5')) + ' |')
    lines += ['', 'Full 和 Sparse 的问题不同，这张表不能说明哪种方法在同题条件下更好。', '',
              '## 待复核题目', '']
    for row in pending:
        lines += [f"- {row['id']}：{row['question']} 原因：{row['review_reason']}"]
    lines += ['', '参考答案仍由机器编写，尚未全部人工核实。当前 Judge 对比问题、参考答案与模型回答的文字含义，'
              '没有重新查看视频；这些分数是现有 240 条回答的结果，不能代表更大视频库的最终完整 benchmark，'
              '也不能单独证明反事实或因果推断能力。', '']
    (out / '结果统计.md').write_text('\n'.join(lines), encoding='utf-8')
    write(out / '待复核题目.json', pending)
    write(out / '统计核验.json', {'passed': True, 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
          'items': len(items), 'response_records_verified': responses, 'scanned_files_without_credentials': scanned,
          'input_sha256': digest(source), 'results_sha256': digest(out/'results.json'),
          'summary_sha256': digest(out/'summary.json'), 'report_sha256': digest(out/'结果统计.md'),
          'validator_sha256': digest(Path(__file__)), 'checks': ['source preserved', 'unique 240-item coverage',
          'frozen prompts and payloads', 'response model and parsed scores', 'human layer separated',
          'pending and failures excluded from means', 'independent arithmetic and subgroups',
          'two-turn denominators', 'credential scan']})
    print(json.dumps({'verified': True, 'with_human_review': main_stats, 'raw_judge': raw_stats,
                      'calibration': summary['calibration'], 'pending_ids': [r['id'] for r in pending]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
