"""Full current benchmark, immutable inputs/prompt, resumable per-item API audit."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time

import requests

from holistic_protocol import VERSION, MODEL, SCOPE_GUIDANCE, USER_TEMPLATE
from run_omnimmi_pilot import sha, encoded, save, read, official_system, SOURCE_URL


def load_all(document):
    items = []
    seen = set()
    for video in document["videos"]:
        for method, points in video["methods"].items():
            history = []
            for point in points:
                for turn in point["turns"]:
                    tid = turn["id"]
                    if tid in seen:
                        raise ValueError("Duplicate id: " + tid)
                    seen.add(tid)
                    prediction = turn.get("omni", {}).get("response")
                    if not isinstance(prediction, str) or not prediction.strip() or turn["omni"].get("error"):
                        raise ValueError("Missing saved prediction: " + tid)
                    if turn["evidence_cutoff_s"] > turn["question_time_s"] or any(
                            e["clip_s"] > turn["evidence_cutoff_s"] for e in turn["evidence"]):
                        raise ValueError("Future evidence metadata: " + tid)
                    if any(h["question_time_s"] > turn["question_time_s"] for h in history):
                        raise ValueError("Future history metadata: " + tid)
                    items.append({"id": tid, "clip_id": video["clip_id"], "method": method,
                        "point_id": point["id"], "source_group": video.get("source_group"),
                        "kind": turn["kind"], "question": turn["question"],
                        "reference_answer": turn["reference_answer"], "candidate_answer": prediction,
                        "question_time_s": turn["question_time_s"],
                        "evidence_cutoff_s": turn["evidence_cutoff_s"], "evidence": turn["evidence"],
                        "prior_dialogue_for_human_only": list(history),
                        "reference_status": "machine_authored_pending_human_review"})
                    history.append({"question": turn["question"], "answer": prediction,
                                    "question_time_s": turn["question_time_s"]})
    return items


def load_feedback(folder, dataset_hash, items):
    summary = read(folder / "评分汇总.json")
    item_map = {i["id"]: i for i in items}
    reviews = {}
    for source in summary["sources"]:
        path = folder / source["feedback_file"]
        if sha(path.read_bytes()) != source["feedback_sha256"] or source["input_sha256"] != dataset_hash:
            raise ValueError("Feedback provenance mismatch")
        feedback = read(path)
        if feedback["input_sha256"] != dataset_hash or feedback["sample_sha256"] != source["sample_sha256"] or feedback["results_sha256"] != source["results_sha256"]:
            raise ValueError("Feedback source mismatch")
        for row in feedback["items"]:
            tid = row["id"]
            if tid in reviews or tid not in item_map:
                raise ValueError("Duplicate/unknown feedback id")
            rev = row["review"]
            suggested = rev["suggested_score"]
            if suggested is not None and (type(suggested) is not int or not 0 <= suggested <= 5):
                raise ValueError("Invalid human score")
            if rev["needs_qa_review"] or rev["rating"] == "qa_issue":
                score, status = None, "pending_qa_review"
            elif suggested is not None:
                score, status = suggested, "reviewed"
            elif rev["rating"] == "appropriate":
                score, status = row["judge_original"]["score"], "reviewed"
            else:
                raise ValueError("Incomplete feedback: " + tid)
            if score is not None and (type(score) is not int or not 0 <= score <= 5):
                raise ValueError("Invalid retained human score")
            reviews[tid] = {"score": score, "status": status, "rating": rev["rating"],
                           "note": rev["note"], "updated_at": rev["updated_at"],
                           "source_feedback_sha256": source["feedback_sha256"]}
    expected = {r["id"]: r for r in summary["items"]}
    if set(reviews) != set(expected):
        raise ValueError("Feedback summary coverage mismatch")
    for tid, r in reviews.items():
        if r["score"] != expected[tid]["effective_reviewed_score"] or item_map[tid]["question"] != expected[tid]["question"]:
            raise ValueError("Feedback summary data mismatch")
    return reviews


def payload_for(item, system):
    return {"model": MODEL, "messages": [{"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(question=item["question"],
                      answer=item["reference_answer"], pred=item["candidate_answer"])}],
        "temperature": 0, "max_tokens": 900, "response_format": {"type": "json_object"}, "stream": False}


def validate_judgment(j):
    if not isinstance(j, dict) or set(j) != {"pred", "score", "reason", "needs_qa_review", "qa_review_reason"}:
        raise ValueError("Invalid judgment fields")
    if not isinstance(j["reason"], str) or not j["reason"].strip():
        raise ValueError("Missing reason")
    if type(j["needs_qa_review"]) is not bool or not isinstance(j["qa_review_reason"], str):
        raise ValueError("Invalid review flag")
    if j["needs_qa_review"]:
        if j["score"] is not None or j["pred"] is not None or not j["qa_review_reason"].strip():
            raise ValueError("Pending QA must have null score/pred and an explanation")
    elif type(j["score"]) is not int or not 0 <= j["score"] <= 5 or j["pred"] not in ("yes", "no") or j["qa_review_reason"]:
        raise ValueError("Invalid scored judgment")
    return j


def effective_result(result, human):
    if human is not None:
        return {"score": human["score"], "status": human["status"], "source": "human_review",
                "reason": human["note"] or ("人工确认沿用原分。" if human["rating"] == "appropriate" else "采用人工复核建议。")}
    if result["status"] == "api_failed":
        return {"score": None, "status": "api_failed", "source": "api_error", "reason": result["error"]}
    j = result["judgment"]
    return {"score": j["score"], "status": "pending_qa_review" if j["needs_qa_review"] else "scored",
            "source": "judge_v2", "reason": j["reason"]}


def metrics(rows, layer):
    if layer == "reviewed":
        statuses = Counter(r["effective"]["status"] for r in rows)
        scores = [r["effective"]["score"] for r in rows if r["effective"]["score"] is not None]
    else:
        statuses = Counter("api_failed" if r["status"] == "api_failed" else
            "pending_qa_review" if r["judgment"]["needs_qa_review"] else "scored" for r in rows)
        scores = [r["judgment"]["score"] for r in rows if r["judgment"] is not None and r["judgment"]["score"] is not None]
    dist = Counter(scores)
    return {"total_items": len(rows), "scored_items": len(scores),
        "pending_qa_review": statuses["pending_qa_review"], "api_failed": statuses["api_failed"],
        "score_sum": sum(scores), "maximum_score": len(scores)*5,
        "mean_score": sum(scores)/len(scores) if scores else None,
        "normalized_score": sum(scores)/(5*len(scores)) if scores else None,
        "coverage": len(scores)/len(rows) if rows else None,
        "score_distribution": {str(n): dist[n] for n in range(6)},
        "score_bands": {"0–1": dist[0]+dist[1], "2–3": dist[2]+dist[3], "4–5": dist[4]+dist[5]}}


def summarize(items, results):
    by_id = {i["id"]: i for i in items}
    raw = metrics(results, "judge")
    scored = [r for r in results if r["judgment"] is not None and r["judgment"]["score"] is not None]
    yes = sum(r["judgment"]["pred"] == "yes" for r in scored)
    raw.update(yes=yes, no=len(scored)-yes, yes_rate=yes/len(scored) if scored else None)
    groups = defaultdict(list)
    for r in results:
        groups[by_id[r["id"]]["point_id"]].append(r)
    eligible = [g for g in groups.values() if all(r["judgment"] is not None and r["judgment"]["pred"] is not None for r in g)]
    all_yes = sum(all(r["judgment"]["pred"] == "yes" for r in g) for g in eligible)
    multi = {"groups": len(groups), "eligible_groups": len(eligible), "unscored_groups": len(groups)-len(eligible),
             "all_yes_groups": all_yes, "all_yes_rate": all_yes/len(eligible) if eligible else None,
             "basis": "raw Judge yes/no; human numeric scores are not converted to yes/no"}
    calibration = [r for r in results if r.get("human_review") and r["human_review"]["score"] is not None]
    comparable = [r for r in calibration if r["judgment"] is not None and r["judgment"]["score"] is not None]
    diffs = [{"id": r["id"], "judge_score": r["judgment"]["score"], "human_score": r["human_review"]["score"]}
             for r in comparable if r["judgment"]["score"] != r["human_review"]["score"]]
    return {"raw_judge": raw, "with_human_review": metrics(results, "reviewed"),
        "by_method": {m: {"raw_judge": metrics([r for r in results if by_id[r["id"]]["method"] == m], "judge"),
                          "with_human_review": metrics([r for r in results if by_id[r["id"]]["method"] == m], "reviewed")}
                       for m in sorted({i["method"] for i in items})},
        "by_kind": {k: metrics([r for r in results if by_id[r["id"]]["kind"] == k], "reviewed")
                    for k in sorted({i["kind"] for i in items})},
        "multi_turn_raw_judge": multi,
        "calibration": {"human_scored_items": len(calibration), "comparable_items": len(comparable),
                        "exact_agreement": len(comparable)-len(diffs), "differences": diffs,
                        "interpretation": "development/calibration examples, not independent validation"}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", required=True, type=Path)
    ap.add_argument("--official-source", required=True, type=Path)
    ap.add_argument("--feedback-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--concurrency", type=int, default=6)
    ap.add_argument("--attempts", type=int, default=3)
    args = ap.parse_args()
    if not 1 <= args.concurrency <= 12 or not 1 <= args.attempts <= 4:
        ap.error("Invalid concurrency/attempts")
    raw_input = (args.input_root / "omni_review.json").read_bytes()
    items = load_all(json.loads(raw_input))
    if len(items) != 240:
        raise ValueError("This frozen run expects the current 240-turn benchmark")
    reviews = load_feedback(args.feedback_dir, sha(raw_input), items)
    system = official_system(args.official_source) + SCOPE_GUIDANCE
    sources = [Path(__file__), Path(__file__).with_name("holistic_protocol.py"), Path(__file__).with_name("run_omnimmi_pilot.py")]
    config = {"version": VERSION, "model": MODEL, "input_sha256": sha(raw_input),
        "system_prompt_sha256": sha(system.encode()), "user_template_sha256": sha(USER_TEMPLATE.encode()),
        "official_source_url": SOURCE_URL, "code_sha256": {p.name: sha(p.read_bytes()) for p in sources},
        "feedback_summary_sha256": sha((args.feedback_dir / "评分汇总.json").read_bytes()),
        "feedback_sha256": sha(encoded(reviews)), "items": len(items), "clips": len({i["clip_id"] for i in items}),
        "human_review_count": len(reviews), "known_pending_ids": [tid for tid, r in reviews.items() if r["score"] is None],
        "judge_inputs": ["question", "reference_answer", "candidate_answer"],
        "temperature": 0, "max_tokens": 900, "concurrency": args.concurrency, "attempt_limit": args.attempts,
        "score_scale": [0,1,2,3,4,5], "pending_score": None,
        "references": "machine_authored_pending_human_review", "judge_modality": "text_only",
        "human_scores_sent_to_judge": False, "old_scores_or_rubric_sent": False,
        "question_reference_candidate_unchanged": True,
        "scope": "all 240 saved QA turns across the current ten clips; not the larger video corpus"}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "configuration.json").exists() and read(out / "configuration.json") != json.loads(encoded(config)):
        raise ValueError("Resume requires identical frozen configuration")
    if (out / "sample.json").exists() and (out / "sample.json").read_bytes() != encoded(items):
        raise ValueError("Frozen input changed")
    save(out / "configuration.json", config)
    save(out / "sample.json", items)
    save(out / "human_review.json", reviews)
    (out / "system_prompt.txt").write_text(system, encoding="utf-8")
    (out / "user_prompt_template.txt").write_text(USER_TEMPLATE, encoding="utf-8")
    audit = out / "requests"
    audit.mkdir(exist_ok=True)
    excluded = {}
    for item in items:
        h = reviews.get(item["id"])
        if h and h["score"] is None:
            excluded[item["id"]] = {"id": item["id"], "status": "excluded_by_human",
                "judgment": {"pred": None, "score": None, "needs_qa_review": True,
                    "reason": "用户已标记题目表述不清，待复核。", "qa_review_reason": h["note"]}}
            continue
        request = payload_for(item, system)
        path = audit / (item["id"] + ".request.json")
        if path.exists() and path.read_bytes() != encoded(request):
            raise ValueError("Frozen request changed")
        save(path, request)
    if args.prepare_only:
        print(json.dumps({"items": len(items), "api_items": len(items)-len(excluded), "known_pending": len(excluded)}))
        return 0
    key = os.environ.get("OPENROUTER_KEY1", "")
    if not re.fullmatch(r"sk-or-v1-[a-f0-9]{64}", key):
        raise SystemExit("OPENROUTER_KEY1 missing or malformed")
    proxy = os.environ.get("OPENROUTER_PROXY", "").strip()
    if (out / "manifest.json").exists():
        existing = read(out / "manifest.json")
        if existing.get("api_failed") == 0 and existing.get("processed_items") == len(items):
            if sha((out / "results.json").read_bytes()) != existing["results_sha256"]:
                raise ValueError("Completed results changed")
            print("Completed immutable run already exists; no requests sent.")
            return 0

    def evaluate(item):
        tid = item["id"]
        path = audit / (tid + ".request.json")
        request = read(path)
        request_hash = sha(path.read_bytes())
        last_error = "No valid response"
        for attempt in range(1, args.attempts+1):
            destination = audit / (tid + f".attempt_{attempt}.json")
            if destination.exists():
                record = read(destination)
                if record["request_sha256"] != request_hash:
                    raise ValueError("Request/response hash mismatch")
            else:
                start = time.perf_counter()
                record = {"request_sha256": request_hash, "started_at_utc": datetime.now(timezone.utc).isoformat()}
                # A new connection per attempt helps recover proxy resets.
                with requests.Session() as session:
                    session.trust_env = False
                    if proxy:
                        session.proxies = {"http": proxy, "https": proxy}
                    session.headers.update({"Authorization": "Bearer " + key})
                    try:
                        response = session.post("https://openrouter.ai/api/v1/chat/completions", json=request,
                                                timeout=(10, 70), allow_redirects=False)
                        record["http_status"] = response.status_code
                        record["body"] = response.json()
                    except Exception as exc:
                        record["transport_error"] = str(exc).replace(key, "[REDACTED]")[:500]
                record["elapsed_s"] = round(time.perf_counter()-start, 3)
                record = json.loads(encoded(record).replace(key.encode(), b"[REDACTED]"))
                save(destination, record)
            try:
                if record.get("http_status") != 200:
                    raise ValueError("HTTP/transport failure: " + str(record.get("http_status")))
                body = record["body"]
                if body.get("model") != MODEL:
                    raise ValueError("Model mismatch")
                choice = body["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError("Incomplete response")
                judgment = validate_judgment(json.loads(choice["message"]["content"]))
                return {"id": tid, "status": "ok", "judgment": judgment, "attempt": attempt,
                    "response_path": str(destination.relative_to(out)), "request_sha256": request_hash,
                    "returned_model": body["model"], "usage": body.get("usage")}
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                last_error = str(exc)
                if record.get("http_status") in (400,401,402,403,404):
                    break
        return {"id": tid, "status": "api_failed", "judgment": None, "error": last_error}

    started = time.perf_counter()
    results = dict(excluded)
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Prior reviewed cases run first, but their labels are not in the payload.
        runnable = sorted([i for i in items if i["id"] not in excluded], key=lambda i: i["id"] not in reviews)
        futures = [pool.submit(evaluate, item) for item in runnable]
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            progress = {"processed": len(results), "total": len(items),
                        "api_failed": sum(r["status"] == "api_failed" for r in results.values()),
                        "pending_qa_review": sum(r["judgment"] is not None and r["judgment"]["needs_qa_review"] for r in results.values())}
            save(out / "progress.json", progress)
            if len(results) % 10 == 0 or result["status"] == "api_failed":
                print(json.dumps(progress), flush=True)
    ordered = [results[i["id"]] for i in items]
    for r in ordered:
        h = reviews.get(r["id"])
        r["human_review"] = h
        r["effective"] = effective_result(r, h)
    save(out / "results.json", ordered)
    summary = summarize(items, ordered)
    save(out / "summary.json", summary)
    attempts = [read(p) for p in audit.glob("*.attempt_*.json")]
    costs = [(a.get("body", {}).get("usage") or {}).get("cost") for a in attempts]
    manifest = dict(config, created_at_utc=datetime.now(timezone.utc).isoformat(),
        sample_sha256=sha((out / "sample.json").read_bytes()), results_sha256=sha((out / "results.json").read_bytes()),
        summary_sha256=sha((out / "summary.json").read_bytes()), processed_items=len(ordered),
        api_completed=sum(r["status"] == "ok" for r in ordered),
        api_failed=sum(r["status"] == "api_failed" for r in ordered),
        excluded_by_human=len(excluded), attempts_recorded=len(attempts),
        failed_attempts=sum(a.get("http_status") != 200 for a in attempts),
        reported_cost_usd=sum(c for c in costs if isinstance(c, (int,float))),
        cost_note="Only provider-returned usage is included; disconnected calls may have unknown charges.",
        elapsed_s=round(time.perf_counter()-started, 3), summary=summary)
    save(out / "manifest.json", manifest)
    print(json.dumps({"processed": len(ordered), "api_failed": manifest["api_failed"],
                       "summary": summary["with_human_review"], "cost_usd": manifest["reported_cost_usd"]}, ensure_ascii=False))
    return int(manifest["api_failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
