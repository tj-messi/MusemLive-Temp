"""Small, holistic 0-5 semantic Judge pilot adapted from OmniMMI.

Use the unmodified saved questions, references and predictions. Sampling is
fixed before any API calls; no old rubric, score, model name or image is sent.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time

import requests

VERSION = "museumlive_omnimmi_holistic_pilot_v1"
MODEL = "x-ai/grok-4.6"
SOURCE_URL = "https://github.com/OmniMMI/OmniMMI/blob/2b93f6ad63126581769151df635b2234f25c2c33/evaluations/evaluate.py"
SOURCE_SHA = "390028e506ea4f383322853b073b5469310ab8d0887b7c2aa3861a6974b0a12c"
SEED = 20260927
# One known concern plus nine coverage samples, not a population estimate.
STRATA = [
    ("N004_K002", "sparse", "current_view", "N004_K002_sparse_P03_T1"),
    ("N004_K001", "full", "temporal_order", None),
    ("N004_K003", "full", "identity_boundary", None),
    ("N004_K004", "sparse", "follow_up", None),
    ("N004_K005", "full", "long_memory", None),
    ("N004_K006", "sparse", "ocr", None),
    ("N005_K004", "sparse", "uncertainty", None),
    ("N006_K001", "full", "ocr", None),
    ("N006_K002", "full", "revisit", None),
    ("N006_K003", "sparse", "current_view", None),
]
USER_TEMPLATE = """Please evaluate the following video-based question-answer pair:

Question: {question}
Correct Answer: {answer}
Predicted Answer: {pred}

Provide your evaluation as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match.
Return only a JSON object with keys 'pred', 'score', and 'reason'. The value of 'pred' must be 'yes' or 'no'; 'score' must be an INTEGER, not a string or a fraction. The 'reason' must briefly explain your judgment in Chinese.
Do not provide a code block or text outside the JSON object.
Example of output format: {{"pred": "yes", "score": 5, "reason": "回答与参考答案的意思一致。"}}
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encoded(obj):
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def save(path, obj):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(encoded(obj))
    temp.replace(path)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def official_system(source):
    if sha(source.read_bytes()) != SOURCE_SHA:
        raise ValueError("Pinned official source hash mismatch")
    tree = ast.parse(source.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "openai_eval")
    assignment = next(n for n in fn.body if isinstance(n, ast.Assign) and
                      any(isinstance(t, ast.Name) and t.id == "messages" for t in n.targets))
    return ast.literal_eval(assignment.value.elts[0])["content"]


def select_items(document, selected_ids=None):
    # The default pilot is answer-blind. Explicit selections carry their own
    # sampling provenance; they may intentionally target prior failure cases.
    candidates = []
    for video in document["videos"]:
        for method, points in video["methods"].items():
            for point in points:
                for turn in point["turns"]:
                    candidates.append({"id": turn["id"], "clip_id": video["clip_id"],
                                       "method": method, "kind": turn["kind"]})
    if len({x["id"] for x in candidates}) != len(candidates):
        raise ValueError("Duplicate dataset ids")
    if selected_ids is None:
        rng = random.Random(SEED)
        selected = []
        for clip, method, kind, fixed_id in STRATA:
            pool = sorted([x for x in candidates if x["clip_id"] == clip and
                           x["method"] == method and x["kind"] == kind], key=lambda x: x["id"])
            selected.append(next(x for x in pool if x["id"] == fixed_id) if fixed_id else rng.choice(pool))
    else:
        if not isinstance(selected_ids, list) or not 1 <= len(selected_ids) <= 20 or any(
                not isinstance(i, str) for i in selected_ids) or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Explicit selection requires 1-20 unique string ids")
        by_id = {x["id"]: x for x in candidates}
        if not set(selected_ids).issubset(by_id):
            raise ValueError("Unknown selected ids")
        selected = [by_id[tid] for tid in selected_ids]
    ids = {x["id"] for x in selected}
    full_items = {}
    for video in document["videos"]:
        for method, points in video["methods"].items():
            history = []
            for point in points:
                for turn in point["turns"]:
                    if turn["id"] in ids:
                        answer = turn["omni"].get("response")
                        if not isinstance(answer, str) or not answer or turn["omni"].get("error"):
                            raise ValueError("Missing saved prediction: " + turn["id"])
                        if turn["evidence_cutoff_s"] > turn["question_time_s"] or any(
                                e["clip_s"] > turn["evidence_cutoff_s"] for e in turn["evidence"]):
                            raise ValueError("Future evidence metadata: " + turn["id"])
                        if any(h["question_time_s"] > turn["question_time_s"] for h in history):
                            raise ValueError("Future dialogue metadata")
                        full_items[turn["id"]] = {
                            "id": turn["id"], "clip_id": video["clip_id"], "method": method,
                            "kind": turn["kind"], "point_id": point["id"],
                            "question": turn["question"], "reference_answer": turn["reference_answer"],
                            "candidate_answer": answer, "question_time_s": turn["question_time_s"],
                            "evidence_cutoff_s": turn["evidence_cutoff_s"], "evidence": turn["evidence"],
                            "prior_dialogue_for_human_only": list(history),
                            "reference_status": "machine_authored_pending_human_review",
                        }
                    history.append({"question": turn["question"],
                                    "answer": turn["omni"].get("response", ""),
                                    "question_time_s": turn["question_time_s"]})
    return [full_items[x["id"]] for x in selected]


def payload_for(item, system):
    return {"model": MODEL, "messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(
            question=item["question"], answer=item["reference_answer"], pred=item["candidate_answer"])}],
        "temperature": 0, "max_tokens": 700, "response_format": {"type": "json_object"},
        "stream": False}


def validate(value):
    if not isinstance(value, dict) or set(value) != {"pred", "score", "reason"}:
        raise ValueError("Expected pred, score, reason")
    if value["pred"] not in ("yes", "no"):
        raise ValueError("Invalid yes/no decision")
    if type(value["score"]) is not int or not 0 <= value["score"] <= 5:
        raise ValueError("Score must be an integer from 0 to 5")
    if not isinstance(value["reason"], str) or not value["reason"].strip():
        raise ValueError("Missing explanation")
    return value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--official-source", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--selection-file", type=Path,
                    help="Optional frozen JSON containing input_sha256, selected_ids and sampling")
    ap.add_argument("--prepare-only", action="store_true")
    args = ap.parse_args()
    system = official_system(args.official_source)
    input_bytes = (args.input_root / "omni_review.json").read_bytes()
    selection = read(args.selection_file) if args.selection_file else None
    if selection is not None and (selection.get("input_sha256") != sha(input_bytes) or
                                  not isinstance(selection.get("sampling"), str) or
                                  not selection["sampling"].strip()):
        raise ValueError("Selection input hash or sampling description is invalid")
    selected = select_items(json.loads(input_bytes), selection["selected_ids"] if selection is not None else None)
    config = {"version": VERSION, "model": MODEL, "input_sha256": sha(input_bytes),
              "source_url": SOURCE_URL, "official_source_sha256": SOURCE_SHA,
              "system_prompt_sha256": sha(system.encode()),
              "user_template_sha256": sha(USER_TEMPLATE.encode()),
              "runner_sha256": sha(Path(__file__).read_bytes()),
              "selection_seed": SEED if selection is None else None,
              "selection_strata": STRATA if selection is None else None,
              "selection_file_sha256": sha(args.selection_file.read_bytes()) if args.selection_file else None,
              "sampling": selection["sampling"] if selection is not None else
                          "one known concern plus nine fixed coverage samples; not representative",
              "selected_ids": [x["id"] for x in selected],
              "question_reference_candidate_unchanged": True,
              "judge_inputs": ["question", "reference_answer", "candidate_answer"],
              "prior_dialogue_sent": False, "images_sent": False, "old_rubric_sent": False,
              "reference_status": "machine_authored_pending_human_review",
              "score_scale": [0, 1, 2, 3, 4, 5], "max_score": 5,
              "prompt_changes": ["JSON instead of Python dictionary",
                                 "integer example 5 replaces contradictory example 4.8",
                                 "brief Chinese reason added for human review"],
              "temperature": 0, "max_tokens": 700, "concurrency": 4, "attempt_limit": 2}
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "configuration.json").exists() and read(out / "configuration.json") != json.loads(encoded(config)):
        raise ValueError("Existing run has different configuration")
    save(out / "configuration.json", config)
    if (out / "sample.json").exists() and (out / "sample.json").read_bytes() != encoded(selected):
        raise ValueError("Frozen sample changed")
    save(out / "sample.json", selected)
    if selection is not None:
        save(out / "selection.json", selection)
    (out / "system_prompt.txt").write_text(system, encoding="utf-8")
    (out / "user_prompt_template.txt").write_text(USER_TEMPLATE, encoding="utf-8")
    audit = out / "requests"
    audit.mkdir(exist_ok=True)
    for item in selected:
        request_path = audit / (item["id"] + ".request.json")
        payload = payload_for(item, system)
        if request_path.exists() and request_path.read_bytes() != encoded(payload):
            raise ValueError("Frozen request changed")
        save(request_path, payload)
    if args.prepare_only:
        print(json.dumps({"status": "prepared", "items": len(selected), "ids": config["selected_ids"]}))
        return 0
    key = os.environ.get("OPENROUTER_KEY1", "")
    if not re.fullmatch(r"sk-or-v1-[a-f0-9]{64}", key):
        raise SystemExit("OPENROUTER_KEY1 missing or malformed")
    proxy = os.environ.get("OPENROUTER_PROXY", "").strip()

    def call(item):
        path = audit / (item["id"] + ".request.json")
        payload = read(path)
        request_hash = sha(path.read_bytes())
        error = "No completed request"
        with requests.Session() as session:
            session.trust_env = False
            if proxy:
                session.proxies = {"http": proxy, "https": proxy}
            session.headers.update({"Authorization": "Bearer " + key})
            for attempt in (1, 2):
                response_path = audit / (item["id"] + f".attempt_{attempt}.json")
                if response_path.exists():
                    record = read(response_path)
                    if record["request_sha256"] != request_hash:
                        raise ValueError("Recorded response/request hash mismatch")
                else:
                    started = time.perf_counter()
                    record = {"request_sha256": request_hash,
                              "started_at_utc": datetime.now(timezone.utc).isoformat()}
                    try:
                        response = session.post("https://openrouter.ai/api/v1/chat/completions", json=payload,
                                                timeout=(10, 90), allow_redirects=False)
                        record["http_status"] = response.status_code
                        record["body"] = response.json()
                    except Exception as exc:
                        record["transport_error"] = str(exc).replace(key, "[REDACTED]")[:500]
                    record["elapsed_s"] = round(time.perf_counter() - started, 3)
                    record = json.loads(encoded(record).replace(key.encode(), b"[REDACTED]"))
                    save(response_path, record)
                try:
                    if record.get("http_status") != 200:
                        raise ValueError("HTTP/transport failure: " + str(record.get("http_status")))
                    body = record["body"]
                    if body.get("model") != MODEL:
                        raise ValueError("Returned model mismatch")
                    choice = body["choices"][0]
                    if choice.get("finish_reason") != "stop":
                        raise ValueError("Incomplete generation")
                    judgment = validate(json.loads(choice["message"]["content"]))
                    return {"id": item["id"], "status": "ok", "judgment": judgment,
                            "returned_model": body["model"], "attempt": attempt,
                            "response_path": str(response_path.relative_to(out)),
                            "request_sha256": request_hash, "usage": body.get("usage")}
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    error = str(exc)
                    if record.get("http_status") in (400, 401, 402, 403, 404):
                        break
        return {"id": item["id"], "status": "failed", "error": error, "judgment": None}

    results = {}
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(call, item) for item in selected]
        for future in as_completed(futures):
            result = future.result()
            results[result["id"]] = result
            save(out / "progress.json", {"completed": len(results), "total": len(selected),
                                         "failed": sum(r["status"] != "ok" for r in results.values())})
            print(json.dumps({"completed": len(results), "total": len(selected),
                              "id": result["id"], "status": result["status"]}), flush=True)
    ordered = [results[item["id"]] for item in selected]
    save(out / "results.json", ordered)
    success = [r["judgment"] for r in ordered if r["status"] == "ok"]
    score_sum = sum(r["score"] for r in success)
    yes = sum(r["pred"] == "yes" for r in success)
    attempts = [read(p) for p in audit.glob("*.attempt_*.json")]
    costs = [(a.get("body", {}).get("usage") or {}).get("cost") for a in attempts]
    manifest = dict(config, created_at_utc=datetime.now(timezone.utc).isoformat(),
                    sample_sha256=sha((out / "sample.json").read_bytes()),
                    results_sha256=sha((out / "results.json").read_bytes()),
                    elapsed_s=round(time.perf_counter() - started, 3),
                    completed_items=len(success), failed_items=len(selected) - len(success),
                    requests_recorded=len(attempts),
                    cost_usd_reported=sum(c for c in costs if isinstance(c, (int, float))),
                    summary={"score_sum": score_sum, "max_score_sum": 5 * len(success),
                             "mean_score": score_sum / len(success) if success else None,
                             "yes": yes, "no": len(success) - yes,
                             "score_distribution": dict(sorted(Counter(r["score"] for r in success).items()))})
    save(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("completed_items", "failed_items", "summary", "cost_usd_reported")}, ensure_ascii=False))
    return 0 if len(success) == len(selected) else 2


if __name__ == "__main__":
    raise SystemExit(main())
