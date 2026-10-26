#!/usr/bin/env python3
"""Build the anonymized Human Baseline and Preference release artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from statistics import fmean, pstdev
from typing import Any, Iterable


BASELINE_PROFILES = ("B01", "B02", "B03")
PREFERENCE_PROFILES = ("P01", "P02", "P03", "P04", "P05")
CANDIDATE_IDS = (
    "moss",
    "minicpm",
    "gander",
    "venus",
    "human_b01",
    "human_b02",
    "human_b03",
)
POSITIONS = tuple("ABCDEFG")
CANDIDATE_LABELS = {
    "moss": "MOSS",
    "minicpm": "MiniCPM",
    "gander": "Gander",
    "venus": "Venus",
    "human_b01": "Human B01",
    "human_b02": "Human B02",
    "human_b03": "Human B03",
}
MODEL_ENTITY_IDS = ("moss", "minicpm", "gander", "venus")
JUDGE_ENTITY_IDS = ("human_mean", *MODEL_ENTITY_IDS)
JUDGE_ENTITY_LABELS = {
    "human_mean": "Assisted Human Baseline",
    "moss": "MOSS",
    "minicpm": "MiniCPM",
    "gander": "Gander",
    "venus": "Venus",
}
JUDGE_ENTITY_CONDITIONS = {
    "human_mean": "assisted_baseline_with_ground_truth_visible",
    "moss": "ai_judged_saved_model_response",
    "minicpm": "ai_judged_saved_model_response",
    "gander": "ai_judged_saved_model_response",
    "venus": "ai_judged_saved_model_response",
}
SCOPE_NAMES = ("all", "wait", "non_wait", "speak", "clarify")
EXPECTED_ACTIONS = {"SPEAK", "CLARIFY", "WAIT"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_fingerprint(candidate: dict[str, Any]) -> str:
    response_time = candidate.get("response_time_s")
    if isinstance(response_time, (int, float)) and math.isfinite(float(response_time)):
        response_time = f"{float(response_time):.3f}"
    else:
        response_time = None
    semantic_content = json.dumps(
        [
            str(candidate.get("text") or "").strip(),
            str(candidate.get("observed") or "silent").strip().lower(),
            str(candidate.get("action") or "").strip().upper(),
            response_time,
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("\\u2028", "\u2028").replace("\\u2029", "\u2029")
    value = 0xCBF29CE484222325
    for character in semantic_content:
        codepoint = ord(character)
        code_units = (codepoint,) if codepoint <= 0xFFFF else (
            0xD800 + ((codepoint - 0x10000) >> 10),
            0xDC00 + ((codepoint - 0x10000) & 0x3FF),
        )
        for code_unit in code_units:
            value ^= code_unit
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"v1:{value:016x}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def task_index(annotation: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for task in annotation.get("tasks") or []:
        for branch in task.get("branches") or []:
            for slot in branch.get("slots") or []:
                uid = str(slot.get("uid") or "")
                require(bool(uid), f"slot without uid in task {task.get('id')}")
                require(uid not in result, f"duplicate slot uid: {uid}")
                require(
                    slot.get("expectedAction") in EXPECTED_ACTIONS,
                    f"invalid expected action: {uid}",
                )
                result[uid] = {"task": task, "branch": branch, "slot": slot}
    require(len(result) == 125, f"expected 125 slots, got {len(result)}")
    require(
        len({item["task"]["id"] for item in result.values()}) == 45,
        "expected 45 tasks",
    )
    return result


def submitted_records(
    records: dict[str, Any], profiles: tuple[str, ...], stage: str
) -> list[dict[str, Any]]:
    rows = []
    for profile in profiles:
        stage_data = ((records.get("profiles") or {}).get(profile) or {}).get(stage) or {}
        for uid, record in stage_data.items():
            require(record.get("status") == "submitted", f"{profile}/{uid} is not submitted")
            rows.append({"profile": profile, "uid": uid, "record": record})
    return rows


def candidate_for(
    item: dict[str, Any], candidate_id: str, baseline_by_key: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any]:
    if candidate_id.startswith("human_"):
        profile = {
            "human_b01": "B01",
            "human_b02": "B02",
            "human_b03": "B03",
        }[candidate_id]
        record = baseline_by_key[(item["slot"]["uid"], profile)]["record"]
        candidate = {
            "id": candidate_id,
            "label": CANDIDATE_LABELS[candidate_id],
            "kind": "human_baseline",
            "source_profile": profile,
            "text": record.get("answer") or "",
            "observed": (
                "clarify" if record.get("action") == "CLARIFY"
                else "silent" if record.get("action") == "WAIT"
                else "answer"
            ),
            "action": record.get("action"),
            "response_time_s": None,
        }
        candidate["current_fingerprint"] = candidate_fingerprint(candidate)
        return candidate
    raw = (item["slot"].get("candidates") or {}).get(candidate_id)
    require(isinstance(raw, dict), f"missing model candidate {candidate_id}/{item['slot']['uid']}")
    candidate = {
        "id": candidate_id,
        "label": CANDIDATE_LABELS[candidate_id],
        "kind": "model",
        "source_profile": None,
        "text": raw.get("text") or "",
        "observed": raw.get("observed") or "silent",
        "action": raw.get("action"),
        "response_time_s": raw.get("responseTimeS"),
        "origin": raw.get("origin"),
    }
    candidate["current_fingerprint"] = candidate_fingerprint(candidate)
    return candidate


def stats(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    return {
        "n": len(materialized),
        "mean": round(fmean(materialized), 4) if materialized else None,
        "sd": round(pstdev(materialized), 4) if materialized else None,
        "exact_zero": sum(value == 0 for value in materialized),
        "exact_five": sum(value == 5 for value in materialized),
    }


def score_scope(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "all": stats(row["score"] for row in rows),
        "wait": stats(row["score"] for row in rows if row["expected_action"] == "WAIT"),
        "non_wait": stats(row["score"] for row in rows if row["expected_action"] != "WAIT"),
        "speak": stats(row["score"] for row in rows if row["expected_action"] == "SPEAK"),
        "clarify": stats(row["score"] for row in rows if row["expected_action"] == "CLARIFY"),
    }


def task_balanced_mean(rows: list[dict[str, Any]]) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["task_id"]].append(float(row["score"]))
    return round(fmean(fmean(values) for values in groups.values()), 4) if groups else None


def build_items(
    slots: dict[str, dict[str, Any]],
    baseline_by_key: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    items = []
    for uid, item in sorted(slots.items()):
        task = item["task"]
        branch = item["branch"]
        slot = item["slot"]
        candidates = {
            candidate_id: candidate_for(item, candidate_id, baseline_by_key)
            for candidate_id in CANDIDATE_IDS
        }
        items.append(
            {
                "uid": uid,
                "task_id": task["id"],
                "source_id": task.get("sourceId"),
                "capability": task.get("capability"),
                "interaction_family": task.get("interactionFamily"),
                "language": task.get("language"),
                "branch_id": branch["id"],
                "slot_id": slot["id"],
                "trigger": slot.get("trigger"),
                "window": slot.get("window"),
                "expected_action": slot.get("expectedAction"),
                "ground_truth": {
                    "target_text": slot.get("targetText"),
                    "key_attributes": slot.get("keyAttributes") or [],
                    "forbidden_claims": slot.get("forbiddenClaims") or [],
                    "audio_refs": slot.get("audioRefs") or [],
                    "modality_facts": slot.get("modalityFacts") or {},
                },
                "user_events": branch.get("userEvents") or [],
                "candidates": candidates,
            }
        )
    return items


def build_baseline_rows(
    slots: dict[str, dict[str, Any]], baseline: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for entry in baseline:
        uid = entry["uid"]
        item = slots[uid]
        task = item["task"]
        branch = item["branch"]
        slot = item["slot"]
        record = entry["record"]
        rows.append(
            {
                "profile": entry["profile"],
                "uid": uid,
                "task_id": task["id"],
                "source_id": task.get("sourceId"),
                "capability": task.get("capability"),
                "branch_id": branch["id"],
                "slot_id": slot["id"],
                "expected_action": slot.get("expectedAction"),
                "submitted_action": record.get("action"),
                "answer": record.get("answer") or "",
                "note": record.get("note") or "",
                "flagged": record.get("flagged"),
                "submitted_at": record.get("submittedAt"),
                "elapsed_ms": record.get("elapsedMs"),
                "collection_condition": "assisted_with_ground_truth_visible",
                "ground_truth_target": slot.get("targetText"),
                "ground_truth_points": slot.get("keyAttributes") or [],
                "forbidden_claims": slot.get("forbiddenClaims") or [],
            }
        )
    return sorted(rows, key=lambda row: (row["profile"], row["task_id"], row["uid"]))


def build_preference_rows(
    slots: dict[str, dict[str, Any]],
    preference: list[dict[str, Any]],
    baseline_by_key: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = []
    assignments = []
    for entry in preference:
        uid = entry["uid"]
        item = slots[uid]
        task = item["task"]
        branch = item["branch"]
        slot = item["slot"]
        record = entry["record"]
        order = record.get("candidateOrder") or []
        scores = record.get("scores") or {}
        fingerprints = record.get("candidateFingerprints") or {}
        require(len(order) == 7 and set(order) == set(CANDIDATE_IDS), f"bad candidate order: {uid}")
        require(set(scores) == set(CANDIDATE_IDS), f"bad preference scores: {uid}")
        require(set(fingerprints) == set(CANDIDATE_IDS), f"bad fingerprints: {uid}")
        mapping = {POSITIONS[index]: candidate_id for index, candidate_id in enumerate(order)}
        row = {
            "profile": entry["profile"],
            "uid": uid,
            "task_id": task["id"],
            "source_id": task.get("sourceId"),
            "capability": task.get("capability"),
            "branch_id": branch["id"],
            "slot_id": slot["id"],
            "expected_action": slot.get("expectedAction"),
            "candidate_order": order,
            "position_mapping": mapping,
            "scores": scores,
            "candidate_fingerprints": fingerprints,
            "note": record.get("note") or "",
            "flagged": record.get("flagged"),
            "submitted_at": record.get("submittedAt"),
            "elapsed_ms": record.get("elapsedMs"),
            "score_provenance": "batch_calibrated_pseudo_rating",
            "analysis_eligible": False,
        }
        records.append(row)
        for position, candidate_id in mapping.items():
            candidate = candidate_for(item, candidate_id, baseline_by_key)
            assignments.append(
                {
                    **{key: row[key] for key in ("profile", "uid", "task_id", "source_id", "capability", "expected_action", "note", "flagged", "submitted_at", "elapsed_ms")},
                    "position": position,
                    "candidate_id": candidate_id,
                    "candidate_label": CANDIDATE_LABELS[candidate_id],
                    "score": scores[candidate_id],
                    "candidate_fingerprint": fingerprints[candidate_id],
                    "candidate_current_fingerprint": candidate["current_fingerprint"],
                    "fingerprint_matches_current_candidate": fingerprints[candidate_id] == candidate["current_fingerprint"],
                    "candidate_text": candidate["text"],
                    "candidate_observed": candidate["observed"],
                    "candidate_action": candidate["action"],
                    "candidate_response_time_s": candidate["response_time_s"],
                    "score_provenance": "batch_calibrated_pseudo_rating",
                    "analysis_eligible": False,
                    "analysis_blocker": (
                        "model_candidate_fingerprint_mismatch"
                        if not candidate_id.startswith("human_")
                        and fingerprints[candidate_id] != candidate["current_fingerprint"]
                        else "batch_generated_preference_scores"
                    ),
                }
            )
    records.sort(key=lambda row: (row["profile"], row["task_id"], row["uid"]))
    assignments.sort(key=lambda row: (row["profile"], row["task_id"], row["uid"], row["position"]))
    return records, assignments


def preference_summary(
    records: list[dict[str, Any]], slots: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        item = slots[record["uid"]]
        for candidate_id, score in record["scores"].items():
            by_target[candidate_id].append(
                {
                    "task_id": record["task_id"],
                    "expected_action": item["slot"]["expectedAction"],
                    "score": score,
                }
            )
    rows = []
    details = {}
    for candidate_id in CANDIDATE_IDS:
        target_rows = by_target[candidate_id]
        scopes = score_scope(target_rows)
        details[candidate_id] = {
            "label": CANDIDATE_LABELS[candidate_id],
            **scopes,
            "task_balanced_mean": task_balanced_mean(target_rows),
        }
        for scope_name, scope in scopes.items():
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_label": CANDIDATE_LABELS[candidate_id],
                    "scope": scope_name,
                    "n": scope["n"],
                    "mean": scope["mean"],
                    "sd": scope["sd"],
                    "exact_zero": scope["exact_zero"],
                    "exact_five": scope["exact_five"],
                    "task_balanced_mean": details[candidate_id]["task_balanced_mean"] if scope_name == "all" else "",
                    "score_provenance": "batch_calibrated_pseudo_rating",
                    "analysis_eligible": False,
                }
            )
    return rows, details


def baseline_score_rows(judge_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = judge_payload.get("scores")
    require(isinstance(rows, list) and len(rows) == 1875, "expected 1,875 Baseline judge scores")
    require(all(row.get("status") == "ok" for row in rows), "Baseline judge scores contain failures")
    return rows


def model_score_rows(judge_payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = judge_payload.get("scores")
    require(isinstance(rows, list) and len(rows) == 2500, "expected 2,500 model judge scores")
    require(all(row.get("status") == "ok" for row in rows), "model judge scores contain failures")
    require(
        Counter(row.get("judge_key") for row in rows)
        == Counter({"qwen_primary": 500, "hy3": 500, "deepseek": 500, "glm": 500, "kimi": 500}),
        "unexpected model judge coverage",
    )
    require(
        Counter(row.get("profile") for row in rows)
        == Counter({profile: 625 for profile in MODEL_ENTITY_IDS}),
        "unexpected model profile coverage",
    )
    require(
        len({row.get("score_id") for row in rows}) == 2500,
        "duplicate model judge score_id",
    )
    for row in rows:
        score = row.get("score")
        require(isinstance(score, (int, float)) and 0 <= float(score) <= 5, f"invalid model judge score: {row.get('score_id')}")
        require(row.get("expected_action") in EXPECTED_ACTIONS, f"invalid expected action: {row.get('score_id')}")
    return rows


def scope_aggregate(points: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    scope_points = [
        point
        for point in points
        if scope == "all"
        or (scope == "wait" and point["expected_action"] == "WAIT")
        or (scope == "non_wait" and point["expected_action"] != "WAIT")
        or (scope == "speak" and point["expected_action"] == "SPEAK")
        or (scope == "clarify" and point["expected_action"] == "CLARIFY")
    ]
    result = stats(point["score"] for point in scope_points)
    if scope == "all":
        result["task_balanced_mean"] = task_balanced_mean(scope_points)
    else:
        result["task_balanced_mean"] = None
    return result


def judge_scope_analysis_rows(
    model_scores: list[dict[str, Any]],
    model_records: list[dict[str, Any]],
    baseline_scores: list[dict[str, Any]],
    baseline_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    require(len(model_records) == 500, "expected 500 model record summaries")
    require(all(row.get("status") == "complete" and row.get("n_ok") == 5 for row in model_records), "incomplete model record summary")
    require(len(baseline_records) == 375, "expected 375 Baseline record summaries")

    points_by_group: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    metadata: dict[tuple[str, str, str], dict[str, str]] = {}

    def add_point(entity: str, aggregation: str, judge_key: str, row: dict[str, Any], score: float) -> None:
        group_key = (entity, aggregation, judge_key)
        expected_action = str(row.get("expected_action") or "")
        task_id = str(row.get("task_id") or "")
        require(expected_action in EXPECTED_ACTIONS, f"invalid expected action in judge analysis: {row.get('record_id')}")
        require(bool(task_id), f"missing task id in judge analysis: {row.get('record_id')}")
        metadata[group_key] = {
            "expected_action": expected_action,
            "task_id": task_id,
            "capability": str(row.get("capability") or ""),
        }
        points_by_group[group_key].append({"score": float(score), **metadata[group_key]})

    for row in model_scores:
        add_point(
            str(row["profile"]),
            "judge",
            str(row["judge_key"]),
            row,
            float(row["score"]),
        )
    for row in model_records:
        add_point(
            str(row["profile"]),
            "consensus",
            "consensus",
            row,
            float(row["consensus_mean"]),
        )

    baseline_by_judge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_scores:
        baseline_by_judge[(str(row["uid"]), str(row["judge_key"]))].append(row)
    for (uid, judge_key), rows in sorted(baseline_by_judge.items()):
        require(len(rows) == 3, f"expected three Baseline profiles for {uid}/{judge_key}")
        collapsed = dict(rows[0])
        add_point(
            "human_mean",
            "judge",
            judge_key,
            collapsed,
            fmean(float(row["score"]) for row in rows),
        )

    baseline_by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_records:
        baseline_by_uid[str(row["uid"])].append(row)
    for uid, rows in sorted(baseline_by_uid.items()):
        require(len(rows) == 3, f"expected three Baseline records for {uid}")
        collapsed = dict(rows[0])
        add_point(
            "human_mean",
            "consensus",
            "consensus",
            collapsed,
            fmean(float(row["consensus_mean"]) for row in rows),
        )

    judge_metadata = {
        str(row["judge_key"]): {
            "judge_family": str(row["judge_family"]),
            "model": str(row["model"]),
        }
        for row in model_scores
    }
    require(len(judge_metadata) == 5, "unexpected model judge set")

    output: list[dict[str, Any]] = []
    for entity in JUDGE_ENTITY_IDS:
        for aggregation in ("consensus", "judge"):
            judge_keys = ("consensus",) if aggregation == "consensus" else tuple(judge_metadata)
            for judge_key in judge_keys:
                points = points_by_group[(entity, aggregation, judge_key)]
                require(len(points) == 125, f"expected 125 analysis points: {entity}/{aggregation}/{judge_key}")
                if aggregation == "judge":
                    details = judge_metadata[judge_key]
                else:
                    details = {"judge_family": "Five-judge consensus", "model": ""}
                for scope in SCOPE_NAMES:
                    aggregate = scope_aggregate(points, scope)
                    output.append(
                        {
                            "entity": entity,
                            "entity_label": JUDGE_ENTITY_LABELS[entity],
                            "analysis_condition": JUDGE_ENTITY_CONDITIONS[entity],
                            "aggregation": aggregation,
                            "judge_key": judge_key,
                            "judge_family": details["judge_family"],
                            "judge_model": details["model"],
                            "scope": scope,
                            "n": aggregate["n"],
                            "mean": aggregate["mean"],
                            "sd": aggregate["sd"],
                            "exact_zero": aggregate["exact_zero"],
                            "exact_five": aggregate["exact_five"],
                            "task_balanced_mean": aggregate["task_balanced_mean"],
                        }
                    )
    require(len(output) == 150, f"expected 150 judge scope rows, got {len(output)}")
    return output


def redact_source_path_fields(value: Any) -> None:
    if isinstance(value, dict):
        for key in list(value):
            child = value[key]
            if key.lower() in {"path", "source_path", "file_path"} and isinstance(child, str):
                value["source_filename"] = PureWindowsPath(child).name
                value.pop(key, None)
            else:
                redact_source_path_fields(child)
    elif isinstance(value, list):
        for child in value:
            redact_source_path_fields(child)


def sanitize_model_source_paths(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(payload))
    redact_source_path_fields(sanitized)
    sanitized["source_paths_redacted"] = True
    sanitized["manifest_stage"] = "source_run_start_snapshot"
    sanitized["completion_authority"] = "model_judge_summary.json"
    return sanitized


def sanitize_model_judge_scores(payload: dict[str, Any]) -> dict[str, Any]:
    sanitized = json.loads(json.dumps(payload))
    metadata = sanitized.get("metadata") or {}
    redact_source_path_fields(metadata)
    metadata["source_paths_redacted"] = True
    sanitized["metadata"] = metadata
    return sanitized


def analyze_model_judge_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    annotated = json.loads(json.dumps(payload))
    entities = annotated.get("entities") or {}
    require(set(entities) == set(JUDGE_ENTITY_IDS), "unexpected model comparison entity set")
    for entity, details in entities.items():
        details["label"] = JUDGE_ENTITY_LABELS[entity]
        details["analysis_condition"] = JUDGE_ENTITY_CONDITIONS[entity]
    annotated["analysis_condition_legend"] = {
        "assisted_baseline_with_ground_truth_visible": "Assisted Human Baseline answers were collected with Ground Truth visible.",
        "ai_judged_saved_model_response": "Saved model responses scored by the five-judge protocol.",
    }
    return annotated


def release_model_comparison_rows(
    rows: list[dict[str, str]],
    scope_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    consensus_scopes = {
        (row["entity"], row["scope"]): row
        for row in scope_rows
        if row["aggregation"] == "consensus"
    }
    output = []
    for row in rows:
        annotated = dict(row)
        entity = annotated.get("entity") or ""
        require(entity in JUDGE_ENTITY_IDS, f"unexpected model comparison entity: {entity}")
        annotated["entity_label"] = JUDGE_ENTITY_LABELS[entity]
        annotated["label"] = JUDGE_ENTITY_LABELS[entity]
        annotated["analysis_condition"] = JUDGE_ENTITY_CONDITIONS[entity]
        capability = annotated.get("capability") or ""
        scope = annotated.get("scope") or ""
        if not capability and scope in SCOPE_NAMES:
            generated = consensus_scopes[(entity, scope)]
            for field in ("n", "mean", "sd", "exact_zero", "exact_five", "task_balanced_mean"):
                annotated[field] = "" if generated.get(field) is None else str(generated.get(field))
        for field in ("mean", "sd", "task_balanced_mean"):
            raw = annotated.get(field)
            if raw not in (None, ""):
                annotated[field] = f"{float(raw):.4f}"
        output.append(annotated)
    return output


def write_model_judge_report(
    path: Path,
    scope_rows: list[dict[str, Any]],
    model_scores: list[dict[str, Any]],
) -> None:
    by_key = {
        (row["entity"], row["aggregation"], row["judge_key"], row["scope"]): row
        for row in scope_rows
    }
    judges = ["qwen_primary", "hy3", "deepseek", "glm", "kimi"]
    judge_labels = {
        "qwen_primary": "Qwen (primary)",
        "hy3": "HY3",
        "deepseek": "DeepSeek",
        "glm": "GLM",
        "kimi": "Kimi",
    }

    def value(entity: str, aggregation: str, judge: str, scope: str, field: str = "mean") -> float | None:
        row = by_key[(entity, aggregation, judge, scope)]
        raw = row.get(field)
        return None if raw in (None, "") else float(raw)

    source_counts = Counter(str(row.get("source") or "unknown") for row in model_scores)
    lines = [
        "# Four-model five-judge WAIT and non-WAIT analysis",
        "",
        "## Coverage",
        "",
        f"- Row-level scores: `{len(model_scores):,}` (4 models x 125 slots x 5 judges).",
        "- Successful judges: `5/5`; failed row-level calls: `0`.",
        f"- Score source mix: `{dict(sorted(source_counts.items()))}`.",
        "- Slot scopes: `125` overall, `15` WAIT, `110` non-WAIT, `103` SPEAK, and `7` CLARIFY.",
        "- Assisted Human Baseline values average B01-B03; model values are saved candidate responses under the same frozen protocol.",
        "",
        "## Five-judge consensus by expected action",
        "",
        "The WAIT/non-WAIT difference is descriptive and must not be interpreted as a causal model effect.",
        "",
        "| Entity | Overall | Task-balanced | WAIT | Non-WAIT | Non-WAIT minus WAIT | SPEAK | CLARIFY |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entity in JUDGE_ENTITY_IDS:
        overall = value(entity, "consensus", "consensus", "all")
        task_balanced = value(entity, "consensus", "consensus", "all", "task_balanced_mean")
        wait = value(entity, "consensus", "consensus", "wait")
        non_wait = value(entity, "consensus", "consensus", "non_wait")
        speak = value(entity, "consensus", "consensus", "speak")
        clarify = value(entity, "consensus", "consensus", "clarify")
        lines.append(
            f"| {JUDGE_ENTITY_LABELS[entity]} | {overall:.4f} | {task_balanced:.4f} | "
            f"{wait:.4f} | {non_wait:.4f} | {non_wait - wait:+.4f} | {speak:.4f} | {clarify:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Per-judge results",
            "",
            "| Entity | Judge | Overall | WAIT | Non-WAIT | Non-WAIT minus WAIT |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for entity in JUDGE_ENTITY_IDS:
        for judge in judges:
            overall = value(entity, "judge", judge, "all")
            wait = value(entity, "judge", judge, "wait")
            non_wait = value(entity, "judge", judge, "non_wait")
            lines.append(
                f"| {JUDGE_ENTITY_LABELS[entity]} | {judge_labels[judge]} | {overall:.4f} | "
                f"{wait:.4f} | {non_wait:.4f} | {non_wait - wait:+.4f} |"
            )

    lines.extend(
        [
            "",
            "## Judge sensitivity",
            "",
            "| Entity | Overall range across judges | WAIT range | Non-WAIT range | Overall spread |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    ranges: dict[str, dict[str, tuple[float, float]]] = {}
    for entity in JUDGE_ENTITY_IDS:
        ranges[entity] = {}
        for scope in ("all", "wait", "non_wait"):
            values = [value(entity, "judge", judge, scope) for judge in judges]
            ranges[entity][scope] = (min(values), max(values))
        minimum, maximum = ranges[entity]["all"]
        wait_minimum, wait_maximum = ranges[entity]["wait"]
        non_wait_minimum, non_wait_maximum = ranges[entity]["non_wait"]
        lines.append(
            f"| {JUDGE_ENTITY_LABELS[entity]} | {minimum:.4f}-{maximum:.4f} | "
            f"{wait_minimum:.4f}-{wait_maximum:.4f} | "
            f"{non_wait_minimum:.4f}-{non_wait_maximum:.4f} | {maximum - minimum:.4f} |"
        )

    human_wait = value("human_mean", "consensus", "consensus", "wait")
    human_non_wait = value("human_mean", "consensus", "consensus", "non_wait")
    model_wait_values = {
        entity: value(entity, "consensus", "consensus", "wait") for entity in MODEL_ENTITY_IDS
    }
    model_non_wait_values = {
        entity: value(entity, "consensus", "consensus", "non_wait") for entity in MODEL_ENTITY_IDS
    }
    best_wait = max(model_wait_values, key=model_wait_values.get)
    best_non_wait = max(model_non_wait_values, key=model_non_wait_values.get)
    widest_entity = max(ranges, key=lambda entity: ranges[entity]["all"][1] - ranges[entity]["all"][0])
    widest_minimum, widest_maximum = ranges[widest_entity]["all"]
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- Assisted Human Baseline scores `{human_wait:.4f}` on WAIT and `{human_non_wait:.4f}` on non-WAIT; the `{human_non_wait - human_wait:+.4f}` gap reflects the frozen action and content rubric, not a claim that all Baseline answers are uniformly strong.",
            f"- Among the four models, `{JUDGE_ENTITY_LABELS[best_wait]}` has the highest consensus WAIT mean (`{model_wait_values[best_wait]:.4f}`), while `{JUDGE_ENTITY_LABELS[best_non_wait]}` has the highest non-WAIT mean (`{model_non_wait_values[best_non_wait]:.4f}`). WAIT rankings therefore cannot stand in for overall response quality.",
            f"- The largest overall spread across judge families is for `{JUDGE_ENTITY_LABELS[widest_entity]}`: `{widest_minimum:.4f}` to `{widest_maximum:.4f}`. Judge-specific tables and row-level evidence should accompany any single-score ranking.",
            "- These are AI-judge results on an assisted Baseline condition. They are not independent human validation; retain the Baseline and Preference provenance warnings in the archive README.",
            "",
            "## Source files",
            "",
            "- `source/model_judge_scores.json`: full 2,500-row normalized judge output with per-point evidence and forbidden-claim evidence.",
            "- `source/model_judge_record_scores.json`: 500 records with all five judge scores and consensus values.",
            "- `source/model_judge_comparison.json` and `.md`: frozen model/Human comparison analysis.",
            "- `source/model_judge_summary.json`: source-run summary.",
            "- `source/model_judge_manifest.json`: path-redacted run manifest.",
            "- The raw API cache is intentionally excluded; it can contain provider payloads and is not needed to audit normalized scores.",
            "",
        ]
    )
    write_text_file(path, "\n".join(lines))


def write_readme(
    path: Path,
    completeness: dict[str, Any],
    preference: dict[str, Any],
    model_csv_rows: list[dict[str, str]],
    judge_scope_rows: list[dict[str, Any]],
) -> None:
    main_rows = [
        row for row in model_csv_rows
        if row.get("scope") in {"all", "wait", "non_wait", "speak", "clarify"}
    ]
    by_entity: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in main_rows:
        by_entity[row["entity"]][row["scope"]] = row

    def cell(entity: str, scope: str, field: str = "mean") -> str:
        row = by_entity.get(entity, {}).get(scope)
        return "--" if not row or row.get(field) in (None, "") else row[field]

    labels = {
        "human_mean": "Assisted Human Baseline",
        "moss": "MOSS",
        "minicpm": "MiniCPM",
        "gander": "Gander",
        "venus": "Venus",
    }
    judge_rows_by_entity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in judge_scope_rows:
        if row["aggregation"] == "judge" and row["scope"] in {"all", "wait", "non_wait"}:
            judge_rows_by_entity[row["entity"]].append(row)

    def judge_range(entity: str, scope: str) -> tuple[float, float]:
        values = [
            float(row["mean"])
            for row in judge_rows_by_entity[entity]
            if row["scope"] == scope
        ]
        require(len(values) == 5, f"expected five judge values for {entity}/{scope}")
        return min(values), max(values)
    fingerprint_audit = completeness["candidate_fingerprint_audit"]
    lines = [
        "# Assisted Human Baseline and batch-calibrated Preference archive",
        "",
        "**Provenance warning:** this is not an independent human-validation release.",
        "The 375 Baseline answers were collected with GroundTruth visible, so they are an",
        "assisted-reference condition. The 625 Preference records are batch-generated,",
        "GT-calibrated pseudo-ratings rather than independent rating sessions. Every derived",
        "Preference row is marked `analysis_eligible=false` and must not be used as human",
        "preference evidence without recollection and revalidation.",
        "",
        "This archive preserves the complete 45-task, 125-slot source snapshot for audit and",
        "reuse. All annotator identifiers are pseudonymous. No media, credentials, names,",
        "email addresses, model weights, or API caches are included.",
        "",
        "## Snapshot",
        "",
        f"- Annotation record revision: `{completeness['annotation_revision']}`.",
        f"- Assisted Human Baseline: `{completeness['baseline_records']['total']}` records from B01-B03, with GroundTruth visible.",
        f"- Batch-calibrated Preference: `{completeness['preference_records']['total']}` pseudo-ratings from P01-P05; not independent human sessions.",
        "- Five judges: Qwen primary, HY3, DeepSeek, GLM, and Kimi.",
        "- Preference ratings: seven candidates scored from 0 to 5 by every rater.",
        f"- Candidate snapshot generated: `{completeness['candidate_snapshot_generated_at']}`.",
        f"- Preference submissions: `{completeness['preference_submission_window']['first']}` to "
        f"`{completeness['preference_submission_window']['last']}`.",
        f"- Batch audit: `{completeness['preference_batch_audit']['submission_span_seconds']}` s submission span, "
        f"`{completeness['preference_batch_audit']['records_with_placeholder_elapsed_12000_ms']}` placeholder `12000 ms` durations, "
        f"`{completeness['preference_batch_audit']['records_with_gt_calibration_note']}` GT-calibration notes.",
        "",
        "## Five-judge Human-vs-model comparison",
        "",
        "Same 45 tasks and 125 slots. Assisted Human Baseline averages B01-B03 after",
        "taking the five-judge consensus for each slot.",
        "",
        "| Entity | Overall | Task-balanced | WAIT | Non-WAIT | SPEAK | CLARIFY |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for entity in ("human_mean", "moss", "minicpm", "gander", "venus"):
        lines.append(
            f"| {labels[entity]} | {cell(entity, 'all')} | "
            f"{cell(entity, 'all', 'task_balanced_mean')} | {cell(entity, 'wait')} | "
            f"{cell(entity, 'non_wait')} | {cell(entity, 'speak')} | {cell(entity, 'clarify')} |"
        )
    lines.extend(
        [
            "",
            "The score distribution is strongly bimodal. Assisted Human Baseline is `4.89` on",
            "`WAIT` slots but `2.69` on slots that require speech or clarification.",
            "",
            "## Four-model five-judge detail",
            "",
            f"The release now includes all `{completeness['model_judge_scores']:,}` row-level scores",
            f"and `{completeness['model_judge_complete_records']}` five-judge record summaries.",
            "No model judge call failed. `data/judge_scope_analysis.csv` separates five-judge",
            "consensus and each judge family across overall, WAIT, non-WAIT, SPEAK, and CLARIFY.",
            "",
            "| Entity | Judge overall range | Judge WAIT range | Judge non-WAIT range |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for entity in JUDGE_ENTITY_IDS:
        overall_min, overall_max = judge_range(entity, "all")
        wait_min, wait_max = judge_range(entity, "wait")
        non_wait_min, non_wait_max = judge_range(entity, "non_wait")
        lines.append(
            f"| {labels[entity]} | {overall_min:.4f}-{overall_max:.4f} | "
            f"{wait_min:.4f}-{wait_max:.4f} | {non_wait_min:.4f}-{non_wait_max:.4f} |"
        )
    lines.extend(
        [
            "",
            "WAIT and non-WAIT are reported separately because the frozen rubric rewards a",
            "correct silent action differently from content completeness. The detailed report",
            "also shows every judge family and the score spread around each consensus mean.",
            "",
            "**Scoring-snapshot caveat:** the Baseline judge summary records annotation revision",
            "`3538`, while the packaged final annotation snapshot is revision `4723`. Judge rows are",
            "joined to Baseline records by pseudonymous profile, slot UID, and submission time. The",
            "earlier raw annotation snapshot is not bundled, so the archive supports row-level score",
            "audit but not a byte-for-byte reconstruction from that earlier revision.",
            "",
            "## Fingerprint integrity note",
            "",
            "Preference scores are joined to candidates by `uid + candidate_id`; the resolved",
            "candidate text, observed state, action, response time, recorded fingerprint, and",
            "current fingerprint are all retained in `preference_assignments.csv`.",
            "",
            f"Of {fingerprint_audit['assignments']:,} candidate assignments, "
            f"{fingerprint_audit['matches']:,} recorded fingerprints equal the current `v1` serializer "
            f"and {fingerprint_audit['mismatches']:,} do not. All mismatches are model assignments "
            f"(`560` per model); all {fingerprint_audit['human_matches']:,} Human Baseline assignments match. "
            "The mismatches come from the batch score migration rather than a later candidate update: "
            "the frozen candidate snapshot was generated before the earliest Preference submission. "
            "All 2,240 mismatches are explained by historical response-time serialization: the current "
            "resolved response time is empty for those assignments, while candidate text, observed state, "
            "and action still match. "
            "Raw fingerprints and scores are preserved rather than rewritten, and the boolean "
            "`fingerprint_matches_current_candidate` marks the distinction explicitly.",
            "",
            "## Batch-calibrated Preference diagnostic (not human validation)",
            "",
            "These raw score means are included only for provenance and debugging. They are not",
            "valid Human Preference results; all rows have `analysis_eligible=false`.",
            "",
            "| Candidate | Overall | WAIT | Non-WAIT | SPEAK | CLARIFY |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for candidate_id in CANDIDATE_IDS:
        value = preference[candidate_id]
        lines.append(
            f"| {value['label']} | {value['all']['mean']:.2f} | {value['wait']['mean']:.2f} | "
            f"{value['non_wait']['mean']:.2f} | {value['speak']['mean']:.2f} | {value['clarify']['mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `data/items.jsonl`: 125 slots, ground truth, causal events, and seven candidate responses.",
            "- `data/baseline_records.jsonl` and `.csv`: 375 assisted Human Baseline answers.",
            "- `data/preference_records.jsonl`: 625 batch-calibrated pseudo-ratings with A-G mappings and seven scores.",
            "- `data/preference_assignments.csv`: long-form 4,375 quarantined rows with A-G mappings, scores, resolved candidate responses, fingerprint-match audit flags, and analysis blockers.",
            "- `data/baseline_judge_details.jsonl`: 1,875 normalized five-judge scores with semantic evidence.",
            "- `data/model_judge_details.jsonl`: 2,500 normalized four-model five-judge scores with per-point evidence.",
            "- `data/model_judge_record_scores.csv`: 500 model records with five-judge consensus scores.",
            "- `data/judge_scope_analysis.csv`: entity x judge/consensus x scope aggregate analysis.",
            "- `data/model_comparison.csv`: five-judge Human-vs-model table and capability breakdown.",
            "- `data/preference_summary.csv`: diagnostic batch-score aggregates, explicitly marked ineligible for human-validation analysis.",
            "- `analysis/wait_nonwait_report.md`: WAIT/non-WAIT findings, per-judge results, and judge sensitivity.",
            "- `analysis/model_judge_comparison.md` and `.json`: frozen source comparison analysis.",
            "- `source/`: immutable source snapshots and selected upstream score artifacts.",
            "- `manifests/`: provenance, completeness checks, and SHA-256 checksums.",
            "",
            "## Reproduce packaged artifacts",
            "",
            "From the repository root:",
            "",
            "```bash",
            "python Experiment/human_evaluation_2026-10-24/scripts/build_release.py",
            "python Experiment/human_evaluation_2026-10-24/scripts/verify_release.py",
            "```",
            "",
            "The build script reads only checked-in snapshots and standard-library Python. It rebuilds",
            "the packaged tables from normalized source rows; it does not call judge APIs. Re-running",
            "the judges or changing the annotation source requires the frozen scoring harness and",
            "rubric outside this archive.",
            "",
        ]
    )
    write_text_file(path, "\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--judge-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    source_root = (args.source_root or output / "source").resolve()
    judge_root = args.judge_root.resolve() if args.judge_root else None
    annotation_path = source_root / "annotation-data.json"
    runtime_path = source_root / "runtime" / "annotation-records.json"
    records_path = runtime_path if runtime_path.exists() else source_root / "annotation-records.json"
    baseline_scores_path = (judge_root / "scores.json") if judge_root else source_root / "baseline_scores.json"
    baseline_record_scores_path = (judge_root / "record_scores.json") if judge_root else source_root / "baseline_record_scores.json"
    baseline_summary_path = (judge_root / "summary.json") if judge_root else source_root / "baseline_summary.json"
    model_root = (judge_root / "models-five-judge") if judge_root else source_root
    model_comparison_path = (model_root / "comparison.csv") if judge_root else source_root / "model_comparison.csv"
    model_record_scores_path = (model_root / "record_scores.csv") if judge_root else source_root / "model_record_scores.csv"
    model_judge_scores_path = (model_root / "scores.json") if judge_root else source_root / "model_judge_scores.json"
    model_judge_records_json_path = (model_root / "record_scores.json") if judge_root else source_root / "model_judge_record_scores.json"
    model_judge_comparison_json_path = (model_root / "comparison.json") if judge_root else source_root / "model_judge_comparison.json"
    model_judge_comparison_md_path = (model_root / "comparison.md") if judge_root else source_root / "model_judge_comparison.md"
    model_judge_summary_path = (model_root / "summary.json") if judge_root else source_root / "model_judge_summary.json"
    model_judge_manifest_path = (model_root / "manifest.json") if judge_root else source_root / "model_judge_manifest.json"

    annotation = read_json(annotation_path)
    records = read_json(records_path)
    baseline_scores = read_json(baseline_scores_path)
    baseline_record_scores = read_json(baseline_record_scores_path)
    baseline_summary = read_json(baseline_summary_path)
    model_judge_scores = sanitize_model_judge_scores(read_json(model_judge_scores_path))
    model_judge_records_json = read_json(model_judge_records_json_path)
    model_judge_comparison = read_json(model_judge_comparison_json_path)
    model_judge_summary = read_json(model_judge_summary_path)
    model_judge_manifest = sanitize_model_source_paths(read_json(model_judge_manifest_path))
    slots = task_index(annotation)
    baseline = submitted_records(records, BASELINE_PROFILES, "baseline")
    preference = submitted_records(records, PREFERENCE_PROFILES, "preference")
    require(len(baseline) == 375, f"expected 375 Baseline records, got {len(baseline)}")
    require(len(preference) == 625, f"expected 625 Preference records, got {len(preference)}")
    baseline_by_key = {(row["uid"], row["profile"]): row for row in baseline}
    require(len(baseline_by_key) == 375, "duplicate Baseline profile/uid")

    items = build_items(slots, baseline_by_key)
    baseline_rows = build_baseline_rows(slots, baseline)
    preference_rows, assignments = build_preference_rows(slots, preference, baseline_by_key)
    preference_csv_rows, preference_details = preference_summary(preference_rows, slots)
    judge_rows = baseline_score_rows(baseline_scores)
    baseline_record_rows = baseline_record_scores.get("records") or []
    require(len(baseline_record_rows) == 375, "expected 375 Baseline record summaries")
    require(all(row.get("status") == "complete" for row in baseline_record_rows), "incomplete Baseline record summary")
    require(baseline_summary.get("complete") is True, "Baseline summary is not complete")
    model_judge_rows = model_score_rows(model_judge_scores)
    model_judge_record_rows = model_judge_records_json.get("records") or []
    require(len(model_judge_record_rows) == 500, "expected 500 model judge record summaries")
    require(all(row.get("status") == "complete" for row in model_judge_record_rows), "incomplete model judge record summary")
    require(model_judge_summary.get("complete") is True, "model judge summary is not complete")
    require(model_judge_summary.get("successful_scores") == 2500, "unexpected model judge success count")
    require(model_judge_summary.get("failed_scores") == 0, "model judge summary reports failures")
    judge_scope_rows = judge_scope_analysis_rows(
        model_judge_rows,
        model_judge_record_rows,
        baseline_score_rows(baseline_scores),
        baseline_record_rows,
    )
    comparison_entities = model_judge_comparison.get("entities") or {}
    require(set(comparison_entities) == set(JUDGE_ENTITY_IDS), "unexpected model judge comparison entity set")
    generated_consensus = {
        (row["entity"], row["scope"]): row
        for row in judge_scope_rows
        if row["aggregation"] == "consensus"
    }
    for entity in JUDGE_ENTITY_IDS:
        for scope in SCOPE_NAMES:
            require(
                abs(float(comparison_entities[entity][scope]["mean"]) - float(generated_consensus[(entity, scope)]["mean"])) < 0.0001,
                f"comparison mean does not match generated consensus: {entity}/{scope}",
            )
        require(
            abs(float(comparison_entities[entity]["task_balanced_mean"]) - float(generated_consensus[(entity, "all")]["task_balanced_mean"])) < 0.0001,
            f"comparison task-balanced mean does not match generated consensus: {entity}",
        )

    with model_comparison_path.open("r", encoding="utf-8-sig", newline="") as handle:
        model_rows = list(csv.DictReader(handle))
    require(model_rows, "model comparison CSV is empty")
    model_rows = release_model_comparison_rows(model_rows, judge_scope_rows)
    model_judge_comparison_analysis = analyze_model_judge_comparison(model_judge_comparison)
    with model_record_scores_path.open("r", encoding="utf-8-sig", newline="") as handle:
        model_record_rows = list(csv.DictReader(handle))
    require(len(model_record_rows) == 500, "expected 500 model record summaries")

    data = output / "data"
    source = output / "source"
    manifests = output / "manifests"
    analysis = output / "analysis"
    source.mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)

    def copy_snapshot(source_path: Path, destination: Path) -> None:
        if source_path.resolve() != destination.resolve():
            shutil.copy2(source_path, destination)

    copy_snapshot(annotation_path, source / "annotation-data.json")
    copy_snapshot(records_path, source / "annotation-records.json")
    copy_snapshot(baseline_scores_path, source / "baseline_scores.json")
    copy_snapshot(baseline_record_scores_path, source / "baseline_record_scores.json")
    copy_snapshot(baseline_summary_path, source / "baseline_summary.json")
    copy_snapshot(model_comparison_path, source / "model_comparison.csv")
    copy_snapshot(model_record_scores_path, source / "model_record_scores.csv")
    write_json(source / "model_judge_scores.json", model_judge_scores)
    copy_snapshot(model_judge_records_json_path, source / "model_judge_record_scores.json")
    copy_snapshot(model_judge_comparison_json_path, source / "model_judge_comparison.json")
    copy_snapshot(model_judge_comparison_md_path, source / "model_judge_comparison.md")
    copy_snapshot(model_judge_summary_path, source / "model_judge_summary.json")
    write_json(source / "model_judge_manifest.json", model_judge_manifest)

    write_jsonl(data / "items.jsonl", items)
    write_jsonl(data / "baseline_records.jsonl", baseline_rows)
    write_csv(
        data / "baseline_records.csv",
        baseline_rows,
        [
            "profile", "uid", "task_id", "source_id", "capability", "branch_id", "slot_id",
            "expected_action", "submitted_action", "answer", "note", "flagged", "submitted_at", "elapsed_ms",
            "collection_condition",
        ],
    )
    write_jsonl(data / "preference_records.jsonl", preference_rows)
    write_csv(
        data / "preference_assignments.csv",
        assignments,
        [
            "profile", "uid", "task_id", "source_id", "capability", "expected_action", "position",
            "candidate_id", "candidate_label", "score", "candidate_observed", "candidate_action",
            "candidate_response_time_s", "candidate_fingerprint", "candidate_current_fingerprint",
            "fingerprint_matches_current_candidate", "candidate_text", "note", "flagged", "submitted_at", "elapsed_ms",
            "score_provenance", "analysis_eligible", "analysis_blocker",
        ],
    )
    write_jsonl(data / "baseline_judge_details.jsonl", judge_rows)
    write_jsonl(
        data / "model_judge_details.jsonl",
        (
            {
                **row,
                "entity_label": JUDGE_ENTITY_LABELS[str(row["profile"])],
                "analysis_condition": JUDGE_ENTITY_CONDITIONS[str(row["profile"])],
            }
            for row in model_judge_rows
        ),
    )
    write_csv(
        data / "baseline_record_scores.csv",
        baseline_record_rows,
        ["record_id", "profile", "uid", "task_id", "branch_id", "capability", "expected_action", "status", "n_judges", "n_ok", "consensus_mean"],
    )
    write_csv(
        data / "model_comparison.csv",
        model_rows,
        list(model_rows[0]),
    )
    write_csv(
        data / "model_judge_record_scores.csv",
        model_judge_record_rows,
        ["record_id", "profile", "uid", "task_id", "branch_id", "capability", "expected_action", "status", "n_judges", "n_ok", "consensus_mean"],
    )
    write_csv(
        data / "judge_scope_analysis.csv",
        judge_scope_rows,
        [
            "entity", "entity_label", "analysis_condition", "aggregation", "judge_key", "judge_family", "judge_model",
            "scope", "n", "mean", "sd", "exact_zero", "exact_five", "task_balanced_mean",
        ],
    )
    write_csv(
        data / "preference_summary.csv",
        preference_csv_rows,
        [
            "candidate_id", "candidate_label", "scope", "n", "mean", "sd",
            "exact_zero", "exact_five", "task_balanced_mean", "score_provenance",
            "analysis_eligible",
        ],
    )
    comparison_markdown = model_judge_comparison_md_path.read_text(encoding="utf-8-sig")
    comparison_markdown = comparison_markdown.replace("Assisted Human Baseline", "Human Baseline")
    comparison_markdown = comparison_markdown.replace("Human Baseline", "Assisted Human Baseline")
    write_text_file(analysis / "model_judge_comparison.md", comparison_markdown)
    write_json(analysis / "model_judge_comparison.json", model_judge_comparison_analysis)
    write_model_judge_report(analysis / "wait_nonwait_report.md", judge_scope_rows, model_judge_rows)

    fingerprints_by_uid_candidate: dict[tuple[str, str], str] = {}
    for row in preference_rows:
        for candidate_id, fingerprint in row["candidate_fingerprints"].items():
            key = (row["uid"], candidate_id)
            previous = fingerprints_by_uid_candidate.setdefault(key, fingerprint)
            require(previous == fingerprint, f"inconsistent fingerprint across raters: {key}")

    fingerprint_by_candidate: dict[str, Counter[str]] = defaultdict(Counter)
    for row in assignments:
        match_key = "matches" if row["fingerprint_matches_current_candidate"] else "mismatches"
        fingerprint_by_candidate[row["candidate_id"]][match_key] += 1
    model_mismatches = sum(
        counts["mismatches"]
        for candidate_id, counts in fingerprint_by_candidate.items()
        if not candidate_id.startswith("human_")
    )
    human_matches = sum(
        counts["matches"]
        for candidate_id, counts in fingerprint_by_candidate.items()
        if candidate_id.startswith("human_")
    )
    require(model_mismatches == 2240, f"unexpected model fingerprint mismatches: {model_mismatches}")
    require(human_matches == 1875, f"unexpected Human fingerprint matches: {human_matches}")
    candidate_fingerprint_audit = {
        "assignments": len(assignments),
        "matches": sum(counts["matches"] for counts in fingerprint_by_candidate.values()),
        "mismatches": sum(counts["mismatches"] for counts in fingerprint_by_candidate.values()),
        "human_matches": human_matches,
        "by_candidate": {
            candidate_id: {
                "matches": counts["matches"],
                "mismatches": counts["mismatches"],
            }
            for candidate_id, counts in sorted(fingerprint_by_candidate.items())
        },
        "interpretation": (
            "Scores are joined by uid + candidate_id. Recorded v1 fingerprints are retained; "
            "model assignments migrated from the earlier batch writer do not match the current "
            "client serializer. The candidate snapshot predates all Preference submissions."
        ),
    }

    preference_times = sorted(
        row["submitted_at"] for row in preference_rows if row.get("submitted_at")
    )
    require(bool(preference_times), "Preference submission times are missing")
    preference_datetimes = [
        datetime.fromisoformat(value.replace("Z", "+00:00")) for value in preference_times
    ]
    submission_span_seconds = round(
        (max(preference_datetimes) - min(preference_datetimes)).total_seconds(), 3
    )
    placeholder_elapsed_records = sum(row.get("elapsed_ms") == 12000 for row in preference_rows)
    calibration_note_records = sum(
        "GT" in str(row.get("note") or "") and "校准" in str(row.get("note") or "")
        for row in preference_rows
    )
    preference_batch_audit = {
        "records": len(preference_rows),
        "first_submitted_at": preference_times[0],
        "last_submitted_at": preference_times[-1],
        "submission_span_seconds": submission_span_seconds,
        "records_with_placeholder_elapsed_12000_ms": placeholder_elapsed_records,
        "records_with_gt_calibration_note": calibration_note_records,
        "records_with_independent_session_evidence": 0,
        "classification": "batch_calibrated_pseudo_ratings",
        "analysis_eligible": False,
    }
    require(submission_span_seconds < 2, "Preference records are no longer a single batch")
    require(placeholder_elapsed_records == 560, "unexpected placeholder elapsed count")
    require(calibration_note_records == 624, "unexpected calibration-note count")

    completeness = {
        "generated_at": utc_now(),
        "annotation_revision": records.get("revision"),
        "annotation_updated_at": records.get("updatedAt"),
        "candidate_snapshot_generated_at": annotation.get("generatedAt"),
        "preference_submission_window": {
            "first": preference_times[0],
            "last": preference_times[-1],
        },
        "tasks": len({item["task_id"] for item in items}),
        "slots": len(items),
        "expected_action_counts": dict(Counter(item["expected_action"] for item in items)),
        "baseline_records": {"total": len(baseline_rows), **dict(Counter(row["profile"] for row in baseline_rows))},
        "preference_records": {"total": len(preference_rows), **dict(Counter(row["profile"] for row in preference_rows))},
        "preference_assignments": len(assignments),
        "baseline_judge_scores": len(judge_rows),
        "baseline_complete_records": len(baseline_record_rows),
        "baseline_judge_annotation_revision": (baseline_summary.get("annotator_activity") or {}).get("annotation_records_revision"),
        "model_comparison_records": len(model_record_rows),
        "model_judge_scores": len(model_judge_rows),
        "model_judge_complete_records": len(model_judge_record_rows),
        "judge_scope_analysis_rows": len(judge_scope_rows),
        "candidate_ids": list(CANDIDATE_IDS),
        "candidate_fingerprint_audit": candidate_fingerprint_audit,
        "source_classification": {
            "baseline": "assisted_human_baseline_with_ground_truth_visible",
            "preference": "batch_calibrated_pseudo_ratings",
            "official_human_validation_ready": False,
        },
        "preference_batch_audit": preference_batch_audit,
        "checks": {
            "all_records_submitted": len(baseline) == 375 and len(preference) == 625,
            "all_preference_records_have_seven_scores": all(len(row["scores"]) == 7 for row in preference_rows),
            "all_preference_records_have_seven_fingerprints": all(len(row["candidate_fingerprints"]) == 7 for row in preference_rows),
            "fingerprints_consistent_across_raters": len(fingerprints_by_uid_candidate) == 125 * 7,
            "five_judges_complete_for_baseline": Counter(row.get("judge_key") for row in judge_rows) == Counter({"qwen_primary": 375, "hy3": 375, "deepseek": 375, "glm": 375, "kimi": 375}),
            "no_judge_failures": all(row.get("status") == "ok" for row in judge_rows),
            "five_judges_complete_for_models": Counter(row.get("judge_key") for row in model_judge_rows) == Counter({"qwen_primary": 500, "hy3": 500, "deepseek": 500, "glm": 500, "kimi": 500}),
            "no_model_judge_failures": all(row.get("status") == "ok" for row in model_judge_rows),
            "judge_scope_analysis_complete": len(judge_scope_rows) == len(JUDGE_ENTITY_IDS) * 6 * len(SCOPE_NAMES),
        },
    }
    require(all(completeness["checks"].values()), f"completeness check failed: {completeness['checks']}")
    provenance = {
        "source_annotation_sha256": sha256_file(annotation_path),
        "source_annotation_generated_at": annotation.get("generatedAt"),
        "baseline_collection_condition": "assisted_with_ground_truth_visible",
        "preference_collection_condition": "batch_calibrated_pseudo_ratings",
        "preference_independent_human_sessions": False,
        "official_human_validation_ready": False,
        "source_records_sha256": sha256_file(records_path),
        "baseline_scores_sha256": sha256_file(baseline_scores_path),
        "baseline_record_scores_sha256": sha256_file(baseline_record_scores_path),
        "baseline_summary_sha256": sha256_file(baseline_summary_path),
        "baseline_judge_annotation_records_sha256": (baseline_summary.get("provenance") or {}).get("annotation_records_sha256"),
        "model_comparison_sha256": sha256_file(model_comparison_path),
        "model_record_scores_sha256": sha256_file(model_record_scores_path),
        "model_judge_scores_sha256": sha256_file(source / "model_judge_scores.json"),
        "model_judge_record_scores_sha256": sha256_file(model_judge_records_json_path),
        "model_judge_comparison_sha256": sha256_file(model_judge_comparison_json_path),
        "model_judge_summary_sha256": sha256_file(model_judge_summary_path),
        "model_judge_manifest_sha256": sha256_file(source / "model_judge_manifest.json"),
        "frozen_protocol": "museumlive_point_judge_lenient_v3",
        "baseline_judge_models": [
            "qwen/qwen3.8-omni-flash",
            "tencent/hy3",
            "deepseek/deepseek-v4.1-flash",
            "z-ai/glm-5.3-flash",
            "moonshotai/kimi-k3",
        ],
        "api_key_persisted": False,
    }
    write_json(manifests / "completeness.json", completeness)
    write_json(manifests / "provenance.json", provenance)
    write_readme(output / "README.md", completeness, preference_details, model_rows, judge_scope_rows)

    checksum_paths = sorted(
        path
        for directory in (data, source, analysis)
        for path in directory.glob("*")
        if path.is_file()
    )
    checksum_lines = [f"{sha256_file(path)}  {path.relative_to(output).as_posix()}" for path in checksum_paths]
    (manifests / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "items": len(items),
        "baseline_records": len(baseline_rows),
        "preference_records": len(preference_rows),
        "preference_assignments": len(assignments),
        "baseline_judge_scores": len(judge_rows),
        "model_judge_scores": len(model_judge_rows),
        "judge_scope_analysis_rows": len(judge_scope_rows),
        "files_checksummed": len(checksum_paths),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
