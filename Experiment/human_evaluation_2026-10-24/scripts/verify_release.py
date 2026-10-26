#!/usr/bin/env python3
"""Verify the integrity and completeness of the human-evaluation release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


BASELINE_PROFILES = ("B01", "B02", "B03")
PREFERENCE_PROFILES = ("P01", "P02", "P03", "P04", "P05")
CANDIDATE_IDS = {
    "moss",
    "minicpm",
    "gander",
    "venus",
    "human_b01",
    "human_b02",
    "human_b03",
}
MODEL_ENTITY_IDS = {"moss", "minicpm", "gander", "venus"}
JUDGE_ENTITY_IDS = {"human_mean", *MODEL_ENTITY_IDS}
SCOPE_NAMES = {"all", "wait", "non_wait", "speak", "clarify"}
JUDGE_KEYS = {"qwen_primary", "hy3", "deepseek", "glm", "kimi"}
EXPECTED_ACTIONS = {"SPEAK", "CLARIFY", "WAIT"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(isinstance(value, dict), f"{path.name}:{line_number} is not an object")
            rows.append(value)
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def verify_checksums(root: Path) -> int:
    manifest = root / "manifests" / "SHA256SUMS"
    require(manifest.is_file(), "missing manifests/SHA256SUMS")
    seen: set[str] = set()
    count = 0
    for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        require(match is not None, f"invalid checksum line {line_number}")
        expected_hash, relative = match.groups()
        require(relative not in seen, f"duplicate checksum path: {relative}")
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts, f"unsafe path: {relative}")
        path = root / relative
        require(path.is_file(), f"checksummed file is missing: {relative}")
        require(sha256_file(path) == expected_hash, f"checksum mismatch: {relative}")
        seen.add(relative)
        count += 1

    expected_paths = {
        path.relative_to(root).as_posix()
        for directory in ("data", "source", "analysis")
        for path in (root / directory).glob("*")
        if path.is_file()
    }
    require(seen == expected_paths, "SHA256SUMS does not cover exactly the data, source, and analysis files")
    require(count == 28, f"expected 28 checksummed files, got {count}")
    return count


def verify_snapshot(completeness: dict[str, Any], provenance: dict[str, Any]) -> None:
    require(completeness.get("annotation_revision") == 4723, "unexpected annotation revision")
    require(completeness.get("tasks") == 45, "unexpected task count")
    require(completeness.get("slots") == 125, "unexpected slot count")
    require(bool(completeness.get("candidate_snapshot_generated_at")), "missing candidate snapshot timestamp")
    require(
        bool((completeness.get("preference_submission_window") or {}).get("first")),
        "missing Preference submission window",
    )
    require(
        completeness.get("expected_action_counts") == {"SPEAK": 103, "CLARIFY": 7, "WAIT": 15},
        "unexpected expected-action counts",
    )
    require(
        completeness.get("baseline_records") == {"total": 375, "B01": 125, "B02": 125, "B03": 125},
        "unexpected Baseline counts",
    )
    require(
        completeness.get("preference_records")
        == {"total": 625, "P01": 125, "P02": 125, "P03": 125, "P04": 125, "P05": 125},
        "unexpected Preference counts",
    )
    require(completeness.get("preference_assignments") == 4375, "unexpected Preference assignment count")
    require(completeness.get("baseline_judge_scores") == 1875, "unexpected judge-score count")
    require(completeness.get("baseline_complete_records") == 375, "unexpected complete-record count")
    require(completeness.get("baseline_judge_annotation_revision") == 3538, "unexpected Baseline judge annotation revision")
    require(completeness.get("model_comparison_records") == 500, "unexpected model-record count")
    require(completeness.get("model_judge_scores") == 2500, "unexpected model judge-score count")
    require(completeness.get("model_judge_complete_records") == 500, "unexpected model judge-record count")
    require(completeness.get("judge_scope_analysis_rows") == 150, "unexpected judge scope-analysis count")
    require(set(completeness.get("candidate_ids") or []) == CANDIDATE_IDS, "unexpected candidate IDs")
    require(
        completeness.get("source_classification")
        == {
            "baseline": "assisted_human_baseline_with_ground_truth_visible",
            "preference": "batch_calibrated_pseudo_ratings",
            "official_human_validation_ready": False,
        },
        "unexpected source classification",
    )
    batch_audit = completeness.get("preference_batch_audit") or {}
    require(batch_audit.get("records") == 625, "unexpected batch-record count")
    require(batch_audit.get("submission_span_seconds") == 1.248, "unexpected batch submission span")
    require(batch_audit.get("records_with_placeholder_elapsed_12000_ms") == 560, "unexpected placeholder duration count")
    require(batch_audit.get("records_with_gt_calibration_note") == 624, "unexpected GT-calibration note count")
    require(batch_audit.get("records_with_independent_session_evidence") == 0, "unexpected independent-session evidence")
    require(batch_audit.get("classification") == "batch_calibrated_pseudo_ratings", "unexpected batch classification")
    require(batch_audit.get("analysis_eligible") is False, "batch ratings must not be analysis eligible")
    fingerprint_audit = completeness.get("candidate_fingerprint_audit") or {}
    require(fingerprint_audit.get("assignments") == 4375, "unexpected fingerprint-audit assignment count")
    require(fingerprint_audit.get("matches") == 2135, "unexpected fingerprint match count")
    require(fingerprint_audit.get("mismatches") == 2240, "unexpected fingerprint mismatch count")
    require(fingerprint_audit.get("human_matches") == 1875, "unexpected Human fingerprint match count")
    by_candidate = fingerprint_audit.get("by_candidate") or {}
    require(set(by_candidate) == CANDIDATE_IDS, "unexpected fingerprint-audit candidate set")
    for candidate_id in CANDIDATE_IDS:
        expected = ({"matches": 625, "mismatches": 0} if candidate_id.startswith("human_")
                    else {"matches": 65, "mismatches": 560})
        require(by_candidate[candidate_id] == expected, f"unexpected fingerprint audit: {candidate_id}")
    require(all((completeness.get("checks") or {}).values()), "a completeness check is false")

    require(provenance.get("api_key_persisted") is False, "provenance reports a persisted API key")
    require(provenance.get("frozen_protocol") == "museumlive_point_judge_lenient_v3", "unexpected Judge protocol")
    require(bool(provenance.get("source_annotation_generated_at")), "missing annotation generation timestamp")
    require(provenance.get("baseline_collection_condition") == "assisted_with_ground_truth_visible", "unexpected Baseline condition")
    require(provenance.get("preference_collection_condition") == "batch_calibrated_pseudo_ratings", "unexpected Preference condition")
    require(provenance.get("preference_independent_human_sessions") is False, "Preference sessions must not be labeled independent")
    require(provenance.get("official_human_validation_ready") is False, "archive must not be labeled official human validation")
    require(len(provenance.get("baseline_judge_models") or []) == 5, "expected five Baseline judges")
    for key in (
        "source_annotation_sha256",
        "source_records_sha256",
        "baseline_scores_sha256",
        "baseline_record_scores_sha256",
        "baseline_summary_sha256",
        "baseline_judge_annotation_records_sha256",
        "model_comparison_sha256",
        "model_record_scores_sha256",
        "model_judge_scores_sha256",
        "model_judge_record_scores_sha256",
        "model_judge_comparison_sha256",
        "model_judge_summary_sha256",
        "model_judge_manifest_sha256",
    ):
        require(re.fullmatch(r"[0-9a-f]{64}", str(provenance.get(key) or "")) is not None, f"invalid provenance hash: {key}")


def verify_items(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    require(len(rows) == 125, f"expected 125 items, got {len(rows)}")
    by_uid = {row.get("uid"): row for row in rows}
    require(len(by_uid) == 125 and None not in by_uid, "duplicate or missing item UID")
    require(len({row.get("task_id") for row in rows}) == 45, "unexpected item task count")
    require(
        Counter(row.get("expected_action") for row in rows)
        == Counter({"SPEAK": 103, "CLARIFY": 7, "WAIT": 15}),
        "unexpected item expected-action counts",
    )
    for row in rows:
        candidates = row.get("candidates") or {}
        require(set(candidates) == CANDIDATE_IDS, f"bad candidate set: {row.get('uid')}")
        require(isinstance(row.get("ground_truth"), dict), f"missing ground truth: {row.get('uid')}")
        for candidate_id, candidate in candidates.items():
            require(
                candidate.get("current_fingerprint") == candidate_fingerprint(candidate),
                f"bad candidate fingerprint in items: {row.get('uid')}/{candidate_id}",
            )
    return by_uid


def verify_baseline_rows(rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]) -> None:
    require(len(rows) == 375, f"expected 375 Baseline rows, got {len(rows)}")
    keys = [(row.get("profile"), row.get("uid")) for row in rows]
    require(len(set(keys)) == 375, "duplicate Baseline profile/UID")
    require(Counter(row.get("profile") for row in rows) == Counter({profile: 125 for profile in BASELINE_PROFILES}), "unexpected Baseline profile counts")
    for row in rows:
        require(row.get("uid") in items, f"unknown Baseline UID: {row.get('uid')}")
        require(row.get("submitted_action") in EXPECTED_ACTIONS, f"invalid Baseline action: {row.get('uid')}")
        require(isinstance(row.get("answer"), str), f"invalid Baseline answer: {row.get('uid')}")
        require(row.get("collection_condition") == "assisted_with_ground_truth_visible", f"bad Baseline condition: {row.get('uid')}")


def verify_preference_rows(
    rows: list[dict[str, Any]], items: dict[str, dict[str, Any]]
) -> dict[tuple[str, str], dict[str, Any]]:
    require(len(rows) == 625, f"expected 625 Preference rows, got {len(rows)}")
    keys = [(row.get("profile"), row.get("uid")) for row in rows]
    require(len(set(keys)) == 625, "duplicate Preference profile/UID")
    require(Counter(row.get("profile") for row in rows) == Counter({profile: 125 for profile in PREFERENCE_PROFILES}), "unexpected Preference profile counts")
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    fingerprints: dict[tuple[str, str], str] = {}
    for row in rows:
        uid = row.get("uid")
        require(uid in items, f"unknown Preference UID: {uid}")
        require(row.get("score_provenance") == "batch_calibrated_pseudo_rating", f"bad Preference provenance: {uid}")
        require(row.get("analysis_eligible") is False, f"batch Preference marked analysis eligible: {uid}")
        order = row.get("candidate_order") or []
        scores = row.get("scores") or {}
        mapping = row.get("position_mapping") or {}
        row_fingerprints = row.get("candidate_fingerprints") or {}
        require(len(order) == 7 and set(order) == CANDIDATE_IDS, f"bad candidate order: {uid}")
        require(set(scores) == CANDIDATE_IDS, f"bad Preference scores: {uid}")
        require(set(row_fingerprints) == CANDIDATE_IDS, f"bad Preference fingerprints: {uid}")
        require(mapping == dict(zip("ABCDEFG", order)), f"bad A-G mapping: {uid}")
        for candidate_id, score in scores.items():
            require(isinstance(score, int) and 0 <= score <= 5, f"invalid score for {uid}/{candidate_id}")
        for candidate_id, fingerprint in row_fingerprints.items():
            previous = fingerprints.setdefault((uid, candidate_id), fingerprint)
            require(previous == fingerprint, f"inconsistent fingerprint: {uid}/{candidate_id}")
        by_key[(row["profile"], uid)] = row
    return by_key


def verify_assignments(
    rows: list[dict[str, Any]],
    preference: dict[tuple[str, str], dict[str, Any]],
    items: dict[str, dict[str, Any]],
) -> dict[str, Counter[str]]:
    require(len(rows) == 4375, f"expected 4,375 Preference assignments, got {len(rows)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    audit: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        grouped[(row.get("profile"), row.get("uid"))].append(row)
    require(set(grouped) == set(preference), "Preference assignment keys do not match records")
    for key, assigned in grouped.items():
        record = preference[key]
        require(len(assigned) == 7 and {row.get("position") for row in assigned} == set("ABCDEFG"), f"bad assignment positions: {key}")
        for row in assigned:
            candidate_id = row.get("candidate_id")
            require(record["position_mapping"][row["position"]] == candidate_id, f"mapping mismatch: {key}/{row.get('position')}")
            require(int(row.get("score") or -1) == record["scores"][candidate_id], f"score mismatch: {key}/{candidate_id}")
            require(row.get("candidate_fingerprint") == record["candidate_fingerprints"][candidate_id], f"fingerprint mismatch: {key}/{candidate_id}")
            candidate = items[row["uid"]]["candidates"][candidate_id]
            current_fingerprint = candidate_fingerprint({
                "text": row.get("candidate_text"),
                "observed": row.get("candidate_observed"),
                "action": row.get("candidate_action"),
                "response_time_s": row.get("candidate_response_time_s"),
            })
            require(row.get("candidate_current_fingerprint") == current_fingerprint, f"current fingerprint mismatch: {key}/{candidate_id}")
            require(current_fingerprint == candidate.get("current_fingerprint"), f"item fingerprint mismatch: {key}/{candidate_id}")
            expected_match = row.get("candidate_fingerprint") == current_fingerprint
            require(
                (row.get("fingerprint_matches_current_candidate") == "True") == expected_match,
                f"fingerprint-match flag mismatch: {key}/{candidate_id}",
            )
            require(row.get("score_provenance") == "batch_calibrated_pseudo_rating", f"bad assignment provenance: {key}/{candidate_id}")
            require(row.get("analysis_eligible") == "False", f"assignment marked analysis eligible: {key}/{candidate_id}")
            expected_blocker = (
                "model_candidate_fingerprint_mismatch"
                if not candidate_id.startswith("human_") and not expected_match
                else "batch_generated_preference_scores"
            )
            require(row.get("analysis_blocker") == expected_blocker, f"bad analysis blocker: {key}/{candidate_id}")
            audit[candidate_id]["matches" if expected_match else "mismatches"] += 1
    return audit


def verify_judges(
    rows: list[dict[str, Any]], record_rows: list[dict[str, str]], baseline_rows: list[dict[str, Any]]
) -> None:
    require(len(rows) == 1875, f"expected 1,875 judge details, got {len(rows)}")
    require(Counter(row.get("judge_key") for row in rows) == Counter({judge: 375 for judge in JUDGE_KEYS}), "unexpected judge coverage")
    require(all(row.get("status") == "ok" for row in rows), "a Baseline judge call failed")
    keys = [(row.get("profile"), row.get("uid"), row.get("judge_key")) for row in rows]
    require(len(set(keys)) == 1875, "duplicate judge detail")
    score_keys = [(row.get("profile"), row.get("uid")) for row in rows]
    baseline_keys = {(row["profile"], row["uid"]) for row in baseline_rows}
    require(set(score_keys) == baseline_keys, "judge details do not match Baseline records")

    require(len(record_rows) == 375, f"expected 375 Baseline record summaries, got {len(record_rows)}")
    require(all(row.get("status") == "complete" for row in record_rows), "an incomplete Baseline record summary")
    require(len({(row.get("profile"), row.get("uid")) for row in record_rows}) == 375, "duplicate Baseline record summary")


def verify_model_judges(
    score_rows: list[dict[str, Any]],
    record_rows: list[dict[str, str]],
    scope_rows: list[dict[str, str]],
) -> None:
    require(len(score_rows) == 2500, f"expected 2,500 model judge details, got {len(score_rows)}")
    require(all(row.get("status") == "ok" for row in score_rows), "a model judge call failed")
    require(Counter(row.get("judge_key") for row in score_rows) == Counter({judge: 500 for judge in JUDGE_KEYS}), "unexpected model judge coverage")
    require(Counter(row.get("profile") for row in score_rows) == Counter({profile: 625 for profile in MODEL_ENTITY_IDS}), "unexpected model judge profile coverage")
    require(len({row.get("score_id") for row in score_rows}) == 2500, "duplicate model judge score_id")
    for row in score_rows:
        score = row.get("score")
        require(isinstance(score, (int, float)) and 0 <= float(score) <= 5, f"invalid model judge score: {row.get('score_id')}")
        require(row.get("expected_action") in EXPECTED_ACTIONS, f"invalid model judge action: {row.get('score_id')}")
        require(row.get("analysis_condition") == "ai_judged_saved_model_response", f"bad model judge analysis condition: {row.get('score_id')}")
        require(row.get("entity_label"), f"missing model judge entity label: {row.get('score_id')}")

    require(len(record_rows) == 500, f"expected 500 model judge records, got {len(record_rows)}")
    require(all(row.get("status") == "complete" and row.get("n_judges") == "5" and row.get("n_ok") == "5" for row in record_rows), "incomplete model judge record")
    require(Counter(row.get("profile") for row in record_rows) == Counter({profile: 125 for profile in MODEL_ENTITY_IDS}), "unexpected model judge record coverage")
    require(len({row.get("record_id") for row in record_rows}) == 500, "duplicate model judge record")
    grouped_scores: dict[str, list[float]] = defaultdict(list)
    for row in score_rows:
        grouped_scores[str(row["record_id"])].append(float(row["score"]))
    for row in record_rows:
        values = grouped_scores.get(row["record_id"]) or []
        require(len(values) == 5, f"model record does not have five judge scores: {row.get('record_id')}")
        require(abs(fmean(values) - float(row["consensus_mean"])) < 0.00001, f"model record consensus mismatch: {row.get('record_id')}")

    require(len(scope_rows) == 150, f"expected 150 judge scope rows, got {len(scope_rows)}")
    require({row.get("entity") for row in scope_rows} == JUDGE_ENTITY_IDS, "unexpected judge scope entity set")
    require({row.get("aggregation") for row in scope_rows} == {"consensus", "judge"}, "unexpected judge aggregation set")
    require({row.get("scope") for row in scope_rows} == SCOPE_NAMES, "unexpected judge scope set")
    require({row.get("judge_key") for row in scope_rows if row.get("aggregation") == "judge"} == JUDGE_KEYS, "unexpected per-judge scope coverage")
    expected_labels = {
        "human_mean": "Assisted Human Baseline",
        "moss": "MOSS",
        "minicpm": "MiniCPM",
        "gander": "Gander",
        "venus": "Venus",
    }
    require(all(row.get("entity_label") == expected_labels[row.get("entity")] for row in scope_rows), "judge scope entity labels are not release-safe")
    require(
        {row.get("entity"): row.get("analysis_condition") for row in scope_rows}
        == {
            "human_mean": "assisted_baseline_with_ground_truth_visible",
            "moss": "ai_judged_saved_model_response",
            "minicpm": "ai_judged_saved_model_response",
            "gander": "ai_judged_saved_model_response",
            "venus": "ai_judged_saved_model_response",
        },
        "unexpected judge analysis conditions",
    )
    expected_counts = {"all": 125, "wait": 15, "non_wait": 110, "speak": 103, "clarify": 7}
    for row in scope_rows:
        require(int(row.get("n") or -1) == expected_counts[row["scope"]], f"unexpected judge scope n: {row}")
        require(row.get("mean") not in (None, ""), f"missing judge scope mean: {row}")


def verify_summaries(model_rows: list[dict[str, str]], preference_rows: list[dict[str, str]]) -> None:
    require(len(model_rows) == 70, f"expected 70 model comparison rows, got {len(model_rows)}")
    require(
        Counter(row.get("entity") for row in model_rows)
        == Counter({entity: 14 for entity in ("human_mean", "moss", "minicpm", "gander", "venus")}),
        "unexpected model comparison coverage",
    )
    require(len(preference_rows) == 35, f"expected 35 Preference summary rows, got {len(preference_rows)}")
    require(Counter(row.get("candidate_id") for row in preference_rows) == Counter({candidate: 5 for candidate in CANDIDATE_IDS}), "unexpected Preference summary coverage")
    require({row.get("scope") for row in preference_rows} == {"all", "wait", "non_wait", "speak", "clarify"}, "unexpected Preference summary scopes")
    require(all(row.get("score_provenance") == "batch_calibrated_pseudo_rating" for row in preference_rows), "bad Preference summary provenance")
    require(all(row.get("analysis_eligible") == "False" for row in preference_rows), "Preference summary marked analysis eligible")
    require(all(row.get("entity_label") for row in model_rows), "model comparison rows are missing entity labels")
    require(all(row.get("analysis_condition") for row in model_rows), "model comparison rows are missing analysis conditions")
    require(
        all(
            row.get("entity_label") == row.get("label") == "Assisted Human Baseline"
            for row in model_rows
            if row.get("entity") == "human_mean"
        ),
        "Human Baseline label is missing the assisted condition",
    )


def verify_analysis_comparison(payload: dict[str, Any]) -> None:
    entities = payload.get("entities") or {}
    require(set(entities) == JUDGE_ENTITY_IDS, "unexpected analysis comparison entity set")
    require(entities["human_mean"].get("label") == "Assisted Human Baseline", "analysis comparison lost the assisted condition")
    for entity in MODEL_ENTITY_IDS:
        require(entities[entity].get("analysis_condition") == "ai_judged_saved_model_response", f"bad analysis condition: {entity}")


def verify_no_sensitive_patterns(root: Path) -> None:
    patterns = (
        re.compile(r"ghp_[A-Za-z0-9]{30,}"),
        re.compile(r"sk-or-v1-[A-Za-z0-9]{20,}"),
        re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        re.compile(r"[A-Za-z]:\\\\[^\r\n\"]+", re.IGNORECASE),
        re.compile(r"\\\\[^\\\s\"]+\\[^\r\n\"]+"),
        re.compile(r"/(?:home|Users|root|mnt|tmp)/[^\s\"]+", re.IGNORECASE),
    )
    files = [root / "README.md"]
    for directory in ("data", "source", "analysis", "manifests"):
        files.extend(path for path in (root / directory).glob("*") if path.is_file())
    for path in files:
        text = path.read_text(encoding="utf-8-sig")
        for pattern in patterns:
            require(pattern.search(text) is None, f"sensitive-looking value in {path.relative_to(root)}")


def verify_redacted_sources(root: Path) -> None:
    scores = read_json(root / "source" / "model_judge_scores.json")
    manifest = read_json(root / "source" / "model_judge_manifest.json")
    require((scores.get("metadata") or {}).get("source_paths_redacted") is True, "model judge score paths were not redacted")
    require(manifest.get("source_paths_redacted") is True, "model judge manifest paths were not redacted")


def independent_stats(values: Iterable[float]) -> dict[str, Any]:
    materialized = [float(value) for value in values]
    return {
        "n": len(materialized),
        "mean": round(fmean(materialized), 4) if materialized else None,
        "sd": round(pstdev(materialized), 4) if materialized else None,
        "exact_zero": sum(value == 0 for value in materialized),
        "exact_five": sum(value == 5 for value in materialized),
    }


def independent_task_balanced_mean(points: list[dict[str, Any]]) -> float | None:
    groups: dict[str, list[float]] = defaultdict(list)
    for point in points:
        groups[point["task_id"]].append(point["score"])
    return round(fmean(fmean(values) for values in groups.values()), 4) if groups else None


def select_scope(points: list[dict[str, Any]], scope: str) -> list[dict[str, Any]]:
    return [
        point
        for point in points
        if scope == "all"
        or (scope == "wait" and point["expected_action"] == "WAIT")
        or (scope == "non_wait" and point["expected_action"] != "WAIT")
        or (scope == "speak" and point["expected_action"] == "SPEAK")
        or (scope == "clarify" and point["expected_action"] == "CLARIFY")
    ]


def build_verification_points(
    model_scores: list[dict[str, Any]],
    model_records: list[dict[str, str]],
    baseline_scores: list[dict[str, Any]],
    baseline_records: list[dict[str, str]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []

    def append_point(
        entity: str,
        aggregation: str,
        judge_key: str,
        score: float,
        row: dict[str, Any],
    ) -> None:
        points.append(
            {
                "entity": entity,
                "aggregation": aggregation,
                "judge_key": judge_key,
                "score": float(score),
                "task_id": str(row["task_id"]),
                "capability": str(row.get("capability") or ""),
                "expected_action": str(row["expected_action"]),
            }
        )

    for row in model_scores:
        append_point(str(row["profile"]), "judge", str(row["judge_key"]), float(row["score"]), row)
    for row in model_records:
        append_point(str(row["profile"]), "consensus", "consensus", float(row["consensus_mean"]), row)

    baseline_by_judge: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in baseline_scores:
        baseline_by_judge[(str(row["uid"]), str(row["judge_key"]))].append(row)
    for (_, judge_key), rows in baseline_by_judge.items():
        require(len(rows) == 3, f"Baseline judge verification expected three profiles: {judge_key}")
        append_point(
            "human_mean",
            "judge",
            judge_key,
            fmean(float(row["score"]) for row in rows),
            rows[0],
        )

    baseline_by_uid: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in baseline_records:
        baseline_by_uid[str(row["uid"])].append(row)
    for rows in baseline_by_uid.values():
        require(len(rows) == 3, "Baseline consensus verification expected three profiles")
        append_point(
            "human_mean",
            "consensus",
            "consensus",
            fmean(float(row["consensus_mean"]) for row in rows),
            rows[0],
        )
    return points


def expected_scope_rows(points: list[dict[str, Any]]) -> dict[tuple[str, str, str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[(point["entity"], point["aggregation"], point["judge_key"])].append(point)
    judges = ("qwen_primary", "hy3", "deepseek", "glm", "kimi")
    output: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for entity in sorted(JUDGE_ENTITY_IDS):
        keys = (("consensus", "consensus"),) + tuple(("judge", judge) for judge in judges)
        for aggregation, judge_key in keys:
            entity_points = grouped[(entity, aggregation, judge_key)]
            require(len(entity_points) == 125, f"incomplete verification points: {entity}/{aggregation}/{judge_key}")
            for scope in SCOPE_NAMES:
                selected = select_scope(entity_points, scope)
                value = independent_stats(point["score"] for point in selected)
                value["task_balanced_mean"] = independent_task_balanced_mean(selected) if scope == "all" else None
                output[(entity, aggregation, judge_key, scope)] = value
    return output


def verify_scope_numbers(
    scope_rows: list[dict[str, str]],
    expected: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    require(len(scope_rows) == len(expected) == 150, "unexpected independent scope coverage")
    seen = set()
    for row in scope_rows:
        key = (row["entity"], row["aggregation"], row["judge_key"], row["scope"])
        require(key in expected and key not in seen, f"unexpected or duplicate scope row: {key}")
        seen.add(key)
        target = expected[key]
        for field in ("n", "mean", "sd", "exact_zero", "exact_five"):
            require(abs(float(row[field]) - float(target[field])) < 0.00001, f"scope value mismatch: {key}/{field}")
        raw_task_balanced = row.get("task_balanced_mean")
        if target["task_balanced_mean"] is None:
            require(raw_task_balanced == "", f"unexpected task-balanced value: {key}")
        else:
            require(abs(float(raw_task_balanced) - float(target["task_balanced_mean"])) < 0.00001, f"task-balanced mismatch: {key}")
    require(seen == set(expected), "scope verification did not cover every aggregate")


def verify_model_comparison_numbers(rows: list[dict[str, str]], points: list[dict[str, Any]]) -> None:
    for row in rows:
        entity = row["entity"]
        capability = row.get("capability") or ""
        scope = row["scope"]
        selected = [
            point
            for point in points
            if point["entity"] == entity
            and point["aggregation"] == "consensus"
            and point["judge_key"] == "consensus"
            and (not capability or point["capability"] == capability)
        ]
        selected = select_scope(selected, "all" if scope == "capability" else scope)
        target = independent_stats(point["score"] for point in selected)
        for field in ("n", "mean", "sd", "exact_zero", "exact_five"):
            require(abs(float(row[field]) - float(target[field])) < 0.00001, f"model comparison mismatch: {entity}/{capability}/{scope}/{field}")
        raw_task_balanced = row.get("task_balanced_mean") or ""
        target_task_balanced = independent_task_balanced_mean(selected) if scope == "all" else None
        if target_task_balanced is None:
            require(raw_task_balanced == "", f"unexpected model comparison task-balanced value: {entity}/{capability}/{scope}")
        else:
            require(abs(float(raw_task_balanced) - target_task_balanced) < 0.00001, f"model comparison task-balanced mismatch: {entity}/{capability}/{scope}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    return parser.parse_args()


def main() -> None:
    root = parse_args().root.resolve()
    checksum_files = verify_checksums(root)
    completeness = read_json(root / "manifests" / "completeness.json")
    provenance = read_json(root / "manifests" / "provenance.json")
    verify_snapshot(completeness, provenance)

    items = verify_items(read_jsonl(root / "data" / "items.jsonl"))
    baseline_rows = read_jsonl(root / "data" / "baseline_records.jsonl")
    verify_baseline_rows(baseline_rows, items)
    preference = verify_preference_rows(read_jsonl(root / "data" / "preference_records.jsonl"), items)
    assignment_audit = verify_assignments(
        read_csv(root / "data" / "preference_assignments.csv"), preference, items
    )
    expected_audit = completeness["candidate_fingerprint_audit"]["by_candidate"]
    require(
        {
            candidate_id: {"matches": counts["matches"], "mismatches": counts["mismatches"]}
            for candidate_id, counts in assignment_audit.items()
        }
        == expected_audit,
        "assignment fingerprint audit does not match completeness manifest",
    )
    baseline_judge_rows = read_jsonl(root / "data" / "baseline_judge_details.jsonl")
    baseline_record_rows = read_csv(root / "data" / "baseline_record_scores.csv")
    verify_judges(baseline_judge_rows, baseline_record_rows, baseline_rows)
    model_judge_rows = read_jsonl(root / "data" / "model_judge_details.jsonl")
    model_judge_record_rows = read_csv(root / "data" / "model_judge_record_scores.csv")
    model_scope_rows = read_csv(root / "data" / "judge_scope_analysis.csv")
    verify_model_judges(model_judge_rows, model_judge_record_rows, model_scope_rows)
    verification_points = build_verification_points(
        model_judge_rows,
        model_judge_record_rows,
        baseline_judge_rows,
        baseline_record_rows,
    )
    expected_scope = expected_scope_rows(verification_points)
    verify_scope_numbers(model_scope_rows, expected_scope)
    model_comparison_rows = read_csv(root / "data" / "model_comparison.csv")
    verify_summaries(
        model_comparison_rows,
        read_csv(root / "data" / "preference_summary.csv"),
    )
    verify_model_comparison_numbers(model_comparison_rows, verification_points)
    verify_analysis_comparison(read_json(root / "analysis" / "model_judge_comparison.json"))
    verify_redacted_sources(root)
    verify_no_sensitive_patterns(root)

    print(json.dumps({
        "status": "ok",
        "checksummed_files": checksum_files,
        "tasks": 45,
        "slots": 125,
        "baseline_records": 375,
        "preference_records": 625,
        "preference_assignments": 4375,
        "baseline_judge_scores": 1875,
        "model_judge_scores": 2500,
        "judge_scope_analysis_rows": 150,
    }, indent=2))


if __name__ == "__main__":
    main()
