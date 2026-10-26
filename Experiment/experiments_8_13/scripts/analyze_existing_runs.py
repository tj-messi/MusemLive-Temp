#!/usr/bin/env python3
"""Experiments 8--10 and 12--13 from existing 1 FPS / 10 FPS artifacts.

No evaluated-model inference and no Judge API calls are performed here.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


MODELS = {
    "minicpm": "MiniCPM-o 4.5",
    "venus": "Venus",
    "gander": "Gander",
    "moss": "MOSS*",
}
TRACK_MODEL_DIR = {
    "1fps": lambda root, model: root / model,
    "10fps": lambda root, model: root / f"main_{model}",
}
WINDOW_PROFILES = {
    "tight": {"user_deadline_s": 1.0, "visual_slack_s": 0.0},
    "medium": {"user_deadline_s": 3.0, "visual_slack_s": 1.0},
    "formal": {"user_deadline_s": 6.0, "visual_slack_s": 3.0},
}
FORMAL_USER_DEADLINE = 6.0
FORMAL_VISUAL_SLACK = 3.0
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261021


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def load_jsonl(path: Path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def round_or_none(x, n=4):
    return None if x is None else round(x, n)


def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * p
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def geom(xs):
    xs = list(xs)
    if not xs:
        return 0.0
    if any(x <= 0 for x in xs):
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def strict_slot_pass(row):
    if not row or not row.get("scorable") or row.get("status") != "ok":
        return False
    if row.get("action") != "correct":
        return False
    if row.get("forbidden_hits", 0) or row.get("contradict", 0):
        return False
    statuses = row.get("statuses") or []
    core = row.get("core") or []
    return all((not is_core) or (i < len(statuses) and statuses[i] == "hit")
               for i, is_core in enumerate(core))


def normalized_score(row):
    if not row or row.get("status") != "ok":
        return 0.0
    return max(0.0, min(1.0, float(row.get("score", 0.0)) / 5.0))


def strict_v2_rescore(row, core):
    """Recompute the deterministic strict_v2 score from saved match labels."""
    if row["expected_action"] != "WAIT" and response_is_silent(row):
        return 0.0
    action_credit = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}.get(row.get("action"), 0.0)
    point_credit = {"hit": 1.0, "partial": 0.7, "miss": 0.0, "contradict": 0.0}
    statuses = row.get("statuses") or []
    core_values = [point_credit.get(status, 0.0) for status, is_core in zip(statuses, core) if is_core]
    support_values = [point_credit.get(status, 0.0) for status, is_core in zip(statuses, core) if not is_core]
    if not statuses:
        coverage = None
    elif support_values:
        coverage = 0.8 * (sum(core_values) / len(core_values)) + 0.2 * (sum(support_values) / len(support_values))
    else:
        coverage = sum(core_values) / len(core_values)
    if row.get("correction_mode") and row["expected_action"] != "WAIT":
        mode = row["correction_mode"]
        status = row.get("correction") or "not_corrected"
        maps = {
            "must_correct": {"corrected": 1.0, "partial": 0.4, "not_corrected": 0.0, "reinforced": 0.0},
            "must_follow": {"corrected": 1.0, "partial": 0.3, "not_corrected": 0.0, "reinforced": 0.0},
        }
        correction_credit = maps[mode].get(status, 0.0)
        base = 5.0 * (0.5 * correction_credit + 0.15 * action_credit + 0.35 * (coverage if coverage is not None else action_credit))
        if status == "reinforced":
            base -= 0.5
    elif row["expected_action"] == "WAIT":
        base = 5.0 * action_credit
    elif coverage is None:
        base = 5.0 * action_credit
    else:
        base = 5.0 * (0.3 * action_credit + 0.7 * coverage)
    base -= float(row.get("forbidden_hits", 0))
    base -= 0.25 * float(row.get("contradict", 0))
    return round(max(0.0, min(5.0, base)), 1)


def response_is_silent(row):
    if not row:
        return True
    response = row.get("response")
    return not response or not str(response).strip() or "SILENCE" in str(response).upper()


def response_is_substantive(row):
    if not row:
        return False
    return row.get("observed") not in {"silent", "ack_only", None} and not response_is_silent(row)


def action_prediction(row):
    observed = (row.get("observed") or "").lower()
    if observed == "answer":
        return "SPEAK"
    if observed == "clarify":
        return "CLARIFY"
    if observed in {"silent", "ack_only"} or response_is_silent(row):
        return "WAIT"
    return "OTHER"


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class TrackData:
    def __init__(self, name, root: Path, tasks):
        self.name = name
        self.root = root.resolve()
        self.tasks = tasks
        self.judge = {}
        self.eval = {}
        self.model_meta = {}
        for model in MODELS:
            model_dir = TRACK_MODEL_DIR[name](self.root, model)
            judge_path = model_dir / "judge_points" / "results.jsonl"
            summary_path = model_dir / "judge_points" / "summary.json"
            rubric_path = model_dir / "judge_points" / "rubric.json"
            if not judge_path.exists():
                raise FileNotFoundError(judge_path)
            rows = [r for r in load_jsonl(judge_path) if r.get("status") == "ok"]
            self.judge[model] = {r["judge_id"]: r for r in rows}
            branches = {}
            # The finalized 1 FPS MiniCPM tree preserves 250 already-complete
            # branches under ``existing/`` and the remaining branches under
            # ``shard*/``. Other runs use shards only. Merge every immediate
            # formal eval directory by branch key and reject duplicates.
            files = sorted(glob.glob(str(model_dir / "*" / "eval" / "evaluation_results.json")))
            if not files:
                raise FileNotFoundError(f"no eval files under {model_dir}")
            for path in files:
                for row in load_json(Path(path)).get("results", []):
                    key = row["branch_key"]
                    if key in branches:
                        raise RuntimeError(f"duplicate branch {name}/{model}/{key}")
                    branches[key] = row
            self.eval[model] = branches
            summary = load_json(summary_path)
            self.model_meta[model] = {
                "judge_results": judge_path.relative_to(self.root).as_posix(),
                "judge_results_sha256": sha256(judge_path),
                "rubric_sha256": sha256(rubric_path),
                "slots": len(rows),
                "branches": len(branches),
                "summary_slots": summary.get("slots"),
                "summary_api_failed": summary.get("api_failed"),
                "strict_mean": summary.get("mean"),
            }
            if len(rows) != 1250 or len(branches) != 900:
                raise RuntimeError(f"unexpected completeness {name}/{model}: {len(rows)} slots, {len(branches)} branches")


def harmonize_strict_labels(canonical: TrackData, target: TrackData):
    """Use the 1 FPS point/core schema for cross-FPS deterministic scoring.

    The 10 FPS Gander rubric contains one regenerated core/support label. The
    response match labels themselves are complete, so the affected score can
    be deterministically recomputed without another Judge call.
    """
    audit = {}
    for model in MODELS:
        left = canonical.judge[model]
        right = target.judge[model]
        if set(left) != set(right):
            raise RuntimeError(f"judge-id mismatch while harmonizing {model}")
        changed = []
        for judge_id in sorted(left):
            a, b = left[judge_id], right[judge_id]
            if a.get("points") != b.get("points"):
                raise RuntimeError(f"point text mismatch {model}/{judge_id}")
            canonical_core = a.get("core") or []
            if canonical_core != (b.get("core") or []):
                before = float(b["score"])
                b["core"] = list(canonical_core)
                b["score"] = strict_v2_rescore(b, canonical_core)
                changed.append({"judge_id": judge_id, "score_before": before, "score_after": b["score"]})
        target.model_meta[model]["canonicalized_core_rows"] = changed
        target.model_meta[model]["analysis_strict_mean"] = round_or_none(mean(r["score"] for r in right.values()), 6)
        canonical.model_meta[model]["analysis_strict_mean"] = round_or_none(mean(r["score"] for r in left.values()), 6)
        audit[model] = {"changed_rows": len(changed), "rows": changed}
    return audit


def build_indexes(tasks):
    task_by_id = {t["task_id"]: t for t in tasks}
    slot_meta = {}
    branch_meta = {}
    for task in tasks:
        for branch in task["branches"]:
            bk = f'{task["task_id"]}::{branch["branch_id"]}'
            branch_meta[bk] = (task, branch)
            for slot in branch["slots"]:
                jid = f'{bk}::{slot["slot_id"]}'
                slot_meta[jid] = (task, branch, slot)
    return task_by_id, branch_meta, slot_meta


def source_cluster_bootstrap(task_values, task_by_id, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    """Bootstrap source IDs and keep all task values under sampled sources."""
    by_source = defaultdict(list)
    for task_id, value in task_values.items():
        if value is None:
            continue
        by_source[task_by_id[task_id]["source_id"]].append(value)
    sources = sorted(by_source)
    if not sources:
        return {"n_tasks": 0, "n_sources": 0, "mean": None, "ci95": [None, None]}
    rng = random.Random(seed)
    draws = []
    for _ in range(reps):
        sampled = [rng.choice(sources) for _ in sources]
        vals = [v for source in sampled for v in by_source[source]]
        draws.append(sum(vals) / len(vals))
    all_vals = [v for vals in by_source.values() for v in vals]
    return {
        "n_tasks": len(all_vals),
        "n_sources": len(sources),
        "mean": round_or_none(mean(all_vals)),
        "ci95": [round_or_none(quantile(draws, 0.025)), round_or_none(quantile(draws, 0.975))],
        "bootstrap_reps": reps,
        "seed": seed,
    }


def paired_source_bootstrap(delta_by_task, task_by_id, reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED):
    return source_cluster_bootstrap(delta_by_task, task_by_id, reps=reps, seed=seed)


def experiment8(track: TrackData, tasks, task_by_id):
    result = {"definition": "geometric mean of normalized strict_v2 slot scores over every multi-slot branch", "models": {}}
    task_values_by_model = {}
    for model in MODELS:
        groups = defaultdict(list)
        per_task = defaultdict(list)
        for task in tasks:
            for branch in task["branches"]:
                if len(branch["slots"]) < 2:
                    continue
                rows = [track.judge[model][f'{task["task_id"]}::{branch["branch_id"]}::{s["slot_id"]}']
                        for s in branch["slots"]]
                first, final = rows[0], rows[-1]
                value = {
                    "task_id": task["task_id"],
                    "source_id": task["source_id"],
                    "capability": task["capability"],
                    "branch_id": branch["branch_id"],
                    "first_score": float(first["score"]),
                    "final_score": float(final["score"]),
                    "chain_score": geom(normalized_score(r) for r in rows),
                    "first_strict": strict_slot_pass(first),
                    "final_strict": strict_slot_pass(final),
                    "strict_chain": all(strict_slot_pass(r) for r in rows),
                    "missed_final": response_is_silent(final),
                }
                groups[(task["capability"], branch["branch_id"])].append(value)
                per_task[task["task_id"]].append(value["chain_score"])
        summaries = {}
        for (capability, branch_id), rows in sorted(groups.items()):
            first_ok = sum(r["first_strict"] for r in rows)
            summaries[f"{capability}/{branch_id}"] = {
                "capability": capability,
                "branch_id": branch_id,
                "n": len(rows),
                "first_mean_0_5": round_or_none(mean(r["first_score"] for r in rows)),
                "final_mean_0_5": round_or_none(mean(r["final_score"] for r in rows)),
                "soft_chain_0_1": round_or_none(mean(r["chain_score"] for r in rows)),
                "first_strict_rate": round_or_none(first_ok / len(rows)),
                "final_strict_rate": round_or_none(sum(r["final_strict"] for r in rows) / len(rows)),
                "strict_chain_rate": round_or_none(sum(r["strict_chain"] for r in rows) / len(rows)),
                "missed_final_rate": round_or_none(sum(r["missed_final"] for r in rows) / len(rows)),
                "final_failure_given_first_strict": round_or_none(
                    sum(r["first_strict"] and not r["final_strict"] for r in rows) / first_ok
                ) if first_ok else None,
            }
        task_values = {task_id: mean(values) for task_id, values in per_task.items()}
        task_values_by_model[model] = task_values
        all_rows = [r for rows in groups.values() for r in rows]
        result["models"][model] = {
            "label": MODELS[model],
            "eligible_chains": len(all_rows),
            "soft_chain_0_1": round_or_none(mean(r["chain_score"] for r in all_rows)),
            "strict_chain_rate": round_or_none(sum(r["strict_chain"] for r in all_rows) / len(all_rows)),
            "source_cluster_ci": source_cluster_bootstrap(task_values, task_by_id),
            "by_chain": summaries,
        }
    return result, task_values_by_model


def experiment9(track: TrackData, tasks, task_by_id):
    result = {"definition": "two-level geometric mean across diagnostic slots and the two counterfactual branches", "models": {}}
    soft_task_values = {}
    strict_task_values = {}
    for model in MODELS:
        per_cap = defaultdict(list)
        per_task_soft = {}
        per_task_strict = {}
        for task in tasks:
            diag = set(task.get("diagnostic_slot_ids") or [])
            branches = []
            for branch in task["branches"]:
                rows = [track.judge[model][f'{task["task_id"]}::{branch["branch_id"]}::{s["slot_id"]}']
                        for s in branch["slots"] if s["slot_id"] in diag]
                if not rows:
                    raise RuntimeError(f'no diagnostic rows for {task["task_id"]}/{branch["branch_id"]}')
                branches.append({
                    "branch_id": branch["branch_id"],
                    "soft": geom(normalized_score(r) for r in rows),
                    "strict": all(strict_slot_pass(r) for r in rows),
                })
            if len(branches) != 2:
                raise RuntimeError(f'expected two branches for {task["task_id"]}')
            left, right = branches
            row = {
                "task_id": task["task_id"],
                "left_branch": left["branch_id"],
                "right_branch": right["branch_id"],
                "left_strict": left["strict"],
                "right_strict": right["strict"],
                "cjcs": math.sqrt(left["soft"] * right["soft"]),
                "pair_success": left["strict"] and right["strict"],
            }
            per_cap[task["capability"]].append(row)
            per_task_soft[task["task_id"]] = row["cjcs"]
            per_task_strict[task["task_id"]] = float(row["pair_success"])

        def summarize(rows):
            n = len(rows)
            return {
                "n": n,
                "left_branch": sorted({r["left_branch"] for r in rows}),
                "right_branch": sorted({r["right_branch"] for r in rows}),
                "soft_cjcs_0_1": round_or_none(mean(r["cjcs"] for r in rows)),
                "strict_pair_success": round_or_none(sum(r["pair_success"] for r in rows) / n),
                "both": sum(r["left_strict"] and r["right_strict"] for r in rows),
                "left_only": sum(r["left_strict"] and not r["right_strict"] for r in rows),
                "right_only": sum(not r["left_strict"] and r["right_strict"] for r in rows),
                "neither": sum(not r["left_strict"] and not r["right_strict"] for r in rows),
            }

        all_rows = [r for rows in per_cap.values() for r in rows]
        soft_task_values[model] = per_task_soft
        strict_task_values[model] = per_task_strict
        result["models"][model] = {
            "label": MODELS[model],
            "overall": summarize(all_rows),
            "soft_source_cluster_ci": source_cluster_bootstrap(per_task_soft, task_by_id),
            "strict_source_cluster_ci": source_cluster_bootstrap(per_task_strict, task_by_id),
            "by_capability": {cap: summarize(rows) for cap, rows in sorted(per_cap.items())},
        }
    return result, soft_task_values, strict_task_values


def class_metrics(confusion, labels=("SPEAK", "WAIT", "CLARIFY")):
    out = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[gold][label] for gold in confusion if gold != label)
        fn = sum(confusion[label][pred] for pred in confusion[label] if pred != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        score = f1(precision, recall)
        out[label] = {"precision": round_or_none(precision), "recall": round_or_none(recall), "f1": round_or_none(score), "support": sum(confusion[label].values())}
        f1s.append(score)
    return out, mean(f1s)


def experiment10(track: TrackData, tasks, task_by_id):
    result = {"definition": "content-independent action calibration over SPEAK/WAIT/CLARIFY", "models": {}}
    task_values_by_model = {}
    task_index = {t["task_id"]: t for t in tasks}
    for model in MODELS:
        confusion = defaultdict(Counter)
        per_task = defaultdict(list)
        for row in track.judge[model].values():
            gold = row["expected_action"]
            pred = action_prediction(row)
            confusion[gold][pred] += 1
            per_task[row["task_id"]].append(float(row.get("action") == "correct"))
        classes, macro = class_metrics(confusion)
        task_values = {task_id: mean(vals) for task_id, vals in per_task.items()}
        task_values_by_model[model] = task_values

        def row_for(task_id, branch_id, slot_id):
            return track.judge[model][f"{task_id}::{branch_id}::{slot_id}"]

        dual = {}
        specs = {
            "monitor_keep_cancel": (("keep_monitor", ["sl_alert"]), ("cancel_monitor", ["sl_alert"]), "monitor_cancel"),
            "clarify_ambiguous_specific": (("ambiguous", ["sl_clarify"]), ("specific", ["sl_direct"]), "proactive_clarify"),
            "guidance_needed_direct": (("need_guidance", ["sl_guide", "sl_answer"]), ("no_guidance", ["sl_direct"]), "proactive_guidance"),
        }
        for name, (left_spec, right_spec, capability) in specs.items():
            rows = []
            for task in tasks:
                if task["capability"] != capability:
                    continue
                left = [row_for(task["task_id"], left_spec[0], slot_id) for slot_id in left_spec[1]]
                right = [row_for(task["task_id"], right_spec[0], slot_id) for slot_id in right_spec[1]]
                left_ok = all(r.get("action") == "correct" for r in left)
                right_ok = all(r.get("action") == "correct" for r in right)
                rows.append((left_ok, right_ok, left, right))
            n = len(rows)
            dual[name] = {
                "n": n,
                "left_condition": left_spec[0],
                "right_condition": right_spec[0],
                "left_action_success": round_or_none(sum(r[0] for r in rows) / n),
                "right_action_success": round_or_none(sum(r[1] for r in rows) / n),
                "both_action_success": round_or_none(sum(r[0] and r[1] for r in rows) / n),
                "left_only": sum(r[0] and not r[1] for r in rows),
                "right_only": sum(not r[0] and r[1] for r in rows),
                "neither": sum(not r[0] and not r[1] for r in rows),
            }
            if name == "monitor_keep_cancel":
                cancel_rows = [rr for _, _, _, right in rows for rr in right]
                dual[name]["post_cancel_substantive_false_alert"] = round_or_none(
                    sum(response_is_substantive(r) for r in cancel_rows) / len(cancel_rows)
                )

        result["models"][model] = {
            "label": MODELS[model],
            "confusion": {gold: dict(sorted(preds.items())) for gold, preds in sorted(confusion.items())},
            "per_class": classes,
            "macro_action_f1": round_or_none(macro),
            "action_accuracy": round_or_none(mean(v for vals in per_task.values() for v in vals)),
            "source_cluster_ci": source_cluster_bootstrap(task_values, task_by_id),
            "dual_conditions": dual,
        }
    return result, task_values_by_model


def tts_span(eval_row, event_id):
    for span in (eval_row.get("omni") or {}).get("tts_spans") or []:
        if span.get("event_id") == event_id:
            return span
    return None


def user_event(eval_row, event_id):
    for event in eval_row.get("timeline") or []:
        if event.get("type") == "user_event" and event.get("event_id") == event_id:
            return event
    return None


def response_onset(eval_row, slot, judge_row, visual_slack=FORMAL_VISUAL_SLACK):
    if response_is_silent(judge_row):
        return None
    trigger = slot["trigger"]
    if trigger["kind"] == "user_query":
        event = user_event(eval_row, trigger.get("event_id"))
        return None if not event else event.get("response_time_s")
    lo = float(slot["window"]["start_s"])
    hi = float(slot["window"]["end_s"]) + visual_slack
    turns = [turn for turn in eval_row.get("duplex_turns") or [] if turn["start_s"] <= hi and turn["end_s"] >= lo]
    return min((turn["start_s"] for turn in turns), default=None)


def formal_timing_valid(eval_row, slot, judge_row):
    onset = response_onset(eval_row, slot, judge_row)
    if onset is None:
        return False
    if slot["trigger"]["kind"] == "user_query":
        span = tts_span(eval_row, slot["trigger"].get("event_id"))
        if not span:
            return False
        return span["tts_end"] - 0.5 <= onset <= span["tts_end"] + FORMAL_USER_DEADLINE
    return float(slot["window"]["start_s"]) <= onset <= float(slot["window"]["end_s"]) + FORMAL_VISUAL_SLACK


def branch_unsolicited_turns(eval_row):
    spans = (eval_row.get("omni") or {}).get("tts_spans") or []
    allowed = [(float(s["tts_start"]), float(s["tts_end"]) + FORMAL_USER_DEADLINE) for s in spans]
    for wrapped in eval_row.get("slots") or []:
        slot = wrapped["slot"]
        if slot["trigger"]["kind"] != "user_query":
            allowed.append((float(slot["window"]["start_s"]), float(slot["window"]["end_s"]) + FORMAL_VISUAL_SLACK))
    return sum(not any(lo <= turn["start_s"] <= hi + 1 for lo, hi in allowed)
               for turn in eval_row.get("duplex_turns") or [])


def timing_profile(track: TrackData, model, profile):
    user_deadline = profile["user_deadline_s"]
    visual_slack = profile["visual_slack_s"]
    total_events = hit_events = 0
    speaking_positive = speaking_counted = 0
    for eval_row in track.eval[model].values():
        chunks = eval_row.get("duplex_chunks") or []
        positive = set()
        dont_count = set()
        slot_user_events = set()
        for wrapped in eval_row.get("slots") or []:
            slot = wrapped["slot"]
            if slot["expected_action"] == "WAIT":
                continue
            if slot["trigger"]["kind"] == "user_query":
                event_id = slot["trigger"].get("event_id")
                slot_user_events.add(event_id)
                span = tts_span(eval_row, event_id)
                if not span:
                    continue
                lo, hi = float(span["tts_end"]), float(span["tts_end"]) + user_deadline
            else:
                lo = float(slot["window"]["start_s"])
                hi = float(slot["window"]["end_s"]) + visual_slack
            seconds = {chunk["t"] for chunk in chunks if lo < chunk["t"] <= hi + 1}
            positive |= seconds
            total_events += 1
            if any(not chunk["listen"] for chunk in chunks if chunk["t"] in seconds):
                hit_events += 1
        for span in (eval_row.get("omni") or {}).get("tts_spans") or []:
            if span.get("event_id") in slot_user_events:
                continue
            dont_count |= {chunk["t"] for chunk in chunks
                           if float(span["tts_start"]) < chunk["t"] <= float(span["tts_end"]) + user_deadline + 1}
        dont_count -= positive
        for chunk in chunks:
            if chunk["listen"] or chunk["t"] in dont_count:
                continue
            speaking_counted += 1
            speaking_positive += chunk["t"] in positive
    recall = hit_events / total_events if total_events else 0.0
    precision = speaking_positive / speaking_counted if speaking_counted else 0.0
    return {
        "event_recall": round_or_none(recall),
        "speaking_time_precision": round_or_none(precision),
        "timing_f1": round_or_none(f1(precision, recall)),
        "required_events": total_events,
        "counted_speaking_bins": speaking_counted,
    }


def experiment12(track: TrackData, tasks, slot_meta):
    result = {
        "definition": (
            "For each required SPEAK/CLARIFY slot with normalized strict_v2 quality q, "
            "an in-window response contributes soft TP=q, FP=1-q, FN=1-q; silence "
            "contributes FN=1; an early/late response contributes FP=1 and FN=1. "
            "A substantive response in a WAIT slot and each unmatched model turn add FP=1."
        ),
        "formal_window": {
            "user_query": "response onset from 0.5 s before TTS end through 6.0 s after TTS end",
            "visual_event": "response onset from slot start through 3.0 s after slot end",
        },
        "window_sensitivity_definition": (
            "Timing-only event recall and speaking-bin precision under tight, medium, "
            "and formal windows; saved response content and Judge labels are unchanged."
        ),
        "models": {},
    }
    for model in MODELS:
        tp = fp = fn = 0.0
        invalid_early_or_late = 0
        required = wait_slots = 0
        for judge_id, row in track.judge[model].items():
            task, branch, slot = slot_meta[judge_id]
            eval_row = track.eval[model][f'{task["task_id"]}::{branch["branch_id"]}']
            if row["expected_action"] == "WAIT":
                wait_slots += 1
                if response_is_substantive(row):
                    fp += 1.0
                continue
            required += 1
            q = normalized_score(row)
            if response_is_silent(row):
                fn += 1.0
            elif formal_timing_valid(eval_row, slot, row):
                tp += q
                fp += 1.0 - q
                fn += 1.0 - q
            else:
                invalid_early_or_late += 1
                fp += 1.0
                fn += 1.0
        unsolicited = sum(branch_unsolicited_turns(row) for row in track.eval[model].values())
        fp += unsolicited
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        result["models"][model] = {
            "label": MODELS[model],
            "quality_time_soft_tp": round_or_none(tp),
            "quality_time_soft_fp": round_or_none(fp),
            "quality_time_soft_fn": round_or_none(fn),
            "quality_time_precision": round_or_none(precision),
            "quality_time_recall": round_or_none(recall),
            "quality_time_f1": round_or_none(f1(precision, recall)),
            "required_slots": required,
            "wait_slots": wait_slots,
            "invalid_early_or_late_attempts": invalid_early_or_late,
            "unmatched_turns": unsolicited,
            "window_sensitivity": {name: timing_profile(track, model, profile) for name, profile in WINDOW_PROFILES.items()},
        }
    return result


def main_score_task_values(track: TrackData):
    out = {}
    by_model_cap = {}
    for model in MODELS:
        by_task = defaultdict(list)
        by_cap_task = defaultdict(lambda: defaultdict(list))
        for row in track.judge[model].values():
            by_task[row["task_id"]].append(float(row["score"]))
            by_cap_task[row["capability"]][row["task_id"]].append(float(row["score"]))
        out[model] = {task_id: mean(vals) for task_id, vals in by_task.items()}
        by_model_cap[model] = {cap: {task_id: mean(vals) for task_id, vals in tasks.items()}
                               for cap, tasks in by_cap_task.items()}
    return out, by_model_cap


def experiment13(tracks, task_by_id, task_metrics, harmonization_audit):
    result = {
        "definition": "source-video cluster bootstrap and paired 10 FPS minus 1 FPS deltas under strict_v2",
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "rubric_hash_check": {},
        "strict_label_harmonization": harmonization_audit,
        "models": {},
    }
    for model in MODELS:
        h1 = tracks["1fps"].model_meta[model]["rubric_sha256"]
        h10 = tracks["10fps"].model_meta[model]["rubric_sha256"]
        result["rubric_hash_check"][model] = {
            "one_fps": h1,
            "ten_fps": h10,
            "byte_equal": h1 == h10,
            "canonicalized_core_rows": len(harmonization_audit[model]["rows"]),
        }
        result["models"][model] = {"label": MODELS[model], "metrics": {}}
        for metric_name in ["task_macro_score", "soft_chain", "soft_cjcs", "strict_pair", "action_accuracy"]:
            one = task_metrics["1fps"][metric_name][model]
            ten = task_metrics["10fps"][metric_name][model]
            common = sorted(set(one) & set(ten))
            deltas = {task_id: ten[task_id] - one[task_id] for task_id in common}
            result["models"][model]["metrics"][metric_name] = {
                "n_tasks": len(common),
                "one_fps_mean": round_or_none(mean(one[t] for t in common)),
                "ten_fps_mean": round_or_none(mean(ten[t] for t in common)),
                "paired_delta_10_minus_1": round_or_none(mean(deltas.values())),
                "source_cluster_delta_ci": paired_source_bootstrap(deltas, task_by_id),
            }
    return result


def format_pct(x):
    return "--" if x is None else f"{100*x:.1f}%"


def format_num(x, n=3):
    return "--" if x is None else f"{x:.{n}f}"


def build_track_report(track_name, payload):
    e8, e9, e10, e12 = payload["experiment8"], payload["experiment9"], payload["experiment10"], payload["experiment12"]
    lines = [
        f"# {track_name.upper()} Experiments 8--12 (no model inference)",
        "",
        "All score-based analyses use the frozen `strict_v2` point labels. Experiment 11 is reported separately because it performs resumable sibling-rubric Judge calls.",
        "",
        "## Experiment 8: sequential chain completion",
        "",
        "| Model | Eligible chains | Soft chain | Strict all-stage | Source-cluster 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        row = e8["models"][model]
        ci = row["source_cluster_ci"]["ci95"]
        lines.append(f'| {row["label"]} | {row["eligible_chains"]} | {format_num(row["soft_chain_0_1"])} | {format_pct(row["strict_chain_rate"])} | [{format_num(ci[0])}, {format_num(ci[1])}] |')
    lines += ["", "## Experiment 9: counterfactual joint completion", "",
              "| Model | Soft CJCS | Strict PairSuccess | Both / left-only / right-only / neither |",
              "|---|---:|---:|---:|"]
    for model in MODELS:
        row = e9["models"][model]
        o = row["overall"]
        lines.append(f'| {row["label"]} | {format_num(o["soft_cjcs_0_1"])} | {format_pct(o["strict_pair_success"])} | {o["both"]} / {o["left_only"]} / {o["right_only"]} / {o["neither"]} |')
    lines += ["", "## Experiment 10: interaction-control calibration", "",
              "| Model | Macro action F1 | Action accuracy | Monitor keep | Cancel specificity | Both | Post-cancel substantive alert |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = e10["models"][model]
        mon = row["dual_conditions"]["monitor_keep_cancel"]
        lines.append(f'| {row["label"]} | {format_num(row["macro_action_f1"])} | {format_pct(row["action_accuracy"])} | {format_pct(mon["left_action_success"])} | {format_pct(mon["right_action_success"])} | {format_pct(mon["both_action_success"])} | {format_pct(mon["post_cancel_substantive_false_alert"])} |')
    lines += ["", "## Experiment 12: quality-weighted temporal alignment", "",
              "| Model | QT precision | QT recall | QT-F1 | Tight timing F1 | Medium timing F1 | Formal timing F1 |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for model in MODELS:
        row = e12["models"][model]
        w = row["window_sensitivity"]
        lines.append(f'| {row["label"]} | {format_num(row["quality_time_precision"])} | {format_num(row["quality_time_recall"])} | {format_num(row["quality_time_f1"])} | {format_num(w["tight"]["timing_f1"])} | {format_num(w["medium"]["timing_f1"])} | {format_num(w["formal"]["timing_f1"])} |')
    lines += ["", "## Interpretation boundary", "",
              "For an in-window required response with normalized quality q, the soft counts are TP=q, FP=1-q, FN=1-q. Silence adds FN=1; early/late attempts add FP=1 and FN=1; substantive WAIT violations and unmatched turns add FP=1.",
              "",
              "The quality-time score is a diagnostic soft F1. It does not replace the separately reported content, coverage, waiting, or timing measurements. Tight/medium windows reuse the formal response labels and only test timing sensitivity; they do not re-Judge truncated utterances.", ""]
    return "\n".join(lines)


def build_cross_report(payload):
    lines = ["# Experiment 13: strict-v2 cross-FPS sensitivity", "",
             "Paired deltas are 10 FPS minus 1 FPS. Confidence intervals resample source videos and retain all tasks sharing each sampled source.",
             "`task_macro_score` first averages strict-v2 slot scores within each task, then averages the 450 paired task values; it is not the slot-weighted Experiment 1 leaderboard mean.", ""]
    for metric in ["task_macro_score", "soft_chain", "soft_cjcs", "strict_pair", "action_accuracy"]:
        lines += [f"## {metric}", "", "| Model | 1 FPS | 10 FPS | Paired delta | Source-cluster 95% CI |", "|---|---:|---:|---:|---:|"]
        for model in MODELS:
            row = payload["models"][model]["metrics"][metric]
            ci = row["source_cluster_delta_ci"]["ci95"]
            lines.append(f'| {payload["models"][model]["label"]} | {format_num(row["one_fps_mean"])} | {format_num(row["ten_fps_mean"])} | {format_num(row["paired_delta_10_minus_1"])} | [{format_num(ci[0])}, {format_num(ci[1])}] |')
        lines.append("")
    return "\n".join(lines)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=Path, required=True)
    p.add_argument("--one-fps-root", type=Path, required=True)
    p.add_argument("--ten-fps-root", type=Path, required=True)
    p.add_argument("--repo-root", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    tasks = load_json(args.tasks)
    if len(tasks) != 450:
        raise RuntimeError(f"expected 450 tasks, got {len(tasks)}")
    task_by_id, branch_meta, slot_meta = build_indexes(tasks)
    tracks = {
        "1fps": TrackData("1fps", args.one_fps_root, tasks),
        "10fps": TrackData("10fps", args.ten_fps_root, tasks),
    }
    harmonization_audit = harmonize_strict_labels(tracks["1fps"], tracks["10fps"])
    task_metrics = {}
    provenance = {
        "schema_version": 1,
        "tasks": args.tasks.name,
        "tasks_sha256": sha256(args.tasks),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "tracks": {},
    }
    for name, track in tracks.items():
        e8, chain_values = experiment8(track, tasks, task_by_id)
        e9, cjcs_values, strict_pair_values = experiment9(track, tasks, task_by_id)
        e10, action_values = experiment10(track, tasks, task_by_id)
        e12 = experiment12(track, tasks, slot_meta)
        overall_values, by_cap_values = main_score_task_values(track)
        payload = {
            "schema_version": 1,
            "track": name,
            "judge_profile": "strict_v2",
            "experiment8": e8,
            "experiment9": e9,
            "experiment10": e10,
            "experiment11": {"status": "generated_by_run_swap_judge.py"},
            "experiment12": e12,
        }
        out_dir = args.repo_root / "Experiment" / ("1FPS" if name == "1fps" else "10FPS") / "experiments_8_13"
        write_json(out_dir / "results" / "analysis_8_10_12.json", payload)
        report = build_track_report(name, payload)
        (out_dir / "reports").mkdir(parents=True, exist_ok=True)
        (out_dir / "reports" / "summary_8_12.md").write_text(report.rstrip() + "\n")
        task_metrics[name] = {
            "task_macro_score": overall_values,
            "soft_chain": chain_values,
            "soft_cjcs": cjcs_values,
            "strict_pair": strict_pair_values,
            "action_accuracy": action_values,
        }
        provenance["tracks"][name] = track.model_meta
    e13 = experiment13(tracks, task_by_id, task_metrics, harmonization_audit)
    shared = args.repo_root / "Experiment" / "experiments_8_13"
    write_json(shared / "results" / "analysis_13_cross_fps.json", e13)
    (shared / "reports").mkdir(parents=True, exist_ok=True)
    (shared / "reports" / "experiment_13_cross_fps.md").write_text(
        build_cross_report(e13).rstrip() + "\n"
    )
    write_json(shared / "results" / "provenance.json", provenance)
    print(json.dumps({
        "status": "complete",
        "tracks": {name: {m: tracks[name].model_meta[m] for m in MODELS} for name in tracks},
        "outputs": {
            "1fps": str(args.repo_root / "Experiment/1FPS/experiments_8_13"),
            "10fps": str(args.repo_root / "Experiment/10FPS/experiments_8_13"),
            "cross_fps": str(shared / "results/analysis_13_cross_fps.json"),
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
