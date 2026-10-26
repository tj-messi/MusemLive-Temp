#!/usr/bin/env python3
"""Experiment 11: cross-score paired branch responses against sibling rubrics.

This script does not run evaluated-model inference.  It reuses the frozen
strict_v2 point rubrics and deterministic score function, and only asks the
same point-matching Judge to label response/rubric combinations that do not
already exist.  Raw cross-Judge labels are cached outside the repository.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import sys
import tempfile
import time
import unicodedata
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MODELS = {
    "minicpm": "MiniCPM-o 4.5",
    "venus": "Venus",
    "gander": "Gander",
    "moss": "MOSS*",
}
TRACK_LABELS = {"1fps": "1 FPS", "10fps": "10 FPS"}
TRACK_MODEL_DIR = {
    "1fps": lambda root, model: root / model,
    "10fps": lambda root, model: root / f"main_{model}",
}
PAIR_SLOT = {
    "constraint_update": "sl_main",
    "context_correction": "sl_main",
    "historical_reference": "sl_recall_appearance",
    "monitor_cancel": "sl_alert",
    "task_resume": "sl_main",
    "selective_attention": "sl_main",
}
EXCLUDED_CAPABILITIES = {
    "temporal_update": "paired final targets are identical rather than target-incompatible",
    "proactive_clarify": "asymmetric multi-stage clarification chains are not a direct target swap",
    "proactive_guidance": "asymmetric multi-stage guidance chains are not a direct target swap",
}
CACHE_SCHEMA = "experiment11-swap-v1"
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


def atomic_write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def sha256_bytes(data: bytes):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def quantile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    pos = (len(xs) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return xs[lo]
    return xs[lo] * (hi - pos) + xs[hi] * (pos - lo)


def rounded(x, n=4):
    return None if x is None else round(x, n)


def normalize_text(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    return "".join(ch for ch in text if ch.isalnum())


def tokenize(text):
    text = unicodedata.normalize("NFKC", str(text or "")).lower()
    # Chinese characters are individual lexical units; Latin/digit spans are
    # kept as words.  Punctuation and whitespace do not affect the metric.
    return re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+", text)


def jaccard(left, right):
    left, right = set(left), set(right)
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def response_similarity(left, right):
    nl, nr = normalize_text(left), normalize_text(right)
    return {
        "exact_normalized_identity": nl == nr,
        "token_jaccard": jaccard(tokenize(left), tokenize(right)),
        "character_jaccard": jaccard(nl, nr),
    }


def source_cluster_ci(rows, tasks_by_id, key="swap_gap"):
    by_source = defaultdict(list)
    for row in rows:
        by_source[tasks_by_id[row["task_id"]]["source_id"]].append(float(row[key]))
    sources = sorted(by_source)
    rng = random.Random(BOOTSTRAP_SEED)
    draws = []
    for _ in range(BOOTSTRAP_REPS):
        sampled = [rng.choice(sources) for _ in sources]
        values = [v for source in sampled for v in by_source[source]]
        draws.append(mean(values))
    return {
        "n_sources": len(sources),
        "bootstrap_reps": BOOTSTRAP_REPS,
        "seed": BOOTSTRAP_SEED,
        "ci95": [rounded(quantile(draws, 0.025)), rounded(quantile(draws, 0.975))],
    }


def load_judge_module(repo_root: Path):
    path = repo_root / "pipelines" / "10-20" / "evaluators" / "judge_points.py"
    spec = importlib.util.spec_from_file_location("museumlive_judge_points", path)
    if not spec or not spec.loader:
        raise RuntimeError(f"cannot import Judge module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.set_profile("strict_v2")
    return module


class TrackData:
    def __init__(self, track, root: Path, canonical_rows):
        self.track = track
        self.root = root.resolve()
        self.eval = {}
        self.judge = {}
        self.inputs = {}
        for model in MODELS:
            model_dir = TRACK_MODEL_DIR[track](self.root, model)
            judge_path = model_dir / "judge_points" / "results.jsonl"
            rows = [r for r in load_jsonl(judge_path) if r.get("status") == "ok"]
            judged = {r["judge_id"]: r for r in rows}
            branches = {}
            files = sorted(glob.glob(str(model_dir / "*" / "eval" / "evaluation_results.json")))
            for filename in files:
                for row in load_json(Path(filename)).get("results", []):
                    key = row["branch_key"]
                    if key in branches:
                        raise RuntimeError(f"duplicate branch {track}/{model}/{key}")
                    branches[key] = row
            if len(judged) != 1250 or len(branches) != 900:
                raise RuntimeError(
                    f"unexpected completeness {track}/{model}: "
                    f"{len(judged)} slots, {len(branches)} branches"
                )
            if set(judged) != set(canonical_rows[model]):
                raise RuntimeError(f"Judge ID mismatch against 1 FPS canonical schema: {track}/{model}")
            for judge_id, row in judged.items():
                canonical = canonical_rows[model][judge_id]
                if row.get("points") != canonical.get("points"):
                    raise RuntimeError(f"point-rubric mismatch {track}/{model}/{judge_id}")
            self.eval[model] = branches
            self.judge[model] = judged
            self.inputs[model] = {
                "judge_results": judge_path.relative_to(self.root).as_posix(),
                "judge_results_sha256": sha256_file(judge_path),
                "eval_files": len(files),
                "branches": len(branches),
                "slots": len(judged),
            }


def load_canonical_rows(one_fps_root: Path):
    output = {}
    for model in MODELS:
        path = one_fps_root / model / "judge_points" / "results.jsonl"
        rows = [r for r in load_jsonl(path) if r.get("status") == "ok"]
        output[model] = {r["judge_id"]: r for r in rows}
    return output


def locate_slot(eval_row, slot_id):
    found = [x for x in eval_row["slots"] if x["slot"]["slot_id"] == slot_id]
    if len(found) != 1:
        raise RuntimeError(f"expected one {slot_id} in {eval_row['branch_key']}, got {len(found)}")
    return found[0]


def make_target_row(judge_module, eval_row, eval_slot, target_judge, response, canonical):
    slot = eval_slot["slot"]
    trigger = slot["trigger"]
    history = [
        {
            "t": item["time_s"],
            "user": item.get("utterance", ""),
            "model": item.get("response", ""),
        }
        for item in eval_slot.get("timeline_prefix") or []
        if item.get("type") == "user_event" and item.get("event_id") != trigger.get("event_id")
    ][-4:]
    points = list(slot.get("key_attributes") or canonical.get("points") or [])
    row = {
        "capability": eval_row["capability"],
        "slot_id": slot["slot_id"],
        "expected_action": slot["expected_action"],
        "trigger_kind": trigger["kind"],
        "target_text": slot.get("target_text") or "",
        "key_attributes": slot.get("key_attributes") or [],
        "forbidden_claims": slot.get("forbidden_claims") or [],
        "guidance_direction": slot.get("guidance_direction"),
        "history": history,
        "utterance": target_judge.get("utterance") or "",
        "response": response,
    }
    if eval_row["capability"] == "context_correction":
        row["correction_check"] = judge_module.correction_ctx(eval_row, trigger.get("event_id"))
        if not row["correction_check"]:
            raise RuntimeError(f"missing correction context for {eval_row['branch_key']}")
    return row, points, list(canonical.get("core") or [True] * len(points))


def cache_identity(track, model, target_id, candidate_id, row, points, core, judge_model):
    payload = {
        "schema": CACHE_SCHEMA,
        "track": track,
        "model": model,
        "target_id": target_id,
        "candidate_id": candidate_id,
        "judge_model": judge_model,
        "profile": "strict_v2",
        "expected_action": row["expected_action"],
        "target_text": row["target_text"],
        "points": points,
        "core": core,
        "forbidden_claims": row["forbidden_claims"],
        "utterance": row["utterance"],
        "history": row["history"],
        "response": row["response"],
        "correction_check": row.get("correction_check"),
    }
    digest = sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
    return digest, payload


def deterministic_silent_match(judge_module, row):
    if not judge_module.silent(row["response"]):
        return None
    if row["expected_action"] == "WAIT":
        return {"action": "correct", "observed": "silent", "points": [], "forbidden": []}
    return {"action": "wrong", "observed": "silent", "points": [], "forbidden": []}


def unicode_escape_value(value):
    """Preserve text exactly while making non-ASCII content provider-filter safe.

    The Qwen endpoint occasionally rejects historical/war-song text before it
    reaches the model.  Unicode escape notation is reversible and leaves the
    rubric, response, Judge model, prompt, and deterministic scoring rule
    unchanged.  The fallback is used only after an explicit provider
    ``data_inspection_failed`` error.
    """
    if isinstance(value, str):
        return value.encode("unicode_escape").decode("ascii")
    if isinstance(value, list):
        return [unicode_escape_value(item) for item in value]
    if isinstance(value, dict):
        return {key: unicode_escape_value(item) for key, item in value.items()}
    return value


def provider_filter_fallback_match(judge_module, row, points):
    escaped_row = unicode_escape_value(row)
    escaped_points = unicode_escape_value(points)
    match = judge_module.match(escaped_row, escaped_points)
    match["_judge_fallback"] = "unicode_escape_after_data_inspection_failed"
    return match


def known_provider_filter_failure(cache_root, job):
    path = cache_root / job["track"] / job["model"] / "failures.json"
    if not path.exists():
        return False
    try:
        rows = load_json(path)
    except Exception:
        return False
    return any(
        row.get("target_id") == job["target_id"]
        and row.get("candidate_id") == job["candidate_id"]
        and "data_inspection_failed" in row.get("error", "")
        for row in rows
    )


def cross_job(job, judge_module, cache_root: Path, max_usd):
    digest, cache_payload = cache_identity(
        job["track"], job["model"], job["target_id"], job["candidate_id"],
        job["row"], job["points"], job["core"], judge_module.MODEL,
    )
    cache_path = cache_root / job["track"] / job["model"] / f"{digest}.json"
    deterministic = deterministic_silent_match(judge_module, job["row"])
    if deterministic is not None:
        match = deterministic
        source = "deterministic_silence"
    elif cache_path.exists():
        cached = load_json(cache_path)
        if cached.get("schema") != CACHE_SCHEMA or cached.get("identity") != cache_payload:
            raise RuntimeError(f"cache identity mismatch: {cache_path}")
        match = cached["match"]
        if not judge_module.valid_match(match, job["row"], job["points"]):
            raise RuntimeError(f"invalid cached match: {cache_path}")
        source = "cache"
    else:
        with judge_module.LOCK:
            spent = judge_module.SPENT["usd"]
        if max_usd and spent >= max_usd:
            raise RuntimeError(f"current-run Judge cost cap reached: ${spent:.4f} >= ${max_usd:.4f}")
        if known_provider_filter_failure(cache_root, job):
            match = provider_filter_fallback_match(judge_module, job["row"], job["points"])
        else:
            try:
                match = judge_module.match(job["row"], job["points"])
            except Exception as exc:
                if "data_inspection_failed" not in str(exc):
                    raise
                match = provider_filter_fallback_match(judge_module, job["row"], job["points"])
        atomic_write_json(cache_path, {
            "schema": CACHE_SCHEMA,
            "identity": cache_payload,
            "match": match,
        })
        source = "api"
    score, detail = judge_module.score(job["row"], job["points"], match, job["core"])
    return {
        "direction": job["direction"],
        "target_id": job["target_id"],
        "candidate_id": job["candidate_id"],
        "score": score,
        "source": source,
        "detail": detail,
    }


def summarize_pairs(rows, tasks_by_id):
    def one(group):
        positive = sum(r["swap_gap"] > 1e-9 for r in group)
        negative = sum(r["swap_gap"] < -1e-9 for r in group)
        ties = len(group) - positive - negative
        out = {
            "pairs": len(group),
            "directions": 2 * len(group),
            "own_target_mean": rounded(mean(r["own_target_mean"] for r in group)),
            "swapped_target_mean": rounded(mean(r["swapped_target_mean"] for r in group)),
            "swap_gap": rounded(mean(r["swap_gap"] for r in group)),
            "positive_discrimination_rate": rounded(positive / len(group)),
            "tie_rate": rounded(ties / len(group)),
            "negative_discrimination_rate": rounded(negative / len(group)),
            "exact_normalized_output_identity": rounded(mean(
                float(r["similarity"]["exact_normalized_identity"]) for r in group
            )),
            "mean_token_jaccard": rounded(mean(r["similarity"]["token_jaccard"] for r in group)),
            "mean_character_jaccard": rounded(mean(r["similarity"]["character_jaccard"] for r in group)),
        }
        out["swap_gap_source_cluster"] = source_cluster_ci(group, tasks_by_id)
        return out

    result = one(rows)
    result["by_capability"] = {
        capability: one([r for r in rows if r["capability"] == capability])
        for capability in PAIR_SLOT
    }
    return result


def render_report(track, aggregate):
    lines = [
        f"# Experiment 11: counterfactual branch sensitivity ({TRACK_LABELS[track]})",
        "",
        "Each response is scored once against its own branch target and once, in the",
        "opposite direction, against the sibling branch target. `SwapGap` is",
        "`((own_A + own_B) - (cross_A(y_B) + cross_B(y_A))) / 2` on the 0--5",
        "strict-v2 scale. Positive values indicate branch-sensitive behavior.",
        "",
        "| Model | Pairs | Own target | Swapped target | SwapGap | 95% source CI | Positive / tie / negative | Exact same | Token Jaccard | Char Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, name in MODELS.items():
        row = aggregate["models"][model]
        ci = row["swap_gap_source_cluster"]["ci95"]
        lines.append(
            f"| {name} | {row['pairs']} | {row['own_target_mean']:.3f} | "
            f"{row['swapped_target_mean']:.3f} | {row['swap_gap']:+.3f} | "
            f"[{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{100*row['positive_discrimination_rate']:.1f}% / "
            f"{100*row['tie_rate']:.1f}% / {100*row['negative_discrimination_rate']:.1f}% | "
            f"{100*row['exact_normalized_output_identity']:.1f}% | "
            f"{row['mean_token_jaccard']:.3f} | {row['mean_character_jaccard']:.3f} |"
        )
    lines.extend(["", "## Capability breakdown", ""])
    for capability in PAIR_SLOT:
        lines.extend([
            f"### {capability}",
            "",
            "| Model | Own target | Swapped target | SwapGap | Positive rate | Exact same |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for model, name in MODELS.items():
            row = aggregate["models"][model]["by_capability"][capability]
            lines.append(
                f"| {name} | {row['own_target_mean']:.3f} | {row['swapped_target_mean']:.3f} | "
                f"{row['swap_gap']:+.3f} | {100*row['positive_discrimination_rate']:.1f}% | "
                f"{100*row['exact_normalized_output_identity']:.1f}% |"
            )
        lines.append("")
    lines.extend([
        "## Interpretation boundary",
        "",
        "The cross-score is a rubric-sensitivity diagnostic, not a replacement for",
        "the original leaderboard. It is restricted to six target-incompatible paired",
        "final slots. Temporal-update pairs and asymmetric proactive chains are excluded.",
        "Raw cross-Judge labels are stored only in the external resumable cache.",
        "",
    ])
    return "\n".join(lines)


def validate_tasks(tasks):
    eligible = [t for t in tasks if t["capability"] in PAIR_SLOT]
    counts = defaultdict(int)
    for task in eligible:
        counts[task["capability"]] += 1
        if len(task["branches"]) != 2:
            raise RuntimeError(f"Experiment 11 requires exactly two branches: {task['task_id']}")
        slot_id = PAIR_SLOT[task["capability"]]
        for branch in task["branches"]:
            matches = [s for s in branch["slots"] if s["slot_id"] == slot_id]
            if len(matches) != 1:
                raise RuntimeError(f"missing unique {slot_id}: {task['task_id']}/{branch['branch_id']}")
    if counts != {capability: 50 for capability in PAIR_SLOT}:
        raise RuntimeError(f"unexpected eligible task counts: {dict(counts)}")
    return eligible


def run_model(track, model, tasks, tasks_by_id, data, canonical_rows, judge_module,
              cache_root, workers, max_usd, progress_every):
    pair_records = []
    jobs = []
    own_scores = {}
    for task in tasks:
        slot_id = PAIR_SLOT[task["capability"]]
        branch_a, branch_b = task["branches"]
        ids = []
        responses = []
        target_specs = []
        for branch in (branch_a, branch_b):
            branch_key = f"{task['task_id']}::{branch['branch_id']}"
            judge_id = f"{branch_key}::{slot_id}"
            eval_row = data.eval[model][branch_key]
            eval_slot = locate_slot(eval_row, slot_id)
            own_judge = data.judge[model][judge_id]
            canonical = canonical_rows[model][judge_id]
            own_row, points, core = make_target_row(
                judge_module, eval_row, eval_slot, own_judge, own_judge.get("response"), canonical
            )
            own_score, _ = judge_module.score(own_row, points, own_judge["match"], core)
            own_scores[judge_id] = own_score
            ids.append(judge_id)
            responses.append(own_judge.get("response"))
            target_specs.append((eval_row, eval_slot, own_judge, canonical))

        for target_idx, candidate_idx in ((0, 1), (1, 0)):
            eval_row, eval_slot, target_judge, canonical = target_specs[target_idx]
            cross_row, points, core = make_target_row(
                judge_module, eval_row, eval_slot, target_judge,
                responses[candidate_idx], canonical,
            )
            jobs.append({
                "track": track,
                "model": model,
                "task_id": task["task_id"],
                "capability": task["capability"],
                "direction": f"{branch_a['branch_id']}<-{branch_b['branch_id']}" if target_idx == 0
                             else f"{branch_b['branch_id']}<-{branch_a['branch_id']}",
                "target_id": ids[target_idx],
                "candidate_id": ids[candidate_idx],
                "row": cross_row,
                "points": points,
                "core": core,
            })

        pair_records.append({
            "task_id": task["task_id"],
            "capability": task["capability"],
            "branch_ids": [branch_a["branch_id"], branch_b["branch_id"]],
            "judge_ids": ids,
            "responses": responses,
            "similarity": response_similarity(*responses),
        })

    before_calls = judge_module.SPENT["calls"]
    before_usd = judge_module.SPENT["usd"]
    started = time.time()
    outputs = {}
    failures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_job = {
            executor.submit(cross_job, job, judge_module, cache_root, max_usd): job
            for job in jobs
        }
        for completed, future in enumerate(as_completed(future_to_job), 1):
            job = future_to_job[future]
            try:
                result = future.result()
                outputs[(job["task_id"], result["direction"])] = result
            except Exception as exc:
                failures.append((job, str(exc)))
            if completed % progress_every == 0 or completed == len(jobs):
                calls = judge_module.SPENT["calls"] - before_calls
                usd = judge_module.SPENT["usd"] - before_usd
                print(
                    f"{track}/{model}: {completed}/{len(jobs)} directions, "
                    f"failures={len(failures)}, requests={calls}, ${usd:.4f}, "
                    f"elapsed={time.time() - started:.0f}s",
                    flush=True,
                )

    # One serial retry pass handles transient provider failures without losing
    # successful cached work.
    if failures:
        retry_failures = []
        print(f"{track}/{model}: retrying {len(failures)} failed directions serially", flush=True)
        for job, _ in failures:
            try:
                result = cross_job(job, judge_module, cache_root, max_usd)
                outputs[(job["task_id"], result["direction"])] = result
            except Exception as exc:
                retry_failures.append((job, str(exc)))
        failures = retry_failures
    if failures:
        failure_path = cache_root / track / model / "failures.json"
        atomic_write_json(failure_path, [{
            "target_id": job["target_id"],
            "candidate_id": job["candidate_id"],
            "error": error,
        } for job, error in failures])
        raise RuntimeError(f"{len(failures)} cross-Judge directions failed; see {failure_path}")
    failure_path = cache_root / track / model / "failures.json"
    if failure_path.exists():
        failure_path.unlink()

    source_counts = defaultdict(int)
    for record in pair_records:
        branch_a, branch_b = record["branch_ids"]
        out_a = outputs[(record["task_id"], f"{branch_a}<-{branch_b}")]
        out_b = outputs[(record["task_id"], f"{branch_b}<-{branch_a}")]
        for item in (out_a, out_b):
            source_counts[item["source"]] += 1
        own_a, own_b = (own_scores[x] for x in record["judge_ids"])
        cross_a, cross_b = out_a["score"], out_b["score"]
        record["own_target_mean"] = (own_a + own_b) / 2
        record["swapped_target_mean"] = (cross_a + cross_b) / 2
        record["swap_gap"] = ((own_a + own_b) - (cross_a + cross_b)) / 2
        # Responses are needed only to create cross-Judge jobs; do not include
        # model text in the committed aggregate.
        del record["responses"]

    summary = summarize_pairs(pair_records, tasks_by_id)
    summary["execution"] = {
        "cross_directions": len(jobs),
        "sources": dict(sorted(source_counts.items())),
        "current_run_api_requests": judge_module.SPENT["calls"] - before_calls,
        "current_run_cost_usd": rounded(judge_module.SPENT["usd"] - before_usd, 6),
        "elapsed_s": rounded(time.time() - started, 2),
    }
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--one-fps-root", required=True, type=Path)
    parser.add_argument("--ten-fps-root", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--tracks", nargs="+", choices=TRACK_LABELS, default=list(TRACK_LABELS))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max-usd", type=float, default=150.0,
                        help="safety cap for cost incurred by this process; 0 disables")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = args.repo_root.resolve()
    tasks = load_json(args.tasks.resolve())
    tasks_by_id = {t["task_id"]: t for t in tasks}
    eligible = validate_tasks(tasks)
    print(
        f"Experiment 11 plan: {len(eligible)} pairs/track/model, "
        f"{2 * len(eligible)} cross directions/track/model; "
        f"tracks={args.tracks}, models={args.models}",
        flush=True,
    )
    if args.dry_run:
        return
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required for uncached cross-Judge directions")

    judge_module = load_judge_module(repo_root)
    canonical_rows = load_canonical_rows(args.one_fps_root.resolve())
    roots = {"1fps": args.one_fps_root.resolve(), "10fps": args.ten_fps_root.resolve()}
    cache_root = args.cache_root.resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    for track in args.tracks:
        data = TrackData(track, roots[track], canonical_rows)
        aggregate = {
            "experiment": 11,
            "name": "Counterfactual branch sensitivity",
            "track": track,
            "scale": "strict_v2 0--5",
            "definition": "SwapGap=((own_A+own_B)-(cross_A(y_B)+cross_B(y_A)))/2",
            "judge_model": judge_module.MODEL,
            "judge_profile": "strict_v2",
            "pair_slots": PAIR_SLOT,
            "excluded_capabilities": EXCLUDED_CAPABILITIES,
            "tasks_per_capability": 50,
            "models": {},
            "inputs": {},
        }
        for model in args.models:
            print(f"starting {track}/{model}", flush=True)
            aggregate["models"][model] = run_model(
                track, model, eligible, tasks_by_id, data, canonical_rows,
                judge_module, cache_root, args.workers, args.max_usd,
                args.progress_every,
            )
            aggregate["models"][model]["display_name"] = MODELS[model]
            aggregate["inputs"][model] = data.inputs[model]

        # A partial model selection is useful for diagnostics but must not
        # overwrite the finalized four-model aggregate/report.
        if set(args.models) == set(MODELS):
            track_dir = repo_root / "Experiment" / ("1FPS" if track == "1fps" else "10FPS") / "experiments_8_13"
            result_path = track_dir / "results" / "analysis_11_swap.json"
            report_path = track_dir / "reports" / "experiment_11_swap.md"
            write_json(result_path, aggregate)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_report(track, aggregate))
            print(f"wrote {result_path}", flush=True)
            print(f"wrote {report_path}", flush=True)
        else:
            print(f"not writing finalized {track} aggregate for partial model selection", flush=True)

    print(
        f"current process Judge usage: {judge_module.SPENT['calls']} requests, "
        f"${judge_module.SPENT['usd']:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
