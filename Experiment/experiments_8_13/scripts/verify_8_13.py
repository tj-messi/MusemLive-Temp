#!/usr/bin/env python3
"""Validate Experiments 8--13 aggregates and build their public manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


MODELS = {"minicpm", "venus", "gander", "moss"}
CAPABILITIES = {
    "constraint_update",
    "context_correction",
    "historical_reference",
    "monitor_cancel",
    "task_resume",
    "selective_attention",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def close(left, right, tolerance=5e-4):
    return abs(float(left) - float(right)) <= tolerance


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def validate_track_8_10_12(path: Path, track):
    payload = load_json(path)
    require(payload["track"] == track, f"track mismatch in {path}")
    require(payload["judge_profile"] == "strict_v2", f"Judge profile mismatch in {path}")
    for experiment in ("experiment8", "experiment9", "experiment10", "experiment12"):
        require(set(payload[experiment]["models"]) == MODELS, f"model set mismatch: {path}/{experiment}")
    for model in MODELS:
        e8 = payload["experiment8"]["models"][model]
        require(e8["eligible_chains"] == 350, f"Experiment 8 chain count: {track}/{model}")
        require(0 <= e8["soft_chain_0_1"] <= 1, f"Experiment 8 score range: {track}/{model}")
        require(0 <= e8["strict_chain_rate"] <= 1, f"Experiment 8 strict range: {track}/{model}")

        e9 = payload["experiment9"]["models"][model]["overall"]
        require(e9["n"] == 450, f"Experiment 9 pair count: {track}/{model}")
        require(e9["both"] + e9["left_only"] + e9["right_only"] + e9["neither"] == 450,
                f"Experiment 9 contingency total: {track}/{model}")

        e10 = payload["experiment10"]["models"][model]
        confusion_total = sum(sum(row.values()) for row in e10["confusion"].values())
        require(confusion_total == 1250, f"Experiment 10 confusion total: {track}/{model}")
        require(0 <= e10["macro_action_f1"] <= 1, f"Experiment 10 F1 range: {track}/{model}")

        e12 = payload["experiment12"]["models"][model]
        require(e12["required_slots"] + e12["wait_slots"] == 1250,
                f"Experiment 12 slot total: {track}/{model}")
        tp, fp, fn = (e12[k] for k in (
            "quality_time_soft_tp", "quality_time_soft_fp", "quality_time_soft_fn"
        ))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        require(close(precision, e12["quality_time_precision"]), f"Experiment 12 precision: {track}/{model}")
        require(close(recall, e12["quality_time_recall"]), f"Experiment 12 recall: {track}/{model}")
        require(close(f1, e12["quality_time_f1"]), f"Experiment 12 F1: {track}/{model}")
        require(set(e12["window_sensitivity"]) == {"tight", "medium", "formal"},
                f"Experiment 12 windows: {track}/{model}")
    return payload


def validate_swap(path: Path, track):
    payload = load_json(path)
    require(payload["experiment"] == 11 and payload["track"] == track, f"Experiment 11 identity: {path}")
    require(payload["judge_profile"] == "strict_v2", f"Experiment 11 profile: {path}")
    require(set(payload["pair_slots"]) == CAPABILITIES, f"Experiment 11 capabilities: {path}")
    require(set(payload["models"]) == MODELS, f"Experiment 11 models: {path}")
    for model, row in payload["models"].items():
        require(row["pairs"] == 300 and row["directions"] == 600,
                f"Experiment 11 completeness: {track}/{model}")
        require(set(row["by_capability"]) == CAPABILITIES,
                f"Experiment 11 capability set: {track}/{model}")
        require(close(row["own_target_mean"] - row["swapped_target_mean"], row["swap_gap"]),
                f"Experiment 11 gap identity: {track}/{model}")
        rate_sum = sum(row[k] for k in (
            "positive_discrimination_rate", "tie_rate", "negative_discrimination_rate"
        ))
        require(close(rate_sum, 1), f"Experiment 11 discrimination rates: {track}/{model}")
        require(sum(row["execution"]["sources"].values()) == 600,
                f"Experiment 11 execution sources: {track}/{model}")
        require(0 <= row["own_target_mean"] <= 5 and 0 <= row["swapped_target_mean"] <= 5,
                f"Experiment 11 score range: {track}/{model}")
        for capability, cap in row["by_capability"].items():
            require(cap["pairs"] == 50 and cap["directions"] == 100,
                    f"Experiment 11 per-cap completeness: {track}/{model}/{capability}")
            require(close(cap["own_target_mean"] - cap["swapped_target_mean"], cap["swap_gap"]),
                    f"Experiment 11 per-cap gap: {track}/{model}/{capability}")
    # The committed aggregate must not contain per-response text or raw matches.
    serialized = json.dumps(payload, ensure_ascii=False)
    require('"match"' not in serialized and '"responses"' not in serialized,
            f"raw Experiment 11 material leaked into aggregate: {path}")
    return payload


def validate_cross_fps(path: Path):
    payload = load_json(path)
    require(set(payload["models"]) == MODELS, f"Experiment 13 models: {path}")
    require(payload["bootstrap_reps"] == 2000, f"Experiment 13 bootstrap reps: {path}")
    for model, model_row in payload["models"].items():
        metrics = model_row["metrics"]
        require("task_macro_score" in metrics and "overall_score" not in metrics,
                f"Experiment 13 task-macro naming: {model}")
        require(metrics["task_macro_score"]["n_tasks"] == 450,
                f"Experiment 13 paired task count: {model}")
        for metric, row in metrics.items():
            expected = row["ten_fps_mean"] - row["one_fps_mean"]
            require(close(expected, row["paired_delta_10_minus_1"]),
                    f"Experiment 13 delta identity: {model}/{metric}")
            ci = row["source_cluster_delta_ci"]["ci95"]
            require(len(ci) == 2 and ci[0] <= ci[1], f"Experiment 13 CI: {model}/{metric}")
    return payload


def validate_provenance(path: Path, external_inputs=None):
    payload = load_json(path)
    require(payload["bootstrap_reps"] == 2000, f"provenance bootstrap reps: {path}")
    require(set(payload["tracks"]) == {"1fps", "10fps"}, f"provenance tracks: {path}")
    if external_inputs:
        tasks_path = external_inputs["tasks"]
        require(tasks_path.exists(), f"missing external task input: {tasks_path}")
        require(sha256(tasks_path) == payload["tasks_sha256"],
                f"external task checksum mismatch: {tasks_path}")
    for track, models in payload["tracks"].items():
        require(set(models) == MODELS, f"provenance model set: {track}")
        for model, row in models.items():
            require(row["slots"] == 1250 and row["branches"] == 900,
                    f"provenance completeness: {track}/{model}")
            require(row["summary_api_failed"] == 0, f"input API failure: {track}/{model}")
            if external_inputs:
                judge_path = external_inputs[track] / row["judge_results"]
                require(judge_path.exists(), f"missing external Judge input: {judge_path}")
                require(sha256(judge_path) == row["judge_results_sha256"],
                        f"external Judge checksum mismatch: {judge_path}")
    return payload


def validate_experiment11_fallback(path: Path):
    payload = load_json(path)
    require(payload["experiment"] == 11 and payload["status"] == "complete",
            f"Experiment 11 fallback status: {path}")
    require(payload["judge_model"] == "qwen/qwen3.8-omni-flash",
            f"Experiment 11 fallback Judge: {path}")
    require(payload["judge_profile"] == "strict_v2" and payload["fallback_api_requests"] == 1,
            f"Experiment 11 fallback protocol: {path}")
    require(payload["provider_error"] == "data_inspection_failed",
            f"Experiment 11 fallback reason: {path}")
    require(payload["raw_cache_committed"] is False,
            f"Experiment 11 fallback cache boundary: {path}")
    return payload


def public_files(repo_root: Path):
    paths = [repo_root / "Experiment" / "experiments_8_13" / "README.md"]
    for base in [
        repo_root / "Experiment" / "experiments_8_13" / "scripts",
        repo_root / "Experiment" / "experiments_8_13" / "results",
        repo_root / "Experiment" / "experiments_8_13" / "reports",
        repo_root / "Experiment" / "1FPS" / "experiments_8_13",
        repo_root / "Experiment" / "10FPS" / "experiments_8_13",
    ]:
        paths.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    manifest_dir = repo_root / "Experiment" / "experiments_8_13" / "manifests"
    paths.append(manifest_dir / "validation.json")
    return sorted(set(p for p in paths if p.exists() and manifest_dir not in p.parents), key=str)


def write_manifest(repo_root: Path, validation):
    manifest_dir = repo_root / "Experiment" / "experiments_8_13" / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    validation_path = manifest_dir / "validation.json"
    with validation_path.open("w") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")

    files = public_files(repo_root)
    # public_files intentionally excludes the manifest directory; include the
    # deterministic validation record explicitly.
    files.append(validation_path)
    files = sorted(set(files), key=str)
    rows = [{
        "path": str(path.relative_to(repo_root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    } for path in files]
    manifest = {
        "schema_version": 1,
        "experiments": [8, 9, 10, 11, 12, 13],
        "tracks": ["1fps", "10fps"],
        "raw_api_cache_committed": False,
        "files": rows,
    }
    manifest_path = manifest_dir / "artifact_manifest.json"
    with manifest_path.open("w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    checksum_rows = rows + [{
        "path": str(manifest_path.relative_to(repo_root)),
        "sha256": sha256(manifest_path),
    }]
    checksum_path = manifest_dir / "SHA256SUMS"
    checksum_path.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in checksum_rows))
    return manifest_path, checksum_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--check-external", action="store_true")
    parser.add_argument("--tasks", type=Path)
    parser.add_argument("--one-fps-root", type=Path)
    parser.add_argument("--ten-fps-root", type=Path)
    parser.add_argument("--write-manifest", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.repo_root.resolve()
    external_inputs = None
    if args.check_external:
        require(args.tasks and args.one_fps_root and args.ten_fps_root,
                "--check-external requires --tasks, --one-fps-root, and --ten-fps-root")
        external_inputs = {
            "tasks": args.tasks.resolve(),
            "1fps": args.one_fps_root.resolve(),
            "10fps": args.ten_fps_root.resolve(),
        }
    checked = []
    for track, dirname in (("1fps", "1FPS"), ("10fps", "10FPS")):
        base = root / "Experiment" / dirname / "experiments_8_13"
        validate_track_8_10_12(base / "results" / "analysis_8_10_12.json", track)
        validate_swap(base / "results" / "analysis_11_swap.json", track)
        for report in (base / "reports" / "summary_8_12.md", base / "reports" / "experiment_11_swap.md"):
            require(report.exists() and report.stat().st_size > 0, f"missing report: {report}")
        checked.append(track)
    shared = root / "Experiment" / "experiments_8_13"
    validate_cross_fps(shared / "results" / "analysis_13_cross_fps.json")
    validate_provenance(shared / "results" / "provenance.json", external_inputs)
    validate_experiment11_fallback(shared / "results" / "experiment_11_fallback.json")
    validation = {
        "status": "complete",
        "experiments": [8, 9, 10, 11, 12, 13],
        "tracks": checked,
        "models_per_track": 4,
        "experiment11_pairs_per_model_track": 300,
        "experiment11_cross_directions_per_model_track": 600,
        "external_input_checksums_verified": bool(args.check_external),
        "raw_api_cache_committed": False,
    }
    if args.write_manifest:
        manifest_path, checksum_path = write_manifest(root, validation)
        validation["manifest"] = str(manifest_path.relative_to(root))
        validation["checksums"] = str(checksum_path.relative_to(root))
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
