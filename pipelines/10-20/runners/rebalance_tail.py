#!/usr/bin/env python3
"""Redistribute unfinished shard6 branches across GPUs 2-7.

The regular orchestrator must be SIGSTOP'ed before this helper starts. Split
outputs live outside runs/<run> so the later judge glob cannot score both
the merged result and the temporary per-GPU results.  Partial split results are
atomically merged into the original shard6 result throughout the run so the
normal status monitor remains accurate.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
EVALUATORS = HERE.parent / "evaluators"
ROOT = Path(os.environ["MUSEUMLIVE_10FPS_RUN_ROOT"]).expanduser().resolve()
PYTHON = os.environ.get("MUSEUMLIVE_PYTHON", sys.executable)
GPUS = [int(item) for item in os.environ.get("EVAL_GPUS", "2,3,4,5,6,7").split(",")]


def now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".rebalance.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    os.replace(temporary, path)


def read_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def result_map(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    payload = read_payload(path)
    return {item["branch_key"]: item for item in payload.get("results", [])}


def branch_order(tasks: list[dict]) -> tuple[list[str], dict[str, int]]:
    keys = [
        f"{task['task_id']}::{branch['branch_id']}"
        for task in tasks
        for branch in task["branches"]
    ]
    return keys, {key: index for index, key in enumerate(keys)}


def write_status(path: Path, **fields: object) -> None:
    fields["updated_at"] = now()
    atomic_json(path, fields)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--orchestrator-pid", type=int)
    parser.add_argument("--restart-orchestrator", action="store_true")
    parser.add_argument("--run-name", default="main_gander")
    parser.add_argument(
        "--model", choices=("gander", "moss", "venus", "minicpm"), default="gander"
    )
    args = parser.parse_args()

    original = ROOT / "runs" / args.run_name / "shard6"
    task_file = original / "tasks.json"
    result_file = original / "eval/evaluation_results.json"
    if args.model in {"venus", "minicpm"}:
        eval_script = EVALUATORS / "eval_duplex.py"
        eval_cwd = EVALUATORS
    else:
        eval_script = EVALUATORS / f"eval_{args.model}.py"
        eval_cwd = EVALUATORS

    tasks = read_payload(task_file)
    original_payload = read_payload(result_file)
    baseline = {item["branch_key"]: item for item in original_payload["results"]}
    ordered_keys, order = branch_order(tasks)
    pending_tasks: list[dict] = []
    for task in tasks:
        for branch in task["branches"]:
            key = f"{task['task_id']}::{branch['branch_id']}"
            if key in baseline:
                continue
            item = copy.deepcopy(task)
            item["branches"] = [copy.deepcopy(branch)]
            pending_tasks.append(item)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    split_root = ROOT / "rebalance" / f"{args.run_name}_shard6_tail_{stamp}"
    split_root.mkdir(parents=True, exist_ok=False)
    status_path = ROOT / "rebalance" / f"{args.run_name}_shard6_tail.status.json"
    manifest = {
        "created_at": now(),
        "orchestrator_pid": args.orchestrator_pid,
        "restart_orchestrator": args.restart_orchestrator,
        "run_name": args.run_name,
        "model": args.model,
        "original_total": len(ordered_keys),
        "baseline_done": len(baseline),
        "pending": len(pending_tasks),
        "split_root": str(split_root),
        "assignments": {},
    }

    assignments: dict[int, list[dict]] = {gpu: [] for gpu in GPUS}
    for index, task in enumerate(pending_tasks):
        assignments[GPUS[index % len(GPUS)]].append(task)
    for gpu in GPUS:
        directory = split_root / f"gpu{gpu}"
        directory.mkdir(parents=True)
        atomic_json(directory / "tasks.json", assignments[gpu])
        manifest["assignments"][str(gpu)] = len(assignments[gpu])
    atomic_json(split_root / "manifest.json", manifest)

    processes: dict[int, subprocess.Popen] = {}
    attempts = {gpu: 0 for gpu in GPUS}

    def launch(gpu: int) -> None:
        directory = split_root / f"gpu{gpu}"
        environment = dict(
            os.environ,
            EVAL_TASKS=str(directory / "tasks.json"),
            EVAL_OUT=str(directory),
            TTS_BACKEND="qwen",
            VISUAL_FPS="10",
        )
        if args.model in {"venus", "minicpm"}:
            environment.update(MODEL_DEVICE=str(gpu), DUPLEX_MODEL=args.model)
            if args.run_name.startswith("force_"):
                environment["EXP_FORCE"] = "1"
        elif args.model == "moss":
            environment["MODEL_DEVICE"] = str(gpu)
        else:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
            environment["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        handle = (directory / "run.log").open("a")
        attempts[gpu] += 1
        processes[gpu] = subprocess.Popen(
            [PYTHON, "-u", str(eval_script)],
            cwd=eval_cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    for gpu in GPUS:
        if assignments[gpu]:
            launch(gpu)

    pending_keys = {
        f"{task['task_id']}::{task['branches'][0]['branch_id']}"
        for task in pending_tasks
    }
    write_status(
        status_path,
        state="running",
        split_root=str(split_root),
        baseline_done=len(baseline),
        pending=len(pending_keys),
        completed_pending=0,
        assignments=manifest["assignments"],
        pids={str(gpu): process.pid for gpu, process in processes.items()},
    )

    try:
        while True:
            merged = dict(baseline)
            per_gpu_done: dict[str, int] = {}
            for gpu in GPUS:
                directory = split_root / f"gpu{gpu}"
                items = result_map(directory / "eval/evaluation_results.json")
                per_gpu_done[str(gpu)] = len(items)
                merged.update(items)

            merged_payload = dict(original_payload)
            merged_payload["results"] = sorted(
                merged.values(), key=lambda item: order[item["branch_key"]]
            )
            atomic_json(result_file, merged_payload)
            completed_pending = len(pending_keys.intersection(merged))

            for gpu, process in list(processes.items()):
                if process.poll() is None:
                    continue
                expected = len(assignments[gpu])
                completed = per_gpu_done.get(str(gpu), 0)
                if completed < expected and attempts[gpu] < 4:
                    time.sleep(2)
                    launch(gpu)

            write_status(
                status_path,
                state="running",
                split_root=str(split_root),
                baseline_done=len(baseline),
                pending=len(pending_keys),
                completed_pending=completed_pending,
                merged_done=len(merged),
                assignments=manifest["assignments"],
                per_gpu_done=per_gpu_done,
                attempts={str(gpu): count for gpu, count in attempts.items()},
                pids={str(gpu): process.pid for gpu, process in processes.items()},
            )

            if completed_pending == len(pending_keys):
                break

            failed = [
                gpu
                for gpu, process in processes.items()
                if process.poll() is not None
                and per_gpu_done.get(str(gpu), 0) < len(assignments[gpu])
                and attempts[gpu] >= 4
            ]
            if failed:
                raise RuntimeError(f"split workers exhausted retries: {failed}")
            time.sleep(10)

        for process in processes.values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)

        final = result_map(result_file)
        missing = [key for key in ordered_keys if key not in final]
        if missing or len(final) != len(ordered_keys):
            raise RuntimeError(
                f"invalid final merge: results={len(final)} missing={len(missing)}"
            )
        write_status(
            status_path,
            state="completed",
            split_root=str(split_root),
            baseline_done=len(baseline),
            pending=len(pending_keys),
            completed_pending=len(pending_keys),
            merged_done=len(final),
            assignments=manifest["assignments"],
        )
        restarted_pid = None
        if args.restart_orchestrator:
            handle = (ROOT / "orchestrate_10fps.restarted.log").open("a")
            restarted = subprocess.Popen(
                [PYTHON, "-u", str(HERE / "orchestrate_10fps.py")],
                cwd=HERE,
                stdout=handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            restarted_pid = restarted.pid
        elif args.orchestrator_pid is not None:
            os.kill(args.orchestrator_pid, signal.SIGCONT)
        if restarted_pid is not None:
            completed = read_payload(status_path)
            completed["restarted_orchestrator_pid"] = restarted_pid
            completed["updated_at"] = now()
            atomic_json(status_path, completed)
        return 0
    except Exception as exc:
        for process in processes.values():
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        write_status(
            status_path,
            state="failed",
            split_root=str(split_root),
            error=repr(exc),
            assignments=manifest["assignments"],
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
