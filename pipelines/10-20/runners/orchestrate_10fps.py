#!/usr/bin/env python3
"""Run MuseumLive main evaluation and Experiments 5–7 with 10-fps vision.

Every model decision tick remains one second, but the model receives ten
ordered release JPEGs from that second. Outputs are isolated from the finalized
1-fps experiment tree under an external run root.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
EVALUATORS = PIPELINE_ROOT / "evaluators"
ROOT = Path(os.environ["MUSEUMLIVE_10FPS_RUN_ROOT"]).expanduser().resolve()
RUN_ROOT = ROOT / "runs"
TASKS = Path(os.environ["MUSEUMLIVE_TASKS_FILE"]).expanduser().resolve()
BLIND_TASKS = Path(os.environ["MUSEUMLIVE_BLIND_TASKS_FILE"]).expanduser().resolve()
PYTHON = os.environ.get("MUSEUMLIVE_PYTHON", sys.executable)
GPUS = [int(item) for item in os.environ.get("EVAL_GPUS", "2,3,4,5,6,7").split(",")]
NEED_GB = {"minicpm": 28, "venus": 25, "gander": 22, "moss": 38}
LAUNCH_INTERVAL_S = 15
JUDGE_DIR_NAME = "judge_points_lenient_v3"
JUDGE_PROFILE = "lenient_v3"

# name: (model, extra environment, shard count, task file, scored slots)
# Seven persistent shards preserve restartability; six GPUs execute them in
# strict run order and pick up shard 6 when one GPU becomes free.
RUNS = {
    "main_minicpm": ("minicpm", {}, 7, TASKS, 1250),
    "main_venus": ("venus", {}, 7, TASKS, 1250),
    "main_gander": ("gander", {}, 7, TASKS, 1250),
    "main_moss": ("moss", {}, 7, TASKS, 1250),
    "force_venus": ("venus", {"EXP_FORCE": "1"}, 7, TASKS, 1250),
    "force_gander": ("gander", {"EXP_FORCE": "1"}, 7, TASKS, 1250),
    "force_minicpm": ("minicpm", {"EXP_FORCE": "1"}, 7, TASKS, 1250),
    "mute_minicpm": ("minicpm", {"EXP_ABLATE": "mute_ambient"}, 7, TASKS, 1250),
    "black_minicpm": ("minicpm", {"EXP_ABLATE": "black_video"}, 7, TASKS, 1250),
    "notext_moss": ("moss", {"EXP_ABLATE": "no_transcript"}, 7, TASKS, 1250),
    "black_moss": ("moss", {"EXP_ABLATE": "black_video"}, 7, TASKS, 1250),
    "blind_minicpm": ("minicpm", {}, 7, BLIND_TASKS, 1000),
    "blind_moss": ("moss", {}, 7, BLIND_TASKS, 1000),
}
ORDER = list(RUNS)


def log(message: str) -> None:
    line = time.strftime("[%F %T] ") + message
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "orchestrate_10fps.log").open("a") as handle:
        handle.write(line + "\n")


def shard_dirs(name: str) -> list[Path]:
    _, _, count, task_file, _ = RUNS[name]
    tasks = json.loads(task_file.read_text())
    output = []
    for index in range(count):
        directory = RUN_ROOT / name / f"shard{index}"
        directory.mkdir(parents=True, exist_ok=True)
        shard_file = directory / "tasks.json"
        if not shard_file.exists():
            shard_file.write_text(json.dumps(tasks[index::count], ensure_ascii=False))
        output.append(directory)
    return output


def progress(directory: Path) -> tuple[int, int]:
    total = sum(len(task["branches"]) for task in json.loads((directory / "tasks.json").read_text()))
    result_file = directory / "eval/evaluation_results.json"
    try:
        done = len(json.loads(result_file.read_text())["results"]) if result_file.exists() else 0
    except Exception:
        done = -1
    return done, total


def running_outputs() -> dict[str, tuple[int, int | None]]:
    active = {}
    for process in Path("/proc").glob("[0-9]*"):
        try:
            command = (process / "cmdline").read_bytes().split(b"\0")
            if not any(item.endswith((b"eval_duplex.py", b"eval_gander.py", b"eval_moss.py")) for item in command):
                continue
            environment = dict(
                item.split("=", 1)
                for item in (process / "environ").read_bytes().decode(errors="ignore").split("\0")
                if "=" in item
            )
            output = environment.get("EVAL_OUT")
            if not output or str(RUN_ROOT) not in str(Path(output).resolve()):
                continue
            gpu = environment.get("MODEL_DEVICE") or environment.get("CUDA_VISIBLE_DEVICES")
            active[str(Path(output).resolve())] = (
                int(process.name),
                int(gpu) if gpu and gpu.isdigit() else None,
            )
        except Exception:
            continue
    return active


def free_gb() -> dict[int, float]:
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {
        int(index): (int(total) - int(used)) / 1024
        for index, used, total in (line.split(", ") for line in query.strip().splitlines())
    }


def launch(name: str, directory: Path, gpu: int) -> None:
    model, extra, _, _, _ = RUNS[name]
    environment = dict(
        os.environ,
        EVAL_TASKS=str(directory / "tasks.json"),
        EVAL_OUT=str(directory),
        TTS_BACKEND="qwen",
        VISUAL_FPS="10",
        **extra,
    )
    if model in {"minicpm", "venus"}:
        environment.update(MODEL_DEVICE=str(gpu), DUPLEX_MODEL=model)
        command = [PYTHON, "-u", str(EVALUATORS / "eval_duplex.py")]
        cwd = EVALUATORS
    elif model == "moss":
        environment.update(MODEL_DEVICE=str(gpu))
        command = [PYTHON, "-u", str(EVALUATORS / "eval_moss.py")]
        cwd = EVALUATORS
    else:
        environment.update(
            CUDA_VISIBLE_DEVICES=str(gpu),
            PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        )
        command = [PYTHON, "-u", str(EVALUATORS / "eval_gander.py")]
        cwd = EVALUATORS
    with (directory / "run.log").open("a") as handle:
        subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    log(f"launch {name}/{directory.name} on GPU {gpu}")


def start_judge(name: str) -> subprocess.Popen:
    out = RUN_ROOT / name / JUDGE_DIR_NAME
    out.mkdir(parents=True, exist_ok=True)
    pattern = str(RUN_ROOT / name / "shard*/eval/evaluation_results.json")
    if name.startswith("blind_"):
        files = sorted(glob.glob(pattern))
        subprocess.run([PYTHON, str(HERE / "blind_tasks.py"), "restore", *files], check=True)
        pattern = pattern.replace("evaluation_results.json", "evaluation_results.restored.json")
    log(f"{name}: eval complete, judge started")
    handle = (out / "judge.log").open("a")
    return subprocess.Popen(
        [
            PYTHON,
            "-u",
            str(EVALUATORS / "judge_points.py"),
            "run",
            "--eval-glob",
            pattern,
            "--out",
            str(out),
            "--workers",
            "16",
            "--profile",
            JUDGE_PROFILE,
        ],
        cwd=EVALUATORS,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )


def judge_complete(name: str) -> bool:
    summary = RUN_ROOT / name / JUDGE_DIR_NAME / "summary.json"
    if not summary.exists():
        return False
    try:
        payload = json.loads(summary.read_text())
    except Exception:
        return False
    return (
        payload.get("slots") == RUNS[name][4]
        and payload.get("scorable_ok") == RUNS[name][4]
        and payload.get("api_failed") == 0
    )


def status() -> None:
    active = running_outputs()
    for name in ORDER:
        done_sum = total_sum = 0
        pieces = []
        for directory in shard_dirs(name):
            done, total = progress(directory)
            done_sum += max(done, 0)
            total_sum += total
            running = active.get(str(directory.resolve()))
            if running:
                pieces.append(f"{directory.name}=GPU{running[1]}")
        judged = " judge=done" if judge_complete(name) else ""
        print(f"{name:16s} {done_sum:4d}/{total_sum:4d}{judged} {' '.join(pieces)}")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "status":
        status()
        return
    retries: dict[str, int] = {}
    judged = set()
    judging: dict[str, subprocess.Popen] = {}
    last_launch = 0.0
    while True:
        active = running_outputs()
        pending = []
        # Strict run priority: do not place a later model/experiment on a GPU
        # while an earlier run still has unfinished branches.
        for name in ORDER:
            directories = shard_dirs(name)
            unfinished = []
            for directory in directories:
                done, total = progress(directory)
                key = str(directory.resolve())
                if done != total and key not in active:
                    unfinished.append((name, directory))
            if any(progress(directory)[0] != progress(directory)[1] for directory in directories):
                pending.extend(unfinished)
                break
        if pending and time.time() - last_launch >= LAUNCH_INTERVAL_S:
            available = free_gb()
            for name, directory in pending:
                model = RUNS[name][0]
                key = str(directory.resolve())
                if retries.get(key, 0) >= 4:
                    continue
                gpu = max(
                    (gpu for gpu in GPUS if available.get(gpu, 0) >= NEED_GB[model]),
                    key=lambda item: available[item],
                    default=None,
                )
                if gpu is None:
                    break
                retries[key] = retries.get(key, 0) + 1
                launch(name, directory, gpu)
                last_launch = time.time()
                break
        if not judging:
            for name in ORDER:
                if name in judged:
                    continue
                if judge_complete(name):
                    judged.add(name)
                    continue
                if all(progress(directory)[0] == progress(directory)[1] for directory in shard_dirs(name)):
                    judging[name] = start_judge(name)
                    break
        for name, process in list(judging.items()):
            if process.poll() is not None:
                log(f"{name}: judge finished rc={process.returncode}")
                if process.returncode == 0 and judge_complete(name):
                    judged.add(name)
                else:
                    log(f"{name}: judge output incomplete; it will be retried")
                judging.pop(name)
        if len(judged) == len(ORDER):
            log("all 10-fps main and scheduled experiment runs complete")
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
