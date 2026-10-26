#!/usr/bin/env python3
"""Freeze source video at 25 FPS and annotate independent one-second windows.\n\nThis is the executed 2026-10-07 visual-annotation implementation with only\nthe filename/location changed for publication. Input selection JSON contains\na videos list with clip_id and media_path. Credentials come from\nOPENROUTER_API_KEY and outputs stay under --output.\n"""

import argparse
import base64
import concurrent.futures
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import requests

MODEL = "qwen/qwen3.8-omni-flash"
URL = "https://openrouter.ai/api/v1/chat/completions"
SYSTEM = """你是视频逐秒标注员。输入是同一秒内按时间排序的25帧，每张图附有帧ID。
只根据当前窗口可见内容，用中文写出主体、状态、动作、镜头变化和可读文字。不要猜测身份、年代、材质、用途，不借用其他秒的信息。不要逐帧机械复述。
只输出JSON：{"summary":"这一秒的概述","events":[{"frame_start":"帧ID","frame_end":"帧ID","description":"可见事实","objects":[],"ocr":[],"uncertainty":"不确定处"}],"limitations":"局限"}。
即使没有变化，也输出一个描述稳定状态的事件；证据范围只能引用本次给出的帧。"""


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as out:
        json.dump(value, out, ensure_ascii=False, indent=2)
        out.write("\n")
        tmp = Path(out.name)
    tmp.replace(path)


def extract(clip, root):
    frame_dir = root / "frames" / clip["clip_id"]
    manifest = root / "manifests" / (clip["clip_id"] + ".json")
    if manifest.exists():
        return json.loads(manifest.read_text(encoding="utf-8"))
    frame_dir.mkdir(parents=True, exist_ok=True)
    if list(frame_dir.glob("F*.jpg")):
        raise RuntimeError("Partial frame extraction without manifest: " + str(frame_dir))
    subprocess.run(["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-i", clip["media_path"],
                    "-vf", "fps=25,scale=640:-2", "-q:v", "4", "-start_number", "1",
                    str(frame_dir / "F%06d.jpg")], check=True)
    frames = sorted(frame_dir.glob("F*.jpg"))
    if not frames:
        raise RuntimeError("No frames: " + clip["clip_id"])
    windows = []
    for second, offset in enumerate(range(0, len(frames) - 24, 25)):
        ids = [frame.name for frame in frames[offset:offset + 25]]
        windows.append({"clip_id": clip["clip_id"], "window_id": f"{clip['clip_id']}_S{second:06d}",
                        "start_s": second, "end_s": second + 1, "frame_ids": ids})
    write_json(manifest, windows)
    print(json.dumps({"stage": "sampling", "clip": clip["clip_id"], "frames": len(frames),
                      "windows": len(windows), "tail_frames_excluded": len(frames) % 25}), flush=True)
    return windows


def validate(value, ids):
    if not isinstance(value, dict) or not isinstance(value.get("summary"), str) or not value.get("events"):
        raise ValueError("Missing summary/events")
    for event in value["events"]:
        if not isinstance(event.get("description"), str) or event.get("frame_start") not in ids or event.get("frame_end") not in ids:
            raise ValueError("Invalid event/evidence")
        if ids.index(event["frame_start"]) > ids.index(event["frame_end"]):
            raise ValueError("Reversed evidence")


def normalize_frame_ids(value, ids):
    by_stem = {Path(frame_id).stem: frame_id for frame_id in ids}
    for event in value.get("events", []):
        for field in ("frame_start", "frame_end"):
            if event.get(field) in by_stem:
                event[field] = by_stem[event[field]]


def annotate(window, root, key, retries):
    path = root / "annotations" / window["clip_id"] / (window["window_id"] + ".json")
    ids = window["frame_ids"]
    if path.exists():
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            validate(old, ids)
            if old.get("model") == MODEL and old.get("frame_ids") == ids:
                return "skipped"
        except (ValueError, KeyError, TypeError):
            pass
    content = [{"type": "text", "text": f"视频 {window['clip_id']}，时间 {window['start_s']} 至 {window['end_s']} 秒；以下帧按时间排序。"}]
    for frame_id in ids:
        encoded = base64.b64encode((root / "frames" / window["clip_id"] / frame_id).read_bytes()).decode("ascii")
        content.extend([{"type": "text", "text": frame_id},
                        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}}])
    body = {"model": MODEL, "temperature": 0.1, "max_tokens": 2000,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]}
    session = requests.Session()
    session.trust_env = False
    for attempt in range(retries):
        try:
            response = session.post(URL, json=body, headers={"Authorization": "Bearer " + key}, timeout=(30, 240))
            if response.status_code == 429 or response.status_code >= 500:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:200]}")
            if response.status_code >= 400:
                raise requests.HTTPError(f"HTTP {response.status_code}: {response.text[:500]}")
            payload = response.json()
            raw = payload["choices"][0]["message"]["content"]
            if raw is None:
                raise ValueError("Empty model content")
            if isinstance(raw, list):
                raw = "".join(part.get("text", "") for part in raw if isinstance(part, dict))
            raw = raw.strip()
            if raw.startswith("~~~") or raw.startswith(chr(96) * 3):
                raw = raw.split("\n", 1)[1].rsplit(raw[:3], 1)[0].strip()
            value = json.loads(raw)
            if attempt == 0:
                write_json(root / "diagnostics" / (window["window_id"] + ".json"), value)
            normalize_frame_ids(value, ids)
            validate(value, ids)
            value.update({"schema_version": "museumlive_basic_understanding_1s25_v1", "model": MODEL,
                          "clip_id": window["clip_id"], "window_id": window["window_id"],
                          "start_s": window["start_s"], "end_s": window["end_s"],
                          "fps": 25, "frame_count": len(ids), "frame_ids": ids,
                          "usage": payload.get("usage", {})})
            write_json(path, value)
            return "ok"
        except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:250]}"
            time.sleep(min(30, 2 ** attempt))
    return "failed: " + error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clip", action="append")
    parser.add_argument("--limit-seconds", type=int)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        parser.error("OPENROUTER_API_KEY required")
    clips = json.loads(args.selection.read_text(encoding="utf-8"))["videos"]
    if args.clip:
        clips = [clip for clip in clips if clip["clip_id"] in args.clip]
    if not clips:
        parser.error("No clips selected")
    failures = 0
    total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(args.workers, 64))) as pool:
        for clip in clips:
            windows = extract(clip, args.output)
            if args.limit_seconds is not None:
                windows = windows[:args.limit_seconds]
            print(json.dumps({"stage": "annotation", "clip": clip["clip_id"], "total": len(windows), "status": "start"}), flush=True)
            tasks = {pool.submit(annotate, window, args.output, key, args.retries): window for window in windows}
            for done, future in enumerate(concurrent.futures.as_completed(tasks), 1):
                result = future.result()
                total += 1
                if result.startswith("failed"):
                    failures += 1
                    print(json.dumps({"window": tasks[future]["window_id"], "status": result}), flush=True)
                if done % 25 == 0 or done == len(windows):
                    print(json.dumps({"stage": "annotation", "clip": clip["clip_id"], "done": done,
                                      "total": len(windows), "failures": failures}), flush=True)
    print(json.dumps({"stage": "complete", "windows": total, "failed": failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
