#!/usr/bin/env python3
"""Validate the frozen MuseumLive-Bench-450 task contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

CAPABILITIES = {
    "constraint_update",
    "context_correction",
    "historical_reference",
    "monitor_cancel",
    "proactive_clarify",
    "proactive_guidance",
    "selective_attention",
    "task_resume",
    "temporal_update",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args()

    tasks = json.loads(args.tasks.read_text(encoding="utf-8"))
    if not isinstance(tasks, list):
        raise SystemExit("task file must contain a JSON list")

    errors = []
    task_ids = [task.get("task_id") for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        errors.append("task_id values are not unique")

    capabilities = Counter(task.get("capability") for task in tasks)
    expected_capabilities = {name: 50 for name in CAPABILITIES}
    if dict(capabilities) != expected_capabilities:
        errors.append(f"capability counts differ: {dict(capabilities)}")

    branches = []
    slots = []
    for task in tasks:
        if not {"source_id", "task_id", "capability", "session", "branches"} <= task.keys():
            errors.append(f"missing task fields: {task.get('task_id')}")
            continue
        session = task["session"]
        if float(session["start_s"]) >= float(session["end_s"]):
            errors.append(f"invalid session: {task['task_id']}")
        for branch in task["branches"]:
            key = (task["task_id"], branch.get("branch_id"))
            branches.append(key)
            events = branch.get("user_events", [])
            if [event["time_s"] for event in events] != sorted(event["time_s"] for event in events):
                errors.append(f"unsorted user events: {key}")
            for slot in branch.get("slots", []):
                slots.append((key, slot.get("slot_id")))
                if slot.get("expected_action") not in {"SPEAK", "WAIT", "CLARIFY"}:
                    errors.append(f"invalid expected_action: {key} {slot.get('slot_id')}")

    if len(tasks) != 450:
        errors.append(f"expected 450 tasks, found {len(tasks)}")
    if len(branches) != 900 or len(branches) != len(set(branches)):
        errors.append(f"expected 900 unique branches, found {len(branches)}/{len(set(branches))}")
    if len(slots) != 1250 or len(slots) != len(set(slots)):
        errors.append(f"expected 1250 unique scored slots, found {len(slots)}/{len(set(slots))}")

    actual_hash = digest(args.tasks)
    if args.sha256 and actual_hash != args.sha256:
        errors.append(f"SHA-256 mismatch: {actual_hash}")

    report = {
        "ok": not errors,
        "tasks": len(tasks),
        "branches": len(branches),
        "scored_slots": len(slots),
        "sources": len({task.get("source_id") for task in tasks}),
        "capabilities": dict(sorted(capabilities.items())),
        "sha256": actual_hash,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
