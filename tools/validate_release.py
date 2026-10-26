#!/usr/bin/env python3
"""Validate the frozen benchmark package without external dependencies."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "benchmark/all_tasks.json"
RUBRIC = ROOT / "judge/rubric.json"
SOURCES = ROOT / "benchmark/sources.tsv"
CANDIDATES = ROOT / "benchmark/candidates"
EXPECTED = {
    TASKS: "5e37614d18d900c8c680891ea691911486fabb6d2a4dd575f23604b07648479b",
    RUBRIC: "3f998b202dc885c433315cf5bc1401177ea6600ebb0fcda76e27da00764d017b",
}
TEXT_SUFFIXES = {".md", ".py", ".json", ".csv", ".tsv", ".txt", ".yaml", ".yml", ".example"}
LEAK_PATTERNS = {
    "local Unix project path": re.compile(r"(?:/data/|/home/|/Users/)[^\s\"']+"),
    "local Windows path": re.compile(r"[A-Za-z]:\\\\Users\\\\[^\s\"']+"),
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    "private-key marker": re.compile(r"BEGIN [A-Z ]*PRIVATE KEY"),
    "GitHub token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]+"),
    "OpenRouter key": re.compile(r"sk-or-v1-[A-Za-z0-9]+"),
}
LINK_PATTERN = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot parse {path.relative_to(ROOT)}: {exc}")


def validate_candidates() -> dict[str, int]:
    counts = {}
    for name in ("minicpm", "venus", "gander", "moss"):
        path = CANDIDATES / f"{name}-results.jsonl"
        if not path.is_file():
            fail(f"missing {path.relative_to(ROOT)}")
        records = []
        try:
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            fail(f"cannot parse {path.relative_to(ROOT)} line {line_number}: {exc}")
        if len(records) != 1250:
            fail(f"{path.relative_to(ROOT)} must contain 1250 records, found {len(records)}")
        judge_ids = [record.get("judge_id") for record in records]
        if any(not judge_id for judge_id in judge_ids) or len(set(judge_ids)) != len(judge_ids):
            fail(f"{path.relative_to(ROOT)} must contain unique non-empty judge_id values")
        if any("score" not in record for record in records):
            fail(f"{path.relative_to(ROOT)} contains a record without score")
        counts[name] = len(records)
    return counts


def validate_markdown_links() -> int:
    checked = 0
    broken = []
    for path in ROOT.rglob("*.md"):
        if ".git" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in LINK_PATTERN.finditer(text):
            target = match.group(1).strip()
            if target.startswith(("http://", "https://", "mailto:", "#")):
                continue
            local = target.split("#", 1)[0]
            if not local:
                continue
            checked += 1
            resolved = (path.parent / local).resolve()
            if not resolved.exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    if broken:
        fail("broken local Markdown links:\n" + "\n".join(broken))
    return checked


def main() -> None:
    for path, expected in EXPECTED.items():
        if not path.is_file():
            fail(f"missing {path.relative_to(ROOT)}")
        actual = sha256(path)
        if actual != expected:
            fail(f"hash mismatch for {path.relative_to(ROOT)}: {actual}")

    tasks = load_json(TASKS)
    if not isinstance(tasks, list) or len(tasks) != 450:
        fail("benchmark/all_tasks.json must contain exactly 450 tasks")
    branches = sum(len(task.get("branches", [])) for task in tasks)
    slots = sum(len(branch.get("slots", [])) for task in tasks for branch in task.get("branches", []))
    if branches != 900 or slots != 1250:
        fail(f"unexpected accounting: branches={branches}, slots={slots}")
    languages = Counter(task.get("language") for task in tasks)
    if languages != Counter({"zh": 384, "en": 66}):
        fail(f"unexpected language counts: {dict(languages)}")
    capabilities = Counter(task.get("capability") for task in tasks)
    if len(capabilities) != 9 or set(capabilities.values()) != {50}:
        fail(f"unexpected capability counts: {dict(capabilities)}")

    rubric = load_json(RUBRIC)
    if not isinstance(rubric, dict) or len(rubric) != 4048:
        fail(f"rubric must contain 4048 entries, found {len(rubric) if isinstance(rubric, dict) else 'non-object'}")

    with SOURCES.open(encoding="utf-8-sig", newline="") as handle:
        source_rows = handle.read().splitlines()
    if len(source_rows) != 130:
        fail(f"source manifest must contain 129 data rows, found {len(source_rows) - 1}")

    candidate_counts = validate_candidates()
    link_count = validate_markdown_links()

    leaks = []
    for path in ROOT.rglob("*"):
        if (not path.is_file() or path == Path(__file__).resolve()
                or ".git" in path.parts or path.suffix.lower() not in TEXT_SUFFIXES):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in LEAK_PATTERNS.items():
            if pattern.search(text):
                leaks.append(f"{path.relative_to(ROOT)}: {name}")
    if leaks:
        fail("text files contain sensitive or machine-local content:\n" + "\n".join(leaks))

    print("PASS: MuseumLive frozen release")
    print(f"  tasks={len(tasks)} branches={branches} slots={slots}")
    print(f"  languages={dict(sorted(languages.items()))} capabilities={len(capabilities)}x50")
    print(f"  rubric_entries={len(rubric)} sources={len(source_rows) - 1}")
    print(f"  candidate_records={candidate_counts} markdown_links={link_count}")
    print(f"  task_sha256={EXPECTED[TASKS]}")
    print(f"  rubric_sha256={EXPECTED[RUBRIC]}")


if __name__ == "__main__":
    main()
