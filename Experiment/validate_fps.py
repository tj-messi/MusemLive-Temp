#!/usr/bin/env python3
"""Validate the public FPS protocol metadata and optional 10 FPS status."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MATRIX = ROOT / "10FPS" / "run_matrix.json"
STATUS = ROOT / "10FPS" / "status.json"
RESULTS = ROOT / "10FPS" / "results" / "lenient_v3"


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    matrix = load(MATRIX)
    assert matrix["track"] == "10fps"
    contract = matrix["input_contract"]
    assert contract["decision_hz"] == 1
    assert contract["visual_fps"] == 10
    assert contract["frames_per_tick"] == 10
    assert contract["future_frames_sent"] is False
    assert contract["future_events_sent"] is False

    runs = matrix["runs"]
    assert len(runs) == 13
    assert [run["order"] for run in runs] == list(range(1, 14))
    names = [run["name"] for run in runs]
    assert len(names) == len(set(names))
    assert sum(run["expected_branches"] for run in runs) == 11_900
    assert sum(run["expected_scored_slots"] for run in runs) == 15_750

    if STATUS.exists():
        status = load(STATUS)
        assert status["track"] == "10fps"
        by_name = {item["name"]: item for item in status["runs"]}
        assert set(by_name) == set(names)
        for run in runs:
            item = by_name[run["name"]]
            assert 0 <= item["evaluated_branches"] <= run["expected_branches"]
            assert item["state"] in {
                "queued",
                "evaluating",
                "judging",
                "complete",
            }
            if item["state"] == "complete":
                assert item["evaluated_branches"] == run["expected_branches"]
                assert item["judge_complete"] is True
                summary = load(RESULTS / run["name"] / "summary.json")
                expected_slots = run["expected_scored_slots"]
                assert summary["judge_profile"] == matrix["scheduled_judge_profile"]
                assert summary["slots"] == expected_slots
                assert summary["scorable_ok"] == expected_slots
                assert summary["api_failed"] == 0

    print("FPS metadata validation passed: 13 runs, 11,900 branches")


if __name__ == "__main__":
    main()
