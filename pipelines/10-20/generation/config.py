"""Environment-only configuration for the published 10-20 QA-generation snapshot.

The private runs used the same code with machine-local paths.  This public
snapshot keeps the data/model contract but requires callers to provide all
external roots explicitly.  API credentials are read only from
OPENROUTER_API_KEY by qa_gen/common.py.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = Path(os.environ.get("QA_RUNS", str(HERE / "runs")))
MODEL = os.environ.get("QA_GENERATION_MODEL", "qwen/qwen3.8-omni-flash")

EXTERNAL = Path(os.environ.get("MUSEUMLIVE_WORK_ROOT", "/external/museumlive/work"))
RUN50 = Path(os.environ.get("MUSEUMLIVE_QA50_ROOT", EXTERNAL / "museumlive_50_qa"))
FRAMES25 = Path(os.environ.get("MUSEUMLIVE_25FPS_ROOT", EXTERNAL / "museumlive_25fps"))
AUDIO = Path(os.environ.get("MUSEUMLIVE_AUDIO_ANNOTATION_ROOT", EXTERNAL / "audio_annotations"))
MEDIA50 = Path(os.environ.get("MUSEUMLIVE_MEDIA50_ROOT", EXTERNAL / "media50"))
MEDIA316_MANIFEST = Path(
    os.environ.get("MUSEUMLIVE_MEDIA316_MANIFEST", EXTERNAL / "media316_manifest.json")
)
GAPS316 = Path(os.environ.get("MUSEUMLIVE_GAPS316", EXTERNAL / "gap_seconds_316.json"))

_CONTENT_BLOCKED = {
    value.strip()
    for value in os.environ.get("MUSEUMLIVE_CONTENT_BLOCKED", "").split(",")
    if value.strip()
}

BATCHES = {
    "50": {
        "kind": "zip",
        "windows": Path(os.environ.get("MUSEUMLIVE_WINDOWS50", RUN50 / "inventory/windows.jsonl")),
        "archive": Path(os.environ.get("MUSEUMLIVE_VISUAL50_ARCHIVE", EXTERNAL / "visual50.zip")),
        "media_preflight": Path(os.environ.get("MUSEUMLIVE_MEDIA50_PREFLIGHT", RUN50 / "media_preflight.json")),
        "gap_seconds": None,
        "content_blocked": _CONTENT_BLOCKED,
    },
    "50a": {
        "kind": "dir_audio",
        "src": Path(os.environ.get("MUSEUMLIVE_ANNOTATIONS50A", AUDIO / "annotations_with_audio")),
        "media_dir": MEDIA50,
        "media_preflight": Path(os.environ.get("MUSEUMLIVE_MEDIA50_PREFLIGHT", RUN50 / "media_preflight.json")),
        "gap_seconds": None,
        "content_blocked": _CONTENT_BLOCKED,
    },
    "316": {
        "kind": "dir",
        "src": FRAMES25,
        "media_manifest": MEDIA316_MANIFEST,
        "gap_seconds": GAPS316,
        "content_blocked": set(),
    },
    "316a": {
        "kind": "dir_audio_316",
        "src": Path(os.environ.get("MUSEUMLIVE_ANNOTATIONS316A", AUDIO / "annotations_with_audio_316")),
        "frames": FRAMES25 / "frames",
        "media_manifest": MEDIA316_MANIFEST,
        "gap_seconds": GAPS316,
        "content_blocked": set(),
    },
}
