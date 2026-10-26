#!/usr/bin/env python3
"""Build deterministic checksums for a finalized public Experiment snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


INLINE_LINK = re.compile(
    r"!?\[[^\]]*\]\(\s*(?P<target><[^>]+>|[^\s)]+)",
)
REFERENCE_LINK = re.compile(
    r"^\s{0,3}\[[^\]]+\]:\s*(?P<target><[^>]+>|\S+)",
    re.MULTILINE,
)
RUNTIME_REPORT_REFERENCES = (
    "详细逐项表格见 `analysis_a.md`、`analysis_b.md`；完整机器可读结果见 `report_1_7.json`。"
)
PUBLIC_REPORT_REFERENCES = (
    "详细逐项表格见 [`analysis_1_4.md`](analysis_1_4.md) 和\n"
    "[`analysis_5_7.md`](analysis_5_7.md)；完整机器可读结果见\n"
    "[`../results/summary.json`](../results/summary.json)。"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(root: Path, excluded: set[Path]) -> list[dict[str, object]]:
    output = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root)
        if relative in excluded:
            continue
        output.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": digest(path)})
    return output


def local_markdown_targets(text: str) -> list[str]:
    """Return local inline/image/reference targets from a Markdown document."""
    targets = [match.group("target") for match in INLINE_LINK.finditer(text)]
    targets.extend(match.group("target") for match in REFERENCE_LINK.finditer(text))
    return [target.removeprefix("<").removesuffix(">") for target in targets]


def validate_local_markdown_links(root: Path) -> list[str]:
    """Find relative Markdown links whose targets do not exist."""
    root = root.resolve()
    errors: list[str] = []
    for markdown in sorted(path for path in root.rglob("*.md") if path.is_file()):
        for target in local_markdown_targets(markdown.read_text(errors="replace")):
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            relative_target = Path(unquote(parsed.path))
            if relative_target.is_absolute():
                continue
            resolved = (markdown.parent / relative_target).resolve()
            if not resolved.exists():
                errors.append(
                    f"{markdown.relative_to(root).as_posix()}: missing local target: {target}"
                )
    return errors


def public_report_text(runtime_text: str) -> str:
    """Translate the external-run report links to the public snapshot layout."""
    return runtime_text.replace("](figures_1_7/", "](../figures/").replace(
        RUNTIME_REPORT_REFERENCES, PUBLIC_REPORT_REFERENCES
    )


def publish_runtime_report(runtime_report: Path, root: Path) -> Path:
    """Publish a runtime report under reports/ with Experiment-relative links."""
    if not runtime_report.is_file():
        raise SystemExit(f"runtime report not found: {runtime_report}")
    target = root / "reports" / "summary.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        public_report_text(runtime_report.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    return target


def validate_public_report_layout(root: Path) -> list[str]:
    """Reject runtime-only names that are invalid in Experiment/10FPS."""
    report = root / "reports" / "summary.md"
    if not report.is_file():
        return ["reports/summary.md: missing finalized public report"]
    text = report.read_text(encoding="utf-8")
    errors = []
    if "](figures_1_7/" in text:
        errors.append("reports/summary.md: still uses runtime figures_1_7 links")
    if RUNTIME_REPORT_REFERENCES in text:
        errors.append("reports/summary.md: still uses runtime analysis/result names")
    if PUBLIC_REPORT_REFERENCES not in text:
        errors.append("reports/summary.md: missing public analysis/result links")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--finalized-at", required=True)
    parser.add_argument(
        "--runtime-report",
        type=Path,
        help=(
            "optional external report_1_7.md to publish as reports/summary.md "
            "with Experiment-relative figure and result links"
        ),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if args.runtime_report is not None:
        publish_runtime_report(args.runtime_report.resolve(), root)
    validation_errors = validate_public_report_layout(root)
    validation_errors.extend(validate_local_markdown_links(root))
    if validation_errors:
        raise SystemExit(
            "invalid public snapshot Markdown:\n"
            + "\n".join(f"- {error}" for error in validation_errors)
        )

    manifest_dir = root / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "artifact_manifest.json"
    sums_path = manifest_dir / "SHA256SUMS"

    files = records(
        root,
        {Path("manifests/artifact_manifest.json"), Path("manifests/SHA256SUMS")},
    )
    payload = {
        "schema_version": 1,
        "track": "10fps",
        "finalized_at": args.finalized_at,
        "scheduled_runs": 13,
        "evaluated_branches": 11900,
        "alternate_judge_panels": 16,
        "recent30_panels": 2,
        "figures": 6,
        "files": files,
    }
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    sum_files = records(root, {Path("manifests/SHA256SUMS")})
    sums_path.write_text("".join(f"{item['sha256']}  {item['path']}\n" for item in sum_files))
    print(f"manifest written: {len(files)} artifacts, {len(sum_files)} checksums")


if __name__ == "__main__":
    main()
