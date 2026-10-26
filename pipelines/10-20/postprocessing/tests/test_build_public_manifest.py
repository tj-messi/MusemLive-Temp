"""Regression tests for public-snapshot Markdown link validation."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "build_public_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_public_manifest", SCRIPT)
manifest = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(manifest)


class MarkdownLinkValidationTests(unittest.TestCase):
    def test_published_report_resolves_parent_figure_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reports").mkdir()
            (root / "figures").mkdir()
            (root / "figures" / "chart.svg").write_text("<svg/>")
            (root / "reports" / "summary.md").write_text(
                "[chart](../figures/chart.svg)\n"
            )

            self.assertEqual(manifest.validate_local_markdown_links(root), [])

    def test_missing_published_target_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reports").mkdir()
            (root / "reports" / "summary.md").write_text(
                "![chart](figures_1_7/chart.svg)\n"
            )

            self.assertEqual(
                manifest.validate_local_markdown_links(root),
                [
                    "reports/summary.md: missing local target: "
                    "figures_1_7/chart.svg"
                ],
            )

    def test_runtime_report_is_translated_to_public_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "snapshot"
            runtime_report = parent / "report_1_7.md"
            (root / "figures").mkdir(parents=True)
            (root / "figures" / "chart.svg").write_text("<svg/>")
            (root / "reports").mkdir()
            (root / "reports" / "analysis_1_4.md").write_text("analysis")
            (root / "reports" / "analysis_5_7.md").write_text("analysis")
            (root / "results").mkdir()
            (root / "results" / "summary.json").write_text("{}")
            runtime_report.write_text(
                "[chart](figures_1_7/chart.svg)\n\n"
                + manifest.RUNTIME_REPORT_REFERENCES
                + "\n"
            )

            target = manifest.publish_runtime_report(runtime_report, root)

            self.assertEqual(target, root / "reports" / "summary.md")
            text = target.read_text()
            self.assertIn("[chart](../figures/chart.svg)", text)
            self.assertIn("[`analysis_1_4.md`](analysis_1_4.md)", text)
            self.assertIn("[`../results/summary.json`](../results/summary.json)", text)
            self.assertEqual(manifest.validate_public_report_layout(root), [])
            self.assertEqual(manifest.validate_local_markdown_links(root), [])

    def test_public_report_layout_rejects_runtime_names(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "reports").mkdir()
            (root / "reports" / "summary.md").write_text(
                "[chart](figures_1_7/chart.svg)\n\n"
                + manifest.RUNTIME_REPORT_REFERENCES
                + "\n"
            )

            self.assertEqual(
                manifest.validate_public_report_layout(root),
                [
                    "reports/summary.md: still uses runtime figures_1_7 links",
                    "reports/summary.md: still uses runtime analysis/result names",
                    "reports/summary.md: missing public analysis/result links",
                ],
            )

    def test_external_and_fragment_links_are_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text(
                "[web](https://example.com/file) [section](#results)\n"
            )

            self.assertEqual(manifest.validate_local_markdown_links(root), [])

    def test_existing_relative_target_outside_snapshot_is_allowed(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "snapshot"
            root.mkdir()
            (parent / "pipeline.md").write_text("pipeline\n")
            (root / "README.md").write_text("[pipeline](../pipeline.md)\n")

            self.assertEqual(manifest.validate_local_markdown_links(root), [])

    def test_directory_with_markdown_suffix_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompt.md").mkdir()

            self.assertEqual(manifest.validate_local_markdown_links(root), [])


if __name__ == "__main__":
    unittest.main()
