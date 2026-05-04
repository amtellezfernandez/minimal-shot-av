from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_submission_demos.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_submission_demos", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunSubmissionDemosTests(unittest.TestCase):
    def test_track_readmes_are_separate(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            grand = module._write_track_readme(
                root,
                track="grand_commission",
                title="Grand",
                claim="Architecture claim.",
                demo_names=["grand_spotlight_demo"],
            )
            minor = module._write_track_readme(
                root,
                track="minor_commission",
                title="Minor",
                claim="Simulation claim.",
                demo_names=["minor_eval", "minor_ood_eval", "minor_alpasignal_bridge", "minor_runtime"],
            )

            self.assertIn("Architecture claim.", grand.read_text(encoding="utf-8"))
            self.assertIn("Simulation claim.", minor.read_text(encoding="utf-8"))
            self.assertIn("docs/sota-grand-submission-form.md", grand.read_text(encoding="utf-8"))
            self.assertIn("docs/sota-minor-submission-form.md", minor.read_text(encoding="utf-8"))
            self.assertIn("docs/sota-grand-slide-script.md", grand.read_text(encoding="utf-8"))
            self.assertIn("docs/sota-minor-slide-script.md", minor.read_text(encoding="utf-8"))
            self.assertIn("docs/submission-tracks.md", grand.read_text(encoding="utf-8"))
            self.assertIn("docs/submission-tracks.md", minor.read_text(encoding="utf-8"))
            self.assertIn("docs/grand-submission.md", grand.read_text(encoding="utf-8"))
            self.assertIn("docs/minor-simulation-submission.md", minor.read_text(encoding="utf-8"))
            self.assertIn("minor_ood_eval", minor.read_text(encoding="utf-8"))
            self.assertIn("minor_alpasignal_bridge", minor.read_text(encoding="utf-8"))
            self.assertIn("minor_runtime", minor.read_text(encoding="utf-8"))

    def test_archive_contains_track_readme_and_docs(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            readme = module._write_track_readme(
                root,
                track="minor_commission",
                title="Minor",
                claim="Simulation claim.",
                demo_names=[],
            )

            archive = module._write_archive(root, "minor_commission", readme)

            with tarfile.open(archive, "r:gz") as stream:
                names = set(stream.getnames())
            self.assertIn("minor_commission/README.md", names)
            self.assertIn("minor_commission/repo_README.md", names)
            self.assertIn("minor_commission/docs/sota-minor-submission-form.md", names)
            self.assertIn("minor_commission/docs/sota-minor-slide-script.md", names)
            self.assertIn("minor_commission/docs/submission-tracks.md", names)
            self.assertIn("minor_commission/docs/minor-simulation-submission.md", names)

    def test_archive_report_includes_sha_and_members(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            readme = module._write_track_readme(
                root,
                track="grand_commission",
                title="Grand",
                claim="Architecture claim.",
                demo_names=[],
            )
            archive = module._write_archive(root, "grand_commission", readme)

            report = module._archive_report(archive)

            self.assertEqual(str(archive), report["path"])
            self.assertGreater(report["size_bytes"], 0)
            self.assertEqual(64, len(report["sha256"]))
            self.assertIn("grand_commission/README.md", report["members"])

    def test_submission_index_separates_three_tracks(self) -> None:
        module = _load_module()
        manifest = {
            "grand_commission": {
                "path": "artifacts/grand_commission.tar.gz",
                "sha256": "a" * 64,
            },
            "minor_commission": {
                "path": "artifacts/minor_commission.tar.gz",
                "sha256": "b" * 64,
            },
        }

        index = module._submission_index(manifest)

        self.assertIn("SoTA Grand Commission", index)
        self.assertIn("SoTA Minor Commission", index)
        self.assertIn("Waymo WOD-E2E Challenge", index)
        self.assertIn("Do not merge the claims", index)
        self.assertIn("scripts/prepare_wod_e2e_data.py --data-root /home/amdev/waymo", index)
        self.assertIn("scripts/run_wod_leaderboard_attack.py --data-root /home/amdev/waymo", index)
        self.assertIn("Alba Maria Tellez Fernandez", index)


if __name__ == "__main__":
    unittest.main()
