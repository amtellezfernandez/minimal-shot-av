from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCRIPT = ROOT / "scripts" / "audit_sota_submission_bundles.py"
BUILD_SCRIPT = ROOT / "scripts" / "run_submission_demos.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditSotaSubmissionBundlesTests(unittest.TestCase):
    def test_audit_accepts_generated_archives_and_manifest(self) -> None:
        audit = _load(AUDIT_SCRIPT, "audit_sota_submission_bundles")
        builder = _load(BUILD_SCRIPT, "run_submission_demos")
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_required_artifacts(root)
            grand_readme = builder._write_track_readme(
                root,
                track="grand_commission",
                title="Grand",
                claim="Architecture claim.",
                demo_names=[],
            )
            minor_readme = builder._write_track_readme(
                root,
                track="minor_commission",
                title="Minor",
                claim="Simulation claim.",
                demo_names=[],
            )
            grand_archive = builder._write_archive(root, "grand_commission", grand_readme)
            minor_archive = builder._write_archive(root, "minor_commission", minor_readme)
            _write_manifest(root, builder, grand_archive, minor_archive)

            waymo_root = root / "waymo"
            leaderboard = root / "leaderboard.json"
            leaderboard.write_text(
                (
                    '{"schema":"wod_e2e_leaderboard_results_v1",'
                    '"results":[{"confirmed_hidden_test":true,"submission_sha256":"abc"}]}'
                ),
                encoding="utf-8",
            )

            report = audit.audit_bundles(
                root,
                waymo_data_root=waymo_root,
                waymo_leaderboard_results=leaderboard,
            )

            self.assertTrue(report["valid"])
            self.assertTrue(report["submission_tracks"]["sota_grand"]["valid"])
            self.assertTrue(report["submission_tracks"]["sota_minor"]["valid"])
            self.assertFalse(report["submission_tracks"]["waymo_wod_e2e"]["ready_for_claim"])
            self.assertTrue(report["submission_tracks"]["waymo_wod_e2e"]["separate_from_sota"])
            self.assertFalse(report["submission_tracks"]["waymo_wod_e2e"]["test_split_complete"])
            self.assertEqual([], report["tracks"]["grand_commission"]["missing_required"])
            self.assertTrue(report["tracks"]["minor_commission"]["manifest_sha_matches"])

    def test_waymo_track_rejects_partial_test_split(self) -> None:
        audit = _load(AUDIT_SCRIPT, "audit_sota_submission_bundles")
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            test_dir = root / "waymo" / "test"
            test_dir.mkdir(parents=True)
            for index in (0, 2):
                (test_dir / f"test_000.tfrecord-{index:05d}-of-00003").write_bytes(b"")
            leaderboard = root / "leaderboard.json"
            leaderboard.write_text(
                (
                    '{"schema":"wod_e2e_leaderboard_results_v1",'
                    '"results":[{"confirmed_hidden_test":true,"submission_sha256":"abc"}]}'
                ),
                encoding="utf-8",
            )

            report = audit._waymo_track_report(root / "waymo", leaderboard)

            self.assertFalse(report["ready_for_claim"])
            self.assertFalse(report["test_split_complete"])
            self.assertEqual(3, report["test_expected_shards"])
            self.assertEqual(1, report["test_missing_shards"])
            self.assertEqual([1], report["test_missing_shard_indices_sample"])
            self.assertIn("incomplete WOD-E2E test split", "\n".join(report["blockers"]))

    def test_audit_rejects_missing_required_member(self) -> None:
        audit = _load(AUDIT_SCRIPT, "audit_sota_submission_bundles")
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            archive = root / "grand_commission.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                empty = root / "README.md"
                empty.write_text("x", encoding="utf-8")
                stream.add(empty, arcname="grand_commission/README.md")
            (root / "minor_commission.tar.gz").write_bytes(archive.read_bytes())
            manifest = {
                "grand_commission": {"sha256": audit._sha256(archive)},
                "minor_commission": {"sha256": audit._sha256(root / "minor_commission.tar.gz")},
            }
            (root / "submission_bundles_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            report = audit.audit_bundles(root)

            self.assertFalse(report["valid"])
            self.assertIn(
                "grand_commission/docs/sota-grand-submission-form.md",
                report["tracks"]["grand_commission"]["missing_required"],
            )


def _write_manifest(root: Path, builder, grand_archive: Path, minor_archive: Path) -> None:
    manifest = {
        "grand_commission": builder._archive_report(grand_archive),
        "minor_commission": builder._archive_report(minor_archive),
    }
    (root / "submission_bundles_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _write_required_artifacts(root: Path) -> None:
    for path in [
        "grand_spotlight_demo/latest_rollout.json",
        "grand_spotlight_demo/latest_rollout.svg",
        "grand_baseline_spotlight_demo/latest_rollout.json",
        "grand_baseline_spotlight_demo/latest_rollout.svg",
        "minor_construction_seed1/latest_rollout.json",
        "minor_construction_seed1/latest_rollout.svg",
        "minor_fod_seed2/latest_rollout.json",
        "minor_fod_seed2/latest_rollout.svg",
        "minor_spotlight_seed3/latest_rollout.json",
        "minor_spotlight_seed3/latest_rollout.svg",
        "minor_eval/scenario_eval.csv",
        "minor_eval/scenario_eval.json",
    ]:
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("{}" if target.suffix == ".json" else "x", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
