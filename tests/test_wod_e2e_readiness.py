from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_wod_e2e_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_wod_e2e_readiness", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodE2EReadinessTests(unittest.TestCase):
    def test_readiness_report_flags_missing_submission_inputs(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "waymo" / "val").mkdir(parents=True)
            (root / "waymo" / "val" / "val_000.tfrecord-00000-of-00001").write_bytes(b"")

            report = module.readiness_report(
                data_root=root / "waymo",
                frame_list=root / "frames.json",
                model=root / "model.json",
                candidates=root / "candidates.jsonl",
                submission=root / "submission.tar.gz",
            )

            self.assertFalse(report["ready_for_training"])
            self.assertFalse(report["ready_for_packaging"])
            self.assertFalse(report["minimal_shot_av_readiness"]["ready_for_leaderboard_upload"])
            self.assertEqual(1, report["splits"]["val"]["shard_count"])
            self.assertEqual("split_dir", report["splits"]["val"]["layout"])
            self.assertIn("missing WOD-E2E train split", report["blockers"][0])

    def test_readiness_report_accepts_generation_inputs(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for split in ("train", "test"):
                (root / "waymo" / split).mkdir(parents=True)
                (root / "waymo" / split / f"{split}_000.tfrecord-00000-of-00001").write_bytes(b"")
            frame_list = root / "frames.json"
            model = root / "model.json"
            candidates = root / "candidates.jsonl"
            frame_list.write_text('{"frame_names":["a"]}', encoding="utf-8")
            model.write_text("{}", encoding="utf-8")
            candidates.write_text("", encoding="utf-8")

            report = module.readiness_report(
                data_root=root / "waymo",
                frame_list=frame_list,
                model=model,
                candidates=candidates,
                submission=root / "submission.tar.gz",
            )

            self.assertTrue(report["ready_for_training"])
            self.assertTrue(report["ready_for_generation"])
            self.assertTrue(report["ready_for_packaging"])
            self.assertFalse(report["ready_for_upload"])
            self.assertFalse(report["minimal_shot_av_readiness"]["leaderboard_truth_available"])

    def test_readiness_report_accepts_flat_root_shard_layout(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            waymo = root / "waymo"
            waymo.mkdir()
            for split in ("train", "test"):
                (waymo / f"{split}_000.tfrecord-00000-of-00001").write_bytes(b"")
            frame_list = root / "frames.json"
            model = root / "model.json"
            candidates = root / "candidates.jsonl"
            frame_list.write_text('{"frame_names":["a"]}', encoding="utf-8")
            model.write_text("{}", encoding="utf-8")
            candidates.write_text("", encoding="utf-8")

            report = module.readiness_report(
                data_root=waymo,
                frame_list=frame_list,
                model=model,
                candidates=candidates,
                submission=root / "submission.tar.gz",
            )

            self.assertTrue(report["ready_for_generation"])
            self.assertEqual("root_flat", report["splits"]["train"]["layout"])
            self.assertEqual("root_flat", report["splits"]["test"]["layout"])

    def test_readiness_report_accepts_pre_registered_submission_matrix(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            waymo = root / "waymo"
            for split in ("train", "test"):
                (waymo / split).mkdir(parents=True)
                (waymo / split / f"{split}_000.tfrecord-00000-of-00001").write_bytes(b"")
            matrix = root / "matrix"
            matrix.mkdir()
            (matrix / "manifest.json").write_text(
                (
                    '{"pre_registered":true,'
                    '"submissions":[{"variant":"a","submission_sha256":"abc",'
                    '"branch":"strict_minimal_shot","minimal_shot_claim_allowed":true,'
                    '"validation_tuned":false}]}'
                ),
                encoding="utf-8",
            )
            (matrix / "UPLOAD_NOTES.md").write_text("upload all", encoding="utf-8")
            model = root / "model.json"
            candidates = root / "candidates.jsonl"
            leaderboard = root / "leaderboard.json"
            model.write_text("{}", encoding="utf-8")
            candidates.write_text("", encoding="utf-8")
            leaderboard.write_text(
                (
                    '{"schema":"wod_e2e_leaderboard_results_v1",'
                    '"results":[{"confirmed_hidden_test":true,'
                    '"hidden_test_rfs":7.2,"submission_sha256":"abc"}]}'
                ),
                encoding="utf-8",
            )
            closed_loop = root / "closed_loop"
            (closed_loop / "run").mkdir(parents=True)
            (closed_loop / "run" / "scenario_eval.json").write_text("{}", encoding="utf-8")
            world = root / "world"
            world.mkdir()
            (world / "wod_cv_world.json").write_text("{}", encoding="utf-8")
            solution = root / "solution-reset.md"
            solution.write_text("This is not a complete driving solution.", encoding="utf-8")

            report = module.readiness_report(
                data_root=waymo,
                frame_list=root / "missing_frame_list.json",
                model=model,
                candidates=candidates,
                submission=root / "submission.tar.gz",
                submission_matrix=matrix,
                leaderboard_results=leaderboard,
                closed_loop_dir=closed_loop,
                world_sweep_dir=world,
                solution_reset=solution,
            )

            readiness = report["minimal_shot_av_readiness"]
            self.assertTrue(readiness["ready_for_blind_matrix"])
            self.assertTrue(readiness["ready_for_leaderboard_upload"])
            self.assertTrue(readiness["leaderboard_truth_available"])
            self.assertTrue(readiness["all_evidence_gates_pass"])
            self.assertEqual(1, report["artifacts"]["submission_matrix"]["submission_count"])
            self.assertEqual(1, report["artifacts"]["submission_matrix"]["strict_branch_count"])

    def test_readiness_report_rejects_validation_tuned_minimal_shot_claim_row(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            matrix = root / "matrix"
            matrix.mkdir()
            (matrix / "manifest.json").write_text(
                (
                    '{"pre_registered":true,'
                    '"submissions":[{"variant":"bad","submission_sha256":"abc",'
                    '"branch":"preference_calibrated_leaderboard",'
                    '"minimal_shot_claim_allowed":true,"validation_tuned":true}]}'
                ),
                encoding="utf-8",
            )

            report = module.readiness_report(
                data_root=root / "waymo",
                frame_list=root / "frames.json",
                model=root / "model.json",
                candidates=root / "candidates.jsonl",
                submission=root / "submission.tar.gz",
                submission_matrix=matrix,
            )

            matrix_report = report["artifacts"]["submission_matrix"]
            self.assertFalse(matrix_report["valid"])
            self.assertEqual(["bad"], matrix_report["invalid_claim_rows"])

    def test_readiness_report_rejects_unconfirmed_leaderboard_log(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            leaderboard = root / "leaderboard.json"
            leaderboard.write_text(
                '{"schema":"wod_e2e_leaderboard_results_v1","results":[{"hidden_test_rfs":7.2}]}',
                encoding="utf-8",
            )

            report = module.readiness_report(
                data_root=root / "waymo",
                frame_list=root / "frames.json",
                model=root / "model.json",
                candidates=root / "candidates.jsonl",
                submission=root / "submission.tar.gz",
                leaderboard_results=leaderboard,
            )

            self.assertFalse(report["minimal_shot_av_readiness"]["leaderboard_truth_available"])
            self.assertFalse(report["artifacts"]["leaderboard_results"]["valid"])


if __name__ == "__main__":
    unittest.main()
