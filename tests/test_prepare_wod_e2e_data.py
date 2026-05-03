from __future__ import annotations

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_wod_e2e_data.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_wod_e2e_data", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PrepareWodE2EDataTests(unittest.TestCase):
    def test_download_plan_uses_split_globs_and_skips_existing_split(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "wod"
            (data_root / "train").mkdir(parents=True)
            (data_root / "train" / "train_000.tfrecord-00000-of-00001").write_bytes(b"")

            plan = module.download_plan(
                data_root=data_root,
                bucket="gs://bucket",
                splits=["train", "test"],
            )

            self.assertEqual(1, plan["splits"]["train"]["existing_shards"])
            self.assertEqual(0, plan["splits"]["test"]["existing_shards"])
            self.assertFalse(plan["ready_for_blind_matrix"])
            self.assertFalse((data_root / "test").exists())
            self.assertEqual(
                [["gsutil", "-m", "cp", "-n", "gs://bucket/test*", str(data_root / "test")]],
                plan["commands"],
            )

    def test_download_plan_skips_existing_flat_root_shards(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "wod"
            data_root.mkdir()
            (data_root / "test_000.tfrecord-00000-of-00001").write_bytes(b"")

            plan = module.download_plan(
                data_root=data_root,
                bucket="gs://bucket",
                splits=["test"],
            )

            self.assertEqual(1, plan["splits"]["test"]["existing_shards"])
            self.assertEqual("root_flat", plan["splits"]["test"]["existing_layout"])
            self.assertTrue(plan["ready_for_blind_matrix"])
            self.assertEqual([], plan["commands"])

    def test_download_plan_does_not_skip_partial_split(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "wod"
            (data_root / "train").mkdir(parents=True)
            (data_root / "train" / "train_000.tfrecord-00000-of-00003").write_bytes(b"")

            plan = module.download_plan(
                data_root=data_root,
                bucket="gs://bucket",
                splits=["train"],
            )

            self.assertTrue(plan["splits"]["train"]["present"])
            self.assertFalse(plan["splits"]["train"]["complete"])
            self.assertEqual(3, plan["splits"]["train"]["expected_shards"])
            self.assertEqual(2, plan["splits"]["train"]["missing_shards"])
            self.assertEqual(
                [["gsutil", "-m", "cp", "-n", "gs://bucket/train*", str(data_root / "train")]],
                plan["commands"],
            )

    def test_download_plan_does_not_skip_non_contiguous_split(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "wod"
            (data_root / "train").mkdir(parents=True)
            for index in (0, 2):
                (data_root / "train" / f"train_000.tfrecord-{index:05d}-of-00003").write_bytes(b"")

            plan = module.download_plan(
                data_root=data_root,
                bucket="gs://bucket",
                splits=["train"],
            )

            self.assertFalse(plan["splits"]["train"]["complete"])
            self.assertEqual(1, plan["splits"]["train"]["missing_shards"])
            self.assertEqual([1], plan["splits"]["train"]["missing_shard_indices_sample"])
            self.assertEqual(
                [["gsutil", "-m", "cp", "-n", "gs://bucket/train*", str(data_root / "train")]],
                plan["commands"],
            )

    def test_download_plan_includes_blind_matrix_command_with_author_metadata(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "wod"
            (data_root / "test").mkdir(parents=True)
            (data_root / "test" / "test_000.tfrecord-00000-of-00001").write_bytes(b"")

            plan = module.download_plan(
                data_root=data_root,
                bucket="gs://bucket",
                splits=["test"],
                account_name="alba@example.com",
                authors="Alba Maria Tellez Fernandez",
                method_prefix="alba_blind",
                matrix_output_dir=Path("artifacts/matrix"),
            )

            command = plan["two_gate_attack_command"]
            self.assertIn("scripts/run_wod_leaderboard_attack.py", command)
            self.assertIn("--account-name", command)
            self.assertIn("alba@example.com", command)
            self.assertIn("--authors", command)
            self.assertIn("Alba Maria Tellez Fernandez", command)
            self.assertIn("--method-prefix", command)
            self.assertIn("alba_blind", command)


if __name__ == "__main__":
    unittest.main()
