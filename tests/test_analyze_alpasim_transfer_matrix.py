from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_alpasim_transfer_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_alpasim_transfer_matrix", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyzeAlpaSimTransferMatrixTests(unittest.TestCase):
    def test_mcnemar_exact_p_balanced_discordant_is_one(self) -> None:
        module = _load_module()
        self.assertEqual(1.0, module._mcnemar_exact_p(3, 3))

    def test_axis_conflict_flags_progress_up_collision_up(self) -> None:
        module = _load_module()
        flags = module._axis_conflicts(
            ["clip-a"],
            {
                "clip-a": {
                    "raw_collision_any": 0.0,
                    "raw_offroad": 0.0,
                    "raw_wrong_lane": 1.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.1,
                    "raw_dist_to_gt_trajectory": 10.0,
                }
            },
            {
                "clip-a": {
                    "raw_collision_any": 1.0,
                    "raw_offroad": 0.0,
                    "raw_wrong_lane": 0.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.5,
                    "raw_dist_to_gt_trajectory": 8.0,
                }
            },
        )
        self.assertEqual(1, flags["progress_up_collision_up"])
        self.assertEqual(1, flags["wrong_lane_down_collision_flat_or_up"])
        self.assertEqual(1, flags["wrong_lane_down_collision_flat_or_up_valid_scenes"])
        self.assertEqual(1, flags["dist_to_gt_down_collision_up"])

    def test_axis_conflicts_skip_invalid_offroad_and_wrong_lane_scenes(self) -> None:
        module = _load_module()
        flags = module._axis_conflicts(
            ["clip-a"],
            {
                "clip-a": {
                    "raw_collision_any": 0.0,
                    "raw_offroad": 0.0,
                    "raw_wrong_lane": 1.0,
                    "offroad_metric_valid": False,
                    "wrong_lane_metric_valid": False,
                    "raw_progress": 0.1,
                    "raw_dist_to_gt_trajectory": 10.0,
                }
            },
            {
                "clip-a": {
                    "raw_collision_any": 1.0,
                    "raw_offroad": 1.0,
                    "raw_wrong_lane": 0.0,
                    "offroad_metric_valid": False,
                    "wrong_lane_metric_valid": False,
                    "raw_progress": 0.5,
                    "raw_dist_to_gt_trajectory": 8.0,
                }
            },
        )
        self.assertEqual(1, flags["progress_up_collision_up"])
        self.assertEqual(0, flags["progress_up_offroad_up_valid_scenes"])
        self.assertEqual(0, flags["progress_up_offroad_up"])
        self.assertEqual(0, flags["wrong_lane_down_collision_flat_or_up_valid_scenes"])
        self.assertEqual(0, flags["wrong_lane_down_collision_flat_or_up"])

    def test_analyze_matrix_aligns_common_clips_and_formats_report(self) -> None:
        module = _load_module()
        synthetic = {
            "token_dagger_iter2": {
                "clip-a": {
                    "clipgt_id": "clip-a",
                    "raw_collision_any": 1.0,
                    "raw_offroad": 1.0,
                    "raw_wrong_lane": 0.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.1,
                    "raw_dist_traveled_m": 50.0,
                    "raw_dist_to_gt_trajectory": 10.0,
                },
                "clip-b": {
                    "clipgt_id": "clip-b",
                    "raw_collision_any": 0.0,
                    "raw_offroad": 1.0,
                    "raw_wrong_lane": 1.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.2,
                    "raw_dist_traveled_m": 60.0,
                    "raw_dist_to_gt_trajectory": 11.0,
                },
            },
            "token_dagger_iter2_clamped": {
                "clip-a": {
                    "clipgt_id": "clip-a",
                    "raw_collision_any": 0.0,
                    "raw_offroad": 0.0,
                    "raw_wrong_lane": 0.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.4,
                    "raw_dist_traveled_m": 55.0,
                    "raw_dist_to_gt_trajectory": 6.0,
                },
                "clip-b": {
                    "clipgt_id": "clip-b",
                    "raw_collision_any": 1.0,
                    "raw_offroad": 0.0,
                    "raw_wrong_lane": 0.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.3,
                    "raw_dist_traveled_m": 58.0,
                    "raw_dist_to_gt_trajectory": 7.0,
                },
                "clip-extra": {
                    "clipgt_id": "clip-extra",
                    "raw_collision_any": 0.0,
                    "raw_offroad": 0.0,
                    "raw_wrong_lane": 0.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.9,
                    "raw_dist_traveled_m": 90.0,
                    "raw_dist_to_gt_trajectory": 2.0,
                },
            },
        }

        with patch.object(module, "_discover_batches", return_value={key: Path(key) for key in synthetic}), patch.object(
            module,
            "_load_batch_rows",
            side_effect=lambda batch_dir, max_dist_to_gt: synthetic[batch_dir.name],
        ):
            report = module.analyze_matrix(
                Path("/tmp/matrix"),
                baseline_model="token_dagger_iter2",
                scene_preset=None,
                max_dist_to_gt=4.0,
                bootstrap_samples=32,
                seed=7,
            )

        comparison = report["comparisons_vs_baseline"]["token_dagger_iter2_clamped"]
        self.assertEqual(2, comparison["scene_count"])
        self.assertEqual(["clip-a", "clip-b"], comparison["clip_ids"])
        self.assertEqual(2, report["all_models_common_scene_count"])
        self.assertAlmostEqual(0.2, comparison["continuous_metrics"]["raw_progress"]["mean_delta"], places=6)
        self.assertAlmostEqual(-1.0, comparison["binary_metrics"]["raw_offroad"]["delta_rate"], places=6)
        self.assertEqual(2, report["per_model_on_common_scenes"]["token_dagger_iter2_clamped"]["scene_count"])
        self.assertAlmostEqual(
            1.0 / 3.0,
            report["per_model"]["token_dagger_iter2_clamped"]["raw_collision_any"]["rate"],
            places=6,
        )
        self.assertAlmostEqual(
            0.5,
            report["per_model_on_common_scenes"]["token_dagger_iter2_clamped"]["raw_collision_any"]["rate"],
            places=6,
        )

        markdown = module._markdown_report(report)
        self.assertIn("Paired summary table uses the 2 scenes shared by the 2 started models.", markdown)
        self.assertIn("clamped_iter2", markdown)
        self.assertIn("raw_progress", markdown)

    def test_model_summary_excludes_invalid_offroad_and_wrong_lane_rows(self) -> None:
        module = _load_module()
        summary = module._model_summary(
            {
                "clip-a": {
                    "raw_collision_any": 1.0,
                    "raw_offroad": 1.0,
                    "raw_wrong_lane": 1.0,
                    "offroad_metric_valid": True,
                    "wrong_lane_metric_valid": True,
                    "raw_progress": 0.1,
                    "raw_dist_traveled_m": 10.0,
                    "raw_dist_to_gt_trajectory": 1.0,
                },
                "clip-b": {
                    "raw_collision_any": 0.0,
                    "raw_offroad": None,
                    "raw_wrong_lane": None,
                    "offroad_metric_valid": False,
                    "wrong_lane_metric_valid": False,
                    "raw_progress": 0.2,
                    "raw_dist_traveled_m": 20.0,
                    "raw_dist_to_gt_trajectory": 2.0,
                },
            }
        )
        self.assertEqual(2, summary["scene_count"])
        self.assertEqual(1, summary["raw_offroad"]["valid_count"])
        self.assertEqual(1, summary["raw_offroad"]["invalid_count"])
        self.assertAlmostEqual(1.0, summary["raw_offroad"]["rate"], places=6)
        self.assertEqual(1, summary["metric_warning_counts"]["offroad_metric_invalid"])

    def test_metric_warning_flags_detects_missing_offroad_warning(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate = run_dir / "aggregate"
            aggregate.mkdir()
            (aggregate / "metrics_results.txt").write_text(
                "WARNING: No offroad column found in the metrics dataframe. Adding a default value of 0.0.\n",
                encoding="utf-8",
            )
            flags = module._metric_warning_flags(run_dir)
            self.assertFalse(flags["offroad_metric_valid"])
            self.assertFalse(flags["wrong_lane_metric_valid"])

    def test_discover_batches_requires_scene_preset_when_model_has_multiple_presets(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "token_dagger_iter2__front_camera_10scene_smoke").mkdir()
            (root / "token_dagger_iter2__front_camera_30scene_merged").mkdir()
            with self.assertRaises(SystemExit):
                module._discover_batches(root, scene_preset=None)
            batches = module._discover_batches(root, scene_preset="front_camera_30scene_merged")
            self.assertEqual(
                root / "token_dagger_iter2__front_camera_30scene_merged",
                batches["token_dagger_iter2"],
            )

    def test_load_batch_rows_rejects_duplicate_clip_ids(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_a = root / "run_a" / "aggregate"
            run_b = root / "run_b" / "aggregate"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            (run_a / "metrics_unprocessed.parquet").write_text("", encoding="utf-8")
            (run_b / "metrics_unprocessed.parquet").write_text("", encoding="utf-8")

            duplicate_row = [{
                "clipgt_id": "clip-a",
                "raw_collision_any": 0.0,
                "raw_offroad": 0.0,
                "raw_wrong_lane": 0.0,
                "offroad_metric_valid": True,
                "wrong_lane_metric_valid": True,
                "raw_progress": 0.1,
                "raw_dist_traveled_m": 10.0,
                "raw_dist_to_gt_trajectory": 1.0,
            }]

            with patch.object(module, "summarize_run", return_value=duplicate_row):
                with self.assertRaises(ValueError):
                    module._load_batch_rows(root, max_dist_to_gt=4.0)


if __name__ == "__main__":
    unittest.main()
