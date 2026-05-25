from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nuplan_border_residual_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanBorderResidualSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "train_nuplan_border_residual_selector")
        cls.cal_module = _load_module(
            ROOT / "scripts" / "train_nuplan_replay_calibrated_selector.py",
            "train_nuplan_replay_calibrated_selector_for_border",
        )

    def test_border_residual_examples_include_weights(self) -> None:
        report = _replay_report()
        baseline = _zero_baseline_selector_payload()
        examples = self.module.build_border_residual_examples(
            report,
            baseline_selector=self.module.load_selector(_write_temp_payload(baseline)),
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
            recoverable_bonus=4.0,
            margin_temperature=0.5,
        )

        self.assertEqual(8, len(examples))
        self.assertTrue(all(example["example_weight"] >= 1.0 for example in examples))

    def test_border_residual_can_improve_over_baseline_on_toy_report(self) -> None:
        report = _replay_report()
        baseline = self.module.load_selector(_write_temp_payload(_zero_baseline_selector_payload()))
        train_report, holdout_report, _split = self.module.split_replay_report_by_db(
            report,
            holdout_fraction=0.5,
            seed=0,
        )
        examples = self.module.build_border_residual_examples(
            train_report,
            baseline_selector=baseline,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
            recoverable_bonus=4.0,
            margin_temperature=0.5,
        )
        residual_selector, _metrics = self.module.fit_replay_value_selector(examples, ridge_alpha=1.0e-3)

        evaluation = self.module.evaluate_selector_family(
            holdout_report,
            baseline_selector=baseline,
            residual_selector=residual_selector,
            near_miss_threshold_m=1.0,
            residual_scale=0.5,
            margin_threshold=5.0,
        )

        self.assertGreaterEqual(
            evaluation["border_residual"]["recoverable_proxy_optimism_fixed_count"],
            evaluation["baseline_selector"]["recoverable_proxy_optimism_fixed_count"],
        )

    def test_cli_main_writes_model_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            baseline_path = tmp_path / "baseline.json"
            output_path = tmp_path / "selector.json"
            report_json = tmp_path / "eval.json"
            report_md = tmp_path / "eval.md"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            baseline_path.write_text(json.dumps(_zero_baseline_selector_payload()), encoding="utf-8")
            argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--baseline-selector-model",
                str(baseline_path),
                "--output",
                str(output_path),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_md),
                "--seed",
                "5",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            evaluation = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_replay_value_selector_v1", payload["model_type"])
            self.assertEqual("nuplan_border_residual_selector_eval_v1", evaluation["schema"])
            self.assertIn("Border Residual Selector", report_md.read_text(encoding="utf-8"))


def _write_temp_payload(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    path = Path(tmp.name)
    tmp.close()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _zero_baseline_selector_payload() -> dict:
    feature_names = [
        "obstacle_pressure",
        "route_blockage",
        "corridor_blocked",
        "left_clearance_m",
        "right_clearance_m",
        "escape_side_float",
        "heading_error_rad",
        "lane_offset_m",
        "route_remaining_m",
        "nearest_actor_distance_m",
        "leading_actor_distance_m",
        "rear_closing_actor_count",
        "crossing_actor_count",
        "candidate_speed_scale",
        "candidate_lateral_offset_m",
        "candidate_proxy_safe",
        "candidate_min_proxy_clearance_m",
        "candidate_final_progress_m",
        "candidate_score_heuristic",
    ]
    return {
        "model_type": "nuplan_replay_value_selector_v1",
        "feature_names": feature_names,
        "feature_mean": [0.0 for _ in feature_names],
        "feature_scale": [1.0 for _ in feature_names],
        "weights": [0.0 for _ in feature_names],
        "bias": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
