from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nuplan_replay_calibration_experiments.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunNuPlanReplayCalibrationExperimentsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "run_nuplan_replay_calibration_experiments")

    def test_repeated_split_report_contains_sweep_and_ablations(self) -> None:
        report = self.module.run_replay_calibration_experiments(
            _replay_report(),
            input_replay_json="synthetic.json",
            split_count=2,
            seed_start=0,
            holdout_fraction=0.5,
            near_miss_threshold_m=1.0,
            risk_weights=[0.0, 10.0],
            risk_thresholds=[0.2, 0.5],
            meta_lambda_values=[0.0, 10.0],
            meta_fail_penalty=30.0,
            meta_progress_weight=1.0,
            meta_ade_weight=2.0,
            meta_epochs=120,
            meta_learning_rate=0.08,
            meta_class_balanced=True,
            enable_meta_calibration=False,
            main_risk_weight=10.0,
            hidden_dim=8,
            epochs=150,
            learning_rate=0.05,
            proxy_score_weight=1.0,
            progress_weight=0.02,
            unsafe_penalty=2.0,
        )

        self.assertEqual("nuplan_replay_calibration_experiments_v1", report["schema"])
        self.assertEqual(2, report["split_count"])
        self.assertTrue(report["expert_distance_available"])
        self.assertEqual(2, len(report["risk_weight_sweep_summary"]))
        self.assertEqual(2, len(report["constrained_threshold_sweep_summary"]))
        self.assertFalse(report["meta_calibration_enabled"])
        self.assertEqual([], report["meta_lambda_summary"])
        self.assertEqual([], report["meta_lambda_value_summary"])
        self.assertIsNotNone(report["risk_weight_sweep_summary"][0]["mean_ade_3s_m"]["mean"])
        self.assertTrue(any(row["name"] == "proxy_selector" for row in report["ablation_summary"]))
        self.assertTrue(any(row["name"] == "replay_oracle" for row in report["ablation_summary"]))
        self.assertTrue(report["token_shift_summary"])
        self.assertIn("Risk Weight Sweep", self.module.markdown_report(report))
        self.assertIn("Constrained Risk Threshold Sweep", self.module.markdown_report(report))
        self.assertNotIn("Meta-Calibrated Lambda Policy", self.module.markdown_report(report))
        self.assertNotIn("meta_lambda_value", self.module.markdown_report(report))
        self.assertIn("lambda,replay_infeasible_rate_mean", self.module.pareto_csv(report))
        self.assertIn("<svg", self.module.pareto_svg(report))

    def test_meta_calibration_is_opt_in(self) -> None:
        report = self.module.run_replay_calibration_experiments(
            _replay_report(),
            input_replay_json="synthetic.json",
            split_count=1,
            seed_start=0,
            holdout_fraction=0.5,
            near_miss_threshold_m=1.0,
            risk_weights=[0.0, 10.0],
            risk_thresholds=[0.2],
            meta_lambda_values=[0.0, 10.0],
            meta_fail_penalty=30.0,
            meta_progress_weight=1.0,
            meta_ade_weight=2.0,
            meta_epochs=80,
            meta_learning_rate=0.08,
            meta_class_balanced=True,
            enable_meta_calibration=True,
            main_risk_weight=10.0,
            hidden_dim=8,
            epochs=80,
            learning_rate=0.05,
            proxy_score_weight=1.0,
            progress_weight=0.02,
            unsafe_penalty=2.0,
        )

        self.assertTrue(report["meta_calibration_enabled"])
        self.assertEqual(1, len(report["meta_lambda_summary"]))
        self.assertEqual(1, len(report["meta_lambda_value_summary"]))
        self.assertIn("Exploratory Meta-Calibrated Lambda Policy", self.module.markdown_report(report))

    def test_cli_main_writes_experiment_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_json = tmp_path / "experiments.json"
            output_md = tmp_path / "experiments.md"
            output_pareto_csv = tmp_path / "experiments_pareto.csv"
            output_pareto_svg = tmp_path / "experiments_pareto.svg"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--output-pareto-csv",
                str(output_pareto_csv),
                "--output-pareto-svg",
                str(output_pareto_svg),
                "--split-count",
                "2",
                "--risk-weight-sweep",
                "0,10",
                "--risk-threshold-sweep",
                "0.2,0.5",
                "--meta-lambda-values",
                "0,10",
                "--epochs",
                "120",
                "--meta-epochs",
                "120",
                "--meta-class-balanced",
                "--hidden-dim",
                "8",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_replay_calibration_experiments_v1", payload["schema"])
            self.assertIn("Ablations", output_md.read_text(encoding="utf-8"))
            self.assertIn("replay_infeasible_rate_mean", output_pareto_csv.read_text(encoding="utf-8"))
            self.assertIn("Replay-calibrated selector", output_pareto_svg.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
