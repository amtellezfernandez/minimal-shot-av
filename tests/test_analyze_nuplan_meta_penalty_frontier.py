from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
FRONTIER_SCRIPT = ROOT / "scripts" / "analyze_nuplan_meta_penalty_frontier.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_nuplan_replay_calibrated_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeNuPlanMetaPenaltyFrontierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.frontier_module = _load_module(FRONTIER_SCRIPT, "analyze_nuplan_meta_penalty_frontier")
        cls.train_module = _load_module(TRAIN_SCRIPT, "train_nuplan_replay_calibrated_selector_for_frontier")

    def test_frontier_report_and_strict_audit(self) -> None:
        replay_report = _replay_report()
        train_report, _holdout_report, _split = self.train_module.split_replay_report_by_db(
            replay_report,
            holdout_fraction=0.5,
            seed=0,
        )
        examples = self.train_module.build_replay_calibration_examples(train_report, near_miss_threshold_m=1.0)
        selector, _metrics = self.train_module.fit_replay_calibrated_selector(
            examples,
            hidden_dim=8,
            epochs=200,
            learning_rate=0.05,
            seed=0,
            risk_weight=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            policy_dir = Path(tmp)
            report, strict = self.frontier_module.analyze_meta_penalty_frontier(
                replay_report,
                selector_model=selector,
                penalties=[30.0, 250.0],
                lambda_values=[0.0, 10.0],
                split_count=2,
                seed_start=0,
                holdout_fraction=0.5,
                near_miss_threshold_m=1.0,
                objective_progress_weight=1.0,
                objective_ade_weight=2.0,
                policy_output_dir=policy_dir,
                save_seed0_penalties={250},
                strict_penalty=250.0,
                risk_hidden_dim=8,
                risk_epochs=200,
                risk_learning_rate=0.05,
                seed0_selector_output=policy_dir / "seed0_selector.json",
            )

            self.assertEqual("nuplan_meta_penalty_frontier_v1", report["schema"])
            self.assertEqual(2, len(report["penalties"]))
            self.assertIn("250", report["seed0_policy_paths"])
            self.assertTrue(Path(report["seed0_policy_paths"]["250"]).exists())
            self.assertIn("Lambda hist", self.frontier_module.markdown_report(report))
            self.assertIsNotNone(strict)
            self.assertEqual("nuplan_meta250_strict_adaptation_audit_v1", strict["schema"])
            self.assertTrue(any(row["group"] == "lambda40_fails_lambda80_safe" for row in strict["groups"]))
            self.assertIn("Meta250 Strict Adaptation Audit", self.frontier_module.markdown_strict_report(strict))


if __name__ == "__main__":
    unittest.main()
