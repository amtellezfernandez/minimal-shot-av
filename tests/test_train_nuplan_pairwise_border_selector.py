from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report
from tests.test_train_nuplan_border_residual_selector import _zero_baseline_selector_payload


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nuplan_pairwise_border_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanPairwiseBorderSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "train_nuplan_pairwise_border_selector")

    def test_pairwise_examples_build_on_toy_report(self) -> None:
        baseline = self.module.load_selector(_write_temp_payload(_zero_baseline_selector_payload()))
        examples = self.module.build_pairwise_border_examples(
            _replay_report(),
            baseline_selector=baseline,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
            top_k=2,
            margin_threshold=5.0,
            max_pairs_per_scene=4,
        )
        self.assertTrue(examples)
        self.assertEqual(len(self.module._pair_feature_names()), len(examples[0]["feature_diff"]))

    def test_pairwise_selector_improves_over_zero_baseline_on_toy_report(self) -> None:
        baseline = self.module.load_selector(_write_temp_payload(_zero_baseline_selector_payload()))
        train_report, holdout_report, _split = self.module.split_replay_report_by_db(
            _replay_report(),
            holdout_fraction=0.5,
            seed=0,
        )
        examples = self.module.build_pairwise_border_examples(
            train_report,
            baseline_selector=baseline,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
            top_k=2,
            margin_threshold=5.0,
            max_pairs_per_scene=4,
        )
        policy = self.module.fit_pairwise_border_policy(
            examples,
            iterations=200,
            learning_rate=0.1,
            l2=1.0e-3,
            top_k=2,
            margin_threshold=5.0,
        )
        evaluation = self.module.evaluate_selector_family(
            holdout_report,
            baseline_selector=baseline,
            pairwise_policy=policy,
            near_miss_threshold_m=1.0,
        )
        self.assertGreaterEqual(
            evaluation["pairwise_border"]["recoverable_proxy_optimism_fixed_count"],
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
                "--top-k",
                "2",
                "--margin-threshold",
                "5.0",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            evaluation = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_pairwise_border_selector_v1", payload["model_type"])
            self.assertEqual("nuplan_pairwise_border_selector_eval_v1", evaluation["schema"])
            self.assertIn("Pairwise Border Selector", report_md.read_text(encoding="utf-8"))


def _write_temp_payload(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    path = Path(tmp.name)
    tmp.close()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
