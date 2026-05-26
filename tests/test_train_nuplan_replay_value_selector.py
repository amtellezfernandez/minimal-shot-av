from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_nuplan_replay_value_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanReplayValueSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(TRAIN_SCRIPT, "train_nuplan_replay_value_selector")

    def test_replay_value_examples_train_selector(self) -> None:
        report = _replay_report()
        examples = self.module.build_replay_value_examples(
            report,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
        )

        selector, metrics = self.module.fit_replay_value_selector(examples, ridge_alpha=1.0e-3)

        self.assertEqual(8, len(examples))
        self.assertGreater(metrics["objective_mae"], 0.0)
        payload = selector.to_payload()
        self.assertEqual("nuplan_replay_value_selector_v1", payload["model_type"])
        reloaded = self.module.load_selector(_write_temp_payload(payload))
        safe = report["scenes"][0]["candidates"][1]
        risky = report["scenes"][0]["candidates"][0]
        self.assertGreater(
            reloaded.predict_score(self.module._features(report["scenes"][0], safe)),
            reloaded.predict_score(self.module._features(report["scenes"][0], risky)),
        )

    def test_replay_value_mlp_examples_train_selector(self) -> None:
        report = _replay_report()
        examples = self.module.build_replay_value_examples(
            report,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
        )

        selector, metrics = self.module.fit_replay_value_mlp_selector(
            examples,
            hidden_dim=8,
            epochs=350,
            learning_rate=0.04,
            seed=11,
        )

        self.assertEqual(8, len(examples))
        self.assertLess(metrics["objective_mae"], 80.0)
        payload = selector.to_payload()
        self.assertEqual("nuplan_replay_value_mlp_selector_v1", payload["model_type"])
        reloaded = self.module.load_selector(_write_temp_payload(payload))
        safe = report["scenes"][0]["candidates"][1]
        risky = report["scenes"][0]["candidates"][0]
        self.assertGreater(
            reloaded.predict_score(self.module._features(report["scenes"][0], safe)),
            reloaded.predict_score(self.module._features(report["scenes"][0], risky)),
        )

    def test_db_split_and_evaluation_report_recover_safe_alternative(self) -> None:
        report = _replay_report()
        train_report, holdout_report, split = self.module.split_replay_report_by_db(
            report,
            holdout_fraction=0.5,
            seed=0,
        )
        examples = self.module.build_replay_value_examples(
            train_report,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
        )
        selector, _metrics = self.module.fit_replay_value_selector(examples, ridge_alpha=1.0e-3)

        evaluation = self.module.evaluate_selector_family(
            holdout_report,
            replay_value_selector=selector,
            baseline_selector=None,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual(1, split["holdout_db_count"])
        self.assertIn("proxy_selector", evaluation)
        self.assertIn("replay_value", evaluation)
        self.assertGreaterEqual(
            evaluation["replay_value"]["recoverable_proxy_optimism_fixed_count"],
            evaluation["proxy_selector"]["recoverable_proxy_optimism_fixed_count"],
        )

    def test_cli_main_writes_model_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "selector.json"
            report_json = tmp_path / "eval.json"
            report_md = tmp_path / "eval.md"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            argv = [
                str(TRAIN_SCRIPT),
                "--input-replay-json",
                str(input_path),
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
            self.assertEqual("nuplan_replay_value_selector_eval_v1", evaluation["schema"])
            self.assertIn("Replay-Value Selector", report_md.read_text(encoding="utf-8"))

    def test_cli_main_writes_mlp_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "selector.json"
            report_json = tmp_path / "eval.json"
            report_md = tmp_path / "eval.md"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            argv = [
                str(TRAIN_SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output",
                str(output_path),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_md),
                "--model-kind",
                "mlp",
                "--epochs",
                "120",
                "--hidden-dim",
                "8",
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
            self.assertEqual("nuplan_replay_value_mlp_selector_v1", payload["model_type"])
            self.assertEqual("mlp", evaluation["model_kind"])


def _write_temp_payload(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    path = Path(tmp.name)
    tmp.close()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
