from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _candidate
from tests.test_train_nuplan_replay_calibrated_selector import _replay
from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nuplan_regime_gate_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanRegimeGateSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "train_nuplan_regime_gate_selector")

    def test_marginal_head_respects_progress_retention_floor(self) -> None:
        scene = _regime_scene()

        token = self.module.select_marginal_head_token(
            scene,
            _ProgressTargetSelector(target=13.0),
            min_progress_retention=0.5,
        )

        self.assertEqual("nudge_right", token)

    def test_oracle_regime_distinguishes_marginal_from_intervention(self) -> None:
        scene = _regime_scene()

        marginal_label = self.module.oracle_regime_label(
            scene,
            marginal_token="nudge_right",
            intervention_token="stop",
            near_miss_threshold_m=1.0,
        )
        intervention_label = self.module.oracle_regime_label(
            scene,
            marginal_token="maintain",
            intervention_token="stop",
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("marginal_correction", marginal_label)
        self.assertEqual("intervention", intervention_label)

    def test_regime_gate_learns_marginal_switch_on_fixture(self) -> None:
        replay = _replay_report()
        selector = _ProgressTargetSelector(target=8.0)
        marginal_fn = lambda scene: self.module.select_marginal_head_token(
            scene,
            selector,
            min_progress_retention=0.5,
        )
        intervention_fn = lambda _scene: "slow_yield"
        examples = self.module.build_regime_gate_examples(
            replay,
            marginal_token_fn=marginal_fn,
            intervention_token_fn=intervention_fn,
            marginal_selector=selector,
            intervention_selector=selector,
            near_miss_threshold_m=1.0,
            marginal_weight=4.0,
            intervention_weight=4.0,
            keep_weight=1.0,
        )
        gate, metrics = self.module.fit_regime_gate_mlp(
            examples,
            hidden_dim=8,
            epochs=260,
            learning_rate=0.05,
            seed=13,
        )
        evaluation = self.module.evaluate_regime_gate_family(
            replay,
            gate_model=gate,
            marginal_token_fn=marginal_fn,
            intervention_token_fn=intervention_fn,
            marginal_selector=selector,
            intervention_selector=selector,
            risk_selector=selector,
            risk_threshold=0.5,
            near_miss_threshold_m=1.0,
        )

        self.assertGreater(metrics["regime_accuracy"], 0.7)
        self.assertEqual(4, evaluation["proxy_selector"]["selected_replay_infeasible_count"])
        self.assertEqual(0, evaluation["oracle_regime_gate"]["selected_replay_infeasible_count"])
        self.assertEqual(0, evaluation["regime_gate"]["selected_replay_infeasible_count"])
        self.assertEqual(
            4,
            evaluation["regime_gate"]["regime_diagnostics"]["marginal_recovery_count"],
        )

    def test_cli_main_writes_model_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "regime_gate.json"
            report_json = tmp_path / "eval.json"
            report_md = tmp_path / "eval.md"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output",
                str(output_path),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_md),
                "--hidden-dim",
                "8",
                "--epochs",
                "180",
                "--value-hidden-dim",
                "8",
                "--value-epochs",
                "160",
                "--seed",
                "3",
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_regime_gate_selector_v1", payload["model_type"])
            self.assertEqual("nuplan_regime_gate_selector_eval_v1", report["schema"])
            self.assertIn("Regime-Gated Selector", report_md.read_text(encoding="utf-8"))


class _ProgressTargetSelector:
    def __init__(self, *, target: float) -> None:
        self.target = float(target)

    def predict_score(self, features) -> float:
        progress = float(features["candidate_final_progress_m"])
        return -abs(progress - self.target)

    def predict_risk(self, features) -> float:
        progress = float(features["candidate_final_progress_m"])
        return 0.1 if progress <= self.target + 1.0 else 0.9


def _regime_scene() -> dict:
    scene = _replay_report()["scenes"][0]
    scene = json.loads(json.dumps(scene))
    scene["candidates"] = [
        _candidate("maintain", proxy_safe=True, clearance=2.0, progress=16.0, score=8.0),
        _candidate("nudge_right", proxy_safe=True, clearance=1.4, progress=13.0, score=6.5),
        _candidate("stop", proxy_safe=True, clearance=2.5, progress=2.0, score=3.0),
    ]
    scene["candidate_replay_evaluations"] = [
        _replay("maintain", clearance=0.2),
        _replay("nudge_right", clearance=1.4),
        _replay("stop", clearance=1.8),
    ]
    scene["oracle_log_replay_safe_token"] = "nudge_right"
    return scene


if __name__ == "__main__":
    unittest.main()
