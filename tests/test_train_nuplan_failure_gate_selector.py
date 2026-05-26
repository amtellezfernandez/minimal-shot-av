from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nuplan_failure_gate_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanFailureGateSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "train_nuplan_failure_gate_selector")

    def test_failure_gate_learns_to_switch_when_proxy_fails(self) -> None:
        replay = _replay_report()
        safe_fn = lambda _scene: "slow_yield"
        examples = self.module.build_failure_gate_examples(
            replay,
            safe_token_fn=safe_fn,
            near_miss_threshold_m=1.0,
            positive_weight=4.0,
        )
        gate, metrics = self.module.fit_failure_gate_mlp(
            examples,
            hidden_dim=8,
            epochs=250,
            learning_rate=0.05,
            seed=19,
        )
        evaluation = self.module.evaluate_failure_gate_family(
            replay,
            gate_model=gate,
            safe_token_fn=safe_fn,
            gate_threshold=0.5,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual(4, len(examples))
        self.assertGreater(metrics["gate_accuracy"], 0.7)
        self.assertEqual(4, evaluation["proxy_selector"]["selected_replay_infeasible_count"])
        self.assertEqual(0, evaluation["failure_gate"]["selected_replay_infeasible_count"])
        self.assertEqual(0, evaluation["oracle_failure_gate"]["selected_replay_infeasible_count"])
        self.assertEqual(0, evaluation["progress_oracle_failure_gate"]["selected_replay_infeasible_count"])
        self.assertEqual(4, evaluation["failure_gate"]["gate_diagnostics"]["true_positive_switch_count"])
        self.assertEqual(1.0, evaluation["failure_gate"]["gate_diagnostics"]["switch_recall"])

    def test_safe_token_fn_supports_risk_constrained_fallback(self) -> None:
        class Selector:
            proxy_score_weight = 1.0
            progress_weight = 0.02
            unsafe_penalty = 2.0

            def predict_risk(self, features):
                return 0.1 if features["candidate_final_progress_m"] < 10.0 else 0.9

        token_fn = self.module.build_safe_token_fn(
            Selector(),
            fallback_mode="risk_constrained",
            fallback_risk_threshold=0.5,
        )

        self.assertEqual("slow_yield", token_fn(_replay_report()["scenes"][0]))

    def test_progress_constrained_fallback_prefers_progress_below_risk(self) -> None:
        class Selector:
            def predict_risk(self, features):
                return 0.1 if features["candidate_final_progress_m"] < 20.0 else 0.9

        token = self.module.select_highest_progress_below_risk(
            _replay_report()["scenes"][0],
            Selector(),
            risk_threshold=0.5,
        )

        self.assertEqual("maintain", token)

    def test_progress_retention_guard_blocks_low_progress_switch(self) -> None:
        replay = _replay_report()
        safe_fn = lambda _scene: "slow_yield"
        examples = self.module.build_failure_gate_examples(
            replay,
            safe_token_fn=safe_fn,
            near_miss_threshold_m=1.0,
            positive_weight=4.0,
            min_fallback_progress_retention=0.75,
        )
        gate_model = _always_switch_gate_model(self.module, replay["scenes"][0], "slow_yield")
        selected = self.module.select_with_failure_gate(
            replay["scenes"][0],
            gate_model=gate_model,
            safe_token_fn=safe_fn,
            gate_threshold=0.5,
            min_fallback_progress_retention=0.75,
        )

        self.assertTrue(all(float(example["label"]) == 0.0 for example in examples))
        self.assertEqual("maintain", selected)

    def test_can_fit_replay_value_mlp_fallback_on_train_split(self) -> None:
        args = SimpleNamespace(
            fit_safe_selector=True,
            safe_selector_kind="replay_value_mlp",
            safe_hidden_dim=8,
            safe_epochs=120,
            safe_learning_rate=0.04,
            safe_value_fail_penalty=250.0,
            safe_value_progress_weight=1.0,
            safe_value_ade_weight=2.0,
            safe_value_ridge_alpha=1.0e-3,
            safe_student_positive_weight=8.0,
            seed=23,
        )

        selector, source, metrics = self.module.load_or_fit_safe_selector(
            args,
            train_report=_replay_report(),
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("fit_replay_value_mlp_on_train_split", source)
        self.assertEqual("replay_value_mlp", metrics["safe_selector_kind"])
        self.assertGreater(selector.predict_score(self.module.selector_feature_row(
            _replay_report()["scenes"][0],
            _replay_report()["scenes"][0]["candidates"][1],
        )), selector.predict_score(self.module.selector_feature_row(
            _replay_report()["scenes"][0],
            _replay_report()["scenes"][0]["candidates"][0],
        )))

    def test_failure_gate_policy_embeds_and_loads_fallback_selector(self) -> None:
        report = _replay_report()
        examples = self.module.build_replay_value_examples(
            report,
            near_miss_threshold_m=1.0,
            fail_penalty=250.0,
            objective_progress_weight=1.0,
            objective_ade_weight=2.0,
        )
        safe_selector, _metrics = self.module.fit_replay_value_mlp_selector(
            examples,
            hidden_dim=8,
            epochs=120,
            learning_rate=0.04,
            seed=17,
        )
        gate_model = _always_switch_gate_model(self.module, report["scenes"][0], "slow_yield")
        payload = {
            **self.module.failure_gate_payload(gate_model),
            "training": {
                "safe_selector_payload": safe_selector.to_payload(),
                "fallback_mode": "score",
                "fallback_risk_threshold": 0.5,
                "gate_threshold": 0.5,
            },
        }

        with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
            path = Path(tmp.name)
        path.write_text(json.dumps(payload), encoding="utf-8")
        policy = self.module.load_failure_gate_policy(path)

        self.assertEqual("slow_yield", self.module.select_with_failure_gate_policy(report["scenes"][0], policy))

    def test_threshold_recommendations_are_train_selected(self) -> None:
        rows = [
            {"threshold": 0.2, "selected_replay_infeasible_rate": 0.2, "mean_selected_progress_m": 8.0},
            {"threshold": 0.5, "selected_replay_infeasible_rate": 0.3, "mean_selected_progress_m": 9.0},
            {"threshold": 0.8, "selected_replay_infeasible_rate": 0.4, "mean_selected_progress_m": 11.0},
        ]
        recommendations = self.module.threshold_recommendations(rows, proxy_progress_m=10.0)

        self.assertEqual(0.2, recommendations["safety_first"]["threshold"])
        self.assertEqual(0.2, recommendations["balanced_80pct_proxy_progress"]["threshold"])
        self.assertEqual(0.5, recommendations["utility_90pct_proxy_progress"]["threshold"])


def _always_switch_gate_model(module, scene, fallback_token: str) -> dict:
    features = module.failure_gate_feature_row(scene, fallback_token)
    return {
        "model_type": module.MODEL_TYPE,
        "feature_names": tuple(features),
        "hidden_dim": 1,
        "feature_mean": tuple(0.0 for _ in features),
        "feature_scale": tuple(1.0 for _ in features),
        "w1": tuple((0.0,) for _ in features),
        "b1": (1.0,),
        "w2": (0.0,),
        "b2": 10.0,
    }


if __name__ == "__main__":
    unittest.main()
