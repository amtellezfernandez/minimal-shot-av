from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_nuplan_replay_calibrated_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanReplayCalibratedSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(TRAIN_SCRIPT, "train_nuplan_replay_calibrated_selector")

    def test_replay_examples_train_risk_selector(self) -> None:
        report = _replay_report()
        examples = self.module.build_replay_calibration_examples(report, near_miss_threshold_m=1.0)

        selector, metrics = self.module.fit_replay_calibrated_selector(
            examples,
            hidden_dim=8,
            epochs=300,
            learning_rate=0.05,
            seed=3,
            risk_weight=5.0,
        )

        self.assertEqual(8, len(examples))
        self.assertGreater(metrics["risk_accuracy"], 0.7)
        payload = selector.to_payload()
        self.assertEqual("nuplan_replay_calibrated_selector_v1", payload["model_type"])
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
        examples = self.module.build_replay_calibration_examples(train_report, near_miss_threshold_m=1.0)
        selector, _metrics = self.module.fit_replay_calibrated_selector(
            examples,
            hidden_dim=8,
            epochs=250,
            seed=4,
        )

        evaluation = self.module.evaluate_selector_family(
            holdout_report,
            replay_calibrated_selector=selector,
            baseline_selector=None,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual(1, split["holdout_db_count"])
        self.assertIn("proxy_selector", evaluation)
        self.assertIn("replay_calibrated", evaluation)
        self.assertIsNotNone(evaluation["replay_calibrated"]["mean_ade_3s_m"])
        self.assertIsNotNone(evaluation["replay_calibrated"]["mean_fde_3s_m"])
        self.assertGreaterEqual(
            evaluation["replay_calibrated"]["recoverable_proxy_optimism_fixed_count"],
            evaluation["proxy_selector"]["recoverable_proxy_optimism_fixed_count"],
        )

    def test_constrained_replay_risk_selects_highest_utility_safe_candidate(self) -> None:
        report = _replay_report()
        examples = self.module.build_replay_calibration_examples(report, near_miss_threshold_m=1.0)
        selector, _metrics = self.module.fit_replay_calibrated_selector(
            examples,
            hidden_dim=8,
            epochs=300,
            learning_rate=0.05,
            seed=6,
        )

        token = self.module.select_with_constrained_replay_risk(
            report["scenes"][0],
            selector,
            risk_threshold=0.5,
        )

        self.assertEqual("slow_yield", token)

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
                "--epochs",
                "250",
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
            self.assertEqual("nuplan_replay_calibrated_selector_v1", payload["model_type"])
            self.assertEqual("nuplan_replay_calibrated_selector_eval_v1", evaluation["schema"])
            self.assertIn("Replay-Calibrated Selector", report_md.read_text(encoding="utf-8"))


def _write_temp_payload(payload: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    path = Path(tmp.name)
    tmp.close()
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _replay_report() -> dict:
    scenes = []
    for db_index, db in enumerate(("db_a.db", "db_b.db")):
        for scene_index in range(2):
            scenes.append(_scene(f"{db_index}_{scene_index}", db))
    return {"scene_count": len(scenes), "scenes": scenes}


def _scene(scene_id: str, db: str) -> dict:
    return {
        "scene_id": scene_id,
        "source_db_file": db,
        "selected_token": "maintain",
        "six_scalar_state": {
            "obstacle_pressure": 0.8,
            "route_blockage": 0.5,
            "corridor_blocked": False,
            "left_clearance_m": 6.0,
            "right_clearance_m": 3.0,
            "vector": [0.8, 0.5, 0.0, 6.0, 3.0, 1.0],
        },
        "route_features": {
            "heading_error_rad": 0.05,
            "lane_offset_m": 0.0,
            "route_remaining_m": 20.0,
        },
        "actor_summary": {
            "nearest_actor_distance_m": 3.0,
            "leading_actor_distance_m": 5.0,
            "rear_closing_actor_count": 0,
            "crossing_actor_count": 1,
        },
        "candidates": [
            _candidate("maintain", proxy_safe=True, clearance=2.0, progress=16.0, score=8.0),
            _candidate("slow_yield", proxy_safe=True, clearance=1.5, progress=8.0, score=5.0),
        ],
        "candidate_replay_evaluations": [
            _replay("maintain", clearance=0.25),
            _replay("slow_yield", clearance=2.5),
        ],
        "expert_trajectory": [
            {"x_m": 2.0 * step, "y_m": 0.0, "heading_rad": 0.0}
            for step in range(1, 9)
        ],
        "oracle_log_replay_safe_token": "slow_yield",
    }


def _candidate(token: str, *, proxy_safe: bool, clearance: float, progress: float, score: float) -> dict:
    return {
        "token": token,
        "speed_scale": 1.0 if token == "maintain" else 0.5,
        "lateral_offset_m": 0.0,
        "proxy_safe": proxy_safe,
        "min_proxy_clearance_m": clearance,
        "final_progress_m": progress,
        "score": score,
        "poses": [[2.0 * step * (1.0 if token == "maintain" else 0.5), 0.0, 0.0] for step in range(1, 9)],
    }


def _replay(token: str, *, clearance: float) -> dict:
    return {
        "token": token,
        "realized_min_clearance_m": clearance,
        "realized_near_miss": clearance < 1.0,
    }


if __name__ == "__main__":
    unittest.main()
