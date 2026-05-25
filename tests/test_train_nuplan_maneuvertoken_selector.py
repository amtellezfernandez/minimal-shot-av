from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts" / "train_nuplan_maneuvertoken_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanManeuverTokenSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(TRAIN_SCRIPT, "train_nuplan_maneuvertoken_selector")

    def test_build_training_examples_and_fit_selector(self) -> None:
        scenes = [
            _scene("left_a", "left", "left"),
            _scene("left_b", "left", "left"),
            _scene("right_a", "right", "right"),
            _scene("right_b", "right", "right"),
        ]

        examples = self.module.build_training_examples(scenes)
        selector, metrics = self.module.fit_selector_mlp(
            examples,
            hidden_dim=8,
            epochs=300,
            learning_rate=0.05,
            seed=1,
        )

        self.assertEqual(36, len(examples))
        self.assertGreaterEqual(metrics["scene_accuracy"], 0.99)
        example_tokens = {example["token"] for example in examples if example["label"] == 1.0}
        self.assertTrue(example_tokens.issubset({"evasive_left", "evasive_right", "nudge_left", "nudge_right"}))
        self.assertLess(metrics["example_log_loss"], 0.7)
        payload = selector.to_payload()
        self.assertEqual("nuplan_maneuvertoken_selector_mlp_v1", payload["model_type"])

    def test_cli_main_writes_model_artifact(self) -> None:
        scenes = [
            _scene("left_a", "left", "left"),
            _scene("right_a", "right", "right"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "scenes.json"
            output_path = tmp_path / "selector.json"
            input_path.write_text(json.dumps(scenes), encoding="utf-8")
            argv = [
                str(TRAIN_SCRIPT),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--epochs",
                "200",
                "--hidden-dim",
                "8",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_maneuvertoken_selector_mlp_v1", payload["model_type"])
            self.assertIn("training_metrics", payload)
            self.assertGreaterEqual(payload["training_metrics"]["scene_accuracy"], 0.5)


def _scene(scene_id: str, escape_side: str, route_command: str) -> dict:
    sign = 1.0 if escape_side == "left" else -1.0
    return {
        "scene_id": scene_id,
        "ego_state": {"speed_mps": 6.0},
        "route": {
            "command": route_command,
            "heading_error_rad": 0.05,
            "lane_offset_m": -0.4 if escape_side == "left" else 0.4,
            "remaining_distance_m": 28.0,
        },
        "actors": [
            {"x_m": 5.0, "y_m": 0.2 * sign, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.9, "visible": True},
            {"x_m": 4.5, "y_m": 2.1 * sign, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.8, "visible": True},
            {"x_m": 8.0, "y_m": -4.5 * sign, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.8, "visible": True},
        ],
        "expert_trajectory": _expert_trajectory(escape_side),
    }


def _expert_trajectory(escape_side: str) -> list[list[float]]:
    lateral = 2.2 if escape_side == "left" else -2.2
    return [[2.1 * step, lateral * min(1.0, step / 5.0)] for step in range(1, 9)]


if __name__ == "__main__":
    unittest.main()
