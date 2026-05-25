from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nuplan_maneuvertoken_rollout.py"
TRAIN_SCRIPT = ROOT / "scripts" / "train_nuplan_maneuvertoken_selector.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_nuplan_maneuvertoken_rollout", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunNuPlanManeuverTokenRolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()
        train_spec = importlib.util.spec_from_file_location("train_nuplan_maneuvertoken_selector", TRAIN_SCRIPT)
        if train_spec is None or train_spec.loader is None:
            raise ImportError(TRAIN_SCRIPT)
        cls.train_module = importlib.util.module_from_spec(train_spec)
        sys.modules[train_spec.name] = cls.train_module
        train_spec.loader.exec_module(cls.train_module)

    def test_rollout_report_exports_scene_diagnostics(self) -> None:
        scenes = [_scene("scene_a"), _scene("scene_b")]

        report = self.module.run_rollout(scenes)

        self.assertEqual(2, report["scene_count"])
        self.assertIn("selected_token_histogram", report)
        self.assertIn("failure_rung_table", report)
        self.assertEqual(2, len(report["scenes"]))
        self.assertEqual(9, report["scenes"][0]["candidate_count"])
        self.assertIn("selected_token_rollout", report["scenes"][0])
        self.assertTrue(report["scenes"][0]["selected_token_rollout"]["per_frame_diagnostics"])
        self.assertIn("failure_rung", report["scenes"][0])
        self.assertIn("safe_token_existed", report["scenes"][0])
        self.assertIn("oracle_safe_token", report["scenes"][0])
        self.assertIn("realized_min_clearance", report["scenes"][0])
        self.assertIn("selected_token_realized_min_clearance_m", report["scenes"][0])

    def test_cli_main_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "scenes.json"
            output_json = tmp_path / "out.json"
            output_md = tmp_path / "out.md"
            input_path.write_text(json.dumps([_scene("scene_a")]), encoding="utf-8")
            argv = [
                str(SCRIPT),
                "--input",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(1, payload["scene_count"])
            self.assertEqual(5, len(payload["failure_rung_table"]))
            self.assertIn("Failure rung", output_md.read_text(encoding="utf-8"))
            self.assertIn("Proxy-safe selected count", output_md.read_text(encoding="utf-8"))
            self.assertIn("Proxy-safe rate", output_md.read_text(encoding="utf-8"))

    def test_rollout_can_use_trained_selector_model(self) -> None:
        training_scenes = [
            _training_scene("left_a", "left"),
            _training_scene("left_b", "left"),
            _training_scene("right_a", "right"),
            _training_scene("right_b", "right"),
        ]
        examples = self.train_module.build_training_examples(training_scenes)
        selector, metrics = self.train_module.fit_selector_mlp(
            examples,
            hidden_dim=8,
            epochs=300,
            learning_rate=0.05,
            seed=0,
        )
        probe_scene = _training_scene("right_probe", "right")
        expected_token = next(
            example["token"]
            for example in self.train_module.build_training_examples([probe_scene])
            if float(example["label"]) == 1.0
        )

        report = self.module.run_rollout([probe_scene], selector=selector)

        self.assertEqual("learned_selector", report["selection_mode"])
        self.assertGreaterEqual(metrics["scene_accuracy"], 0.99)
        self.assertEqual(expected_token, report["scenes"][0]["selected_token"])
        self.assertEqual("learned_selector", report["scenes"][0]["selection_source"])
        self.assertEqual("metric/spec ambiguity", report["scenes"][0]["failure_rung"])

    def test_proxy_safe_without_realized_clearance_is_ambiguity(self) -> None:
        rung = self.module.classify_failure_rung(
            actor_visible_count=2,
            actor_count=2,
            safe_token_existed=True,
            selector_chose_safe_token=True,
            selected_token_realized_min_clearance_m=None,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("metric/spec ambiguity", rung)

    def test_proxy_safe_realized_near_is_controller_proxy_mismatch(self) -> None:
        rung = self.module.classify_failure_rung(
            actor_visible_count=2,
            actor_count=2,
            safe_token_existed=True,
            selector_chose_safe_token=True,
            selected_token_realized_min_clearance_m=0.4,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("proxy predicted safe but realized failed", rung)

    def test_proxy_safe_realized_safe_is_resolved_safe(self) -> None:
        rung = self.module.classify_failure_rung(
            actor_visible_count=2,
            actor_count=2,
            safe_token_existed=True,
            selector_chose_safe_token=True,
            selected_token_realized_min_clearance_m=3.2,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("resolved safe", rung)

    def test_no_safe_token_has_precedence(self) -> None:
        rung = self.module.classify_failure_rung(
            actor_visible_count=2,
            actor_count=2,
            safe_token_existed=False,
            selector_chose_safe_token=False,
            selected_token_realized_min_clearance_m=0.2,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("no safe token existed", rung)

    def test_selector_miss_when_safe_token_exists_but_not_selected(self) -> None:
        rung = self.module.classify_failure_rung(
            actor_visible_count=2,
            actor_count=2,
            safe_token_existed=True,
            selector_chose_safe_token=False,
            selected_token_realized_min_clearance_m=None,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("safe token existed but selector missed it", rung)

    def test_current_failure_rung_count_summary(self) -> None:
        records = (
            [{"failure_rung": "no safe token existed"} for _ in range(25)]
            + [{"failure_rung": "metric/spec ambiguity"} for _ in range(25)]
        )
        counts = Counter(record["failure_rung"] for record in records)

        self.assertEqual(0, counts["no actor/state visibility"])
        self.assertEqual(25, counts["no safe token existed"])
        self.assertEqual(0, counts["safe token existed but selector missed it"])
        self.assertEqual(0, counts["proxy predicted safe but realized failed"])
        self.assertEqual(25, counts["metric/spec ambiguity"])


def _scene(scene_id: str) -> dict:
    return {
        "scene_id": scene_id,
        "ego_state": {"speed_mps": 5.0},
        "route": {"command": "left", "heading_error_rad": 0.1, "lane_offset_m": 0.3, "remaining_distance_m": 20.0},
        "actors": [
            {"x_m": 4.0, "y_m": 0.1, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 1.0, "visible": True},
            {"x_m": 6.0, "y_m": -3.0, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.8, "visible": True},
        ],
    }


def _training_scene(scene_id: str, escape_side: str) -> dict:
    sign = 1.0 if escape_side == "left" else -1.0
    route_command = "left" if escape_side == "left" else "right"
    return {
        "scene_id": scene_id,
        "ego_state": {"speed_mps": 6.0},
        "route": {
            "command": route_command,
            "heading_error_rad": 0.05,
            "lane_offset_m": -0.4 if escape_side == "left" else 0.4,
            "remaining_distance_m": 24.0,
        },
        "actors": [
            {"x_m": 5.0, "y_m": 0.2 * sign, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.9, "visible": True},
            {"x_m": 4.4, "y_m": 2.2 * sign, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.8, "visible": True},
            {"x_m": 7.8, "y_m": -4.5 * sign, "vx_mps": 0.0, "vy_mps": 0.0, "radius_m": 0.8, "visible": True},
        ],
        "expert_trajectory": _expert_trajectory(escape_side),
    }


def _expert_trajectory(escape_side: str) -> list[list[float]]:
    lateral = 2.2 if escape_side == "left" else -2.2
    return [[2.1 * step, lateral * min(1.0, step / 5.0)] for step in range(1, 9)]


if __name__ == "__main__":
    unittest.main()
