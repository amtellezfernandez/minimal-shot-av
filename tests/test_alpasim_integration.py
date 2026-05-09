from __future__ import annotations

from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import numpy as np

from tests.pyproject_helpers import load_string_tables

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.neutral.alpasim_metrics import build_alpasim_evidence, load_alpasim_metrics
from minimal_shot_av.simulator.alpasim_signal import extract_alpasim_signal, scenario_from_command
from minimal_shot_av.simulator.alpasim_spotlight import DriveCommand, SpotlightReflexAlpaSimModel


class AlpaSimIntegrationTests(unittest.TestCase):
    def test_pyproject_registers_alpasim_plugin_entrypoints(self) -> None:
        pyproject = load_string_tables("pyproject.toml")
        self.assertEqual(
            pyproject['project.entry-points."alpasim.models"']["spotlight_reflex"],
            "minimal_shot_av.simulator.alpasim_spotlight:SpotlightReflexAlpaSimModel",
        )
        self.assertEqual(
            pyproject['project.entry-points."alpasim.configs"']["spotlight_reflex"],
            "minimal_shot_av.simulator.alpasim_configs",
        )

    def test_alpasim_driver_config_exists(self) -> None:
        config_path = Path("src/minimal_shot_av/simulator/alpasim_configs/driver/spotlight_reflex.yaml")
        config = config_path.read_text()
        self.assertIn("model_type: spotlight_reflex", config)
        self.assertIn("output_frequency_hz: 4", config)

    def test_alpasim_signal_uses_structured_hazards(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=6.0,
            acceleration=0.0,
            ego_pose_history=[object()],
            alpasignal={
                "hazards": [
                    {
                        "forward_m": 12.0,
                        "lateral_m": -1.5,
                        "radius_m": 1.25,
                        "type": "vehicle",
                        "id": "cutin_0",
                    }
                ]
            },
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertEqual(signal["structured_hazards"][0]["label"], "cutin_0")
        self.assertEqual(len(scenario.obstacles), 1)
        self.assertEqual(scenario.obstacles[0].x, 12.0)
        self.assertEqual(scenario.obstacles[0].kind, "vehicle")

    def test_alpasim_signal_preserves_static_hazard_shape_metadata(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=6.0,
            acceleration=0.0,
            ego_pose_history=[object()],
            structured_hazards=[
                {
                    "x": 10.0,
                    "y": 1.5,
                    "radius": 0.8,
                    "length_m": 5.2,
                    "heading_rad": 1.57,
                    "kind": "vehicle",
                    "label": "parked_van",
                }
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertEqual(len(scenario.obstacles), 1)
        obstacle = scenario.obstacles[0]
        self.assertAlmostEqual(obstacle.length or 0.0, 5.2)
        self.assertAlmostEqual(obstacle.heading, 1.57, places=2)

    def test_alpasim_signal_adds_caution_zone_for_low_visibility_braking(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.zeros((4, 4, 3), dtype=np.uint8))]},
            speed=0.2,
            acceleration=-3.0,
            ego_pose_history=[],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertGreaterEqual(signal["visibility_risk"], 0.5)
        self.assertGreaterEqual(signal["dynamics_risk"], 0.5)
        self.assertEqual(scenario.obstacles[0].label, "alpasim_signal_caution")

    def test_alpasim_signal_preserves_moving_hazards_as_actors(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=8.0,
            acceleration=0.0,
            ego_pose_history=[],
            traffic_hazards=[
                {
                    "x": 14.0,
                    "y": 2.0,
                    "radius": 1.0,
                    "kind": "pedestrian",
                    "label": "crossing_0",
                    "vx": 0.0,
                    "vy": -2.0,
                }
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertEqual(len(scenario.obstacles), 0)
        self.assertEqual(len(scenario.actors), 1)
        self.assertEqual(scenario.actors[0].role, "crossing_0")
        self.assertEqual(scenario.actors[0].vy, -2.0)

    def test_alpasim_signal_preserves_moving_hazard_shape_metadata(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=8.0,
            acceleration=0.0,
            ego_pose_history=[],
            traffic_hazards=[
                {
                    "x": 14.0,
                    "y": 2.0,
                    "radius": 0.9,
                    "width_m": 2.2,
                    "length_m": 4.8,
                    "heading_rad": 1.2,
                    "kind": "vehicle",
                    "label": "crossing_vehicle",
                    "vx": 0.0,
                    "vy": -2.0,
                }
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertEqual(len(scenario.actors), 1)
        actor = scenario.actors[0]
        self.assertAlmostEqual(actor.width, 2.2)
        self.assertAlmostEqual(actor.length, 4.8)
        self.assertEqual(actor.role, "crossing_vehicle")

    def test_alpasim_signal_preserves_explicit_moving_behavior(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=8.0,
            acceleration=0.0,
            ego_pose_history=[],
            traffic_hazards=[
                {
                    "x": 14.0,
                    "y": 2.0,
                    "radius": 1.0,
                    "kind": "vehicle",
                    "label": "lead_vehicle",
                    "vx": 0.0,
                    "vy": -2.0,
                    "behavior": "sudden_brake",
                }
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertEqual(scenario.actors[0].behavior, "sudden_brake")

    def test_alpasim_signal_infers_supported_behavior_from_label(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=8.0,
            acceleration=0.0,
            ego_pose_history=[],
            traffic_hazards=[
                {
                    "x": 14.0,
                    "y": 2.0,
                    "radius": 1.0,
                    "kind": "vehicle",
                    "label": "wrong_way_vehicle",
                    "vx": -3.0,
                    "vy": 0.0,
                },
                {
                    "x": 8.0,
                    "y": 3.0,
                    "radius": 0.6,
                    "kind": "pedestrian",
                    "label": "erratic_pedestrian",
                    "vx": 0.0,
                    "vy": -1.5,
                },
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)
        behaviors = {actor.role: actor.behavior for actor in scenario.actors}

        self.assertEqual(behaviors["wrong_way_vehicle"], "wrong_way")
        self.assertEqual(behaviors["erratic_pedestrian"], "erratic_pedestrian")

    def test_alpasim_signal_infers_sudden_brake_from_acceleration(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=8.0,
            acceleration=0.0,
            ego_pose_history=[],
            traffic_hazards=[
                {
                    "x": 14.0,
                    "y": 0.0,
                    "radius": 1.0,
                    "kind": "vehicle",
                    "label": "lead_vehicle",
                    "vx": 2.0,
                    "vy": 0.0,
                    "acceleration_mps2": -3.5,
                }
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertEqual(scenario.actors[0].behavior, "sudden_brake")

    def test_alpasim_signal_preserves_explicit_heading_for_elongated_vehicle(self) -> None:
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            speed=8.0,
            acceleration=0.0,
            ego_pose_history=[],
            traffic_hazards=[
                {
                    "x": 14.0,
                    "y": 2.0,
                    "radius": 0.9,
                    "width_m": 2.0,
                    "length_m": 5.5,
                    "heading_rad": 1.2,
                    "kind": "vehicle",
                    "label": "cut_in_vehicle",
                    "vx": 2.0,
                    "vy": 0.0,
                }
            ],
        )

        signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command("straight", signal)

        self.assertAlmostEqual(scenario.actors[0].heading, 1.2, places=6)

    def test_alpasim_adapter_reasoning_exposes_world_state_summary(self) -> None:
        model = SpotlightReflexAlpaSimModel(camera_ids=["front"], context_length=1, output_frequency_hz=4)
        prediction_input = SimpleNamespace(
            camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
            command=DriveCommand.STRAIGHT,
            speed=6.0,
            acceleration=0.0,
            ego_pose_history=[],
            structured_hazards=[
                {
                    "x": 8.0,
                    "y": 1.0,
                    "radius": 0.9,
                    "length_m": 4.5,
                    "heading_rad": 0.0,
                    "kind": "vehicle",
                    "label": "lead_vehicle",
                }
            ],
        )

        prediction = model.predict(prediction_input)
        reasoning = prediction.reasoning_text

        self.assertIsNotNone(reasoning)
        assert reasoning is not None
        self.assertIn('"route_blockage"', reasoning)
        self.assertIn('"preferred_escape_side"', reasoning)
        self.assertIn('"obstacle_pressure"', reasoning)

    def test_imports_alpasim_aggregate_text_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate_dir = run_dir / "aggregate"
            aggregate_dir.mkdir()
            metrics_file = aggregate_dir / "metrics_results.txt"
            metrics_file.write_text(
                "\n".join(
                    [
                        "Run: spotlight_reflex",
                        "n_clips: 4, n_rollouts/clip: 2",
                        "collision_at_fault        0.00 ± 0.00      MAX",
                        "offroad                   0.01 ± 0.00      MAX",
                        "dist_to_gt_trajectory     2.40 ± 0.50      MAX",
                        "safety_monitor_triggered  0.00 ± 0.00      MAX",
                    ]
                ),
                encoding="utf-8",
            )

            metrics_path, metrics, run_count = load_alpasim_metrics(run_dir)
            evidence = build_alpasim_evidence(run_dir).to_dict()

        self.assertEqual(metrics_path.name, "metrics_results.txt")
        self.assertEqual(run_count, 8)
        self.assertEqual(metrics["collision_at_fault"], 0.0)
        self.assertEqual(evidence["run_count"], 8)
        self.assertTrue(evidence["sensor_realistic"])
        self.assertFalse(evidence["official_compass_score"])
        self.assertTrue(evidence["gates"]["collision_rate"])
        self.assertTrue(evidence["gates"]["route_deviation_m"])

    def test_imports_alpasim_json_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            metrics_file = Path(tmp) / "metrics_results.json"
            metrics_file.write_text(
                """{
                  "run_count": 3,
                  "collision_at_fault": {"mean": 0.02, "std": 0.01},
                  "offroad": 0.0,
                  "dist_to_gt_trajectory": 4.0,
                  "safety_monitor_triggered": 0.0
                }""",
                encoding="utf-8",
            )

            evidence = build_alpasim_evidence(metrics_file).to_dict()

        self.assertEqual(evidence["run_count"], 3)
        self.assertFalse(evidence["gates"]["collision_rate"])
        self.assertFalse(evidence["gates"]["route_deviation_m"])

    def test_imports_alpasim_csv_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            aggregate_dir = Path(tmp) / "aggregate"
            aggregate_dir.mkdir()
            metrics_file = aggregate_dir / "metrics_results.csv"
            header = ",".join(
                [
                    "run_uuid",
                    "n_clips",
                    "n_rollouts",
                    "collision_at_fault",
                    "offroad",
                    "dist_to_gt_trajectory",
                    "safety_monitor_triggered",
                ]
            )
            metrics_file.write_text(
                "\n".join(
                    [
                        header,
                        "a,2,1,0.0,0.0,1.0,0.0",
                        "b,2,1,0.0,0.0,2.0,0.0",
                    ]
                )
            )

            evidence = build_alpasim_evidence(Path(tmp)).to_dict()

        self.assertEqual(evidence["run_count"], 4)
        self.assertEqual(evidence["metrics"]["dist_to_gt_trajectory"], 1.5)
        self.assertTrue(all(value is True for value in evidence["gates"].values()))

if __name__ == "__main__":
    unittest.main()
