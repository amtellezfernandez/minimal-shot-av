from __future__ import annotations

import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

from tests.pyproject_helpers import load_string_tables

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.neutral.alpasim_metrics import build_alpasim_evidence, load_alpasim_metrics
from minimal_shot_av.simulator.alpasim_signal import extract_alpasim_signal, scenario_from_command
from minimal_shot_av.simulator.alpasim_spotlight import DriveCommand, SpotlightReflexAlpaSimModel
from minimal_shot_av.simulator.alpasim_token_bc import TOKEN_ORDER, TokenBCAlpaSimModel, _GeomMLP


class AlpaSimIntegrationTests(unittest.TestCase):
    def test_pyproject_registers_alpasim_plugin_entrypoints(self) -> None:
        pyproject = load_string_tables("pyproject.toml")
        self.assertEqual(
            pyproject['project.entry-points."alpasim.models"']["spotlight_reflex"],
            "minimal_shot_av.simulator.alpasim_spotlight:SpotlightReflexAlpaSimModel",
        )
        self.assertEqual(
            pyproject['project.entry-points."alpasim.models"']["token_dagger_bc"],
            "minimal_shot_av.simulator.alpasim_token_bc:TokenBCAlpaSimModel",
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

    def test_alpasim_token_dagger_configs_exist_and_default_to_cuda(self) -> None:
        for name in (
            "token_dagger_bc.yaml",
            "token_dagger_srcdecay.yaml",
            "token_dagger_bc_clamped.yaml",
            "token_dagger_srcdecay_clamped.yaml",
        ):
            config_path = Path("src/minimal_shot_av/simulator/alpasim_configs/driver") / name
            config = config_path.read_text()
            self.assertIn("model_type: token_dagger_bc", config)
            self.assertIn('device: "cuda"', config)
            if name.endswith("_clamped.yaml"):
                self.assertNotIn("trajectory_mode:", config)
                self.assertNotIn("max_lateral_offset_m:", config)

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

    def test_token_bc_alpasim_adapter_loads_checkpoint_and_predicts(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "token_dagger_bc.pt"
            model = _GeomMLP(len(TOKEN_ORDER))
            for parameter in model.parameters():
                parameter.data.zero_()
            last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[TOKEN_ORDER.index("maintain")] = 5.0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": list(TOKEN_ORDER),
                },
                checkpoint_path,
            )

            adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
            )
            prediction_input = SimpleNamespace(
                camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
                command=DriveCommand.STRAIGHT,
                speed=6.0,
                acceleration=0.0,
                ego_pose_history=[],
            )
            prediction = adapter.predict(prediction_input)

        self.assertEqual((20, 2), prediction.trajectory_xy.shape)
        self.assertEqual((20,), prediction.headings.shape)
        self.assertIsNotNone(prediction.reasoning_text)
        assert prediction.reasoning_text is not None
        self.assertIn('"selected_maneuver": "maintain"', prediction.reasoning_text)

    def test_token_bc_alpasim_adapter_can_clamp_lateral_token_geometry(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "token_dagger_bc.pt"
            model = _GeomMLP(len(TOKEN_ORDER))
            for parameter in model.parameters():
                parameter.data.zero_()
            last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[TOKEN_ORDER.index("evasive_left")] = 5.0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": list(TOKEN_ORDER),
                },
                checkpoint_path,
            )
            prediction_input = SimpleNamespace(
                camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
                command=DriveCommand.STRAIGHT,
                speed=10.0,
                acceleration=0.0,
                ego_pose_history=[],
            )
            raw_adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
                trajectory_mode="token",
            )
            clamped_adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
                trajectory_mode="clamped_lateral",
                max_lateral_offset_m=2.0,
            )

            raw_prediction = raw_adapter.predict(prediction_input)
            clamped_prediction = clamped_adapter.predict(prediction_input)

        self.assertGreater(float(raw_prediction.trajectory_xy[:, 1].max()), 6.0)
        self.assertLessEqual(float(clamped_prediction.trajectory_xy[:, 1].max()), 2.05)
        self.assertIn('"trajectory_mode": "clamped_lateral"', clamped_prediction.reasoning_text or "")

    def test_token_bc_alpasim_adapter_hybrid_logs_selection_trace(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "token_dagger_bc.pt"
            selection_log_path = Path(tmp) / "selection-log.jsonl"
            model = _GeomMLP(len(TOKEN_ORDER))
            for parameter in model.parameters():
                parameter.data.zero_()
            last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[TOKEN_ORDER.index("evasive_left")] = 5.0
            last_linear.bias.data[TOKEN_ORDER.index("maintain")] = 4.95
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": list(TOKEN_ORDER),
                },
                checkpoint_path,
            )

            adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
                selection_mode="hybrid_veto",
                hybrid_top_k=2,
                hybrid_geometric_weight=20.0,
                selection_log_path=selection_log_path,
            )
            prediction_input = SimpleNamespace(
                camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
                command=DriveCommand.STRAIGHT,
                speed=6.0,
                acceleration=0.0,
                ego_pose_history=[],
                scene_id="clipgt-test-scene",
            )

            prediction = adapter.predict(prediction_input)
            reasoning_payload = json.loads(prediction.reasoning_text or "{}")
            trace = reasoning_payload["selection_trace"]
            self.assertEqual("hybrid_veto", reasoning_payload["selection_mode"])
            self.assertEqual("evasive_left", trace["dagger_argmax_token"])
            self.assertEqual("maintain", trace["hybrid_token"])
            self.assertEqual("maintain", trace["spotlight_token"])
            self.assertTrue(selection_log_path.is_file())
            records = [json.loads(line) for line in selection_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(1, len(records))
            self.assertEqual("clipgt-test-scene", records[0]["scene_id"])
            self.assertEqual("evasive_left", records[0]["dagger_argmax_token"])
            self.assertEqual("maintain", records[0]["hybrid_token"])
            self.assertEqual("spotlight_wins", records[0]["decision_type"])

    def test_token_bc_alpasim_adapter_hard_veto_suppresses_bad_argmax(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "token_dagger_bc.pt"
            selection_log_path = Path(tmp) / "selection-log.jsonl"
            model = _GeomMLP(len(TOKEN_ORDER))
            for parameter in model.parameters():
                parameter.data.zero_()
            last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[TOKEN_ORDER.index("evasive_left")] = 5.0
            last_linear.bias.data[TOKEN_ORDER.index("maintain")] = 4.95
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": list(TOKEN_ORDER),
                },
                checkpoint_path,
            )

            adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
                selection_mode="hybrid_veto",
                hybrid_top_k=2,
                hybrid_geometric_weight=0.0,
                hybrid_veto_margin=0.0,
                hybrid_max_geometric_rank=1,
                selection_log_path=selection_log_path,
            )
            prediction_input = SimpleNamespace(
                camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
                command=DriveCommand.STRAIGHT,
                speed=6.0,
                acceleration=0.0,
                ego_pose_history=[],
                scene_id="clipgt-hard-veto",
            )

            prediction = adapter.predict(prediction_input)
            trace = json.loads(prediction.reasoning_text or "{}")["selection_trace"]
            records = [
                json.loads(line)
                for line in selection_log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual("evasive_left", trace["dagger_argmax_token"])
        self.assertEqual("maintain", trace["hybrid_token"])
        self.assertTrue(trace["dagger_argmax_vetoed"])
        self.assertFalse(trace["used_fallback_geometric"])
        self.assertIn(trace["veto_reason"], {"geometric_gap", "geometric_rank"})
        self.assertEqual(1, trace["max_geometric_rank"])
        self.assertEqual(0.0, trace["veto_margin"])
        self.assertNotIn("evasive_left", trace["safe_topk_tokens"])
        self.assertIn("maintain", trace["safe_topk_tokens"])
        self.assertTrue(any(row["token"] == "evasive_left" for row in trace["vetoed_tokens"]))
        self.assertEqual(1, len(records))
        self.assertEqual("clipgt-hard-veto", records[0]["scene_id"])
        self.assertTrue(records[0]["dagger_argmax_vetoed"])
        self.assertIn(records[0]["veto_reason"], {"geometric_gap", "geometric_rank"})

    def test_token_bc_alpasim_adapter_hard_veto_logs_geometric_fallback(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "token_dagger_bc.pt"
            model = _GeomMLP(len(TOKEN_ORDER))
            for parameter in model.parameters():
                parameter.data.zero_()
            last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[TOKEN_ORDER.index("evasive_left")] = 5.0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": list(TOKEN_ORDER),
                },
                checkpoint_path,
            )
            adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
                selection_mode="hybrid_veto",
                hybrid_top_k=1,
                hybrid_veto_margin=0.0,
                hybrid_max_geometric_rank=1,
            )
            prediction_input = SimpleNamespace(
                camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
                command=DriveCommand.STRAIGHT,
                speed=6.0,
                acceleration=0.0,
                ego_pose_history=[],
            )

            prediction = adapter.predict(prediction_input)
            trace = json.loads(prediction.reasoning_text or "{}")["selection_trace"]

        self.assertEqual("evasive_left", trace["dagger_argmax_token"])
        self.assertEqual("maintain", trace["hybrid_token"])
        self.assertTrue(trace["dagger_argmax_vetoed"])
        self.assertTrue(trace["used_fallback_geometric"])
        self.assertEqual("fallback_geometric", trace["decision_type"])

    def test_token_bc_alpasim_adapter_axis_constrained_preserves_safe_dagger_argmax(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "token_dagger_bc.pt"
            model = _GeomMLP(len(TOKEN_ORDER))
            for parameter in model.parameters():
                parameter.data.zero_()
            last_linear = [module for module in model.modules() if isinstance(module, torch.nn.Linear)][-1]
            last_linear.bias.data[TOKEN_ORDER.index("evasive_left")] = 5.0
            last_linear.bias.data[TOKEN_ORDER.index("maintain")] = 4.95
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": list(TOKEN_ORDER),
                },
                checkpoint_path,
            )
            adapter = TokenBCAlpaSimModel(
                checkpoint_path=checkpoint_path,
                device="cpu",
                camera_ids=["front"],
                context_length=1,
                output_frequency_hz=4,
                trajectory_mode="clamped_lateral",
                selection_mode="axis_constrained",
                hybrid_top_k=2,
            )
            prediction_input = SimpleNamespace(
                camera_images={"front": [SimpleNamespace(image=np.full((4, 4, 3), 180, dtype=np.uint8))]},
                command=DriveCommand.STRAIGHT,
                speed=6.0,
                acceleration=0.0,
                ego_pose_history=[],
            )

            prediction = adapter.predict(prediction_input)
            trace = json.loads(prediction.reasoning_text or "{}")["selection_trace"]

        self.assertEqual("evasive_left", trace["dagger_argmax_token"])
        self.assertEqual("evasive_left", trace["hybrid_token"])
        self.assertFalse(trace["dagger_argmax_vetoed"])
        self.assertEqual("dagger_wins", trace["decision_type"])

    def test_token_bc_alpasim_adapter_rejects_unknown_checkpoint_tokens(self) -> None:
        with TemporaryDirectory() as tmp:
            checkpoint_path = Path(tmp) / "bad_token_dagger_bc.pt"
            model = _GeomMLP(len(TOKEN_ORDER))
            bad_tokens = list(TOKEN_ORDER)
            bad_tokens[TOKEN_ORDER.index("maintain")] = "unknown_token"
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "feat_mean": np.zeros(10, dtype=np.float32),
                    "feat_std": np.ones(10, dtype=np.float32),
                    "token_names": bad_tokens,
                },
                checkpoint_path,
            )

            with self.assertRaisesRegex(ValueError, "token_names do not match"):
                TokenBCAlpaSimModel(
                    checkpoint_path=checkpoint_path,
                    device="cpu",
                    camera_ids=["front"],
                    context_length=1,
                    output_frequency_hz=4,
                )

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
