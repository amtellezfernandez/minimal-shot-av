from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_nuplan_closed_loop_bridge.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_nuplan_closed_loop_bridge", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AuditNuPlanClosedLoopBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_select_bridge_scenes_balances_replay_safe_subset(self) -> None:
        scenes = [
            _scene("a", failure_rung=self.module.REPLAY_INFEASIBLE_RUNG, selected_token="stop", near=True),
            _scene("b", failure_rung=self.module.REPLAY_INFEASIBLE_RUNG, selected_token="stop", near=True),
            _scene("c", failure_rung="resolved safe", selected_token="stop", near=False),
            _scene("d", failure_rung="resolved safe", selected_token="evasive_right", near=False),
        ]

        selected = self.module.select_bridge_scenes(scenes, safe_match_limit=1)

        self.assertEqual(3, len(selected))
        self.assertEqual(2, sum(1 for row in selected if row["bridge_replay_class"] == "replay_infeasible"))
        self.assertEqual(1, sum(1 for row in selected if row["bridge_replay_class"] == "replay_safe"))

    def test_enrich_report_with_closed_loop_execution_builds_bridge_table(self) -> None:
        report = {
            "scenes": [
                _scene("a", failure_rung=self.module.REPLAY_INFEASIBLE_RUNG, selected_token="stop", near=True),
                _scene("b", failure_rung="resolved safe", selected_token="evasive_right", near=False),
            ]
        }

        def executor(scene):
            if scene["scene_id"] == "a":
                return {
                    "selected_token_executed_min_clearance_m": 0.2,
                    "selected_token_executed_min_clearance_t": 1.0,
                    "selected_token_executed_near_miss": True,
                    "selected_token_executed_collision": False,
                    "first_erosion_time_s": 0.5,
                    "time_from_first_erosion_to_impact_s": 0.5,
                    "executed_clearance_source": "nuplan_closed_loop",
                }
            return {
                "selected_token_executed_min_clearance_m": 2.0,
                "selected_token_executed_min_clearance_t": 1.0,
                "selected_token_executed_near_miss": False,
                "selected_token_executed_collision": False,
                "first_erosion_time_s": None,
                "time_from_first_erosion_to_impact_s": None,
                "executed_clearance_source": "nuplan_closed_loop",
            }

        enriched = self.module.enrich_report_with_closed_loop_execution(
            report,
            executor=executor,
            safe_match_limit=1,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("nuplan_selected_token_closed_loop_bridge_v1", enriched["schema"])
        self.assertEqual(2, enriched["bridge_scene_count"])
        table = {
            (row["replay_class"], row["executed_class"]): row["count"]
            for row in enriched["closed_loop_bridge_table"]
        }
        self.assertEqual(1, table[("replay_infeasible", "proxy_safe_but_realized_failed")])
        self.assertEqual(1, table[("replay_safe", "executed_safe")])

    def test_build_bridge_calibration_report_emits_metric_variant_table(self) -> None:
        report = {
            "bridge_scenes": [
                {
                    "scene_id": "safe_a",
                    "bridge_replay_class": "replay_safe",
                    "selected_token": "stop",
                    "selected_token_realized_min_clearance_m": 1.4,
                    "selected_token_realized_min_clearance_t": 1.0,
                    "selected_token_executed_min_clearance_m": 0.6,
                    "selected_token_executed_min_clearance_t": 1.0,
                    "replay_nearest_actor_id": "actor_1",
                    "executed_nearest_actor_id": "actor_1",
                    "replay_nearest_actor_type": "vehicle",
                    "executed_nearest_actor_type": "vehicle",
                    "ego_pose_source": "selected_token_rollout_rear_axle",
                    "actor_geometry_source": "nuplan_tracked_object_oriented_box",
                    "ego_footprint_geometry": "pacifica_oriented_box",
                    "actor_footprint_geometry": "oriented_box",
                    "threshold_m": 1.0,
                    "failure_reason": "footprint_inflation_mismatch",
                    "replay_horizon_s": 4.0,
                    "executed_horizon_s": 4.0,
                    "num_replay_steps": 8,
                    "num_executed_steps": 8,
                    "dt_replay_s": 0.5,
                    "dt_executed_s": 0.5,
                    "proxy_metric_variants": {"box_clearance_m": 1.4},
                    "executed_metric_variants": {
                        "center_distance_m": 2.0,
                        "box_clearance_m": 0.6,
                        "inflated_box_clearance_0_0m": 0.6,
                        "inflated_box_clearance_0_5m": 0.1,
                        "inflated_box_clearance_1_0m": -0.4,
                    },
                    "trajectory_agreement": {
                        "mean_trajectory_deviation_m": 0.0,
                        "max_trajectory_deviation_m": 0.0,
                        "final_pose_deviation_m": 0.0,
                    },
                    "executed_failure_rung": "replay_safe_but_executed_failed",
                }
            ]
        }

        calibration = self.module.build_bridge_calibration_report(report, near_miss_threshold_m=1.0)

        self.assertEqual("nuplan_selected_token_closed_loop_bridge_calibration_v1", calibration["schema"])
        self.assertEqual(1, calibration["scene_count"])
        self.assertEqual(
            1,
            next(
                row["replay_safe_controls_failed"]
                for row in calibration["metric_variant_table"]
                if row["metric_variant"] == "box clearance + 1.0m buffer"
                and row["surface"] == "executed_observation"
                ),
        )

    def test_selector_comparison_runs_proxy_calibrated_and_oracle(self) -> None:
        report = {"scenes": [_comparison_scene()]}

        class Selector:
            def predict_score(self, features):
                return float(features["rank"])

        def executor(scene):
            clearance = 2.0 if scene["selected_token"] == "stop" else 0.2
            return {
                "selected_token_executed_min_clearance_m": clearance,
                "selected_token_executed_min_clearance_t": 1.0,
                "selected_token_executed_near_miss": clearance < 1.0,
                "selected_token_executed_collision": False,
                "first_erosion_time_s": None,
                "time_from_first_erosion_to_impact_s": None,
                "executed_clearance_source": "nuplan_closed_loop",
                "aligned_executed_metric_variants": {"box_clearance_m": clearance},
            }

        enriched = self.module.enrich_report_with_closed_loop_selector_comparison(
            report,
            replay_calibrated_selector=Selector(),
            executor=executor,
            replay_infeasible_limit=1,
            safe_match_limit=0,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("nuplan_selector_closed_loop_bridge_comparison_v1", enriched["schema"])
        table = {row["selector"]: row for row in enriched["closed_loop_selector_comparison"]}
        self.assertEqual(1, table["proxy_selector"]["executed_failure_count"])
        self.assertEqual(0, table["replay_calibrated"]["executed_failure_count"])
        self.assertEqual(0, table["replay_oracle"]["executed_failure_count"])
        self.assertEqual(1, table["proxy_selector"]["aligned_logged_actor_failure_count"])
        self.assertEqual(0, table["replay_calibrated"]["aligned_logged_actor_failure_count"])
        self.assertEqual(0, table["replay_oracle"]["aligned_logged_actor_failure_count"])

    def test_selector_comparison_can_use_constrained_risk_selector(self) -> None:
        report = {"scenes": [_comparison_scene()]}

        class Selector:
            def predict_risk(self, features):
                return 0.1 if features["rank"] > 1.0 else 0.9

        def executor(scene):
            clearance = 2.0 if scene["selected_token"] == "stop" else 0.2
            return {
                "selected_token_executed_min_clearance_m": clearance,
                "selected_token_executed_min_clearance_t": 1.0,
                "selected_token_executed_near_miss": clearance < 1.0,
                "selected_token_executed_collision": False,
                "first_erosion_time_s": None,
                "time_from_first_erosion_to_impact_s": None,
                "executed_clearance_source": "nuplan_closed_loop",
                "aligned_executed_metric_variants": {"box_clearance_m": clearance},
            }

        enriched = self.module.enrich_report_with_closed_loop_selector_comparison(
            report,
            replay_calibrated_selector=Selector(),
            executor=executor,
            replay_infeasible_limit=1,
            safe_match_limit=0,
            near_miss_threshold_m=1.0,
            constrained_risk_threshold=0.5,
        )

        table = {row["selector"]: row for row in enriched["closed_loop_selector_comparison"]}
        self.assertEqual(0, table["replay_constrained_risk_0.5"]["aligned_logged_actor_failure_count"])

    def test_selector_comparison_can_use_failure_gate_policy(self) -> None:
        report = {"scenes": [_comparison_scene()]}

        class SafeSelector:
            def predict_score(self, features):
                return float(features["rank"])

        def executor(scene):
            clearance = 2.0 if scene["selected_token"] == "stop" else 0.2
            return {
                "selected_token_executed_min_clearance_m": clearance,
                "selected_token_executed_min_clearance_t": 1.0,
                "selected_token_executed_near_miss": clearance < 1.0,
                "selected_token_executed_collision": False,
                "first_erosion_time_s": None,
                "time_from_first_erosion_to_impact_s": None,
                "executed_clearance_source": "nuplan_closed_loop",
                "aligned_executed_metric_variants": {"box_clearance_m": clearance},
            }

        policy = {
            "gate_model": _always_switch_gate_model(self.module, _comparison_scene(), "stop"),
            "safe_selector": SafeSelector(),
            "fallback_mode": "score",
            "fallback_risk_threshold": 0.5,
            "gate_threshold": 0.5,
        }
        enriched = self.module.enrich_report_with_closed_loop_selector_comparison(
            report,
            replay_calibrated_selector=None,
            failure_gate_policy=policy,
            executor=executor,
            replay_infeasible_limit=1,
            safe_match_limit=0,
            near_miss_threshold_m=1.0,
        )

        table = {row["selector"]: row for row in enriched["closed_loop_selector_comparison"]}
        self.assertEqual(0, table["failure_gate"]["aligned_logged_actor_failure_count"])

    def test_selector_comparison_can_use_selective_regret_selector(self) -> None:
        report = {"scenes": [_comparison_scene()]}

        class BaselineSelector:
            def predict_score(self, features):
                return float(features["candidate_score_heuristic"])

        class RegretSelector:
            def predict_score(self, features):
                return 10.0 if float(features["candidate_score_heuristic"]) < 0.75 else -1.0

        def executor(scene):
            clearance = 2.0 if scene["selected_token"] == "stop" else 0.2
            return {
                "selected_token_executed_min_clearance_m": clearance,
                "selected_token_executed_min_clearance_t": 1.0,
                "selected_token_executed_near_miss": clearance < 1.0,
                "selected_token_executed_collision": False,
                "first_erosion_time_s": None,
                "time_from_first_erosion_to_impact_s": None,
                "executed_clearance_source": "nuplan_closed_loop",
                "aligned_executed_metric_variants": {"box_clearance_m": clearance},
            }

        enriched = self.module.enrich_report_with_closed_loop_selector_comparison(
            report,
            replay_calibrated_selector=BaselineSelector(),
            selective_regret_policy={
                "regret_model": RegretSelector(),
                "config": {"top_k": 2, "margin_threshold": 5.0, "intervention_threshold": 0.0},
            },
            executor=executor,
            replay_infeasible_limit=1,
            safe_match_limit=0,
            near_miss_threshold_m=1.0,
        )

        table = {row["selector"]: row for row in enriched["closed_loop_selector_comparison"]}
        self.assertEqual(0, table["selective_regret"]["aligned_logged_actor_failure_count"])

    def test_selector_comparison_can_use_boundary_intervention_selector(self) -> None:
        report = {"scenes": [_comparison_scene()]}

        class BaselineSelector:
            def predict_score(self, features):
                return float(features["candidate_score_heuristic"])

        def executor(scene):
            clearance = 2.0 if scene["selected_token"] == "stop" else 0.2
            return {
                "selected_token_executed_min_clearance_m": clearance,
                "selected_token_executed_min_clearance_t": 1.0,
                "selected_token_executed_near_miss": clearance < 1.0,
                "selected_token_executed_collision": False,
                "first_erosion_time_s": None,
                "time_from_first_erosion_to_impact_s": None,
                "executed_clearance_source": "nuplan_closed_loop",
                "aligned_executed_metric_variants": {"box_clearance_m": clearance},
            }

        enriched = self.module.enrich_report_with_closed_loop_selector_comparison(
            report,
            replay_calibrated_selector=BaselineSelector(),
            boundary_intervention_policy={
                "feature_names": [
                    "scene_margin",
                    "score_delta",
                    "progress_delta",
                    "proxy_safe_delta",
                    "clearance_delta",
                    "speed_scale_delta",
                    "lateral_offset_delta",
                    "obstacle_pressure",
                    "route_blockage",
                    "nearest_actor_distance_m",
                    "rear_closing_actor_count",
                    "crossing_actor_count",
                    "baseline_progress_m",
                    "baseline_proxy_clearance_m",
                ],
                "feature_mean": [0.0] * 14,
                "feature_scale": [1.0] * 14,
                "weights": [0.0, -5.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "bias": 0.0,
                "top_k": 2,
                "margin_threshold": 5.0,
                "intervention_threshold": 0.0,
            },
            executor=executor,
            replay_infeasible_limit=1,
            safe_match_limit=0,
            near_miss_threshold_m=1.0,
        )

        table = {row["selector"]: row for row in enriched["closed_loop_selector_comparison"]}
        self.assertEqual(0, table["boundary_intervention"]["aligned_logged_actor_failure_count"])

    def test_select_selector_border_scenes_prioritizes_disagreement_then_low_margin(self) -> None:
        scenes = [
            _border_scene("a", maintain_score=1.0, stop_score=0.9),
            _border_scene("b", maintain_score=1.0, stop_score=0.2),
            _border_scene("c", maintain_score=0.6, stop_score=0.59),
        ]

        def selector_a(scene):
            keep = next(candidate for candidate in scene["candidates"] if candidate["token"] == "maintain")
            stop = next(candidate for candidate in scene["candidates"] if candidate["token"] == "stop")
            return {
                "token": "maintain" if keep["score"] >= stop["score"] else "stop",
                "margin": abs(float(keep["score"]) - float(stop["score"])),
            }

        def selector_b(scene):
            keep = next(candidate for candidate in scene["candidates"] if candidate["token"] == "maintain")
            stop = next(candidate for candidate in scene["candidates"] if candidate["token"] == "stop")
            return {
                "token": "stop" if keep["score"] >= stop["score"] else "maintain",
                "margin": (
                    abs(float(keep["score"]) - float(stop["score"]))
                    + (0.01 if scene["scene_id"] == "b" else 0.0)
                ),
            }

        selected, summary = self.module.select_selector_border_scenes(
            scenes,
            selector_a_name="a",
            selector_a_decision=selector_a,
            selector_b_name="b",
            selector_b_decision=selector_b,
            limit=2,
        )

        self.assertEqual(2, len(selected))
        self.assertEqual(["c", "a"], [row["scene_id"] for row in selected])
        self.assertEqual(3, summary["total_disagreement_count"])
        self.assertEqual(2, summary["selected_disagreement_count"])
        self.assertEqual(["replay_safe", "replay_safe"], [row["bridge_replay_class"] for row in selected])


def _scene(scene_id: str, *, failure_rung: str, selected_token: str, near: bool) -> dict[str, object]:
    return {
        "scene_id": scene_id,
        "failure_rung": failure_rung,
        "proxy_safe_selected": True,
        "selected_token_realized_near_miss": near,
        "selected_token": selected_token,
        "selected_token_min_proxy_clearance_m": 1.5,
        "scenario_type": "lane_change",
        "source_db_file": "/tmp/demo.db",
        "scenario_token": "token",
        "sensor_timestamp_us": 10,
        "map_name": "us-ma-boston",
    }


def _comparison_scene() -> dict[str, object]:
    scene = _scene(
        "comparison_a",
        failure_rung="proxy-safe but replay-infeasible",
        selected_token="maintain",
        near=True,
    )
    scene.update(
        {
            "safe_token_existed": True,
            "oracle_log_replay_safe_token": "stop",
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
                {
                    "token": "maintain",
                    "speed_scale": 1.0,
                    "lateral_offset_m": 0.0,
                    "proxy_safe": True,
                    "min_proxy_clearance_m": 1.5,
                    "score": 1.0,
                    "final_progress_m": 8.0,
                    "poses": [[1.0, 0.0, 0.0]],
                    "features": {
                        "rank": 0.0,
                        "candidate_score_heuristic": 1.0,
                        "candidate_final_progress_m": 8.0,
                        "candidate_proxy_safe": 1.0,
                    },
                },
                {
                    "token": "stop",
                    "speed_scale": 0.0,
                    "lateral_offset_m": 0.0,
                    "proxy_safe": True,
                    "min_proxy_clearance_m": 1.2,
                    "score": 0.5,
                    "final_progress_m": 0.0,
                    "poses": [[0.0, 0.0, 0.0]],
                    "features": {
                        "rank": 10.0,
                        "candidate_score_heuristic": 0.5,
                        "candidate_final_progress_m": 0.0,
                        "candidate_proxy_safe": 1.0,
                    },
                },
            ],
            "candidate_replay_evaluations": [
                {
                    "token": "maintain",
                    "realized_min_clearance_m": 0.2,
                    "realized_min_clearance_t": 1.0,
                    "realized_near_miss": True,
                    "realized_collision": False,
                    "realized_clearance_trace": [],
                },
                {
                    "token": "stop",
                    "realized_min_clearance_m": 2.0,
                    "realized_min_clearance_t": 1.0,
                    "realized_near_miss": False,
                    "realized_collision": False,
                    "realized_clearance_trace": [],
                },
            ],
        }
    )
    return scene


def _always_switch_gate_model(module, scene: dict[str, object], fallback_token: str) -> dict[str, object]:
    features = module.failure_gate_feature_row(scene, fallback_token)
    return {
        "feature_names": list(features),
        "hidden_dim": 1,
        "feature_mean": [0.0 for _ in features],
        "feature_scale": [1.0 for _ in features],
        "w1": [[0.0] for _ in features],
        "b1": [1.0],
        "w2": [0.0],
        "b2": 10.0,
    }


def _border_scene(scene_id: str, *, maintain_score: float, stop_score: float) -> dict[str, object]:
    scene = _scene(
        scene_id,
        failure_rung="resolved safe",
        selected_token="maintain",
        near=False,
    )
    scene.update(
        {
            "safe_token_existed": True,
            "candidates": [
                {
                    "token": "maintain",
                    "proxy_safe": True,
                    "min_proxy_clearance_m": 1.5,
                    "score": maintain_score,
                    "final_progress_m": 8.0,
                    "poses": [[1.0, 0.0, 0.0]],
                    "features": {
                        "candidate_score_heuristic": maintain_score,
                        "candidate_final_progress_m": 8.0,
                        "candidate_proxy_safe": 1.0,
                    },
                },
                {
                    "token": "stop",
                    "proxy_safe": True,
                    "min_proxy_clearance_m": 1.6,
                    "score": stop_score,
                    "final_progress_m": 1.0,
                    "poses": [[0.0, 0.0, 0.0]],
                    "features": {
                        "candidate_score_heuristic": stop_score,
                        "candidate_final_progress_m": 1.0,
                        "candidate_proxy_safe": 1.0,
                    },
                },
            ],
        }
    )
    return scene


if __name__ == "__main__":
    unittest.main()
