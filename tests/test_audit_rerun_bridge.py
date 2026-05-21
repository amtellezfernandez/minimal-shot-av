from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.audit import critical_event_bundle, export_alpasim_audit_log, export_internal_audit_log, load_audit_log


class AuditRerunBridgeTests(unittest.TestCase):
    def test_export_internal_audit_log_writes_manifest_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            audit_dir = Path(temp_dir) / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--scenario-cluster",
                    "spotlight",
                    "--rollout-preset",
                    "showcase-spotlight",
                    "--artifacts-dir",
                    str(rollout_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            manifest = export_internal_audit_log(rollout_dir / "latest_rollout.json", audit_dir)
            self.assertEqual("internal", manifest["source"])
            loaded_manifest, frames = load_audit_log(audit_dir)

        self.assertEqual(manifest["frame_count"], len(frames))
        self.assertEqual("spotlight", loaded_manifest["scenario_cluster"])
        self.assertIn("step", frames[0])
        self.assertIn("active_obstacles", frames[0])
        self.assertIn("trigger_state", frames[0])

    def test_summary_only_cli_reads_audit_log_without_rerun_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            audit_dir = Path(temp_dir) / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--scenario-cluster",
                    "intersection",
                    "--rollout-preset",
                    "showcase-intersection",
                    "--artifacts-dir",
                    str(rollout_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_internal_audit.py"),
                    "--rollout-json",
                    str(rollout_dir / "latest_rollout.json"),
                    "--output-dir",
                    str(audit_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "view_audit_rerun.py"),
                    str(audit_dir),
                    "--summary-only",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual("internal", payload["manifest"]["source"])
        self.assertEqual("internal:intersection", payload["severity_policy"]["profile"])
        self.assertGreater(payload["frame_count"], 0)
        self.assertIn("bookmark_count", payload)
        self.assertIn("bookmark_index", payload)

    def test_export_critical_events_cli_writes_compact_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            audit_dir = Path(temp_dir) / "audit"
            output_path = Path(temp_dir) / "critical_events.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--scenario-cluster",
                    "intersection",
                    "--rollout-preset",
                    "showcase-intersection",
                    "--artifacts-dir",
                    str(rollout_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            export_internal_audit_log(rollout_dir / "latest_rollout.json", audit_dir)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_audit_critical_events.py"),
                    str(audit_dir),
                    "--output",
                    str(output_path),
                    "--context-radius",
                    "1",
                    "--min-severity",
                    "medium",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(result.stdout)
            bundle = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(str(output_path), payload["output"])
        self.assertEqual("medium", payload["min_severity"])
        self.assertEqual("internal:intersection", bundle["severity_policy"]["profile"])
        self.assertGreater(bundle["bookmark_count"], 0)
        self.assertGreater(bundle["critical_frame_count"], 0)
        self.assertLessEqual(bundle["critical_frame_count"], bundle["manifest"]["frame_count"])
        self.assertTrue(all(item["severity"] in {"medium", "high"} for item in bundle["bookmarks"]))

    def test_critical_event_bundle_keeps_bookmark_media_refs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            audit_dir = Path(temp_dir) / "audit"
            (run_dir / "driver").mkdir(parents=True)
            (run_dir / "aggregate").mkdir(parents=True)
            (run_dir / "launch-metadata.json").write_text(
                json.dumps({"model": "token_dagger_iter2_hybrid_clamped", "scene_preset": "fresh_3scene", "scene_ids": ["clipgt-test-scene"]}),
                encoding="utf-8",
            )
            (run_dir / "aggregate" / "metrics_results.json").write_text(
                json.dumps({"run_count": 1, "collision_at_fault": 0.0, "offroad": 0.0, "dist_to_gt_trajectory": 1.2}),
                encoding="utf-8",
            )
            selection_row = {
                "frame_index": 1,
                "scene_id": "clipgt-test-scene",
                "command": "straight",
                "speed_mps": 6.0,
                "front_camera_image_path": "frames/front_0001.jpg",
                "selection_mode": "fallback_stop",
                "trajectory_mode": "clamped_lateral",
                "hybrid_token": "slow_yield",
                "spotlight_token": "slow_yield",
                "decision_type": "fallback_stop",
                "dagger_argmax_geo_gap": 0.4,
                "top_logits": [{"token": "slow_yield", "logit": 5.0}],
                "spotlight_top_candidates": [{"token": "slow_yield"}],
                "alpasim_signal": {
                    "route_waypoints": [{"x": 0.0, "y": 0.0}, {"x": 20.0, "y": 0.0}],
                    "structured_hazards": [],
                    "dynamics_risk": 0.85,
                },
            }
            (run_dir / "driver" / "selection-log.jsonl").write_text(json.dumps(selection_row) + "\n", encoding="utf-8")
            export_alpasim_audit_log(run_dir, audit_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_audit_critical_events.py"),
                    str(audit_dir),
                    "--output",
                    str(Path(temp_dir) / "critical_alpasim.json"),
                    "--min-severity",
                    "high",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(result.stdout)
            bundle = json.loads((Path(temp_dir) / "critical_alpasim.json").read_text(encoding="utf-8"))

        self.assertEqual("high", payload["min_severity"])
        self.assertEqual("alpasim:fresh_3scene", bundle["severity_policy"]["profile"])
        self.assertGreater(bundle["bookmark_count"], 0)
        self.assertTrue(all(item["severity"] == "high" for item in bundle["bookmarks"]))
        self.assertTrue(any(item["media"] and item["media"][0]["path"] == "frames/front_0001.jpg" for item in bundle["bookmarks"]))

    def test_critical_event_bundle_uses_scenario_specific_policy(self) -> None:
        manifest = {"source": "internal", "scenario_cluster": "intersection", "frame_count": 1}
        frames = [
            {
                "frame_idx": 0,
                "timestamp_s": 0.0,
                "ego": {"speed": 1.0, "goal_distance": 10.0},
                "step": {"min_obstacle_distance": 1.2, "collision_risk": 0.0, "lane_error": 0.0, "action_mode": "maintain"},
                "trigger_state": {},
                "media": [],
            }
        ]
        bundle = critical_event_bundle(manifest, frames, min_severity="medium")

        self.assertEqual("internal:intersection", bundle["severity_policy"]["profile"])
        self.assertEqual(1, bundle["bookmark_count"])
        self.assertEqual("near_miss", bundle["bookmarks"][0]["kind"])

    def test_export_alpasim_audit_log_from_selection_log(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "run"
            audit_dir = Path(temp_dir) / "audit"
            (run_dir / "driver").mkdir(parents=True)
            (run_dir / "aggregate").mkdir(parents=True)
            (run_dir / "launch-metadata.json").write_text(
                json.dumps({"model": "token_dagger_iter2_hybrid_clamped", "scene_preset": "fresh_3scene", "scene_ids": ["clipgt-test-scene"]}),
                encoding="utf-8",
            )
            (run_dir / "aggregate" / "metrics_results.json").write_text(
                json.dumps({"run_count": 1, "collision_at_fault": 0.0, "offroad": 0.0, "dist_to_gt_trajectory": 1.2}),
                encoding="utf-8",
            )
            selection_row = {
                "frame_index": 1,
                "scene_id": "clipgt-test-scene",
                "command": "straight",
                "speed_mps": 6.0,
                "front_camera_image_path": "frames/front_0001.jpg",
                "selection_mode": "hybrid_veto",
                "trajectory_mode": "clamped_lateral",
                "hybrid_token": "maintain",
                "spotlight_token": "maintain",
                "decision_type": "spotlight_wins",
                "dagger_argmax_geo_gap": 2.5,
                "top_logits": [{"token": "maintain", "logit": 5.0}],
                "spotlight_top_candidates": [{"token": "maintain"}],
                "alpasim_signal": {
                    "route_waypoints": [{"x": 0.0, "y": 0.0}, {"x": 20.0, "y": 0.0}, {"x": 40.0, "y": 1.0}],
                    "structured_hazards": [{"x": 8.0, "y": 1.0, "radius": 1.0, "kind": "vehicle", "label": "crossing_vehicle", "vx": -1.0, "vy": 0.0}],
                    "dynamics_risk": 0.2,
                },
            }
            (run_dir / "driver" / "selection-log.jsonl").write_text(json.dumps(selection_row) + "\n", encoding="utf-8")

            manifest = export_alpasim_audit_log(run_dir, audit_dir)
            loaded_manifest, frames = load_audit_log(audit_dir)

        self.assertEqual("alpasim", manifest["source"])
        self.assertEqual("alpasim", loaded_manifest["source"])
        self.assertEqual(1, len(frames))
        self.assertEqual("maintain", frames[0]["step"]["selected_maneuver"])
        self.assertEqual("clipgt-test-scene", manifest["scene_ids"][0])
        self.assertEqual("spotlight_wins", frames[0]["step"]["decision_type"])
        self.assertEqual("frames/front_0001.jpg", frames[0]["media"][0]["path"])

    def test_compare_audit_logs_cli_reports_summary_delta_and_bookmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            internal_dir = Path(temp_dir) / "internal_audit"
            alpasim_run_dir = Path(temp_dir) / "alpasim_run"
            alpasim_audit_dir = Path(temp_dir) / "alpasim_audit"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--scenario-cluster",
                    "spotlight",
                    "--rollout-preset",
                    "showcase-spotlight",
                    "--artifacts-dir",
                    str(rollout_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            export_internal_audit_log(rollout_dir / "latest_rollout.json", internal_dir)

            (alpasim_run_dir / "driver").mkdir(parents=True)
            (alpasim_run_dir / "aggregate").mkdir(parents=True)
            (alpasim_run_dir / "launch-metadata.json").write_text(
                json.dumps({"model": "token_dagger_iter2_hybrid_clamped", "scene_preset": "fresh_3scene", "scene_ids": ["clipgt-test-scene"]}),
                encoding="utf-8",
            )
            (alpasim_run_dir / "aggregate" / "metrics_results.json").write_text(
                json.dumps({"run_count": 1, "collision_at_fault": 0.0, "offroad": 0.0, "dist_to_gt_trajectory": 1.2}),
                encoding="utf-8",
            )
            (alpasim_run_dir / "driver" / "selection-log.jsonl").write_text(
                json.dumps(
                    {
                        "frame_index": 1,
                        "scene_id": "clipgt-test-scene",
                        "command": "straight",
                        "speed_mps": 6.0,
                        "selection_mode": "hybrid_veto",
                        "trajectory_mode": "clamped_lateral",
                        "hybrid_token": "maintain",
                        "spotlight_token": "maintain",
                        "decision_type": "spotlight_wins",
                        "dagger_argmax_geo_gap": 2.5,
                        "top_logits": [{"token": "maintain", "logit": 5.0}],
                        "spotlight_top_candidates": [{"token": "maintain"}],
                        "alpasim_signal": {
                            "route_waypoints": [{"x": 0.0, "y": 0.0}, {"x": 20.0, "y": 0.0}],
                            "structured_hazards": [],
                            "dynamics_risk": 0.2,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            export_alpasim_audit_log(alpasim_run_dir, alpasim_audit_dir)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "compare_audit_logs.py"),
                    str(internal_dir),
                    str(alpasim_audit_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(result.stdout)

        self.assertIn("delta", payload)
        self.assertIn("min_clearance_delta", payload["delta"])
        self.assertIn(payload["alignment_mode"], {"timestamp", "progress"})
        self.assertIn("aligned_samples", payload)
        self.assertGreater(len(payload["aligned_samples"]), 0)
        self.assertIn("bookmarks", payload["left"])
        self.assertIn("bookmarks", payload["right"])
        self.assertIn("paired_bookmarks", payload)
        self.assertGreater(len(payload["left"]["bookmarks"]), 0)
        self.assertIn(payload["left"]["bookmarks"][0]["kind"], {"trigger_activation", "near_miss", "collision_risk_spike", "intervention"})
        if payload["alignment_mode"] == "timestamp":
            self.assertIn("timestamp_delta_s", payload["aligned_samples"][0])
