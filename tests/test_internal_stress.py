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

from minimal_shot_av.stress import (
    STRESS_LEVELS,
    aggregate_internal_stress,
    compute_internal_stress_metrics,
    evaluate_internal_stress,
    generate_internal_stress_scenario,
)
from minimal_shot_av.simulator.policy import run_spotlight_reflex_policy


class InternalStressTests(unittest.TestCase):
    def test_intersection_audit_trigger_is_ego_relative_and_tighter_than_debug(self) -> None:
        debug = generate_internal_stress_scenario("intersection", seed=3, level="debug")
        audit = generate_internal_stress_scenario("intersection", seed=3, level="audit")
        debug_actor = next(actor for actor in debug.actors if actor.role == "conflicting_vehicle")
        audit_actor = next(actor for actor in audit.actors if actor.role == "conflicting_vehicle")
        self.assertLessEqual(audit_actor.active_from, int(audit.tags["ego_relative_trigger_tick"]))
        self.assertGreater(debug_actor.active_from, audit_actor.active_from)
        self.assertLess(abs(audit_actor.y - audit.lane_center[3][1]), abs(debug_actor.y - debug.lane_center[3][1]))

    def test_spotlight_audit_pulls_hazard_toward_lane_center(self) -> None:
        base = generate_internal_stress_scenario("spotlight", seed=3, level="stress")
        audit = generate_internal_stress_scenario("spotlight", seed=3, level="audit")
        base_actor = next(actor for actor in base.actors if actor.role == "spotlight_hazard")
        audit_actor = next(actor for actor in audit.actors if actor.role == "spotlight_hazard")
        center_y = audit.lane_center[3][1]
        self.assertLess(abs(audit_actor.y - center_y), abs(base_actor.y - center_y))

    def test_metrics_classify_real_response(self) -> None:
        scenario = generate_internal_stress_scenario("intersection", seed=3, level="audit")
        rollout = run_spotlight_reflex_policy(scenario)
        metrics = compute_internal_stress_metrics(scenario, rollout)
        self.assertLess(metrics.min_clearance_m, 2.0)
        self.assertTrue(metrics.hard_response)

    def test_evaluate_internal_stress_produces_summary_rows(self) -> None:
        rows = evaluate_internal_stress(("intersection",), 3, 3, ("audit",), "spotlight-reflex")
        self.assertEqual(1, len(rows))
        summary = aggregate_internal_stress(rows)
        self.assertEqual(1, len(summary))
        self.assertEqual("intersection", summary[0]["cluster"])
        self.assertIn(summary[0]["stressed_pass_rate"], {0.0, 1.0})

    def test_cli_writes_internal_stress_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "evaluate_internal_stress.py"),
                    "--seed-start",
                    "3",
                    "--seed-end",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--clusters",
                    "intersection",
                    "--levels",
                    "audit",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads((Path(temp_dir) / "internal_stress_summary.json").read_text())

        self.assertEqual(["intersection"], payload["manifest"]["clusters"])
        self.assertEqual(["audit"], payload["manifest"]["levels"])
        self.assertEqual(["spotlight-reflex"], payload["manifest"]["policies"])
        self.assertEqual(1, len(payload["runs"]))
        self.assertEqual(1, len(payload["summary"]))


if __name__ == "__main__":
    unittest.main()
