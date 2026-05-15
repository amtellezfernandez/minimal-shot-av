from __future__ import annotations

import math
import unittest

import numpy as np

from scripts import audit_dagger_representation as module


class DaggerRepresentationAuditTests(unittest.TestCase):
    def test_js_divergence_is_zero_for_identical_distributions(self) -> None:
        p = np.array([0.2, 0.3, 0.5], dtype=np.float64)
        self.assertAlmostEqual(module._js_divergence(p, p), 0.0, places=12)

    def test_js_divergence_is_symmetric(self) -> None:
        p = np.array([0.8, 0.1, 0.1], dtype=np.float64)
        q = np.array([0.1, 0.8, 0.1], dtype=np.float64)
        self.assertAlmostEqual(module._js_divergence(p, q), module._js_divergence(q, p), places=12)

    def test_summarize_rollout_extracts_continuous_metrics(self) -> None:
        class Step:
            def __init__(self, min_obstacle_distance, speed, progress, intervention, stall) -> None:
                self.min_obstacle_distance = min_obstacle_distance
                self.speed = speed
                self.progress = progress
                self.intervention = intervention
                self.stall = stall

        class Rollout:
            def __init__(self) -> None:
                self.success = True
                self.collision = False
                self.reached_goal = True
                self.steps = [
                    Step(2.0, 0.0, 0.0, False, False),
                    Step(1.5, 1.0, 0.8, False, False),
                    Step(0.8, 2.0, 0.9, True, False),
                    Step(1.2, 1.0, 0.6, True, True),
                ]

        summary = module._summarize_rollout(Rollout())
        self.assertTrue(summary["pass"])
        self.assertEqual(0.8, summary["min_clearance_m"])
        self.assertAlmostEqual(0.5, summary["intervention_rate"], places=4)
        self.assertAlmostEqual(0.25, summary["stall_rate"], places=4)
        self.assertTrue(math.isfinite(summary["mean_abs_accel"]))
        self.assertTrue(math.isfinite(summary["mean_abs_jerk"]))


if __name__ == "__main__":
    unittest.main()
