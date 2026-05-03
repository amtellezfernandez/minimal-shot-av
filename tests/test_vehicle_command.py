from __future__ import annotations

import math
import unittest

from minimal_shot_av.simulator.safety import SafeAction
from minimal_shot_av.simulator.vehicle_command import (
    ValidationContext,
    VehicleCommand,
    VehicleCommandLimits,
    apply_command_envelope,
    command_from_safe_action,
    command_report,
    command_violations,
)


class VehicleCommandTests(unittest.TestCase):
    def test_safe_action_converts_to_bounded_vehicle_command_shape(self) -> None:
        command = command_from_safe_action(
            SafeAction(direction=(1.0, 0.0), speed=0.8, mode="maintain", intervention=False),
            timestamp_s=1.25,
            previous_speed_mps=0.4,
        )

        self.assertEqual(1.25, command.timestamp_s)
        self.assertEqual("maintain", command.mode)
        self.assertAlmostEqual(0.0, command.steering_rad)
        self.assertAlmostEqual(0.8, command.speed_mps)
        self.assertAlmostEqual(1.6, command.accel_mps2)

    def test_envelope_clamps_speed_steering_and_acceleration(self) -> None:
        command = VehicleCommand(
            timestamp_s=0.0,
            mode="test",
            steering_rad=1.2,
            speed_mps=4.0,
            accel_mps2=5.0,
            brake=0.0,
            source="test",
            estop_required=False,
            geofence_ok=True,
            reason="raw",
        )

        safe = apply_command_envelope(command, limits=VehicleCommandLimits())

        self.assertEqual(2.0, safe.speed_mps)
        self.assertEqual(0.6, safe.steering_rad)
        self.assertEqual(1.0, safe.accel_mps2)
        self.assertEqual([], command_violations(safe))
        self.assertIn("speed_clamped", safe.reason)

    def test_envelope_rejects_invalid_or_unsafe_context(self) -> None:
        command = VehicleCommand(
            timestamp_s=0.0,
            mode="test",
            steering_rad=math.nan,
            speed_mps=1.0,
            accel_mps2=0.0,
            brake=0.0,
            source="test",
            estop_required=False,
            geofence_ok=True,
            reason="raw",
        )

        safe = apply_command_envelope(command, context=ValidationContext(geofence_ok=False))

        self.assertEqual(0.0, safe.speed_mps)
        self.assertEqual(1.0, safe.brake)
        self.assertTrue(safe.estop_required)
        self.assertIn(safe.reason, {"invalid_numeric_command", "geofence_not_ok"})

    def test_shadow_report_requires_nonempty_violation_free_commands(self) -> None:
        command = VehicleCommand(
            timestamp_s=0.0,
            mode="test",
            steering_rad=0.0,
            speed_mps=1.0,
            accel_mps2=0.0,
            brake=0.0,
            source="test",
            estop_required=False,
            geofence_ok=True,
            reason="raw",
        )

        report = command_report(
            [command],
            estop_tested=True,
            manual_takeover_tested=True,
            geofence_tested=True,
        )

        self.assertEqual("vehicle_validation_shadow_report_v1", report["schema"])
        self.assertEqual(1, report["command_count"])
        self.assertEqual(0, report["violation_count"])
        self.assertTrue(report["ready_for_low_speed_actuation"])

    def test_shadow_report_does_not_claim_untested_controls_by_default(self) -> None:
        command = VehicleCommand(
            timestamp_s=0.0,
            mode="test",
            steering_rad=0.0,
            speed_mps=1.0,
            accel_mps2=0.0,
            brake=0.0,
            source="test",
            estop_required=False,
            geofence_ok=True,
            reason="raw",
        )

        report = command_report([command])

        self.assertFalse(report["estop_tested"])
        self.assertFalse(report["manual_takeover_tested"])
        self.assertFalse(report["geofence_tested"])
        self.assertFalse(report["ready_for_low_speed_actuation"])


if __name__ == "__main__":
    unittest.main()
