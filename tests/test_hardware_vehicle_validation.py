from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_hardware_vehicle_validation.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_hardware_vehicle_validation", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HardwareVehicleValidationTests(unittest.TestCase):
    def test_complete_evidence_is_ready_for_controlled_hil_vil_start(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closed_loop = _write_closed_loop(root)
            safety = _write_safety_case(root)
            integration = _write_integration_manifest(root)
            hardware_plan = _write_hardware_plan(root)
            shadow_report = _write_shadow_report(root)

            report = module.hardware_vehicle_validation_report(
                closed_loop_evidence=closed_loop,
                safety_case=safety,
                integration_manifest=integration,
                hardware_plan=hardware_plan,
                shadow_report=shadow_report,
            )

        self.assertTrue(report["ready_to_start_hardware_vehicle_validation"])
        self.assertEqual("controlled_hil_vil_ready", report["disposition"])
        self.assertEqual("go_controlled_validation", report["authorization"]["decision"])
        self.assertTrue(report["authorization"]["hardware_in_the_loop_authorized"])
        self.assertTrue(report["authorization"]["closed_course_vehicle_validation_authorized"])
        self.assertFalse(report["authorization"]["public_road_deployment_authorized"])
        self.assertFalse(report["authorization"]["unsupervised_vehicle_control_authorized"])
        self.assertEqual([], report["blockers"])

    def test_missing_estop_controls_block_vehicle_validation(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closed_loop = _write_closed_loop(root)
            safety = _write_safety_case(root)
            integration = _write_integration_manifest(root)
            hardware_plan = root / "hardware_plan.json"
            hardware_plan.write_text(
                json.dumps(
                    {
                        "schema": "hardware_vehicle_validation_plan_v1",
                        "approval_status": "approved_for_controlled_validation_start",
                        "allowed_modes": ["hardware_in_the_loop", "closed_course_supervised"],
                        "public_road_allowed": False,
                        "unsupervised_control_allowed": False,
                        "mandatory_controls": ["safety_driver", "speed_limit_enforced"],
                    }
                ),
                encoding="utf-8",
            )

            report = module.hardware_vehicle_validation_report(
                closed_loop_evidence=closed_loop,
                safety_case=safety,
                integration_manifest=integration,
                hardware_plan=hardware_plan,
                shadow_report=_write_shadow_report(root),
            )

        self.assertFalse(report["ready_to_start_hardware_vehicle_validation"])
        self.assertFalse(report["gates"]["rollback_and_estop_controls_present"])
        self.assertEqual("no_go", report["authorization"]["decision"])
        self.assertIn("physical_estop", report["evidence"]["hardware_plan"]["missing_controls"])

    def test_public_road_or_production_claim_blocks_controlled_start(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            closed_loop = _write_closed_loop(root)
            safety = _write_safety_case(root, public_road=True)
            integration = _write_integration_manifest(root, production=True)
            hardware_plan = _write_hardware_plan(root)

            report = module.hardware_vehicle_validation_report(
                closed_loop_evidence=closed_loop,
                safety_case=safety,
                integration_manifest=integration,
                hardware_plan=hardware_plan,
                shadow_report=_write_shadow_report(root),
            )

        self.assertFalse(report["ready_to_start_hardware_vehicle_validation"])
        self.assertFalse(report["gates"]["production_or_public_road_control_blocked"])
        self.assertFalse(report["authorization"]["public_road_deployment_authorized"])
        self.assertTrue(any("public-road" in blocker for blocker in report["blockers"]))

    def test_missing_shadow_report_blocks_vehicle_validation(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = module.hardware_vehicle_validation_report(
                closed_loop_evidence=_write_closed_loop(root),
                safety_case=_write_safety_case(root),
                integration_manifest=_write_integration_manifest(root),
                hardware_plan=_write_hardware_plan(root),
                shadow_report=root / "missing_shadow.json",
            )

        self.assertFalse(report["ready_to_start_hardware_vehicle_validation"])
        self.assertFalse(report["gates"]["shadow_command_replay_passed"])
        self.assertEqual("no_go", report["authorization"]["decision"])

    def test_quality_failed_closed_loop_level_blocks_vehicle_validation(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            report = module.hardware_vehicle_validation_report(
                closed_loop_evidence=_write_closed_loop(root, quality_thresholds_met=False),
                safety_case=_write_safety_case(root),
                integration_manifest=_write_integration_manifest(root),
                hardware_plan=_write_hardware_plan(root),
                shadow_report=_write_shadow_report(root),
            )

        self.assertFalse(report["ready_to_start_hardware_vehicle_validation"])
        self.assertFalse(report["gates"]["simulator_sotif_evidence_passed"])
        self.assertEqual("no_go", report["authorization"]["decision"])

    def test_shadow_report_with_mismatched_violation_list_blocks_vehicle_validation(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shadow_report = _write_shadow_report(root)
            payload = json.loads(shadow_report.read_text(encoding="utf-8"))
            payload["violation_count"] = 0
            payload["violations"] = [{"index": 0, "violations": ["speed_out_of_bounds"]}]
            shadow_report.write_text(json.dumps(payload), encoding="utf-8")

            report = module.hardware_vehicle_validation_report(
                closed_loop_evidence=_write_closed_loop(root),
                safety_case=_write_safety_case(root),
                integration_manifest=_write_integration_manifest(root),
                hardware_plan=_write_hardware_plan(root),
                shadow_report=shadow_report,
            )

        self.assertFalse(report["ready_to_start_hardware_vehicle_validation"])
        self.assertFalse(report["gates"]["shadow_command_replay_passed"])
        self.assertFalse(report["evidence"]["shadow_report"]["reported_violation_count_matches"])


def _write_closed_loop(
    root: Path,
    *,
    sample_size_valid: bool = True,
    quality_thresholds_met: bool = True,
) -> Path:
    path = root / "closed_loop.json"
    path.write_text(
        json.dumps(
            {
                "schema": "compass_sotif_evidence_package_v0",
                "evidence_profile": "sotif-v0",
                "evidence_status": "compass_evidence_threshold_met",
                "claim_type": "simulation_evidence_package_not_legal_certification",
                "statistical_evidence": {
                    "official_level_evidence": [
                        {
                            "level": 0,
                            "ranked_runs": 381,
                            "minimum_ranked_runs": 381,
                            "sample_size_valid": sample_size_valid,
                            "quality_thresholds_met": quality_thresholds_met,
                        }
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_safety_case(root: Path, *, public_road: bool = False) -> Path:
    path = root / "safety.json"
    path.write_text(
        json.dumps(
            {
                "schema": "av_safety_case_v1",
                "approved_for_public_road_deployment": public_road,
                "sections": [
                    "operational_design_domain",
                    "hazard_analysis",
                    "verification_validation",
                    "residual_risk_acceptance",
                    "fallback_and_minimal_risk_condition",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_integration_manifest(root: Path, *, production: bool = False) -> Path:
    path = root / "integration.json"
    path.write_text(
        json.dumps(
            {
                "schema": "av_stack_integration_manifest_v1",
                "production_av_stack": production,
                "components": {
                    "perception": "validated",
                    "prediction": "validated",
                    "planning": "validated",
                    "control": "validated",
                    "safety_monitor": "validated",
                },
                "closed_loop_interface": "validated",
                "fault_injection": "validated",
                "real_time_budget": "validated",
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_hardware_plan(root: Path) -> Path:
    path = root / "hardware_plan.json"
    path.write_text(
        json.dumps(
            {
                "schema": "hardware_vehicle_validation_plan_v1",
                "approval_status": "approved_for_controlled_validation_start",
                "allowed_modes": ["hardware_in_the_loop", "closed_course_supervised"],
                "public_road_allowed": False,
                "unsupervised_control_allowed": False,
                "mandatory_controls": [
                    "physical_estop",
                    "remote_estop",
                    "safety_driver",
                    "speed_limit_enforced",
                    "geofenced_closed_course",
                    "manual_takeover_tested",
                    "shadow_mode_before_actuation",
                    "command_logging_enabled",
                    "rollback_plan_present",
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_shadow_report(root: Path, *, violations: int = 0) -> Path:
    path = root / "shadow_report.json"
    path.write_text(
        json.dumps(
            {
                "schema": "vehicle_validation_shadow_report_v1",
                "command_count": 10,
                "violations": [] if violations == 0 else [{"index": 0, "violations": ["speed_out_of_bounds"]}],
                "violation_count": violations,
                "max_speed_mps": 1.5,
                "max_abs_steering_rad": 0.3,
                "estop_tested": True,
                "manual_takeover_tested": True,
                "geofence_tested": True,
                "shadow_mode_only": True,
                "ready_for_low_speed_actuation": violations == 0,
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
