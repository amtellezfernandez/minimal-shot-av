#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLOSED_LOOP_EVIDENCE = ROOT / "artifacts" / "compass_evidence_report_sotif_seed381.json"
DEFAULT_SAFETY_CASE = ROOT / "artifacts" / "safety_case.json"
DEFAULT_INTEGRATION_MANIFEST = ROOT / "artifacts" / "av_stack_integration_manifest.json"
DEFAULT_HARDWARE_PLAN = ROOT / "configs" / "hardware_vehicle_validation_plan.json"
DEFAULT_SHADOW_REPORT = ROOT / "artifacts" / "vehicle_validation_shadow_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit readiness to start controlled hardware-in-the-loop and vehicle validation."
    )
    parser.add_argument("--closed-loop-evidence", type=Path, default=DEFAULT_CLOSED_LOOP_EVIDENCE)
    parser.add_argument("--safety-case", type=Path, default=DEFAULT_SAFETY_CASE)
    parser.add_argument("--integration-manifest", type=Path, default=DEFAULT_INTEGRATION_MANIFEST)
    parser.add_argument("--hardware-plan", type=Path, default=DEFAULT_HARDWARE_PLAN)
    parser.add_argument("--shadow-report", type=Path, default=DEFAULT_SHADOW_REPORT)
    parser.add_argument(
        "--hil-check",
        action="store_true",
        help="Exit as a HIL/VIL start check: 0 when controlled validation may start, 2 when blocked.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = hardware_vehicle_validation_report(
        closed_loop_evidence=args.closed_loop_evidence,
        safety_case=args.safety_case,
        integration_manifest=args.integration_manifest,
        hardware_plan=args.hardware_plan,
        shadow_report=args.shadow_report,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.hil_check:
        return 0 if report["ready_to_start_hardware_vehicle_validation"] else 2
    return 0 if report["ready_to_start_hardware_vehicle_validation"] else 1


def hardware_vehicle_validation_report(
    *,
    closed_loop_evidence: Path,
    safety_case: Path,
    integration_manifest: Path,
    hardware_plan: Path,
    shadow_report: Path,
) -> dict[str, Any]:
    closed_loop = _closed_loop_evidence(closed_loop_evidence)
    safety = _safety_case_evidence(safety_case)
    integration = _integration_evidence(integration_manifest)
    plan = _hardware_plan_evidence(hardware_plan)
    shadow = _shadow_report_evidence(shadow_report, plan)

    gates = {
        "production_or_public_road_control_blocked": (
            not safety["public_road_approved"] and not integration["production_stack_claim_valid"]
        ),
        "simulator_sotif_evidence_passed": closed_loop["closed_loop_threshold_met"],
        "robust_simulator_integration_validated": integration["robust_integration_valid"],
        "safety_case_sections_present": safety["required_sections_present"],
        "controlled_hil_plan_approved": plan["hil_approved"],
        "controlled_vehicle_plan_approved": plan["vehicle_validation_approved"],
        "rollback_and_estop_controls_present": plan["rollback_and_estop_controls_present"],
        "shadow_command_replay_passed": shadow["ready_for_low_speed_actuation"],
    }
    blockers = _blockers(gates, closed_loop, safety, integration, plan, shadow)
    return {
        "schema": "hardware_vehicle_validation_readiness_v1",
        "claim": "controlled_hil_vil_start_readiness",
        "ready_to_start_hardware_vehicle_validation": all(gates.values()),
        "gates": gates,
        "evidence": {
            "closed_loop": closed_loop,
            "safety_case": safety,
            "integration": integration,
            "hardware_plan": plan,
            "shadow_report": shadow,
        },
        "blockers": blockers,
        "authorization": _authorization(gates, blockers, plan),
        "next_actions": _next_actions(gates, closed_loop, safety, integration, plan, shadow),
        "disposition": "controlled_hil_vil_ready" if not blockers else "hardware_validation_blocked",
    }


def _authorization(gates: dict[str, bool], blockers: list[str], plan: dict[str, Any]) -> dict[str, Any]:
    failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
    ready = not failed_gates
    allowed_modes = list(plan["allowed_modes"]) if ready else []
    return {
        "decision": "go_controlled_validation" if ready else "no_go",
        "hardware_in_the_loop_authorized": ready and "hardware_in_the_loop" in allowed_modes,
        "closed_course_vehicle_validation_authorized": ready and "closed_course_supervised" in allowed_modes,
        "public_road_deployment_authorized": False,
        "unsupervised_vehicle_control_authorized": False,
        "allowed_modes": allowed_modes,
        "failed_gates": failed_gates,
        "blocking_reasons": list(blockers),
        "mandatory_controls": list(plan["mandatory_controls"]) if ready else _default_blocking_controls(),
    }


def _closed_loop_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    level_evidence = _closed_loop_level_evidence(payload)
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "evidence_profile": payload.get("evidence_profile"),
        "evidence_status": payload.get("evidence_status"),
        "claim_type": payload.get("claim_type"),
        "closed_loop_threshold_met": (
            payload.get("schema") == "compass_sotif_evidence_package_v0"
            and payload.get("evidence_profile") == "sotif-v0"
            and payload.get("evidence_status") == "compass_evidence_threshold_met"
            and payload.get("claim_type") == "simulation_evidence_package_not_legal_certification"
            and bool(level_evidence)
            and sum(level["missing_ranked_runs"] for level in level_evidence) == 0
            and all(level["sample_size_valid"] for level in level_evidence)
            and all(level["quality_thresholds_met"] for level in level_evidence)
        ),
        "official_level_evidence": level_evidence,
        "total_missing_ranked_runs": sum(level["missing_ranked_runs"] for level in level_evidence),
    }


def _safety_case_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    required_sections = {
        "operational_design_domain",
        "hazard_analysis",
        "verification_validation",
        "residual_risk_acceptance",
        "fallback_and_minimal_risk_condition",
    }
    sections = (
        {str(section) for section in payload.get("sections", [])}
        if isinstance(payload.get("sections"), list)
        else set()
    )
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "public_road_approved": payload.get("approved_for_public_road_deployment") is True,
        "required_sections": sorted(required_sections),
        "present_sections": sorted(sections),
        "missing_sections": sorted(required_sections - sections),
        "required_sections_present": (
            payload.get("schema") == "av_safety_case_v1" and required_sections.issubset(sections)
        ),
    }


def _integration_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    components = payload.get("components", {}) if isinstance(payload.get("components"), dict) else {}
    required_components = ("perception", "prediction", "planning", "control", "safety_monitor")
    component_status = {name: components.get(name) == "validated" for name in required_components}
    robust = (
        payload.get("schema") == "av_stack_integration_manifest_v1"
        and all(component_status.values())
        and payload.get("closed_loop_interface") == "validated"
        and payload.get("fault_injection") == "validated"
        and payload.get("real_time_budget") == "validated"
    )
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "component_status": component_status,
        "closed_loop_interface": payload.get("closed_loop_interface"),
        "fault_injection": payload.get("fault_injection"),
        "real_time_budget": payload.get("real_time_budget"),
        "production_stack_claim_valid": robust and payload.get("production_av_stack") is True,
        "robust_integration_valid": robust,
    }


def _hardware_plan_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    controls = payload.get("mandatory_controls", []) if isinstance(payload.get("mandatory_controls"), list) else []
    allowed_modes = payload.get("allowed_modes", []) if isinstance(payload.get("allowed_modes"), list) else []
    required_controls = {
        "physical_estop",
        "remote_estop",
        "safety_driver",
        "speed_limit_enforced",
        "geofenced_closed_course",
        "manual_takeover_tested",
        "shadow_mode_before_actuation",
        "command_logging_enabled",
        "rollback_plan_present",
    }
    control_set = {str(control) for control in controls}
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "approval_status": payload.get("approval_status"),
        "validation_scope": payload.get("validation_scope"),
        "allowed_modes": [str(mode) for mode in allowed_modes],
        "mandatory_controls": sorted(control_set),
        "missing_controls": sorted(required_controls - control_set),
        "hil_approved": (
            payload.get("schema") == "hardware_vehicle_validation_plan_v1"
            and payload.get("approval_status") == "approved_for_controlled_validation_start"
            and "hardware_in_the_loop" in allowed_modes
        ),
        "vehicle_validation_approved": (
            payload.get("schema") == "hardware_vehicle_validation_plan_v1"
            and payload.get("approval_status") == "approved_for_controlled_validation_start"
            and "closed_course_supervised" in allowed_modes
            and payload.get("public_road_allowed") is False
            and payload.get("unsupervised_control_allowed") is False
        ),
        "rollback_and_estop_controls_present": required_controls.issubset(control_set),
        "max_speed_mps": _float(payload.get("max_speed_mps")),
    }


def _shadow_report_evidence(path: Path, plan: dict[str, Any]) -> dict[str, Any]:
    payload = _read_json(path)
    expected_max_speed = _float(plan.get("max_speed_mps"))
    max_speed = _float(payload.get("max_speed_mps"))
    violations = payload.get("violations", [])
    violations_valid = isinstance(violations, list)
    violation_count = payload.get("violation_count", len(violations) if violations_valid else 1)
    violation_count_int = int(violation_count) if isinstance(violation_count, int) else 1
    ready = (
        payload.get("schema") == "vehicle_validation_shadow_report_v1"
        and int(payload.get("command_count", 0)) > 0
        and violations_valid
        and len(violations) == 0
        and violation_count_int == 0
        and payload.get("estop_tested") is True
        and payload.get("manual_takeover_tested") is True
        and payload.get("geofence_tested") is True
        and payload.get("shadow_mode_only") is True
        and payload.get("ready_for_low_speed_actuation") is True
        and (expected_max_speed <= 0.0 or max_speed <= expected_max_speed + 1e-9)
    )
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "command_count": (
            int(payload.get("command_count", 0)) if isinstance(payload.get("command_count"), int) else 0
        ),
        "violation_count": violation_count_int,
        "violations_valid": violations_valid,
        "reported_violation_count_matches": violations_valid and violation_count_int == len(violations),
        "max_speed_mps": max_speed,
        "estop_tested": payload.get("estop_tested") is True,
        "manual_takeover_tested": payload.get("manual_takeover_tested") is True,
        "geofence_tested": payload.get("geofence_tested") is True,
        "shadow_mode_only": payload.get("shadow_mode_only") is True,
        "ready_for_low_speed_actuation": ready,
    }


def _blockers(
    gates: dict[str, bool],
    closed_loop: dict[str, Any],
    safety: dict[str, Any],
    integration: dict[str, Any],
    plan: dict[str, Any],
    shadow: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not gates["production_or_public_road_control_blocked"]:
        blockers.append("public-road or production control is not explicitly blocked")
    if not gates["simulator_sotif_evidence_passed"]:
        blockers.append(
            "strict simulator SOTIF evidence is missing or incomplete "
            f"({closed_loop.get('evidence_status') or 'no evidence'})"
        )
    if not gates["robust_simulator_integration_validated"]:
        blockers.append(f"validated simulator integration manifest is missing at {integration['path']}")
    if not gates["safety_case_sections_present"]:
        blockers.append("safety case is missing required sections: " + ", ".join(safety["missing_sections"]))
    if not gates["controlled_hil_plan_approved"]:
        blockers.append(f"HIL start is not approved by {plan['path']}")
    if not gates["controlled_vehicle_plan_approved"]:
        blockers.append("closed-course supervised vehicle validation is not approved or not properly constrained")
    if not gates["rollback_and_estop_controls_present"]:
        blockers.append("hardware plan is missing controls: " + ", ".join(plan["missing_controls"]))
    if not gates["shadow_command_replay_passed"]:
        blockers.append(
            "shadow command replay report is missing, unsafe, or not ready for low-speed actuation "
            f"at {shadow['path']}"
        )
    return blockers


def _next_actions(
    gates: dict[str, bool],
    closed_loop: dict[str, Any],
    safety: dict[str, Any],
    integration: dict[str, Any],
    plan: dict[str, Any],
    shadow: dict[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not gates["simulator_sotif_evidence_passed"]:
        actions.append(
            {
                "gate": "simulator_sotif_evidence_passed",
                "action": "rerun sotif-v0 evidence until every official level meets sample-size and quality gates",
                "total_missing_ranked_runs": closed_loop["total_missing_ranked_runs"],
            }
        )
    if not gates["robust_simulator_integration_validated"]:
        actions.append(
            {
                "gate": "robust_simulator_integration_validated",
                "action": (
                    "validate perception, prediction, planning, control, safety monitor, "
                    "fault injection, and timing"
                ),
                "component_status": integration["component_status"],
            }
        )
    if not gates["safety_case_sections_present"]:
        actions.append(
            {
                "gate": "safety_case_sections_present",
                "action": "complete the minimum safety-case sections for controlled test-track validation",
                "missing_sections": safety["missing_sections"],
            }
        )
    if not gates["controlled_hil_plan_approved"] or not gates["controlled_vehicle_plan_approved"]:
        actions.append(
            {
                "gate": "controlled_validation_plan",
                "action": (
                    "approve a constrained HIL/VIL plan with explicit no-public-road "
                    "and no-unsupervised-control limits"
                ),
                "approval_status": plan.get("approval_status"),
                "allowed_modes": plan.get("allowed_modes", []),
            }
        )
    if not gates["rollback_and_estop_controls_present"]:
        actions.append(
            {
                "gate": "rollback_and_estop_controls_present",
                "action": "add missing mandatory hardware controls before actuation",
                "missing_controls": plan["missing_controls"],
            }
        )
    if not gates["shadow_command_replay_passed"]:
        actions.append(
            {
                "gate": "shadow_command_replay_passed",
                "action": "run scripts/run_vehicle_validation_shadow.py and resolve any command-envelope violations",
                "path": shadow["path"],
                "command_count": shadow["command_count"],
                "violation_count": shadow["violation_count"],
            }
        )
    return actions


def _closed_loop_level_evidence(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_levels = payload.get("statistical_evidence", {}).get("official_level_evidence", [])
    if not isinstance(raw_levels, list):
        return []
    levels: list[dict[str, Any]] = []
    for level in raw_levels:
        if not isinstance(level, dict):
            continue
        ranked_runs = int(level.get("ranked_runs", 0))
        minimum = int(level.get("minimum_ranked_runs", 0))
        levels.append(
            {
                "level": level.get("level"),
                "name": level.get("name"),
                "ranked_runs": ranked_runs,
                "minimum_ranked_runs": minimum,
                "missing_ranked_runs": max(0, minimum - ranked_runs),
                "sample_size_valid": bool(level.get("sample_size_valid", False)),
                "quality_thresholds_met": bool(level.get("quality_thresholds_met", False)),
            }
        )
    return levels


def _default_blocking_controls() -> list[str]:
    return [
        "do_not_connect_model_output_to_steering_throttle_or_brake",
        "bench_or_simulation_only_until_failed_gates_are_resolved",
        "require_physical_and_remote_estop_before_vehicle_motion",
        "require_safety_driver_and_geofence_for_closed_course_tests",
    ]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _float(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
