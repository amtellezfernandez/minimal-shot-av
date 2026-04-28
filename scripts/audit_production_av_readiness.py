#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY_REPORT = (
    ROOT / "artifacts" / "wod_bestbase_crossfit_kingate_fastonly_rate020_rerun_ridge175_cv_local.json"
)
DEFAULT_LEADERBOARD_RESULTS = ROOT / "artifacts" / "wod_e2e_leaderboard_results.json"
DEFAULT_CLOSED_LOOP_EVIDENCE = ROOT / "artifacts" / "compass_evidence_report.json"
DEFAULT_SAFETY_CASE = ROOT / "artifacts" / "safety_case.json"
DEFAULT_INTEGRATION_MANIFEST = ROOT / "artifacts" / "av_stack_integration_manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strictly audit whether local evidence supports production AV claims."
    )
    parser.add_argument("--trajectory-report", type=Path, default=DEFAULT_TRAJECTORY_REPORT)
    parser.add_argument("--leaderboard-results", type=Path, default=DEFAULT_LEADERBOARD_RESULTS)
    parser.add_argument("--closed-loop-evidence", type=Path, default=DEFAULT_CLOSED_LOOP_EVIDENCE)
    parser.add_argument("--safety-case", type=Path, default=DEFAULT_SAFETY_CASE)
    parser.add_argument("--integration-manifest", type=Path, default=DEFAULT_INTEGRATION_MANIFEST)
    parser.add_argument("--min-normalized-score", type=float, default=9.0)
    parser.add_argument("--min-oracle-capture", type=float, default=0.95)
    parser.add_argument(
        "--deployment-check",
        action="store_true",
        help="Exit as a deployment authorization check: 0 when authorized, 2 when explicitly blocked.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = production_readiness_report(
        trajectory_report=args.trajectory_report,
        leaderboard_results=args.leaderboard_results,
        closed_loop_evidence=args.closed_loop_evidence,
        safety_case=args.safety_case,
        integration_manifest=args.integration_manifest,
        min_normalized_score=args.min_normalized_score,
        min_oracle_capture=args.min_oracle_capture,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.deployment_check:
        return 0 if report["deployment_authorization"]["real_vehicle_control_authorized"] else 2
    return 0 if report["production_ready"] else 1


def production_readiness_report(
    *,
    trajectory_report: Path,
    leaderboard_results: Path,
    closed_loop_evidence: Path,
    safety_case: Path,
    integration_manifest: Path,
    min_normalized_score: float = 9.0,
    min_oracle_capture: float = 0.95,
) -> dict[str, Any]:
    trajectory = _trajectory_metrics(trajectory_report)
    leaderboard = _leaderboard_evidence(leaderboard_results)
    closed_loop = _closed_loop_evidence(closed_loop_evidence)
    safety = _safety_case_evidence(safety_case)
    integration = _integration_evidence(integration_manifest)

    gates = {
        "production_av_stack": integration["production_stack_claim_valid"],
        "robust_perception_planning_control_integration": integration["robust_integration_valid"],
        "closed_loop_driving_validation": closed_loop["closed_loop_threshold_met"],
        "safety_case": safety["safety_case_valid"],
        "generalization_beyond_local_frame_cache": leaderboard["hidden_test_confirmed"],
        "score_near_oracle": trajectory["oracle_capture_ratio"] >= min_oracle_capture,
        "score_near_normalized_ceiling": trajectory["combined_ranker_mean_normalized_rfs"] >= min_normalized_score,
    }
    blockers = _blockers(gates, trajectory, leaderboard, closed_loop, safety, integration)
    solved_gates = _solved_gates(gates, trajectory, leaderboard, closed_loop, safety, integration)
    next_actions = _next_actions(
        gates,
        trajectory,
        leaderboard,
        closed_loop,
        safety,
        integration,
        min_normalized_score=min_normalized_score,
        min_oracle_capture=min_oracle_capture,
    )
    deployment_authorization = _deployment_authorization(gates, blockers)
    return {
        "schema": "production_av_readiness_audit_v1",
        "claim": "production_av_stack_readiness",
        "production_ready": all(gates.values()),
        "thresholds": {
            "min_normalized_score": float(min_normalized_score),
            "min_oracle_capture": float(min_oracle_capture),
        },
        "gates": gates,
        "trajectory_metrics": trajectory,
        "evidence": {
            "leaderboard": leaderboard,
            "closed_loop": closed_loop,
            "safety_case": safety,
            "integration": integration,
        },
        "blockers": blockers,
        "solved_gates": solved_gates,
        "next_actions": next_actions,
        "deployment_authorization": deployment_authorization,
        "disposition": "research_harness_only" if blockers else "production_claim_evidence_complete",
    }


def _deployment_authorization(gates: dict[str, bool], blockers: list[str]) -> dict[str, Any]:
    failed_gates = sorted(gate for gate, passed in gates.items() if not passed)
    authorized = not failed_gates
    return {
        "decision": "go" if authorized else "no_go",
        "real_vehicle_control_authorized": authorized,
        "public_road_deployment_authorized": authorized,
        "human_subject_or_bystander_risk_authorized": authorized,
        "required_gate_count": len(gates),
        "passed_gate_count": len(gates) - len(failed_gates),
        "failed_gates": failed_gates,
        "blocking_reasons": list(blockers),
        "mandatory_controls": []
        if authorized
        else [
            "do_not_connect_model_output_to_steering_throttle_or_brake",
            "simulation_and_offline_analysis_only",
            "require_independent_safety_case_before_vehicle_tests",
            "require_validated_perception_prediction_planning_control_manifest",
            "require_hidden_test_or_blind_generalization_evidence",
        ],
    }


def _solved_gates(
    gates: dict[str, bool],
    trajectory: dict[str, Any],
    leaderboard: dict[str, Any],
    closed_loop: dict[str, Any],
    safety: dict[str, Any],
    integration: dict[str, Any],
) -> list[dict[str, Any]]:
    solved: list[dict[str, Any]] = []
    if gates["closed_loop_driving_validation"]:
        solved.append(
            {
                "gate": "closed_loop_driving_validation",
                "evidence_path": closed_loop["path"],
                "evidence_profile": closed_loop["evidence_profile"],
                "evidence_status": closed_loop["evidence_status"],
                "total_missing_ranked_runs": closed_loop["total_missing_ranked_runs"],
            }
        )
    if gates["generalization_beyond_local_frame_cache"]:
        solved.append(
            {
                "gate": "generalization_beyond_local_frame_cache",
                "evidence_path": leaderboard["path"],
                "confirmed_hidden_test_count": leaderboard["confirmed_hidden_test_count"],
            }
        )
    if gates["score_near_oracle"]:
        solved.append(
            {
                "gate": "score_near_oracle",
                "current": trajectory["oracle_capture_ratio"],
                "rfs_gap_to_oracle": trajectory["rfs_gap_to_oracle"],
            }
        )
    if gates["score_near_normalized_ceiling"]:
        solved.append(
            {
                "gate": "score_near_normalized_ceiling",
                "current": trajectory["combined_ranker_mean_normalized_rfs"],
                "normalized_ceiling_gap": trajectory["normalized_ceiling_gap"],
            }
        )
    if gates["safety_case"]:
        solved.append(
            {
                "gate": "safety_case",
                "evidence_path": safety["path"],
                "present_sections": safety["present_sections"],
            }
        )
    if gates["robust_perception_planning_control_integration"]:
        solved.append(
            {
                "gate": "robust_perception_planning_control_integration",
                "evidence_path": integration["path"],
                "component_status": integration["component_status"],
            }
        )
    if gates["production_av_stack"]:
        solved.append(
            {
                "gate": "production_av_stack",
                "evidence_path": integration["path"],
                "production_stack_claim_valid": integration["production_stack_claim_valid"],
            }
        )
    return solved


def _trajectory_metrics(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    selected = _float(payload.get("combined_ranker_mean_rfs"))
    oracle = _float(payload.get("combined_oracle_mean_rfs"))
    normalized = _float(payload.get("combined_ranker_mean_normalized_rfs"))
    reference_ceiling = _float(payload.get("reference_ceiling_mean_rfs"))
    oracle_normalized = _float(payload.get("combined_oracle_mean_normalized_rfs"))
    future_normalized = _float(payload.get("future_trajectory_mean_normalized_rfs"))
    label_profile = _trajectory_label_profile(payload, normalized, oracle_normalized, reference_ceiling)
    return {
        "path": str(path),
        "present": bool(payload),
        "combined_ranker_mean_rfs": selected,
        "combined_oracle_mean_rfs": oracle,
        "combined_ranker_mean_normalized_rfs": normalized,
        "combined_oracle_mean_normalized_rfs": oracle_normalized,
        "future_trajectory_mean_normalized_rfs": future_normalized,
        "reference_ceiling_mean_rfs": reference_ceiling,
        "oracle_capture_ratio": selected / oracle if oracle > 0.0 else 0.0,
        "normalized_oracle_capture_ratio": normalized / oracle_normalized if oracle_normalized > 0.0 else 0.0,
        "reference_ceiling_capture_ratio": selected / reference_ceiling if reference_ceiling > 0.0 else 0.0,
        "rfs_gap_to_oracle": max(0.0, oracle - selected),
        "normalized_ceiling_gap": max(0.0, 10.0 - normalized),
        "label_metric_profile": label_profile,
    }


def _trajectory_label_profile(
    payload: dict[str, Any],
    normalized: float,
    oracle_normalized: float,
    reference_ceiling: float,
) -> dict[str, Any]:
    frame_count = int(payload.get("frames", 0)) if isinstance(payload.get("frames"), int) else 0
    hidden_labels = bool(payload.get("hidden_test_confirmed", False))
    local_cv = payload.get("benchmark_type") == "segment_grouped_cross_validation"
    oracle_gap = max(0.0, oracle_normalized - normalized) if oracle_normalized > 0.0 else 0.0
    cautions: list[str] = []
    if local_cv:
        cautions.append("segment-grouped local CV is development evidence, not hidden-test generalization")
    if not hidden_labels:
        cautions.append("no confirmed hidden-test labels or leaderboard result are attached")
    if reference_ceiling <= 0.0:
        cautions.append("reference ceiling is missing, so normalized score calibration is incomplete")
    if oracle_gap > 0.5:
        cautions.append("candidate oracle is much stronger than selected policy, so routing labels remain weak")
    return {
        "frames": frame_count,
        "benchmark_type": payload.get("benchmark_type"),
        "local_cv_only": local_cv and not hidden_labels,
        "hidden_test_confirmed": hidden_labels,
        "oracle_normalized_gap": oracle_gap,
        "cautions": cautions,
    }


def _leaderboard_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    confirmed = [
        item
        for item in results
        if isinstance(item, dict)
        and item.get("confirmed_hidden_test") is True
        and isinstance(item.get("hidden_test_rfs"), (int, float))
        and str(item.get("submission_sha256", "")).strip()
    ]
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "hidden_test_confirmed": payload.get("schema") == "wod_e2e_leaderboard_results_v1" and bool(confirmed),
        "confirmed_hidden_test_count": len(confirmed),
    }


def _closed_loop_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    conservative_profile = payload.get("evidence_profile") == "sotif-v0"
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
            and conservative_profile
            and payload.get("evidence_status") == "compass_evidence_threshold_met"
            and payload.get("claim_type") == "simulation_evidence_package_not_legal_certification"
        ),
        "profile_allowed_for_production_gate": conservative_profile,
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
    sections = set(payload.get("sections", [])) if isinstance(payload.get("sections"), list) else set()
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "required_sections": sorted(required_sections),
        "present_sections": sorted(str(section) for section in sections),
        "missing_sections": sorted(required_sections - {str(section) for section in sections}),
        "safety_case_valid": (
            payload.get("schema") == "av_safety_case_v1"
            and payload.get("approved_for_public_road_deployment") is True
            and required_sections.issubset({str(section) for section in sections})
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


def _blockers(
    gates: dict[str, bool],
    trajectory: dict[str, Any],
    leaderboard: dict[str, Any],
    closed_loop: dict[str, Any],
    safety: dict[str, Any],
    integration: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    if not gates["score_near_normalized_ceiling"]:
        blockers.append(
            "selected normalized RFS is "
            f"{trajectory['combined_ranker_mean_normalized_rfs']:.3f}, below the production audit threshold"
        )
    if not gates["score_near_oracle"]:
        blockers.append(
            "selected/oracle capture ratio is "
            f"{trajectory['oracle_capture_ratio']:.3f}, so routing remains far from candidate oracle"
        )
    if not gates["generalization_beyond_local_frame_cache"]:
        blockers.append(f"missing confirmed hidden-test leaderboard evidence at {leaderboard['path']}")
    if not gates["closed_loop_driving_validation"]:
        blockers.append(
            "closed-loop evidence is missing or below threshold "
            f"({closed_loop.get('evidence_status') or 'no evidence'})"
        )
    if not gates["safety_case"]:
        missing_sections = safety.get("missing_sections", [])
        if missing_sections:
            blockers.append(
                "missing approved safety case with required sections: "
                + ", ".join(missing_sections)
            )
        else:
            blockers.append("safety case sections are present, but public-road approval is false")
    if not gates["robust_perception_planning_control_integration"]:
        blockers.append(f"missing validated perception/prediction/planning/control manifest at {integration['path']}")
    if not gates["production_av_stack"]:
        blockers.append("no validated production AV stack claim is present")
    return blockers


def _next_actions(
    gates: dict[str, bool],
    trajectory: dict[str, Any],
    leaderboard: dict[str, Any],
    closed_loop: dict[str, Any],
    safety: dict[str, Any],
    integration: dict[str, Any],
    *,
    min_normalized_score: float,
    min_oracle_capture: float,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    if not gates["score_near_normalized_ceiling"]:
        actions.append(
            {
                "gate": "score_near_normalized_ceiling",
                "action": "improve WOD selector/routing until normalized CV score reaches threshold",
                "current": trajectory["combined_ranker_mean_normalized_rfs"],
                "target": float(min_normalized_score),
                "gap_to_target": max(
                    0.0,
                    float(min_normalized_score) - trajectory["combined_ranker_mean_normalized_rfs"],
                ),
                "label_metric_profile": trajectory["label_metric_profile"],
            }
        )
    if not gates["score_near_oracle"]:
        actions.append(
            {
                "gate": "score_near_oracle",
                "action": "reduce candidate-selection regret and improve oracle capture",
                "current": trajectory["oracle_capture_ratio"],
                "target": float(min_oracle_capture),
                "gap": max(0.0, float(min_oracle_capture) - trajectory["oracle_capture_ratio"]),
                "rfs_gap_to_oracle": trajectory["rfs_gap_to_oracle"],
                "normalized_oracle_capture_ratio": trajectory["normalized_oracle_capture_ratio"],
            }
        )
    if not gates["closed_loop_driving_validation"]:
        actions.append(
            {
                "gate": "closed_loop_driving_validation",
                "action": "run conservative SOTIF evidence profile with enough ranked runs and no threshold failures",
                "evidence_profile": closed_loop.get("evidence_profile"),
                "evidence_status": closed_loop.get("evidence_status"),
                "total_missing_ranked_runs": closed_loop.get("total_missing_ranked_runs", 0),
                "levels": closed_loop.get("official_level_evidence", []),
            }
        )
    if not gates["generalization_beyond_local_frame_cache"]:
        actions.append(
            {
                "gate": "generalization_beyond_local_frame_cache",
                "action": "record confirmed hidden-test leaderboard result for a packaged blind submission",
                "path": leaderboard["path"],
            }
        )
    if not gates["robust_perception_planning_control_integration"]:
        actions.append(
            {
                "gate": "robust_perception_planning_control_integration",
                "action": (
                    "validate all integration components plus closed-loop interface, "
                    "fault injection, and real-time budget"
                ),
                "component_status": integration.get("component_status", {}),
            }
        )
    if not gates["safety_case"]:
        actions.append(
            {
                "gate": "safety_case",
                "action": (
                    "complete and approve safety case only after validation evidence "
                    "supports public-road deployment"
                ),
                "missing_sections": safety.get("missing_sections", []),
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
        success_rate = level.get("success_rate", {}) if isinstance(level.get("success_rate"), dict) else {}
        collision_rate = level.get("collision_rate", {}) if isinstance(level.get("collision_rate"), dict) else {}
        quality = level.get("quality", {}) if isinstance(level.get("quality"), dict) else {}
        levels.append(
            {
                "level": level.get("level"),
                "name": level.get("name"),
                "ranked_runs": ranked_runs,
                "minimum_ranked_runs": minimum,
                "missing_ranked_runs": max(0, minimum - ranked_runs),
                "sample_size_valid": bool(level.get("sample_size_valid", False)),
                "success_lower_bound": _float(success_rate.get("lower")),
                "collision_upper_bound": _float(collision_rate.get("upper")),
                "quality_thresholds_met": bool(level.get("quality_thresholds_met", False)),
                "safety_score": _float(quality.get("safety_score")),
                "worst_safety_score": _float(quality.get("worst_safety_score")),
            }
        )
    return levels


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
