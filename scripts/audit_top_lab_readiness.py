#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JUDGING_AUDIT = ROOT / "artifacts" / "sota_judging_criteria_audit.json"
DEFAULT_LEADERBOARD_RESULTS = ROOT / "artifacts" / "wod_e2e_leaderboard_results.json"
DEFAULT_EXTERNAL_EMBEDDING_AUDIT = ROOT / "artifacts" / "external_embedding_result_audit.json"
DEFAULT_SIM_EVAL = ROOT / "artifacts" / "sota_submission_bundles" / "minor_eval" / "scenario_eval.json"
DEFAULT_OOD_EVAL = ROOT / "artifacts" / "sota_submission_bundles" / "minor_ood_eval" / "scenario_eval.json"
DEFAULT_SAFETY_CASE = ROOT / "artifacts" / "safety_case.json"
DEFAULT_RLVR_CURRICULUM = ROOT / "artifacts" / "rlvr_curriculum_v1.json"
DEFAULT_ALPASIGNAL_BRIDGE = (
    ROOT / "artifacts" / "sota_submission_bundles" / "minor_alpasignal_bridge" / "alpasignal_bridge_audit.json"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "top_lab_readiness_audit.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether evidence reaches a top-lab autonomy research bar.")
    parser.add_argument("--judging-audit", type=Path, default=DEFAULT_JUDGING_AUDIT)
    parser.add_argument("--leaderboard-results", type=Path, default=DEFAULT_LEADERBOARD_RESULTS)
    parser.add_argument("--external-embedding-audit", type=Path, default=DEFAULT_EXTERNAL_EMBEDDING_AUDIT)
    parser.add_argument("--sim-eval", type=Path, default=DEFAULT_SIM_EVAL)
    parser.add_argument("--ood-eval", type=Path, default=DEFAULT_OOD_EVAL)
    parser.add_argument("--safety-case", type=Path, default=DEFAULT_SAFETY_CASE)
    parser.add_argument("--rlvr-curriculum", type=Path, default=DEFAULT_RLVR_CURRICULUM)
    parser.add_argument("--alpasignal-bridge", type=Path, default=DEFAULT_ALPASIGNAL_BRIDGE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = top_lab_readiness_report(
        judging_audit=args.judging_audit,
        leaderboard_results=args.leaderboard_results,
        external_embedding_audit=args.external_embedding_audit,
        sim_eval=args.sim_eval,
        ood_eval=args.ood_eval,
        safety_case=args.safety_case,
        rlvr_curriculum=args.rlvr_curriculum,
        alpasignal_bridge=args.alpasignal_bridge,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if report["top_lab_ready"] else 1


def top_lab_readiness_report(
    *,
    judging_audit: Path,
    leaderboard_results: Path,
    external_embedding_audit: Path,
    sim_eval: Path,
    ood_eval: Path,
    safety_case: Path,
    rlvr_curriculum: Path,
    alpasignal_bridge: Path,
) -> dict[str, Any]:
    judging = _judging_evidence(judging_audit)
    leaderboard = _leaderboard_evidence(leaderboard_results)
    external = _external_embedding_evidence(external_embedding_audit)
    sim = _sim_evidence(sim_eval)
    ood = _sim_evidence(ood_eval)
    safety = _safety_case_evidence(safety_case)
    curriculum = _rlvr_curriculum_evidence(rlvr_curriculum)
    bridge = _alpasignal_bridge_evidence(alpasignal_bridge)
    gates = {
        "sota_submission_package_valid": judging["valid"],
        "explicit_rlvr_curriculum": curriculum["valid"],
        "hidden_waymo_test_confirmed": leaderboard["hidden_test_confirmed"],
        "beats_sota_snapshot_on_hidden_test": leaderboard["beats_sota_snapshot"],
        "external_embedding_improves_wod": external["passed"],
        "camera_or_scene_grounding_validated": external["passed"] and external["embedding_source_present"],
        "independent_or_external_simulator_evidence": ood["external_or_independent"] or bridge["valid"],
        "closed_loop_ood_safety_floor": ood["min_trajectory_safety_pass_rate"] >= 0.95,
        "hard_ood_not_solved_by_easy_metrics": ood["hard_suite_disclosed"],
        "formalish_safety_case_present": safety["valid"],
        "reproducibility_and_claim_boundary": judging["valid"] and sim["has_statistics"],
    }
    return {
        "schema": "top_lab_readiness_audit_v1",
        "claim": "top_lab_waymo_deepmind_research_bar",
        "top_lab_ready": all(gates.values()),
        "gates": gates,
        "evidence": {
            "judging": judging,
            "leaderboard": leaderboard,
            "external_embedding": external,
            "closed_loop_sim": sim,
            "closed_loop_ood": ood,
            "safety_case": safety,
            "rlvr_curriculum": curriculum,
            "alpasignal_bridge": bridge,
        },
        "blockers": _blockers(gates, leaderboard, external, ood, safety, curriculum, bridge),
        "acceptable_current_claim": "SoTA Commission prototype package" if judging["valid"] else "research draft",
        "unacceptable_claims": [
            "Google DeepMind-level autonomy system",
            "Waymo WOD-E2E winner",
            "state-of-the-art hidden-test driving policy",
            "production autonomous vehicle stack",
        ],
    }


def _judging_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    criteria = payload.get("criteria", {}) if isinstance(payload.get("criteria"), dict) else {}
    failed = [
        name
        for name, row in criteria.items()
        if not isinstance(row, dict) or row.get("status") != "pass"
    ]
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "valid": payload.get("schema") == "sota_judging_criteria_audit_v1"
        and payload.get("valid") is True
        and not failed,
        "failed_criteria": sorted(failed),
    }


def _leaderboard_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    results = payload.get("results", []) if isinstance(payload.get("results"), list) else []
    confirmed = [
        item
        for item in results
        if isinstance(item, dict)
        and item.get("confirmed_hidden_test") is True
        and isinstance(item.get("hidden_test_rfs"), int | float)
        and str(item.get("submission_sha256", "")).strip()
    ]
    best = max((float(item["hidden_test_rfs"]) for item in confirmed), default=None)
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "hidden_test_confirmed": payload.get("schema") == "wod_e2e_leaderboard_results_v1" and bool(confirmed),
        "confirmed_hidden_test_count": len(confirmed),
        "best_hidden_test_rfs": best,
        "beats_sota_snapshot": bool(best is not None and best > 8.0461),
    }


def _external_embedding_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    failures = payload.get("failures", []) if isinstance(payload.get("failures"), list) else []
    checks = payload.get("checks", []) if isinstance(payload.get("checks"), list) else []
    diagnostics = next(
        (check for check in checks if isinstance(check, dict) and check.get("id") == "world_candidate_diagnostics"),
        {},
    )
    return {
        "path": str(path),
        "present": bool(payload),
        "audit_type": payload.get("audit_type"),
        "passed": payload.get("audit_type") == "external_embedding_result_gate"
        and payload.get("status") == "pass"
        and not failures,
        "embedding_source_present": "missing_embedding_source" not in {
            str(item.get("id")) for item in failures if isinstance(item, dict)
        },
        "selected_world_rate": _optional_float(diagnostics.get("selected_world_rate")),
        "oracle_world_rate": _optional_float(diagnostics.get("oracle_world_rate")),
        "failures": failures,
    }


def _sim_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    runs = payload.get("runs", []) if isinstance(payload.get("runs"), list) else []
    summary = payload.get("summary", []) if isinstance(payload.get("summary"), list) else []
    suites = {str(row.get("suite")) for row in runs if isinstance(row, dict)}
    min_safety = min(
        (float(row.get("trajectory_safety_pass_rate", 0.0)) for row in summary if isinstance(row, dict)),
        default=0.0,
    )
    min_benchmark = min(
        (float(row.get("benchmark_pass_rate", 0.0)) for row in summary if isinstance(row, dict)),
        default=0.0,
    )
    has_statistics = (
        isinstance(payload.get("statistics"), dict)
        and payload["statistics"].get("unit") == "closed-loop rollout"
    )
    return {
        "path": str(path),
        "present": bool(payload),
        "run_count": len(runs),
        "summary_count": len(summary),
        "suites": sorted(suites),
        "external_or_independent": "alpasim" in suites or "external" in suites,
        "has_statistics": has_statistics,
        "min_trajectory_safety_pass_rate": min_safety,
        "min_benchmark_pass_rate": min_benchmark,
        "hard_suite_disclosed": min_benchmark < 0.8
        and any(suite in suites for suite in {"gauntlet", "adversarial"}),
    }


def _safety_case_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    required_sections = {
        "operational_design_domain",
        "hazard_analysis",
        "verification_validation",
        "fallback_and_minimal_risk_condition",
        "residual_risk_acceptance",
    }
    present_sections = _present_safety_case_sections(payload, required_sections)
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "present_sections": sorted(present_sections),
        "missing_sections": sorted(required_sections - present_sections),
        "valid": payload.get("schema") == "av_safety_case_v1" and not (required_sections - present_sections),
    }


def _present_safety_case_sections(payload: dict[str, Any], required_sections: set[str]) -> set[str]:
    raw_sections = payload.get("sections")
    if isinstance(raw_sections, dict):
        return {section for section in required_sections if section in raw_sections}
    if isinstance(raw_sections, list):
        named_sections = {str(section) for section in raw_sections}
        return {section for section in required_sections if section in named_sections and section in payload}
    return {section for section in required_sections if section in payload}


def _rlvr_curriculum_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    coverage = payload.get("coverage", {}) if isinstance(payload.get("coverage"), dict) else {}
    tasks = payload.get("tasks", []) if isinstance(payload.get("tasks"), list) else []
    split_counts = coverage.get("split_counts", {}) if isinstance(coverage.get("split_counts"), dict) else {}
    required_splits = {"train_curriculum", "dev_holdout", "blind_holdout", "stress_holdout"}
    present_splits = {split for split in required_splits if int(split_counts.get(split, 0)) > 0}
    valid_tasks = all(
        isinstance(task, dict)
        and task.get("composition_key")
        and isinstance(task.get("axes"), dict)
        and "benchmark_pass" in task.get("grader_ids", [])
        and task.get("observation_contract", {}).get("privileged_state_allowed") is False
        for task in tasks
    )
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "task_count": int(coverage.get("task_count", 0) or 0),
        "composition_count": int(coverage.get("composition_count", 0) or 0),
        "grader_count": int(coverage.get("grader_count", 0) or 0),
        "observation_mode_count": int(coverage.get("observation_mode_count", 0) or 0),
        "present_splits": sorted(present_splits),
        "missing_splits": sorted(required_splits - present_splits),
        "valid": (
            payload.get("schema") == "minimal_shot_av_rlvr_curriculum_v1"
            and int(coverage.get("task_count", 0) or 0) >= 500
            and int(coverage.get("composition_count", 0) or 0) >= 100
            and int(coverage.get("grader_count", 0) or 0) >= 5
            and int(coverage.get("observation_mode_count", 0) or 0) >= 4
            and required_splits.issubset(present_splits)
            and valid_tasks
        ),
    }


def _alpasignal_bridge_evidence(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    gates = payload.get("gates", {}) if isinstance(payload.get("gates"), dict) else {}
    failed_gates = sorted(str(gate) for gate, passed in gates.items() if passed is not True)
    return {
        "path": str(path),
        "present": bool(payload),
        "schema": payload.get("schema"),
        "valid": payload.get("schema") == "alpasignal_bridge_audit_v1"
        and payload.get("valid") is True
        and not failed_gates,
        "failed_gates": failed_gates,
        "case_count": len(payload.get("cases", [])) if isinstance(payload.get("cases"), list) else 0,
        "boundary": payload.get("boundary"),
    }


def _blockers(
    gates: dict[str, bool],
    leaderboard: dict[str, Any],
    external: dict[str, Any],
    ood: dict[str, Any],
    safety: dict[str, Any],
    curriculum: dict[str, Any],
    bridge: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {
            "gate": gate,
            "detail": _blocker_detail(gate, leaderboard, external, ood, safety, curriculum, bridge),
        }
        for gate, passed in gates.items()
        if not passed
    ]


def _blocker_detail(
    gate: str,
    leaderboard: dict[str, Any],
    external: dict[str, Any],
    ood: dict[str, Any],
    safety: dict[str, Any],
    curriculum: dict[str, Any],
    bridge: dict[str, Any],
) -> str:
    details = {
        "explicit_rlvr_curriculum": (
            f"RLVR curriculum invalid: {curriculum['task_count']} tasks, "
            f"{curriculum['composition_count']} compositions, missing splits {curriculum['missing_splits']}."
        ),
        "hidden_waymo_test_confirmed": "No confirmed hidden-test WOD-E2E leaderboard result is recorded.",
        "beats_sota_snapshot_on_hidden_test": (
            f"Best hidden-test RFS is {leaderboard['best_hidden_test_rfs']}; "
            "required > 8.0461 snapshot."
        ),
        "external_embedding_improves_wod": "No passing external/camera-scene embedding audit is attached.",
        "camera_or_scene_grounding_validated": "No passing evidence that pixels or scene embeddings improve decisions.",
        "independent_or_external_simulator_evidence": (
            f"OOD suites are {ood['suites']}; AlpaSignal bridge valid={bridge['valid']}. "
            "No independent/external simulator or structured simulator bridge evidence is present."
        ),
        "closed_loop_ood_safety_floor": (
            f"OOD trajectory-safety floor is {ood['min_trajectory_safety_pass_rate']:.3f}; "
            "required >= 0.950."
        ),
        "hard_ood_not_solved_by_easy_metrics": "Hard-suite failures are not disclosed in OOD summary.",
        "formalish_safety_case_present": f"Safety case missing sections: {safety['missing_sections']}.",
    }
    if gate == "sota_submission_package_valid":
        return "SoTA judging audit does not pass."
    if gate == "reproducibility_and_claim_boundary":
        return "Reproducibility/statistical/claim-boundary evidence is incomplete."
    return details.get(gate, "Gate failed.")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
