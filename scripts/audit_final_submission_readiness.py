#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_ROOT = ROOT / "artifacts" / "sota_submission_bundles"
DEFAULT_MINIMAL_SHOT_AUDIT = ROOT / "artifacts" / "minimal_shot_claim_audit.json"
DEFAULT_JUDGING_AUDIT = ROOT / "artifacts" / "sota_judging_criteria_audit.json"
DEFAULT_PRODUCTION_AUDIT = ROOT / "artifacts" / "production_av_readiness_audit.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "final_submission_readiness_audit.json"

REQUIRED_GRAND_MEMBERS = {
    "grand_commission/notebooks/wod_e2e_analysis.ipynb",
    "grand_commission/artifacts/minimal_shot_claim_audit.json",
    "grand_commission/artifacts/sota_judging_criteria_audit.json",
    "grand_commission/artifacts/production_av_readiness_audit.json",
    "grand_commission/artifacts/wod_e2e_data_restore_plan.json",
    "grand_commission/artifacts/wod_e2e_leaderboard_attack_readiness.json",
}

REQUIRED_MINOR_MEMBERS = {
    "minor_commission/artifacts/minor_eval/scenario_eval.json",
    "minor_commission/artifacts/minor_eval/scenario_eval.csv",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit final SoTA submission readiness in one pass.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--minimal-shot-audit", type=Path, default=DEFAULT_MINIMAL_SHOT_AUDIT)
    parser.add_argument("--judging-audit", type=Path, default=DEFAULT_JUDGING_AUDIT)
    parser.add_argument("--production-audit", type=Path, default=DEFAULT_PRODUCTION_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = final_readiness_report(
        bundle_root=args.bundle_root,
        minimal_shot_audit=args.minimal_shot_audit,
        judging_audit=args.judging_audit,
        production_audit=args.production_audit,
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["valid"] else 1


def final_readiness_report(
    *,
    bundle_root: Path,
    minimal_shot_audit: Path,
    judging_audit: Path,
    production_audit: Path,
) -> dict[str, Any]:
    manifest_path = bundle_root / "submission_bundles_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    grand_archive = bundle_root / "grand_commission.tar.gz"
    minor_archive = bundle_root / "minor_commission.tar.gz"
    grand = _archive_gate(
        grand_archive,
        manifest.get("grand_commission", {}),
        required_members=REQUIRED_GRAND_MEMBERS,
    )
    minor = _archive_gate(
        minor_archive,
        manifest.get("minor_commission", {}),
        required_members=REQUIRED_MINOR_MEMBERS,
    )
    minimal = _minimal_shot_gate(minimal_shot_audit)
    judging = _judging_gate(judging_audit)
    production = _production_boundary_gate(production_audit)
    gates = {
        "manifest_present": manifest_path.is_file(),
        "grand_archive_ready": grand["valid"],
        "minor_archive_ready": minor["valid"],
        "minimal_shot_integrity": minimal["valid"],
        "judging_criteria_evidence": judging["valid"],
        "production_no_go_boundary": production["valid"],
    }
    return {
        "schema": "final_submission_readiness_audit_v1",
        "bundle_root": str(bundle_root),
        "gates": gates,
        "valid": all(gates.values()),
        "archives": {
            "grand_commission": grand,
            "minor_commission": minor,
        },
        "evidence": {
            "minimal_shot": minimal,
            "judging": judging,
            "production_boundary": production,
        },
        "blockers": _blockers(gates, grand, minor, minimal, judging, production),
    }


def _archive_gate(path: Path, manifest_row: dict[str, Any], *, required_members: set[str]) -> dict[str, Any]:
    present = path.is_file()
    members: set[str] = set()
    sha256 = None
    if present:
        sha256 = _sha256(path)
        with tarfile.open(path, "r:gz") as archive:
            members = {member.name for member in archive.getmembers() if member.isfile()}
    missing = sorted(required_members - members)
    manifest_sha = str(manifest_row.get("sha256", ""))
    return {
        "path": str(path),
        "present": present,
        "sha256": sha256,
        "manifest_sha256": manifest_sha or None,
        "manifest_sha_matches": bool(sha256) and sha256 == manifest_sha,
        "required_count": len(required_members),
        "missing_required": missing,
        "valid": present and bool(sha256) and sha256 == manifest_sha and not missing,
    }


def _minimal_shot_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path) if path.is_file() else {}
    checks = payload.get("checks", {})
    required_checks = {
        "active_policy_has_no_episode_lookup",
        "closed_loop_no_near_miss",
        "closed_loop_zero_collision",
        "no_strict_zero_shot_wod_claim",
        "validation_preferences_not_primary_claim",
        "wod_evidence_declared_auxiliary",
    }
    missing_or_false = sorted(check for check in required_checks if checks.get(check) is not True)
    return {
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "valid": payload.get("schema") == "minimal_shot_claim_audit_v1"
        and payload.get("valid") is True
        and not missing_or_false,
        "missing_or_false": missing_or_false,
    }


def _judging_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path) if path.is_file() else {}
    criteria = payload.get("criteria", {})
    failed = sorted(
        name
        for name, row in criteria.items()
        if not isinstance(row, dict) or row.get("status") != "pass"
    )
    return {
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "valid": payload.get("schema") == "sota_judging_criteria_audit_v1"
        and payload.get("valid") is True
        and not failed,
        "failed_criteria": failed,
    }


def _production_boundary_gate(path: Path) -> dict[str, Any]:
    payload = _load_json(path) if path.is_file() else {}
    authorization = payload.get("deployment_authorization", {})
    valid_no_go = (
        payload.get("schema") == "production_av_readiness_audit_v1"
        and payload.get("production_ready") is False
        and authorization.get("decision") == "no_go"
        and authorization.get("real_vehicle_control_authorized") is False
        and authorization.get("public_road_deployment_authorized") is False
    )
    return {
        "path": str(path),
        "present": path.is_file(),
        "schema": payload.get("schema"),
        "valid": valid_no_go,
        "decision": authorization.get("decision"),
        "real_vehicle_control_authorized": authorization.get("real_vehicle_control_authorized"),
        "public_road_deployment_authorized": authorization.get("public_road_deployment_authorized"),
    }


def _blockers(
    gates: dict[str, bool],
    grand: dict[str, Any],
    minor: dict[str, Any],
    minimal: dict[str, Any],
    judging: dict[str, Any],
    production: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    for name, passed in gates.items():
        if not passed:
            blockers.append(f"gate failed: {name}")
    for label, archive in (("grand", grand), ("minor", minor)):
        for member in archive.get("missing_required", []):
            blockers.append(f"{label} archive missing {member}")
    for check in minimal.get("missing_or_false", []):
        blockers.append(f"minimal-shot check failed: {check}")
    for criterion in judging.get("failed_criteria", []):
        blockers.append(f"judging criterion failed: {criterion}")
    if not production.get("valid"):
        blockers.append("production boundary is not an explicit no-go")
    return blockers


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
