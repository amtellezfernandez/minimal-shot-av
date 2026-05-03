#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE_ROOT = ROOT / "artifacts" / "sota_submission_bundles"
DEFAULT_WAYMO_DATA_ROOT = ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEFAULT_WAYMO_LEADERBOARD_RESULTS = ROOT / "artifacts" / "wod_e2e_leaderboard_results.json"

REQUIRED_MEMBERS = {
    "grand_commission": {
        "grand_commission/README.md",
        "grand_commission/repo_README.md",
        "grand_commission/models/DECLARATION.md",
        "grand_commission/docs/sota-grand-submission-form.md",
        "grand_commission/docs/sota-grand-slide-script.md",
        "grand_commission/docs/submission-tracks.md",
        "grand_commission/docs/grand-submission.md",
        "grand_commission/docs/judging-criteria-evidence.md",
        "grand_commission/docs/final-submission-handoff.md",
        "grand_commission/notebooks/wod_e2e_analysis.ipynb",
        "grand_commission/artifacts/wod_fastkin_gate_ridge175_rate020_fallback_cv_official.json",
        "grand_commission/artifacts/wod_fastkin_gate_ridge175_rate020_fallback_breakthrough_audit.json",
        "grand_commission/artifacts/sota_judging_criteria_audit.json",
        "grand_commission/artifacts/grand_spotlight_demo/latest_rollout.json",
        "grand_commission/artifacts/grand_spotlight_demo/latest_rollout.svg",
        "grand_commission/artifacts/grand_baseline_spotlight_demo/latest_rollout.json",
        "grand_commission/artifacts/grand_baseline_spotlight_demo/latest_rollout.svg",
    },
    "minor_commission": {
        "minor_commission/README.md",
        "minor_commission/repo_README.md",
        "minor_commission/docs/sota-minor-submission-form.md",
        "minor_commission/docs/sota-minor-slide-script.md",
        "minor_commission/docs/submission-tracks.md",
        "minor_commission/docs/minor-simulation-submission.md",
        "minor_commission/docs/final-submission-handoff.md",
        "minor_commission/artifacts/minor_construction_seed1/latest_rollout.json",
        "minor_commission/artifacts/minor_construction_seed1/latest_rollout.svg",
        "minor_commission/artifacts/minor_fod_seed2/latest_rollout.json",
        "minor_commission/artifacts/minor_fod_seed2/latest_rollout.svg",
        "minor_commission/artifacts/minor_spotlight_seed3/latest_rollout.json",
        "minor_commission/artifacts/minor_spotlight_seed3/latest_rollout.svg",
        "minor_commission/artifacts/minor_eval/scenario_eval.csv",
        "minor_commission/artifacts/minor_eval/scenario_eval.json",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit separate SoTA Grand and Minor submission bundles.")
    parser.add_argument("--bundle-root", type=Path, default=DEFAULT_BUNDLE_ROOT)
    parser.add_argument("--waymo-data-root", type=Path, default=DEFAULT_WAYMO_DATA_ROOT)
    parser.add_argument("--waymo-leaderboard-results", type=Path, default=DEFAULT_WAYMO_LEADERBOARD_RESULTS)
    args = parser.parse_args()

    report = audit_bundles(
        args.bundle_root,
        waymo_data_root=args.waymo_data_root,
        waymo_leaderboard_results=args.waymo_leaderboard_results,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def audit_bundles(
    bundle_root: Path,
    *,
    waymo_data_root: Path = DEFAULT_WAYMO_DATA_ROOT,
    waymo_leaderboard_results: Path = DEFAULT_WAYMO_LEADERBOARD_RESULTS,
) -> dict[str, Any]:
    manifest_path = bundle_root / "submission_bundles_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.is_file() else {}
    tracks = {
        track: _track_report(bundle_root, track, manifest.get(track, {}))
        for track in ("grand_commission", "minor_commission")
    }
    return {
        "bundle_root": str(bundle_root),
        "manifest": _file_report(manifest_path),
        "submission_tracks": {
            "sota_grand": {
                "type": "sota_commission",
                "archive": str(bundle_root / "grand_commission.tar.gz"),
                "valid": tracks["grand_commission"]["valid"],
            },
            "sota_minor": {
                "type": "sota_commission",
                "archive": str(bundle_root / "minor_commission.tar.gz"),
                "valid": tracks["minor_commission"]["valid"],
            },
            "waymo_wod_e2e": _waymo_track_report(waymo_data_root, waymo_leaderboard_results),
        },
        "tracks": tracks,
        "valid": all(track["valid"] for track in tracks.values()),
    }


def _track_report(bundle_root: Path, track: str, manifest_row: dict[str, Any]) -> dict[str, Any]:
    archive_path = bundle_root / f"{track}.tar.gz"
    archive = _archive_report(archive_path)
    required = REQUIRED_MEMBERS[track]
    members = set(archive.get("members", []))
    missing = sorted(required - members)
    sha_matches = bool(manifest_row) and manifest_row.get("sha256") == archive.get("sha256")
    return {
        "archive": archive,
        "required_count": len(required),
        "missing_required": missing,
        "manifest_sha_matches": sha_matches,
        "valid": archive.get("present", False) and not missing and sha_matches,
    }


def _archive_report(path: Path) -> dict[str, Any]:
    report = _file_report(path)
    if not path.is_file():
        return {**report, "sha256": None, "members": []}
    with tarfile.open(path, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    return {
        **report,
        "sha256": _sha256(path),
        "members": members,
        "file_count": len(members),
    }


def _file_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _waymo_track_report(data_root: Path, leaderboard_results: Path) -> dict[str, Any]:
    test_shards = sorted((data_root / "test").glob("test*.tfrecord-*")) if (data_root / "test").is_dir() else []
    leaderboard = _leaderboard_report(leaderboard_results)
    blockers = []
    if not test_shards:
        blockers.append(f"missing WOD-E2E test split at {data_root / 'test'}")
    if not leaderboard["valid"]:
        blockers.append(f"missing confirmed hidden-test result log at {leaderboard_results}")
    return {
        "type": "waymo_leaderboard",
        "separate_from_sota": True,
        "test_split_present": bool(test_shards),
        "test_shard_count": len(test_shards),
        "leaderboard_results": leaderboard,
        "ready_for_claim": bool(test_shards) and leaderboard["valid"],
        "blockers": blockers,
    }


def _leaderboard_report(path: Path) -> dict[str, Any]:
    report = _file_report(path)
    report.update({"valid": False, "confirmed_hidden_test_count": 0})
    if not path.is_file():
        return report
    payload = _load_json(path)
    results = payload.get("results", [])
    confirmed = [
        item
        for item in results
        if isinstance(item, dict)
        and item.get("confirmed_hidden_test") is True
        and str(item.get("submission_sha256", "")).strip()
    ]
    report.update(
        {
            "schema": payload.get("schema"),
            "confirmed_hidden_test_count": len(confirmed),
            "valid": payload.get("schema") == "wod_e2e_leaderboard_results_v1" and bool(confirmed),
        }
    )
    return report


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
