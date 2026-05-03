#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEFAULT_DATA_ROOT = ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEFAULT_FRAME_LIST = ROOT / "data" / "waymo" / "e2e" / "submission_frames" / "test_frames.json"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_e2e_submission_matrix"
DEFAULT_READINESS_OUTPUT = ROOT / "artifacts" / "wod_e2e_leaderboard_attack_readiness.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the two-gate WOD-E2E leaderboard attack readiness and submission-matrix flow."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frame-list", type=Path, default=DEFAULT_FRAME_LIST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--readiness-output", type=Path, default=DEFAULT_READINESS_OUTPUT)
    parser.add_argument("--account-name", required=True)
    parser.add_argument("--authors", required=True)
    parser.add_argument("--affiliation", default="")
    parser.add_argument("--method-prefix", default="minimal_shot_av_two_gate")
    parser.add_argument("--method-link", default="")
    parser.add_argument("--max-test-shards", type=int)
    parser.add_argument("--max-test-records", type=int)
    parser.add_argument("--strict-spotlight-candidates", type=Path)
    parser.add_argument("--calibrated-verifier-candidates", type=Path)
    parser.add_argument("--allow-missing-frame-list", action="store_true")
    args = parser.parse_args()

    initial = _readiness(args)
    _write_json(args.readiness_output, initial)
    if not _can_build_matrix(initial, allow_missing_frame_list=args.allow_missing_frame_list):
        print(json.dumps(_blocked_payload(args.readiness_output, initial), indent=2, sort_keys=True))
        return 1

    _run_matrix(args)
    final = _readiness(args)
    _write_json(args.readiness_output, final)
    print(
        json.dumps(
            {
                "matrix_manifest": str(args.output_dir / "manifest.json"),
                "readiness_report": str(args.readiness_output),
                "ready_for_leaderboard_upload": final["minimal_shot_av_readiness"]["ready_for_leaderboard_upload"],
                "leaderboard_truth_available": final["minimal_shot_av_readiness"]["leaderboard_truth_available"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if final["minimal_shot_av_readiness"]["ready_for_leaderboard_upload"] else 1


def _readiness(args: argparse.Namespace) -> dict[str, Any]:
    from scripts.audit_wod_e2e_readiness import readiness_report

    return readiness_report(
        data_root=args.data_root,
        frame_list=args.frame_list,
        model=ROOT / "artifacts" / "wod_e2e_submission" / "wod_ridge_trajectory_model.json",
        candidates=ROOT / "artifacts" / "wod_e2e_submission" / "wod_test_candidates.jsonl",
        submission=ROOT / "artifacts" / "wod_e2e_submission" / "wod_e2e_submission.tar.gz",
        submission_matrix=args.output_dir,
    )


def _can_build_matrix(report: dict[str, Any], *, allow_missing_frame_list: bool) -> bool:
    if not report["splits"]["test"]["present"]:
        return False
    if not allow_missing_frame_list and not report["frame_list"]["present"]:
        return False
    return True


def _run_matrix(args: argparse.Namespace) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "prepare_wod_e2e_submission_matrix.py"),
        "--test-dir",
        str(args.data_root / "test"),
        "--output-dir",
        str(args.output_dir),
        "--account-name",
        args.account_name,
        "--authors",
        args.authors,
        "--affiliation",
        args.affiliation,
        "--method-prefix",
        args.method_prefix,
        "--method-link",
        args.method_link,
        *_optional_path("--frame-list", args.frame_list if args.frame_list.is_file() else None),
        *_optional_int("--max-test-shards", args.max_test_shards),
        *_optional_int("--max-test-records", args.max_test_records),
        *_optional_path("--strict-spotlight-candidates", args.strict_spotlight_candidates),
        *_optional_path("--calibrated-verifier-candidates", args.calibrated_verifier_candidates),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / ".wod-protos"), str(ROOT / "src"), env.get("PYTHONPATH", "")])
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _blocked_payload(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "blocked": True,
        "readiness_report": str(path),
        "blockers": report["blockers"],
        "instruction": (
            "Restore WOD-E2E test shards and the official frame list before generating "
            "leaderboard submission tarballs."
        ),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_int(flag: str, value: int | None) -> list[str]:
    return [] if value is None else [flag, str(value)]


def _optional_path(flag: str, value: Path | None) -> list[str]:
    return [] if value is None else [flag, str(value)]


if __name__ == "__main__":
    raise SystemExit(main())
