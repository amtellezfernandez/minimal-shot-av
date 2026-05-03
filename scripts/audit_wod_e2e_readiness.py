#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEFAULT_FRAME_LIST = ROOT / "data" / "waymo" / "e2e" / "submission_frames" / "test_frames.json"
DEFAULT_SUBMISSION_DIR = ROOT / "artifacts" / "wod_e2e_submission"
DEFAULT_SUBMISSION_MATRIX = ROOT / "artifacts" / "wod_e2e_submission_matrix"
DEFAULT_LEADERBOARD_RESULTS = ROOT / "artifacts" / "wod_e2e_leaderboard_results.json"
DEFAULT_CLOSED_LOOP_DIR = ROOT / "artifacts"
DEFAULT_WORLD_SWEEP_DIR = ROOT / "artifacts" / "world_sweeps"
DEFAULT_SOLUTION_RESET = ROOT / "docs" / "solution-reset.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit local WOD-E2E data and submission artifact readiness.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--frame-list", type=Path, default=DEFAULT_FRAME_LIST)
    parser.add_argument("--model", type=Path, default=DEFAULT_SUBMISSION_DIR / "wod_ridge_trajectory_model.json")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_SUBMISSION_DIR / "wod_test_candidates.jsonl")
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION_DIR / "wod_e2e_submission.tar.gz")
    parser.add_argument("--submission-matrix", type=Path, default=DEFAULT_SUBMISSION_MATRIX)
    parser.add_argument("--leaderboard-results", type=Path, default=DEFAULT_LEADERBOARD_RESULTS)
    parser.add_argument("--closed-loop-dir", type=Path, default=DEFAULT_CLOSED_LOOP_DIR)
    parser.add_argument("--world-sweep-dir", type=Path, default=DEFAULT_WORLD_SWEEP_DIR)
    parser.add_argument("--solution-reset", type=Path, default=DEFAULT_SOLUTION_RESET)
    args = parser.parse_args()

    report = readiness_report(
        data_root=args.data_root,
        frame_list=args.frame_list,
        model=args.model,
        candidates=args.candidates,
        submission=args.submission,
        submission_matrix=args.submission_matrix,
        leaderboard_results=args.leaderboard_results,
        closed_loop_dir=args.closed_loop_dir,
        world_sweep_dir=args.world_sweep_dir,
        solution_reset=args.solution_reset,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["minimal_shot_av_readiness"]["ready_for_leaderboard_upload"] else 1


def readiness_report(
    *,
    data_root: Path,
    frame_list: Path,
    model: Path,
    candidates: Path,
    submission: Path,
    submission_matrix: Path = DEFAULT_SUBMISSION_MATRIX,
    leaderboard_results: Path = DEFAULT_LEADERBOARD_RESULTS,
    closed_loop_dir: Path = DEFAULT_CLOSED_LOOP_DIR,
    world_sweep_dir: Path = DEFAULT_WORLD_SWEEP_DIR,
    solution_reset: Path = DEFAULT_SOLUTION_RESET,
) -> dict[str, Any]:
    splits = {
        name: _split_report(data_root, split=name)
        for name in ("train", "val", "test")
    }
    frame_list_report = _file_report(frame_list)
    model_report = _file_report(model)
    candidates_report = _file_report(candidates)
    submission_report = _file_report(submission)
    matrix_report = _submission_matrix_report(submission_matrix)
    leaderboard_report = _leaderboard_results_report(leaderboard_results)
    closed_loop_report = _closed_loop_report(closed_loop_dir)
    world_report = _world_sweep_report(world_sweep_dir)
    safety_report = _safety_report(solution_reset)
    blockers: list[str] = []
    if not splits["train"]["present"]:
        blockers.append(f"missing WOD-E2E train split at {splits['train']['path']}")
    elif not splits["train"]["complete"]:
        blockers.append(
            "incomplete WOD-E2E train split: "
            f"{splits['train']['shard_count']} of {splits['train']['expected_shard_count']} shards"
        )
    if not splits["test"]["present"]:
        blockers.append(f"missing WOD-E2E test split at {splits['test']['path']}")
    elif not splits["test"]["complete"]:
        blockers.append(
            "incomplete WOD-E2E test split: "
            f"{splits['test']['shard_count']} of {splits['test']['expected_shard_count']} shards"
        )
    if not frame_list_report["present"]:
        blockers.append(f"missing optional challenge frame-list JSON at {frame_list}")
    if not model_report["present"]:
        blockers.append(f"missing final trajectory model at {model}")
    if not candidates_report["present"]:
        blockers.append(f"missing generated test candidates at {candidates}")
    if not submission_report["present"]:
        blockers.append(f"missing packaged submission tarball at {submission}")
    if not matrix_report["present"]:
        blockers.append(f"missing blind submission matrix manifest under {submission_matrix}")
    if not leaderboard_report["present"]:
        blockers.append(f"missing hidden-test leaderboard result log at {leaderboard_results}")
    if not closed_loop_report["present"]:
        blockers.append(f"missing closed-loop scenario_eval.json evidence under {closed_loop_dir}")
    if not world_report["present"]:
        blockers.append(f"missing world-model sweep evidence under {world_sweep_dir}")
    if not safety_report["present"]:
        blockers.append(f"missing solution-reset/safety honesty note at {solution_reset}")

    ready_for_generation = (
        splits["test"]["complete"]
        and model_report["present"]
    )
    ready_for_packaging = ready_for_generation and candidates_report["present"]
    ready_for_upload = ready_for_packaging and submission_report["present"]
    ready_for_blind_matrix = splits["test"]["complete"] and matrix_report["valid"]
    ready_for_leaderboard_upload = splits["test"]["complete"] and matrix_report["valid"]
    leaderboard_truth_available = leaderboard_report["valid"]
    minimal_shot_av_gates = {
        "leaderboard_truth": leaderboard_truth_available,
        "blind_submission_protocol": matrix_report["valid"],
        "closed_loop_evidence": closed_loop_report["present"],
        "world_model_evidence": world_report["present"],
        "safety_bias_evidence": safety_report["present"],
    }
    return {
        "data_root": str(data_root),
        "splits": splits,
        "frame_list": frame_list_report,
        "artifacts": {
            "model": model_report,
            "candidates": candidates_report,
            "submission": submission_report,
            "submission_matrix": matrix_report,
            "leaderboard_results": leaderboard_report,
            "closed_loop": closed_loop_report,
            "world_model": world_report,
            "safety_bias": safety_report,
        },
        "ready_for_training": splits["train"]["complete"],
        "ready_for_generation": ready_for_generation,
        "ready_for_packaging": ready_for_packaging,
        "ready_for_upload": ready_for_upload,
        "minimal_shot_av_readiness": {
            "ready_for_blind_matrix": ready_for_blind_matrix,
            "ready_for_leaderboard_upload": ready_for_leaderboard_upload,
            "leaderboard_truth_available": leaderboard_truth_available,
            "all_evidence_gates_pass": all(minimal_shot_av_gates.values()),
            "gates": minimal_shot_av_gates,
        },
        "blockers": blockers,
    }


def _split_report(data_root: Path, *, split: str) -> dict[str, Any]:
    split_dir = data_root / split
    split_dir_shards = sorted(split_dir.glob(f"{split}*.tfrecord-*")) if split_dir.is_dir() else []
    root_shards = sorted(data_root.glob(f"{split}*.tfrecord-*")) if data_root.is_dir() else []
    shards = split_dir_shards or root_shards
    layout = "split_dir" if split_dir_shards else "root_flat" if root_shards else None
    expected, missing_indices = _shard_index_report(shards)
    complete = bool(shards) and (expected is None or not missing_indices)
    return {
        "path": str(split_dir if layout == "split_dir" else data_root / f"{split}*.tfrecord-*"),
        "layout": layout,
        "present": bool(shards),
        "complete": complete,
        "shard_count": len(shards),
        "expected_shard_count": expected,
        "missing_shard_count": len(missing_indices) if expected is not None else None,
        "missing_shard_indices_sample": missing_indices[:20] if expected is not None else None,
        "first_shard": str(shards[0]) if shards else None,
        "last_shard": str(shards[-1]) if shards else None,
    }


def _shard_index_report(shards: list[Path]) -> tuple[int | None, list[int]]:
    totals: set[int] = set()
    indices: set[int] = set()
    for shard in shards:
        name = shard.name
        if "-of-" not in name:
            continue
        prefix, suffix = name.rsplit("-of-", maxsplit=1)
        if suffix.isdigit():
            totals.add(int(suffix))
        index_text = prefix.rsplit("-", maxsplit=1)[-1]
        if index_text.isdigit():
            indices.add(int(index_text))
    if not totals:
        return None, []
    expected = max(totals)
    missing_indices = [index for index in range(expected) if index not in indices]
    return expected, missing_indices


def _file_report(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
    }


def _leaderboard_results_report(path: Path) -> dict[str, Any]:
    report = _file_report(path)
    report.update({"valid": False, "result_count": 0, "confirmed_hidden_test_count": 0})
    if not path.is_file():
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["error"] = str(exc)
        return report
    results = payload.get("results", [])
    confirmed = [
        item
        for item in results
        if isinstance(item, dict)
        and item.get("confirmed_hidden_test") is True
        and isinstance(item.get("hidden_test_rfs"), (int, float))
        and str(item.get("submission_sha256", "")).strip()
    ]
    report.update(
        {
            "schema": payload.get("schema"),
            "valid": payload.get("schema") == "wod_e2e_leaderboard_results_v1" and bool(confirmed),
            "result_count": len(results) if isinstance(results, list) else 0,
            "confirmed_hidden_test_count": len(confirmed),
            "best_hidden_test_rfs": max(
                (float(item["hidden_test_rfs"]) for item in confirmed),
                default=None,
            ),
            "beats_sota_snapshot": any(float(item["hidden_test_rfs"]) > 8.0461 for item in confirmed),
        }
    )
    return report


def _submission_matrix_report(path: Path) -> dict[str, Any]:
    manifest = path / "manifest.json" if path.is_dir() else path
    report = _file_report(manifest)
    report.update(
        {
            "valid": False,
            "submission_count": 0,
            "validation_tuned_count": 0,
            "minimal_shot_claim_allowed_count": 0,
            "strict_branch_count": 0,
            "calibrated_branch_count": 0,
            "missing_sha256": [],
            "invalid_claim_rows": [],
        }
    )
    if not manifest.is_file():
        return report
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report["error"] = str(exc)
        return report
    submissions = payload.get("submissions", [])
    missing_sha256 = [
        str(item.get("variant", index))
        for index, item in enumerate(submissions)
        if not isinstance(item, dict) or not str(item.get("submission_sha256", "")).strip()
    ]
    validation_tuned_count = sum(
        1 for item in submissions if isinstance(item, dict) and bool(item.get("validation_tuned", False))
    )
    minimal_shot_claim_allowed_count = sum(
        1 for item in submissions if isinstance(item, dict) and bool(item.get("minimal_shot_claim_allowed", False))
    )
    strict_branch_count = sum(
        1 for item in submissions if isinstance(item, dict) and item.get("branch") == "strict_minimal_shot"
    )
    calibrated_branch_count = sum(
        1
        for item in submissions
        if isinstance(item, dict) and item.get("branch") == "preference_calibrated_leaderboard"
    )
    invalid_claim_rows = [
        str(item.get("variant", index))
        for index, item in enumerate(submissions)
        if isinstance(item, dict)
        and bool(item.get("validation_tuned", False))
        and bool(item.get("minimal_shot_claim_allowed", False))
    ]
    report.update(
        {
            "pre_registered": bool(payload.get("pre_registered", False)),
            "schema": payload.get("schema"),
            "valid": (
                bool(payload.get("pre_registered", False))
                and bool(submissions)
                and not missing_sha256
                and not invalid_claim_rows
                and strict_branch_count > 0
            ),
            "submission_count": len(submissions) if isinstance(submissions, list) else 0,
            "validation_tuned_count": validation_tuned_count,
            "minimal_shot_claim_allowed_count": minimal_shot_claim_allowed_count,
            "strict_branch_count": strict_branch_count,
            "calibrated_branch_count": calibrated_branch_count,
            "missing_sha256": missing_sha256,
            "invalid_claim_rows": invalid_claim_rows,
            "upload_notes": _file_report(
                path / "UPLOAD_NOTES.md" if path.is_dir() else path.with_name("UPLOAD_NOTES.md")
            ),
        }
    )
    return report


def _closed_loop_report(path: Path) -> dict[str, Any]:
    reports = sorted(path.glob("**/scenario_eval.json")) if path.is_dir() else []
    return {
        "path": str(path),
        "present": bool(reports),
        "report_count": len(reports),
        "first_report": str(reports[0]) if reports else None,
    }


def _world_sweep_report(path: Path) -> dict[str, Any]:
    reports = sorted(path.glob("*.json")) if path.is_dir() else []
    world_reports = [report for report in reports if "world" in report.name or "scene_token" in report.name]
    return {
        "path": str(path),
        "present": bool(world_reports),
        "report_count": len(world_reports),
        "first_report": str(world_reports[0]) if world_reports else None,
    }


def _safety_report(path: Path) -> dict[str, Any]:
    report = _file_report(path)
    if not path.is_file():
        report["declares_not_solution"] = False
        return report
    text = path.read_text(encoding="utf-8").lower()
    report["declares_not_solution"] = (
        "not be presented as a completed autonomy solution" in text
        or "not a complete driving solution" in text
        or "not a solution" in text
    )
    return report


if __name__ == "__main__":
    raise SystemExit(main())
