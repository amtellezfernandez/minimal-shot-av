#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX = ROOT / "artifacts" / "wod_e2e_submission_matrix" / "manifest.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "wod_e2e_leaderboard_results.json"
SOTA_SNAPSHOT_RFS = 8.0461


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a Waymo WOD-E2E hidden-test leaderboard score against a pre-registered submission hash."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--submission-sha256", required=True)
    parser.add_argument("--rfs", type=float, required=True, help="Hidden-test rater feedback score from Waymo.")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--leaderboard-url", default="https://waymo.com/open/challenges/2025/e2e-driving/")
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--confirmed-hidden-test",
        action="store_true",
        help="Required: confirms this score came from Waymo hidden-test leaderboard feedback.",
    )
    args = parser.parse_args()
    if not args.confirmed_hidden_test:
        raise ValueError("--confirmed-hidden-test is required for non-dummy leaderboard result logging")

    record = leaderboard_record(
        manifest_path=args.manifest,
        variant=args.variant,
        submission_sha256=args.submission_sha256,
        rfs=args.rfs,
        rank=args.rank,
        leaderboard_url=args.leaderboard_url,
        notes=args.notes,
        confirmed_hidden_test=args.confirmed_hidden_test,
    )
    payload = append_record(args.output, record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "record_count": len(payload["results"])}, indent=2))
    return 0


def leaderboard_record(
    *,
    manifest_path: Path,
    variant: str,
    submission_sha256: str,
    rfs: float,
    rank: int | None = None,
    leaderboard_url: str = "https://waymo.com/open/challenges/2025/e2e-driving/",
    notes: str = "",
    confirmed_hidden_test: bool = False,
) -> dict[str, Any]:
    if not confirmed_hidden_test:
        raise ValueError("confirmed_hidden_test must be true for leaderboard result logging")
    if not 0.0 <= float(rfs) <= 10.0:
        raise ValueError("rfs must be between 0 and 10")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    matched = _manifest_submission(manifest, variant=variant)
    expected_sha = str(matched.get("submission_sha256", ""))
    if submission_sha256 != expected_sha:
        raise ValueError(
            f"submission SHA mismatch for {variant}: got {submission_sha256}, manifest has {expected_sha}"
        )
    return {
        "recorded_at_utc": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "variant": variant,
        "submission": str(matched.get("submission", "")),
        "submission_sha256": submission_sha256,
        "candidate_name": matched.get("candidate_name"),
        "branch": matched.get("branch"),
        "validation_tuned": bool(matched.get("validation_tuned", False)),
        "minimal_shot_claim_allowed": bool(matched.get("minimal_shot_claim_allowed", False)),
        "hidden_test_rfs": float(rfs),
        "sota_snapshot_rfs": SOTA_SNAPSHOT_RFS,
        "beats_sota_snapshot": float(rfs) > SOTA_SNAPSHOT_RFS,
        "confirmed_hidden_test": True,
        "rank": rank,
        "leaderboard_url": leaderboard_url,
        "notes": notes,
    }


def append_record(output: Path, record: dict[str, Any]) -> dict[str, Any]:
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        payload = {"schema": "wod_e2e_leaderboard_results_v1", "results": []}
    if payload.get("schema") != "wod_e2e_leaderboard_results_v1":
        raise ValueError(f"unsupported leaderboard result schema: {payload.get('schema')!r}")
    results = payload.get("results")
    if not isinstance(results, list):
        raise ValueError("leaderboard result payload must contain a results list")
    duplicate = [
        item
        for item in results
        if isinstance(item, dict)
        and item.get("variant") == record["variant"]
        and item.get("submission_sha256") == record["submission_sha256"]
    ]
    if duplicate:
        raise ValueError(f"leaderboard result already recorded for {record['variant']}")
    results.append(record)
    payload["results"] = sorted(results, key=lambda item: str(item.get("recorded_at_utc", "")))
    return payload


def _manifest_submission(manifest: dict[str, Any], *, variant: str) -> dict[str, Any]:
    submissions = manifest.get("submissions", [])
    matches = [item for item in submissions if isinstance(item, dict) and item.get("variant") == variant]
    if not matches:
        raise ValueError(f"variant {variant!r} not found in submission matrix manifest")
    if len(matches) > 1:
        raise ValueError(f"variant {variant!r} appears multiple times in submission matrix manifest")
    return matches[0]


if __name__ == "__main__":
    raise SystemExit(main())
