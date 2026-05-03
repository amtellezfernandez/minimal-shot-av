#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from dataclasses import dataclass


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0"


@dataclass(frozen=True)
class MatrixVariant:
    name: str
    candidate_name: str
    description: str
    branch: str = "strict_minimal_shot"
    validation_tuned: bool = False
    minimal_shot_claim_allowed: bool = True
    candidates_path: Path | None = None


KINEMATIC_VARIANTS = (
    MatrixVariant(
        name="kinematic_constant_velocity",
        candidate_name="constant_velocity",
        description="Blind kinematic constant-velocity WOD-E2E submission.",
    ),
    MatrixVariant(
        name="kinematic_constant_acceleration",
        candidate_name="constant_acceleration",
        description="Blind kinematic constant-acceleration WOD-E2E submission.",
    ),
    MatrixVariant(
        name="kinematic_hold_position",
        candidate_name="hold_position",
        description="Blind kinematic hold-position WOD-E2E submission.",
    ),
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate pre-registered WOD-E2E submission tarballs for blind leaderboard checks."
    )
    parser.add_argument("--test-dir", type=Path, default=DEFAULT_DATA_ROOT / "test")
    parser.add_argument("--frame-list", type=Path, help="Optional official challenge frame-list JSON.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "wod_e2e_submission_matrix")
    parser.add_argument("--max-test-shards", type=int)
    parser.add_argument("--max-test-records", type=int)
    parser.add_argument("--num-submission-shards", type=int, default=8)
    parser.add_argument("--account-name", required=True)
    parser.add_argument("--authors", required=True)
    parser.add_argument("--affiliation", default="")
    parser.add_argument("--method-prefix", default="minimal_shot_av_blind")
    parser.add_argument("--method-link", default="")
    parser.add_argument("--uses-public-model-pretraining", action="store_true")
    parser.add_argument("--public-model-names", default="")
    parser.add_argument("--num-model-parameters", default="under 1M")
    parser.add_argument(
        "--strict-spotlight-candidates",
        type=Path,
        help="Optional WOD candidate JSONL from the strict episode-free Spotlight Reflex branch.",
    )
    parser.add_argument(
        "--calibrated-verifier-candidates",
        type=Path,
        help="Optional WOD candidate JSONL scored by the small validation-preference verifier branch.",
    )
    args = parser.parse_args()

    _require_dir(args.test_dir, "WOD-E2E test split")
    if args.frame_list is not None:
        _require_file(args.frame_list, "challenge frame-list JSON")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    kinematic_candidates = _generate_kinematic_candidates(args)
    variants = _matrix_variants(args, kinematic_candidates)
    submissions = [
        _run_variant(args, variant=variant)
        for variant in variants
    ]

    manifest = {
        "schema": "wod_e2e_submission_matrix_v2",
        "pre_registered": True,
        "selection_policy": (
            "Generate every configured variant before leaderboard upload; "
            "do not choose by validation after seeing results."
        ),
        "sota_target": {
            "source": "docs/leaderboard.md",
            "metric": "hidden_test_rfs",
            "threshold_rfs": 8.0461,
            "higher_is_better": True,
        },
        "two_gate_policy": {
            "strict_minimal_shot": (
                "eligible only when validation_tuned is false and no WOD preference verifier is used"
            ),
            "preference_calibrated_leaderboard": (
                "allowed to use the declared small verifier but not eligible for strict minimal-shot claims"
            ),
        },
        "test_dir": str(args.test_dir),
        "frame_list": str(args.frame_list) if args.frame_list else None,
        "submissions": submissions,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upload_notes_path = args.output_dir / "UPLOAD_NOTES.md"
    upload_notes_path.write_text(_upload_notes(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "submissions": submissions}, indent=2, sort_keys=True))
    return 0


def _generate_kinematic_candidates(args: argparse.Namespace) -> Path:
    candidates = args.output_dir / "kinematic_candidates.jsonl"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "generate_wod_kinematic_candidates.py"),
            "--data-dir",
            str(args.test_dir),
            "--include-unlabeled",
            "--output",
            str(candidates),
            *_optional_path("--frame-list", args.frame_list),
            *_optional_int("--max-shards", args.max_test_shards),
            *_optional_int("--max-records", args.max_test_records),
        ]
    )
    return candidates


def _matrix_variants(args: argparse.Namespace, kinematic_candidates: Path) -> tuple[MatrixVariant, ...]:
    variants = [
        MatrixVariant(
            name=variant.name,
            candidate_name=variant.candidate_name,
            description=variant.description,
            candidates_path=kinematic_candidates,
        )
        for variant in KINEMATIC_VARIANTS
    ]
    if args.strict_spotlight_candidates is not None:
        _require_file(args.strict_spotlight_candidates, "strict Spotlight Reflex candidate JSONL")
        variants.append(
            MatrixVariant(
                name="strict_spotlight_reflex",
                candidate_name="ranker_score",
                description=(
                    "Strict episode-free Spotlight Reflex branch; no WOD preference verifier, "
                    "no nearest-neighbor episode memory."
                ),
                branch="strict_minimal_shot",
                validation_tuned=False,
                minimal_shot_claim_allowed=True,
                candidates_path=args.strict_spotlight_candidates,
            )
        )
    if args.calibrated_verifier_candidates is not None:
        _require_file(args.calibrated_verifier_candidates, "calibrated verifier candidate JSONL")
        variants.append(
            MatrixVariant(
                name="small_verifier_calibrated",
                candidate_name="ranker_score",
                description=(
                    "Preference-calibrated leaderboard branch using a declared small verifier; "
                    "not strict zero-shot or minimal-shot claim evidence."
                ),
                branch="preference_calibrated_leaderboard",
                validation_tuned=True,
                minimal_shot_claim_allowed=False,
                candidates_path=args.calibrated_verifier_candidates,
            )
        )
    return tuple(variants)


def _run_variant(
    args: argparse.Namespace,
    *,
    variant: MatrixVariant,
) -> dict[str, object]:
    if variant.candidates_path is None:
        raise ValueError(f"variant {variant.name} has no candidate path")
    submission = args.output_dir / f"{variant.name}.tar.gz"
    _write_and_validate_submission(
        args,
        variant=variant,
        candidates=variant.candidates_path,
        submission=submission,
    )
    return _manifest_row(
        variant,
        variant.candidates_path,
        submission,
    )


def _write_and_validate_submission(
    args: argparse.Namespace,
    *,
    variant: MatrixVariant,
    candidates: Path,
    submission: Path,
) -> None:
    method_name = f"{args.method_prefix}_{variant.name}"
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "write_wod_e2e_submission.py"),
            "--candidates",
            str(candidates),
            "--output",
            str(submission),
            "--num-shards",
            str(args.num_submission_shards),
            "--account-name",
            args.account_name,
            "--unique-method-name",
            method_name,
            "--authors",
            args.authors,
            "--affiliation",
            args.affiliation,
            "--description",
            variant.description,
            "--method-link",
            args.method_link,
            "--public-model-names",
            args.public_model_names,
            "--num-model-parameters",
            args.num_model_parameters,
            *_optional_path("--frame-list", args.frame_list),
            "--candidate-name",
            variant.candidate_name,
            *_optional_flag("--uses-public-model-pretraining", args.uses_public_model_pretraining),
        ]
    )
    _run(
        [
            sys.executable,
            str(ROOT / "scripts" / "validate_wod_e2e_submission.py"),
            "--submission",
            str(submission),
            *_optional_path("--frame-list", args.frame_list),
        ]
    )


def _manifest_row(
    variant: MatrixVariant,
    candidates: Path,
    submission: Path,
) -> dict[str, object]:
    return {
        "variant": variant.name,
        "branch": variant.branch,
        "candidates": str(candidates),
        "submission": str(submission),
        "submission_size_bytes": submission.stat().st_size,
        "submission_sha256": _sha256(submission),
        "tar_members": _tar_members(submission),
        "validation_tuned": variant.validation_tuned,
        "minimal_shot_claim_allowed": variant.minimal_shot_claim_allowed,
        "candidate_name": variant.candidate_name,
        "claim_boundary": (
            "strict_minimal_shot_evidence"
            if variant.minimal_shot_claim_allowed
            else "preference_calibrated_leaderboard_only"
        ),
    }


def _upload_notes(manifest: dict[str, object]) -> str:
    lines = [
        "# WOD-E2E Blind Submission Matrix",
        "",
        "Upload every `.tar.gz` listed below before using leaderboard feedback to choose a direction.",
        "Challenge page: https://waymo.com/open/challenges/2025/e2e-driving/",
        "",
        "| variant | branch | minimal_shot_claim_allowed | validation_tuned | sha256 | path |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in manifest["submissions"]:
        item = dict(row)
        lines.append(
            "| {variant} | {branch} | {minimal_shot_claim_allowed} | {validation_tuned} | "
            "`{submission_sha256}` | `{submission}` |".format(**item)
        )
    lines.extend(
        [
            "",
            "Record the hidden-test leaderboard score next to the exact SHA-256 after upload.",
            "Do not rename, regenerate, or filter this matrix after seeing any leaderboard score.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tar_members(path: Path) -> list[str]:
    with tarfile.open(path, "r:gz") as archive:
        return [member.name for member in archive.getmembers() if member.isfile()]


def _run(command: list[str]) -> None:
    env = os.environ.copy()
    source_paths = [str(ROOT / ".wod-protos"), str(ROOT / "src")]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        source_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(source_paths)
    subprocess.run(command, check=True, cwd=ROOT, env=env)


def _require_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _optional_int(flag: str, value: int | None) -> list[str]:
    return [] if value is None else [flag, str(value)]


def _optional_path(flag: str, value: Path | None) -> list[str]:
    return [] if value is None else [flag, str(value)]


def _optional_flag(flag: str, enabled: bool) -> list[str]:
    return [flag] if enabled else []


if __name__ == "__main__":
    raise SystemExit(main())
