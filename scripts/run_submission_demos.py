from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_sota_judging_criteria import build_report
from scripts.audit_minimal_shot_claim import build_report as build_minimal_shot_report


GRAND_RUNS = (
    (
        "grand_spotlight_demo",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "spotlight", "--seed", "3"),
    ),
    (
        "grand_intersection_stress_seed3",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "intersection", "--seed", "3"),
    ),
    (
        "grand_baseline_spotlight_demo",
        ("--policy", "baseline", "--scenario-cluster", "spotlight", "--seed", "3"),
    ),
)

MINOR_RUNS = (
    (
        "minor_construction_seed1",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "construction", "--seed", "1"),
    ),
    (
        "minor_fod_seed2",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "foreign object debris", "--seed", "2"),
    ),
    (
        "minor_spotlight_seed3",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "spotlight", "--seed", "3"),
    ),
)

TRACK_DOCS = {
    "grand_commission": [
        "README.md",
        "models/DECLARATION.md",
        "docs/sota-grand-submission-form.md",
        "docs/sota-grand-slide-deck.md",
        "docs/sota-grand-slide-script.md",
        "docs/submission-tracks.md",
        "docs/grand-submission.md",
        "docs/judging-criteria-evidence.md",
        "docs/preference-calibrated-evidence.md",
        "docs/fast-slow-realtime-architecture.md",
        "docs/two-page-writeup.md",
        "docs/final-submission-handoff.md",
        "docs/final-submission-checklist.md",
        "notebooks/wod_e2e_analysis.ipynb",
        "docs/video-outline.md",
        "docs/solution-reset.md",
        "docs/benchmark-comparison.md",
        "docs/waymo-data-access.md",
    ],
    "minor_commission": [
        "README.md",
        "docs/sota-minor-submission-form.md",
        "docs/sota-minor-slide-script.md",
        "docs/submission-tracks.md",
        "docs/minor-simulation-submission.md",
        "docs/alpasim-integration.md",
        "docs/compositional-ood-eval.md",
        "docs/final-submission-handoff.md",
        "docs/video-outline.md",
    ],
}

TRACK_EVIDENCE = {
    "grand_commission": [
        "artifacts/wod_fastkin_gate_ridge175_scene020_cv_official.json",
        "artifacts/wod_fastkin_gate_ridge175_scene020_repro_audit.json",
        "artifacts/wod_fastkin_gate_ridge175_scene020_breakthrough_audit.json",
        "artifacts/wod_neural_holdout/neural_ensemble3_sourcegate_speedfine_p0_local.json",
        "artifacts/wod_neural_holdout/neural_top1_pc0_familycal_l2_010_heldout_official.json",
        "artifacts/wod_preference_calibrated_ensemble3_full_official.json",
        "artifacts/wod_e2e_data_restore_plan.json",
        "artifacts/wod_e2e_leaderboard_attack_readiness.json",
        "benchmarks/current/wod_fast_slow_runtime.json",
        "benchmarks/current/wod_monolithic_runtime_reference.json",
        "artifacts/minimal_shot_claim_audit.json",
        "artifacts/sota_judging_criteria_audit.json",
    ],
    "minor_commission": [],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate separate Grand and Minor Commission demo artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/submissions"),
        help="Root directory for generated submission demo artifacts.",
    )
    parser.add_argument("--skip-runs", action="store_true", help="Package existing artifacts without rerunning demos.")
    parser.add_argument("--skip-archives", action="store_true", help="Do not create tar.gz bundles.")
    args = parser.parse_args()

    if not args.skip_runs:
        _run_demos(args.artifacts_root, GRAND_RUNS + MINOR_RUNS)
        _run_minor_evaluation(args.artifacts_root / "minor_eval")
    _write_judging_audit(args.artifacts_root)
    grand_readme = _write_track_readme(
        args.artifacts_root,
        track="grand_commission",
        title="SoTA Grand Commission Submission Bundle",
        claim=(
            "Minimal-shot autonomy architecture prototype with closed-loop Spotlight Reflex demos. "
            "WOD-E2E materials are supporting benchmark infrastructure and validation-CV analysis, "
            "not the primary zero-shot autonomy claim."
        ),
        demo_names=[name for name, _ in GRAND_RUNS],
    )
    minor_readme = _write_track_readme(
        args.artifacts_root,
        track="minor_commission",
        title="SoTA Minor Commission Submission Bundle",
        claim="Randomized WOD-E2E-style long-tail simulation environment with reproducible closed-loop evaluation.",
        demo_names=[name for name, _ in MINOR_RUNS] + ["minor_eval"],
    )
    if not args.skip_archives:
        grand_archive = _write_archive(args.artifacts_root, "grand_commission", grand_readme)
        minor_archive = _write_archive(args.artifacts_root, "minor_commission", minor_readme)
        manifest = {
            "grand_commission": _archive_report(grand_archive),
            "minor_commission": _archive_report(minor_archive),
        }
        manifest_path = args.artifacts_root / "submission_bundles_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        index_path = args.artifacts_root / "SUBMISSION_INDEX.md"
        index_path.write_text(_submission_index(manifest), encoding="utf-8")
        print(f"Wrote {grand_archive}")
        print(f"Wrote {minor_archive}")
        print(f"Wrote {manifest_path}")
        print(f"Wrote {index_path}")


def _run_demos(artifacts_root: Path, runs) -> None:
    for name, demo_args in runs:
        output_dir = artifacts_root / name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_demo.py"),
            *demo_args,
            "--artifacts-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


def _run_minor_evaluation(output_dir: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_scenarios.py"),
        "--policy",
        "spotlight-reflex",
        "--seed-start",
        "1",
        "--seed-end",
        "50",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _write_judging_audit(artifacts_root: Path) -> Path:
    minimal_shot_output = ROOT / "artifacts" / "minimal_shot_claim_audit.json"
    minimal_shot_report = build_minimal_shot_report(sim_eval=artifacts_root / "minor_eval" / "scenario_eval.json")
    minimal_shot_output.parent.mkdir(parents=True, exist_ok=True)
    minimal_shot_output.write_text(json.dumps(minimal_shot_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output = ROOT / "artifacts" / "sota_judging_criteria_audit.json"
    report = build_report(sim_eval=artifacts_root / "minor_eval" / "scenario_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _write_track_readme(
    artifacts_root: Path,
    *,
    track: str,
    title: str,
    claim: str,
    demo_names: list[str],
) -> Path:
    output_dir = artifacts_root / track
    output_dir.mkdir(parents=True, exist_ok=True)
    readme = output_dir / "README.md"
    lines = [
        f"# {title}",
        "",
        claim,
        "",
        "## Include In Submission",
        "",
        "- GitHub repository URL",
        "- 1-5 minute video or slide deck",
        "- Short write-up",
        "- Demo artifacts listed below",
        "",
        "## Demo Artifacts",
        "",
    ]
    lines.extend(f"- `{name}/`" for name in demo_names)
    lines.extend(
        [
            "",
            "## Source Documents",
            "",
            *[f"- `{path}`" for path in TRACK_DOCS[track]],
            "",
            "## Verification",
            "",
            "`UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync python scripts/run_tests.py`",
            "",
        ]
    )
    readme.write_text("\n".join(lines), encoding="utf-8")
    return readme


def _write_archive(artifacts_root: Path, track: str, readme: Path) -> Path:
    output = artifacts_root / f"{track}.tar.gz"
    with tarfile.open(output, "w:gz") as archive:
        archive.add(readme, arcname=f"{track}/README.md")
        for source in TRACK_DOCS[track]:
            path = ROOT / source
            if path.exists():
                archive_name = "repo_README.md" if source == "README.md" else source
                archive.add(path, arcname=f"{track}/{archive_name}")
        for source in TRACK_EVIDENCE[track]:
            path = ROOT / source
            if path.exists():
                archive.add(path, arcname=f"{track}/{source}")
        for artifact_dir in _track_artifact_dirs(artifacts_root, track):
            if artifact_dir.exists():
                archive.add(artifact_dir, arcname=f"{track}/artifacts/{artifact_dir.name}")
    return output


def _track_artifact_dirs(artifacts_root: Path, track: str) -> list[Path]:
    runs = GRAND_RUNS if track == "grand_commission" else MINOR_RUNS
    dirs = [artifacts_root / name for name, _ in runs]
    if track == "minor_commission":
        dirs.append(artifacts_root / "minor_eval")
    return dirs


def _archive_report(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "file_count": len(members),
        "members": members,
    }


def _submission_index(manifest: dict[str, dict[str, Any]]) -> str:
    grand = manifest["grand_commission"]
    minor = manifest["minor_commission"]
    return "\n".join(
        [
            "# Submission Index",
            "",
            "There are three separate tracks. Do not merge the claims.",
            "",
            "## 1. SoTA Grand Commission",
            "",
            "Upload / attach:",
            "",
            f"- `{grand['path']}`",
            "- GitHub repository URL",
            "- Video or slide deck based on `docs/sota-grand-slide-script.md`",
            "",
            f"SHA-256: `{grand['sha256']}`",
            "",
            "Claim: minimal-shot autonomy architecture prototype with closed-loop demos.",
            "",
            "## 2. SoTA Minor Commission",
            "",
            "Upload / attach:",
            "",
            f"- `{minor['path']}`",
            "- GitHub repository URL",
            "- Video or slide deck based on `docs/sota-minor-slide-script.md`",
            "",
            f"SHA-256: `{minor['sha256']}`",
            "",
            "Claim: randomized long-tail simulation environment with closed-loop evaluation.",
            "",
            "## 3. Waymo WOD-E2E Challenge",
            "",
            "Separate benchmark track. Do not present it as a SoTA submission artifact.",
            "",
            "Current status:",
            "",
            "- WOD-E2E test split missing locally",
            "- no confirmed hidden-test leaderboard result recorded",
            "",
            "First restore the missing train/test shards and official frame list:",
            "",
            "`python3 scripts/prepare_wod_e2e_data.py --data-root /home/amdev/waymo "
            "--output artifacts/wod_e2e_data_restore_plan.json`",
            "",
            "When ready, generate pre-registered official Waymo tarballs through the two-gate wrapper:",
            "",
            "`PYTHONPATH=src:.wod-protos .venv-wod/bin/python scripts/run_wod_leaderboard_attack.py "
            "--data-root /home/amdev/waymo "
            "--frame-list data/waymo/e2e/submission_frames/test_frames.json "
            "--output-dir artifacts/wod_e2e_submission_matrix "
            "--account-name <account> --authors 'Alba Maria Tellez Fernandez'`",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
