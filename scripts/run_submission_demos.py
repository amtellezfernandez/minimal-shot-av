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
from scripts.audit_production_av_readiness import (
    DEFAULT_CLOSED_LOOP_EVIDENCE,
    DEFAULT_INTEGRATION_MANIFEST,
    DEFAULT_LEADERBOARD_RESULTS,
    DEFAULT_SAFETY_CASE,
    DEFAULT_TRAJECTORY_REPORT,
    production_readiness_report,
)
from scripts.audit_final_submission_readiness import final_readiness_report
from scripts.audit_alpasignal_bridge import build_report as build_alpasignal_bridge_report
from scripts.audit_minor_runtime_constraints import build_report as build_minor_runtime_report
from scripts.build_minor_visual_gallery import build_gallery as build_minor_visual_gallery


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
        "artifacts/novel_object_stress_audit.json",
        "artifacts/sota_judging_criteria_audit.json",
        "artifacts/rlvr_curriculum_v1.json",
        "artifacts/top_lab_readiness_audit.json",
        "artifacts/production_av_readiness_audit.json",
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
        _run_minor_ood_evaluation(args.artifacts_root / "minor_ood_eval")
        _write_minor_alpasignal_bridge_audit(args.artifacts_root / "minor_alpasignal_bridge")
        _write_minor_runtime_audit(args.artifacts_root / "minor_runtime")
        build_minor_visual_gallery(args.artifacts_root, args.artifacts_root / "minor_visual_gallery")
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
        claim=(
            "Randomized long-tail simulation environment with WOD-style cluster coverage, "
            "compositional OOD stress cases, and reproducible closed-loop evaluation."
        ),
        demo_names=[
            name for name, _ in MINOR_RUNS
        ] + ["minor_eval", "minor_ood_eval", "minor_alpasignal_bridge", "minor_runtime", "minor_visual_gallery"],
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
        _write_final_readiness_audit(args.artifacts_root)
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


def _run_minor_ood_evaluation(output_dir: Path) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_scenarios.py"),
        "--policy",
        "spotlight-reflex",
        "--suite",
        "all",
        "--seed-start",
        "1",
        "--seed-end",
        "20",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


def _write_minor_alpasignal_bridge_audit(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "alpasignal_bridge_audit.json"
    report = build_alpasignal_bridge_report()
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise RuntimeError(f"AlpaSignal bridge audit failed: {output}")
    return output


def _write_minor_runtime_audit(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "minor_runtime_constraints.json"
    report = build_minor_runtime_report(seed_start=1, seed_end=3, target_step_ms=50.0)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise RuntimeError(f"Minor runtime constraints audit failed: {output}")
    return output


def _write_judging_audit(artifacts_root: Path) -> Path:
    minimal_shot_output = ROOT / "artifacts" / "minimal_shot_claim_audit.json"
    minimal_shot_report = build_minimal_shot_report(sim_eval=artifacts_root / "minor_eval" / "scenario_eval.json")
    minimal_shot_output.parent.mkdir(parents=True, exist_ok=True)
    minimal_shot_output.write_text(json.dumps(minimal_shot_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    output = ROOT / "artifacts" / "sota_judging_criteria_audit.json"
    report = build_report(sim_eval=artifacts_root / "minor_eval" / "scenario_eval.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    production_output = ROOT / "artifacts" / "production_av_readiness_audit.json"
    production_report = production_readiness_report(
        trajectory_report=DEFAULT_TRAJECTORY_REPORT,
        leaderboard_results=DEFAULT_LEADERBOARD_RESULTS,
        closed_loop_evidence=DEFAULT_CLOSED_LOOP_EVIDENCE,
        safety_case=DEFAULT_SAFETY_CASE,
        integration_manifest=DEFAULT_INTEGRATION_MANIFEST,
    )
    production_output.write_text(json.dumps(production_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _write_final_readiness_audit(artifacts_root: Path) -> Path:
    output = ROOT / "artifacts" / "final_submission_readiness_audit.json"
    report = final_readiness_report(
        bundle_root=artifacts_root,
        minimal_shot_audit=ROOT / "artifacts" / "minimal_shot_claim_audit.json",
        judging_audit=ROOT / "artifacts" / "sota_judging_criteria_audit.json",
        production_audit=ROOT / "artifacts" / "production_av_readiness_audit.json",
    )
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
        dirs.append(artifacts_root / "minor_ood_eval")
        dirs.append(artifacts_root / "minor_alpasignal_bridge")
        dirs.append(artifacts_root / "minor_runtime")
        dirs.append(artifacts_root / "minor_visual_gallery")
    return dirs


def _archive_report(path: Path) -> dict[str, Any]:
    with tarfile.open(path, "r:gz") as archive:
        members = [member.name for member in archive.getmembers() if member.isfile()]
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
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
            f"Git commit: `{grand.get('git_commit', 'unknown')}`",
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
            f"Git commit: `{minor.get('git_commit', 'unknown')}`",
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


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_dirty() -> bool:
    try:
        status = subprocess.check_output(
            ["git", "status", "--short"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return True
    generated_prefixes = (
        "artifacts/sota_submission_bundles/",
        "artifacts/minimal_shot_claim_audit.json",
        "artifacts/novel_object_stress_audit.json",
        "artifacts/sota_judging_criteria_audit.json",
        "artifacts/production_av_readiness_audit.json",
        "artifacts/final_submission_readiness_audit.json",
    )
    for line in status.splitlines():
        path = line[3:] if len(line) > 3 else ""
        if path and not path.startswith(generated_prefixes):
            return True
    return False


if __name__ == "__main__":
    main()
