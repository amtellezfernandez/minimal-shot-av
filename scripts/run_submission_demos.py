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
        "docs/sota-grand-slide-script.md",
        "docs/submission-tracks.md",
        "docs/grand-submission.md",
        "docs/two-page-writeup.md",
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
        "docs/video-outline.md",
    ],
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
    grand_readme = _write_track_readme(
        args.artifacts_root,
        track="grand_commission",
        title="SoTA Grand Commission Submission Bundle",
        claim=(
            "Minimal-shot autonomy architecture prototype with WOD-E2E harness "
            "and closed-loop Spotlight Reflex demos."
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
        "5",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)


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
            "When ready, generate official Waymo tarballs with:",
            "",
            "`PYTHONPATH=src:.wod-protos .venv-wod/bin/python scripts/prepare_wod_e2e_submission_matrix.py "
            "--test-dir waymo_open_dataset_end_to_end_camera_v_1_0_0/test "
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
