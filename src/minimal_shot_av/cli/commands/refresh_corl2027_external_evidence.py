#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh the CoRL 2027 external evidence bundle from AlpaSim runs."
    )
    parser.add_argument("--matrix-dir", type=Path, default=ROOT / "runs" / "alpasim_transfer_matrix_refresh_20260522_smoke10")
    parser.add_argument("--scene-preset", default="front_camera_10scene_smoke")
    parser.add_argument(
        "--models",
        default=(
            "token_dagger_iter2,"
            "token_dagger_iter2_clamped,"
            "token_dagger_iter2_axis_constrained_clamped,"
            "token_dagger_iter2_hybrid_clamped,"
            "token_dagger_srcdecay"
        ),
    )
    parser.add_argument("--mode", choices=("print", "both"), default="both")
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--scene-offset", type=int, default=None)
    parser.add_argument("--allow-existing", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--skip-matrix", action="store_true")
    parser.add_argument("--skip-comparable", action="store_true")
    parser.add_argument("--published-ours-run", type=Path, default=None)
    parser.add_argument("--front-ours-run", type=Path, default=None)
    parser.add_argument("--front-alpamayo-run", type=Path, default=None)
    parser.add_argument("--python-bin", type=Path, default=ROOT / ".venv" / "bin" / "python")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not args.python_bin.is_file():
        raise SystemExit(f"Missing repo python: {args.python_bin}")

    env = os.environ.copy()
    _load_env_file(ROOT / ".env.alpasim_hf", env)

    print("[1/4] AlpaSim readiness")
    _run(
        [
            str(args.python_bin),
            str(ROOT / "scripts" / "check_alpasim_readiness.py"),
            "--skip-scene-artifacts",
        ],
        env=env,
    )

    if not args.skip_matrix:
        print("\n[2/4] Fresh AlpaSim transfer matrix")
        cmd = [
            "bash",
            str(ROOT / "scripts" / "run_alpasim_transfer_matrix_repo.sh"),
            "--mode",
            args.mode,
            "--scene-presets",
            args.scene_preset,
            "--models",
            args.models,
            "--matrix-dir",
            str(args.matrix_dir),
        ]
        if args.allow_existing:
            cmd.append("--allow-existing-matrix-dir")
        if args.rerun_existing:
            cmd.append("--rerun-existing")
        if args.continue_on_error:
            cmd.append("--continue-on-error")
        if args.scene_limit is not None:
            cmd.extend(["--scene-limit", str(args.scene_limit)])
        if args.scene_offset is not None:
            cmd.extend(["--scene-offset", str(args.scene_offset)])
        _run(cmd, env=env)

    print("\n[3/4] Refresh matrix analysis and CoRL audit artifacts")
    _run(
        [
            str(args.python_bin),
            str(ROOT / "scripts" / "analyze_alpasim_transfer_matrix.py"),
            str(args.matrix_dir),
            "--baseline-model",
            "token_dagger_iter2",
            "--scene-preset",
            args.scene_preset,
            "--output-json",
            str(ROOT / "artifacts" / "corl2027" / "alpasim_matrix10_analysis.json"),
            "--output-markdown",
            str(ROOT / "artifacts" / "corl2027" / "alpasim_matrix10_analysis.md"),
        ],
        env=env,
    )
    _run([str(args.python_bin), str(ROOT / "scripts" / "audit_corl_evidence_strength.py")], env=env)
    _run([str(args.python_bin), str(ROOT / "scripts" / "analyze_transfer_predictors.py")], env=env)

    if args.skip_comparable:
        print("\n[4/4] Comparable benchmark reports skipped (--skip-comparable)")
    elif args.published_ours_run and args.front_ours_run and args.front_alpamayo_run:
        print("\n[4/4] Refresh comparable benchmark reports")
        _run(
            [
                str(args.python_bin),
                str(ROOT / "scripts" / "produce_alpasim_comparable_reports.py"),
                "--published-ours-run",
                str(args.published_ours_run),
                "--front-ours-run",
                str(args.front_ours_run),
                "--front-alpamayo-run",
                str(args.front_alpamayo_run),
            ],
            env=env,
        )
    else:
        print("\n[4/4] Comparable benchmark reports skipped (missing run paths)")

    print("\nPrimary refreshed outputs:")
    print("  - artifacts/corl2027/alpasim_matrix10_analysis.md")
    print("  - artifacts/corl2027/corl_evidence_strength_audit.md")
    print("  - artifacts/corl2027/transfer_predictor_analysis.md")
    return 0


def _load_env_file(path: Path, env: dict[str, str]) -> None:
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        env.setdefault(key, value)


def _run(command: list[str], *, env: dict[str, str]) -> None:
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
