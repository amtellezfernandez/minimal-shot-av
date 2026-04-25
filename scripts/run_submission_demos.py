from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


DEMO_RUNS = (
    (
        "grand_spotlight_demo",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "spotlight", "--seed", "3"),
    ),
    (
        "grand_intersection_stress_seed3",
        ("--policy", "spotlight-reflex", "--scenario-cluster", "intersection", "--seed", "3"),
    ),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate separate Grand and Minor Commission demo artifacts.")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts/submissions"),
        help="Root directory for generated submission demo artifacts.",
    )
    args = parser.parse_args()

    for name, demo_args in DEMO_RUNS:
        output_dir = args.artifacts_root / name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_demo.py"),
            *demo_args,
            "--artifacts-dir",
            str(output_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
