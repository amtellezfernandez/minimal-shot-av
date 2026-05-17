#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class CleanupTarget:
    name: str
    paths: tuple[Path, ...]
    description: str


TARGETS = {
    "internvla-models": CleanupTarget(
        name="internvla-models",
        paths=(ROOT / "artifacts" / "models" / "InternVLA-N1-DualVLN",),
        description="Downloaded InternVLA checkpoint shards under artifacts/models.",
    ),
    "candidate-jsonl": CleanupTarget(
        name="candidate-jsonl",
        paths=(
            ROOT / "artifacts" / "wod_test_selector_scored_candidates.jsonl",
            ROOT / "artifacts" / "wod_test_hgb_scored_candidates.jsonl",
            ROOT / "artifacts" / "wod_test_merged_candidates.jsonl",
            ROOT / "artifacts" / "wod_test_learned_candidates.jsonl",
        ),
        description="Large generated WOD test candidate JSONL exports.",
    ),
    "runs": CleanupTarget(
        name="runs",
        paths=(ROOT / "runs",),
        description="Local AlpaSim run directories and rollout logs.",
    ),
    "venv": CleanupTarget(
        name="venv",
        paths=(ROOT / ".venv",),
        description="Repo-local Python environment; fully recreatable with uv.",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reclaim disk space from large local-only repo artifacts.")
    parser.add_argument(
        "--target",
        action="append",
        choices=sorted(TARGETS),
        default=[],
        help="Cleanup target to remove. Repeat to remove multiple targets.",
    )
    parser.add_argument("--list", action="store_true", help="List targets and current sizes.")
    parser.add_argument("--yes", action="store_true", help="Actually delete the selected targets.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list or not args.target:
        for name in sorted(TARGETS):
            target = TARGETS[name]
            size = sum(_path_size(path) for path in target.paths if path.exists())
            print(f"{target.name:16} {_format_size(size):>8}  {target.description}")
        if not args.target:
            return 0

    if not args.yes:
        selected = ", ".join(args.target)
        raise SystemExit(f"Refusing to delete without --yes. Selected targets: {selected}")

    for name in args.target:
        target = TARGETS[name]
        for path in target.paths:
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            print(f"DEL {path}")
    return 0


def _path_size(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "K", "M", "G", "T"):
        if value < 1024.0 or unit == "T":
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}T"


if __name__ == "__main__":
    raise SystemExit(main())
