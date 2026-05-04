#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUCKET = "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEFAULT_STAGE_ROOT = ROOT / "data" / "wod_e2e_stage"


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage a small WOD-E2E shard window from GCS.")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--max-shards", type=int, default=8)
    parser.add_argument("--stage-root", type=Path, default=DEFAULT_STAGE_ROOT)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Copy the selected shards instead of only printing a plan.",
    )
    args = parser.parse_args()

    uris = list_split_shards(args.bucket, args.split)
    selected = select_shard_window(uris, shard_start=args.shard_start, max_shards=args.max_shards)
    target_dir = args.stage_root / args.split
    commands = [["gsutil", "-m", "cp", "-n", uri, str(target_dir)] for uri in selected]
    manifest = {
        "schema": "wod_e2e_gcs_shard_stage_plan_v1",
        "bucket": args.bucket,
        "split": args.split,
        "available_shards": len(uris),
        "shard_start": args.shard_start,
        "max_shards": args.max_shards,
        "selected_shards": len(selected),
        "target_dir": str(target_dir),
        "uris": selected,
        "commands": commands,
        "execute": bool(args.execute),
    }
    text = json.dumps(manifest, indent=2, sort_keys=True)
    if args.manifest_output:
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if args.execute:
        target_dir.mkdir(parents=True, exist_ok=True)
        for command in commands:
            subprocess.run(command, check=True, cwd=ROOT)
    return 0


def list_split_shards(bucket: str, split: str) -> list[str]:
    prefix = split_object_prefix(split)
    result = subprocess.run(
        ["gsutil", "ls", f"{bucket.rstrip('/')}/{prefix}*.tfrecord-*"],
        check=True,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
    )
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def select_shard_window(uris: list[str], *, shard_start: int, max_shards: int) -> list[str]:
    if shard_start < 0:
        raise ValueError("shard_start must be non-negative")
    if max_shards <= 0:
        raise ValueError("max_shards must be positive")
    return uris[shard_start : shard_start + max_shards]


def split_object_prefix(split: str) -> str:
    if split == "train":
        return "training"
    return split


def stage_plan_for_test(
    *,
    bucket: str = DEFAULT_BUCKET,
    split: str = "train",
    uris: list[str],
    shard_start: int = 0,
    max_shards: int = 8,
    stage_root: Path = DEFAULT_STAGE_ROOT,
) -> dict[str, Any]:
    selected = select_shard_window(sorted(uris), shard_start=shard_start, max_shards=max_shards)
    target_dir = stage_root / split
    return {
        "bucket": bucket,
        "split": split,
        "available_shards": len(uris),
        "selected_shards": len(selected),
        "target_dir": str(target_dir),
        "uris": selected,
        "commands": [["gsutil", "-m", "cp", "-n", uri, str(target_dir)] for uri in selected],
    }


if __name__ == "__main__":
    raise SystemExit(main())
