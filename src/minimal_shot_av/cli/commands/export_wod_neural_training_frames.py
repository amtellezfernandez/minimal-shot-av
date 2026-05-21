#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.neural_trajectory_model import save_neural_training_frame_cache  # noqa: E402
from minimal_shot_av.model.wod_e2e import load_preference_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Export compact WOD-E2E frames for neural proposal training.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "train",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_neural_training_frames.jsonl")
    parser.add_argument(
        "--shard-glob",
        help=(
            "Optional local or gs:// glob for TFRecord shards, for example "
            "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_*.tfrecord-*."
        ),
    )
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--record-start", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--append", action="store_true", help="Append this shard window to an existing JSONL cache.")
    args = parser.parse_args()

    frames = load_preference_frames(
        args.train_dir,
        shard_glob=args.shard_glob,
        shard_start=args.shard_start,
        record_start=args.record_start,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=False,
        require_preferences=False,
    )
    exported_rows = save_neural_training_frame_cache(frames, args.output, append=args.append)
    report = {
        "schema": "wod_neural_training_frame_cache_v1",
        "output": str(args.output),
        "exported_rows": exported_rows,
        "train_dir": str(args.train_dir),
        "shard_glob": args.shard_glob,
        "shard_start": args.shard_start,
        "record_start": args.record_start,
        "max_shards": args.max_shards,
        "max_records": args.max_records,
        "append": bool(args.append),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
