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

from minimal_shot_av.model.kinematic_candidates import write_kinematic_candidate_jsonl
from minimal_shot_av.model.wod_e2e import load_preference_frames
from minimal_shot_av.model.wod_submission import load_frame_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate non-text WOD-E2E kinematic candidate trajectories.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
        help="Directory containing WOD-E2E TFRecord shards.",
    )
    parser.add_argument("--val-dir", type=Path, help="Deprecated alias for --data-dir.")
    parser.add_argument(
        "--shard-glob",
        help="GCS or local glob pattern for TFRecord shards (e.g. gs://bucket/test_*.tfrecord-*). "
             "Overrides --data-dir when provided.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_kinematic_candidates.jsonl")
    parser.add_argument("--frame-list", type=Path, help="Optional JSON list of frame names to emit.")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Allow train/test frames without validation preference labels.",
    )
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    args = parser.parse_args()

    if args.shard_glob is not None:
        frames = load_preference_frames(
            ".",
            shard_glob=args.shard_glob,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=False,
            require_preferences=not args.include_unlabeled,
        )
    else:
        data_dir = args.val_dir if args.val_dir is not None else args.data_dir
        frames = load_preference_frames(
            data_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=False,
            require_preferences=not args.include_unlabeled,
        )
    if args.frame_list is not None:
        frame_names = load_frame_names(args.frame_list)
        frames = (frame for frame in frames if frame.frame_name in frame_names)
    if args.max_preference_frames is not None:
        frames = _take(frames, args.max_preference_frames)
    count = write_kinematic_candidate_jsonl(frames, args.output)
    print(json.dumps({"wrote_candidates": count, "path": str(args.output)}, indent=2))
    return 0


def _take(frames, limit: int):
    for index, frame in enumerate(frames):
        if index >= limit:
            break
        yield frame


if __name__ == "__main__":
    raise SystemExit(main())
