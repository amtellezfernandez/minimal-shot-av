#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import islice
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_e2e import load_preference_frames
from minimal_shot_av.model.world_model import scene_token_features, write_scene_token_cache_mapping


def main() -> int:
    parser = argparse.ArgumentParser(description="Build cached WOD-E2E scene-token features for world-model sweeps.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
        help="Directory containing WOD-E2E TFRecord shards.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_scene_tokens_val.json")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=16,
        help="Write a valid partial cache after this many cached frames. Use 0 to write only at the end.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=16,
        help="Print progress after this many cached frames. Use 0 to silence progress.",
    )
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Cache all loadable frames instead of only preference-labeled validation frames.",
    )
    args = parser.parse_args()

    frames = load_preference_frames(
        args.data_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=True,
        require_preferences=not args.include_unlabeled,
    )
    if args.max_preference_frames is not None:
        frames = islice(frames, args.max_preference_frames)

    cache: dict[str, list[float]] = {}
    for count, frame in enumerate(frames, start=1):
        cache[frame.frame_name] = scene_token_features(frame)
        if args.progress_every and count % args.progress_every == 0:
            print(json.dumps({"cached_frames": count, "path": str(args.output)}), flush=True)
        if args.checkpoint_every and count % args.checkpoint_every == 0:
            write_scene_token_cache_mapping(cache, args.output)

    count = write_scene_token_cache_mapping(cache, args.output)
    print(json.dumps({"cached_frames": count, "path": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
