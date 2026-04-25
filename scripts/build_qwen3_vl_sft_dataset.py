#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.qwen3_vl_stack import Qwen3VlWodConfig, build_sft_record
from minimal_shot_av.wod_e2e import load_preference_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build Qwen3-VL SFT JSONL from WOD-E2E preference-labeled validation frames."
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "qwen3_vl_wod_sft.jsonl")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    config = Qwen3VlWodConfig(model_name=args.model_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    with args.output.open("w", encoding="utf-8") as stream:
        for frame in load_preference_frames(args.val_dir, max_shards=args.max_shards, max_records=args.max_records):
            stream.write(json.dumps(build_sft_record(frame, config=config), sort_keys=True) + "\n")
            frame_count += 1
            if args.max_frames is not None and frame_count >= args.max_frames:
                break

    if frame_count == 0:
        raise RuntimeError("no WOD-E2E preference frames were found")
    print(f"wrote {frame_count} Qwen3-VL SFT records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
