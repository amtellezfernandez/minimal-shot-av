#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.anchor_trajectory_model import AnchorResidualTrajectoryModel, anchor_candidate_payloads
from minimal_shot_av.model.wod_e2e import load_preference_frames
from minimal_shot_av.model.wod_submission import load_frame_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate WOD-E2E anchor-residual trajectory candidates.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_anchor_candidates.jsonl")
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--residual-modes-per-anchor", type=int, default=0)
    parser.add_argument("--frame-list", type=Path)
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    model = AnchorResidualTrajectoryModel.load(args.model)
    frames = load_preference_frames(
        args.data_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=False,
        require_preferences=not args.include_unlabeled,
    )
    if args.frame_list is not None:
        frame_names = load_frame_names(args.frame_list)
        frames = (frame for frame in frames if frame.frame_name in frame_names)
    count = _write_jsonl(
        anchor_candidate_payloads(
            frames,
            model,
            top_k=args.top_k,
            residual_modes_per_anchor=args.residual_modes_per_anchor,
        ),
        args.output,
    )
    print(json.dumps({"wrote_candidates": count, "path": str(args.output), "model": str(args.model)}, indent=2))
    return 0


def _write_jsonl(rows: list[dict[str, object]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    return len(rows)


if __name__ == "__main__":
    raise SystemExit(main())
