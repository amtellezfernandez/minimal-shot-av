#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames
from minimal_shot_av.model.wod_ranker import WodPreferenceRanker, candidate_ranker_row, ranker_uses_camera_features
from minimal_shot_av.model.wod_submission import load_frame_names
from minimal_shot_av.model.zero_shot_eval import candidate_record_from_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Score WOD-E2E candidate JSONL rows with a contextual ranker.")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "test")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-list", type=Path, help="Optional JSON list of frame names to score.")
    parser.add_argument("--include-unlabeled", action="store_true", help="Allow frames without preference labels.")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    rows = _load_candidate_rows(args.candidates)
    required = load_frame_names(args.frame_list) if args.frame_list else None
    candidate_frame_names = {str(row["frame_name"]) for row in rows}
    if required is not None:
        candidate_frame_names &= required
    ranker = WodPreferenceRanker.load(args.ranker)
    frames = _load_needed_frames(
        args.data_dir,
        candidate_frame_names,
        max_shards=args.max_shards,
        max_records=args.max_records,
        require_preferences=not args.include_unlabeled,
        include_camera_images=ranker_uses_camera_features(ranker),
    )
    written = score_candidate_rows(rows, frames, ranker, args.output, required_frame_names=required)
    print(
        json.dumps(
            {
                "input": str(args.candidates),
                "ranker": str(args.ranker),
                "output": str(args.output),
                "scored_candidates": written,
                "scored_frames": len(frames),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def score_candidate_rows(
    rows: list[dict[str, Any]],
    frames_by_name: dict[str, WodE2EPreferenceFrame],
    ranker: WodPreferenceRanker,
    output: Path,
    *,
    required_frame_names: set[str] | None = None,
) -> int:
    missing_frames = sorted((required_frame_names or set(frames_by_name)) - set(frames_by_name))
    if missing_frames:
        preview = ", ".join(missing_frames[:5])
        raise ValueError(f"missing frame metadata for {len(missing_frames)} required frame(s): {preview}")
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            frame_name = str(row["frame_name"])
            if required_frame_names is not None and frame_name not in required_frame_names:
                continue
            frame = frames_by_name.get(frame_name)
            if frame is None:
                continue
            candidate = candidate_record_from_json(row)
            ranker_row = candidate_ranker_row(
                frame=frame,
                trajectory=candidate.trajectory,
                candidate_name=candidate.candidate_name,
                candidate_index=candidate.candidate_index,
                source=candidate.source,
            )
            stream.write(
                json.dumps(
                    {
                        **row,
                        "ranker_score": ranker.predict_row(ranker_row),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    if count == 0:
        raise ValueError("no candidate rows were scored")
    return count


def _load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no candidate rows found in {path}")
    return rows


def _load_needed_frames(
    data_dir: Path,
    frame_names: set[str],
    *,
    max_shards: int | None,
    max_records: int | None,
    require_preferences: bool,
    include_camera_images: bool,
) -> dict[str, WodE2EPreferenceFrame]:
    frames: dict[str, WodE2EPreferenceFrame] = {}
    if not frame_names:
        return frames
    for frame in load_preference_frames(
        data_dir,
        max_shards=max_shards,
        max_records=max_records,
        include_camera_images=include_camera_images,
        require_preferences=require_preferences,
    ):
        if frame.frame_name in frame_names:
            frames[frame.frame_name] = frame
            if len(frames) == len(frame_names):
                break
    return frames


if __name__ == "__main__":
    raise SystemExit(main())
