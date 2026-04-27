#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames
from minimal_shot_av.model.wod_preference import trajectory_features
from minimal_shot_av.model.zero_shot_eval import (
    CandidateRecord,
    RfsScorer,
    load_candidate_record_groups,
    load_official_rfs_scorer,
    local_rfs_score,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score an existing WOD-E2E candidate JSONL into ranker-training rows."
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--candidate-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument(
        "--rfs-backend",
        choices=("official", "local"),
        default="official",
        help="Use official Waymo RFS for final datasets; local is for tests and smoke checks.",
    )
    parser.add_argument(
        "--waymo-src",
        type=Path,
        default=ROOT / "waymo-open-dataset" / "src",
        help="Path containing waymo_open_dataset/metrics/python/rater_feedback_utils.py.",
    )
    args = parser.parse_args()

    candidate_groups, invalid = load_candidate_record_groups(args.candidate_jsonl)
    if invalid:
        raise ValueError(f"{args.candidate_jsonl}: invalid candidate rows: {invalid[:5]}")
    scorer = local_rfs_score if args.rfs_backend == "local" else load_official_rfs_scorer(args.waymo_src)
    frames = load_preference_frames(
        args.val_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=False,
    )
    rows = build_scored_candidate_rows(
        frames,
        candidate_groups,
        scorer=scorer,
        max_preference_frames=args.max_preference_frames,
    )
    if not rows:
        raise RuntimeError("no candidate rows matched WOD-E2E preference frames")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    frame_names = {str(row["frame_name"]) for row in rows}
    summary = {
        "frames": len(frame_names),
        "rows": len(rows),
        "output": str(args.output),
        "rfs_backend": args.rfs_backend,
        "mean_first_candidate_rfs": _mean(
            [
                float(row["rfs_score"])
                for row in rows
                if int(row["candidate_index"]) == 0
            ]
        ),
        "mean_oracle_rfs": _mean_best_score_by_frame(rows),
    }
    print(json.dumps(summary, indent=2))
    return 0


def build_scored_candidate_rows(
    frames: Iterable[WodE2EPreferenceFrame],
    candidates_by_frame: dict[str, list[CandidateRecord]],
    *,
    scorer: RfsScorer = local_rfs_score,
    max_preference_frames: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    matched_frames = 0
    for frame in frames:
        candidates = candidates_by_frame.get(frame.frame_name)
        if not candidates:
            continue
        matched_frames += 1
        scored_rows = [_candidate_score_row(frame, candidate, scorer) for candidate in candidates]
        best_score = max(float(row["rfs_score"]) for row in scored_rows)
        for row in scored_rows:
            row["best_candidate_score"] = best_score
            row["is_oracle_best"] = float(row["rfs_score"]) == best_score
            rows.append(row)
        if max_preference_frames is not None and matched_frames >= max_preference_frames:
            break
    return rows


def _candidate_score_row(
    frame: WodE2EPreferenceFrame,
    candidate: CandidateRecord,
    scorer: RfsScorer,
) -> dict[str, object]:
    return {
        "frame_name": frame.frame_name,
        "candidate_name": candidate.candidate_name,
        "candidate_index": candidate.candidate_index,
        "source": candidate.source,
        "rfs_score": float(scorer(candidate, frame)),
        "reference_count": len(frame.references),
        "features": trajectory_features(
            candidate.trajectory,
            candidate_name=candidate.candidate_name,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
        ),
    }


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _mean_best_score_by_frame(rows: list[dict[str, object]]) -> float | None:
    best_by_frame: dict[str, float] = {}
    for row in rows:
        frame_name = str(row["frame_name"])
        best_by_frame[frame_name] = float(row["best_candidate_score"])
    return _mean(list(best_by_frame.values()))


if __name__ == "__main__":
    raise SystemExit(main())
