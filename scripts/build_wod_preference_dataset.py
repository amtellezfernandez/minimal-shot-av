#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WAYMO_SRC = ROOT / "waymo-open-dataset" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.wod_e2e import load_preference_frames
from minimal_shot_av.wod_preference import generate_preference_candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build WOD-E2E preference-ranker JSONL rows with official RFS labels."
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "wod_preference_candidates.jsonl",
    )
    parser.add_argument("--max-shards", type=int, default=None)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-preference-frames", type=int, default=None)
    args = parser.parse_args()

    rater_feedback_utils = _load_rater_feedback_utils()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    frame_count = 0
    row_count = 0
    best_scores: list[float] = []
    logged_scores: list[float] = []

    with args.output.open("w", encoding="utf-8") as stream:
        for frame in load_preference_frames(
            args.val_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
        ):
            frame_count += 1
            candidates = generate_preference_candidates(frame)
            scored_rows = []

            for candidate in candidates:
                rfs_score = _score_candidate(
                    rater_feedback_utils,
                    candidate.trajectory,
                    frame,
                )
                row = {
                    "frame_name": frame.frame_name,
                    "candidate_name": candidate.name,
                    "rfs_score": rfs_score,
                    "reference_count": len(frame.references),
                    "candidate_index": len(scored_rows),
                    "features": candidate.features,
                }
                scored_rows.append(row)

            best_score = max(float(row["rfs_score"]) for row in scored_rows)
            logged_score = float(
                next(row["rfs_score"] for row in scored_rows if row["candidate_name"] == "logged_future")
            )
            best_scores.append(best_score)
            logged_scores.append(logged_score)

            for row in scored_rows:
                row["best_candidate_score"] = best_score
                row["is_oracle_best"] = float(row["rfs_score"]) == best_score
                stream.write(json.dumps(row, sort_keys=True) + "\n")
                row_count += 1

            print(
                f"{frame_count:04d} frame={frame.frame_name} "
                f"candidates={len(candidates)} logged={logged_score:.3f} "
                f"oracle_best={best_score:.3f}"
            )

            if args.max_preference_frames is not None and frame_count >= args.max_preference_frames:
                break

    if frame_count == 0:
        raise RuntimeError("no preference-labeled validation frames were found")

    print(
        "summary "
        f"frames={frame_count} rows={row_count} output={args.output} "
        f"mean_logged={sum(logged_scores) / len(logged_scores):.3f} "
        f"mean_oracle_best={sum(best_scores) / len(best_scores):.3f}"
    )
    return 0


def _score_candidate(rater_feedback_utils, trajectory, frame) -> float:
    inference_trajectory = np.array([[trajectory]], dtype=np.float64)
    inference_probs = np.array([[1.0]], dtype=np.float64)
    rater_trajectories = [
        [np.array(reference.trajectory, dtype=np.float64) for reference in frame.references]
    ]
    rater_labels = [
        np.array([reference.score for reference in frame.references], dtype=np.float64)
    ]
    init_speed = np.array([frame.init_speed_mps], dtype=np.float64)

    outputs = rater_feedback_utils.get_rater_feedback_score(
        inference_trajectory,
        inference_probs,
        rater_trajectories,
        rater_labels,
        init_speed,
    )
    return float(outputs["rater_feedback_score"][0])


def _load_rater_feedback_utils():
    path = WAYMO_SRC / "waymo_open_dataset" / "metrics" / "python" / "rater_feedback_utils.py"
    spec = importlib.util.spec_from_file_location("wod_rater_feedback_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official RFS utility from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(main())
