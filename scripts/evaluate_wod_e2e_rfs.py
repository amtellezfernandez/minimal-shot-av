#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WAYMO_SRC = ROOT / "waymo-open-dataset" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.wod_e2e import load_preference_frames


def _load_rater_feedback_utils():
    path = WAYMO_SRC / "waymo_open_dataset" / "metrics" / "python" / "rater_feedback_utils.py"
    spec = importlib.util.spec_from_file_location("wod_rater_feedback_utils", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load official RFS utility from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score WOD-E2E validation trajectories with official RFS."
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--max-shards", type=int, default=1)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--max-preference-frames", type=int, default=25)
    args = parser.parse_args()

    rater_feedback_utils = _load_rater_feedback_utils()
    scores: list[float] = []
    for frame in load_preference_frames(
        args.val_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
    ):
        inference_trajectory = np.array(
            [[frame.future_trajectory]],
            dtype=np.float64,
        )
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
        score = float(outputs["rater_feedback_score"][0])
        scores.append(score)
        print(
            f"{len(scores):04d} frame={frame.frame_name} "
            f"refs={len(frame.references)} init_speed={frame.init_speed_mps:.3f} "
            f"log_future_rfs={score:.3f}"
        )

        if len(scores) >= args.max_preference_frames:
            break

    if not scores:
        raise RuntimeError("no validation frames with valid preference trajectories were found")

    print(
        "summary "
        f"frames={len(scores)} "
        f"mean_log_future_rfs={sum(scores) / len(scores):.3f} "
        f"min={min(scores):.3f} max={max(scores):.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
