#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.learned_trajectory_model import (
    FEATURE_SET_BASE,
    FEATURE_SET_TEMPORAL,
    fit_ridge_trajectory_model,
)
from minimal_shot_av.model.wod_e2e import load_preference_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a non-text ridge trajectory model from WOD-E2E future labels.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "train",
        help="Directory containing WOD-E2E shards with future trajectories.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_ridge_trajectory_model.json")
    parser.add_argument("--ridge", type=float, default=30.0)
    parser.add_argument("--residual-modes", type=int, default=3)
    parser.add_argument("--feature-set", choices=(FEATURE_SET_BASE, FEATURE_SET_TEMPORAL), default=FEATURE_SET_BASE)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    frames = load_preference_frames(
        args.train_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=False,
        require_preferences=False,
    )
    model = fit_ridge_trajectory_model(
        frames,
        ridge=args.ridge,
        residual_modes=args.residual_modes,
        feature_set=args.feature_set,
    )
    model.save(args.output)
    print(
        json.dumps(
            {
                "model": str(args.output),
                "train_rows": model.train_rows,
                "ridge": model.ridge,
                "feature_set": model.feature_set,
                "residual_modes": len(model.residual_modes),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
