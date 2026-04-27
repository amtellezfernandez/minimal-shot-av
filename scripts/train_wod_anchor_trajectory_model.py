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

from minimal_shot_av.model.anchor_trajectory_model import fit_anchor_residual_trajectory_model
from minimal_shot_av.model.wod_e2e import load_preference_frames


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a non-text WOD-E2E anchor-residual proposal model.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "train",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_anchor_trajectory_model.json")
    parser.add_argument("--anchors", type=int, required=True)
    parser.add_argument(
        "--top-k",
        type=int,
        default=8,
        help="Stored for run metadata; generation controls actual top-k.",
    )
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--anchor-iterations", type=int, default=25)
    parser.add_argument("--residual-modes-per-anchor", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
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
    model = fit_anchor_residual_trajectory_model(
        frames,
        anchor_count=args.anchors,
        ridge=args.ridge,
        anchor_iterations=args.anchor_iterations,
        seed=args.seed,
        residual_modes_per_anchor=args.residual_modes_per_anchor,
    )
    model.save(args.output)
    print(
        json.dumps(
            {
                "model": str(args.output),
                "train_rows": model.train_rows,
                "anchors": len(model.anchors),
                "top_k_metadata": args.top_k,
                "ridge": model.ridge,
                "anchor_iterations": model.anchor_iterations,
                "residual_modes_per_anchor": args.residual_modes_per_anchor,
                "seed": model.seed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
