#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.learned_trajectory_model import (  # noqa: E402
    FEATURE_SET_BASE,
    FEATURE_SET_EXTERNAL_EMBEDDINGS,
    FEATURE_SET_TEMPORAL,
)
from minimal_shot_av.model.neural_trajectory_model import (  # noqa: E402
    fit_neural_anchor_residual_trajectory_model,
    load_neural_training_frame_cache,
)
from minimal_shot_av.model.wod_e2e import load_preference_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a compact neural WOD-E2E anchor-residual proposal model.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "train",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_neural_trajectory_model.json")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "artifacts" / "wod_neural_trajectory_train_report.json",
    )
    parser.add_argument("--anchors", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--anchor-iterations", type=int, default=25)
    parser.add_argument("--residual-modes-per-anchor", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--feature-set",
        choices=(FEATURE_SET_BASE, FEATURE_SET_TEMPORAL, FEATURE_SET_EXTERNAL_EMBEDDINGS),
        default=FEATURE_SET_TEMPORAL,
    )
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--frame-cache",
        type=Path,
        action="append",
        help=(
            "JSONL cache from export_wod_neural_training_frames.py; may be passed multiple times. "
            "Avoids TensorFlow and raw TFRecords in the training environment."
        ),
    )
    args = parser.parse_args()

    if args.frame_cache is not None:
        frames = []
        for frame_cache in args.frame_cache:
            frames.extend(load_neural_training_frame_cache(frame_cache))
    else:
        frames = load_preference_frames(
            args.train_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=False,
            require_preferences=False,
        )
    model, metadata = fit_neural_anchor_residual_trajectory_model(
        frames,
        anchor_count=args.anchors,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        anchor_iterations=args.anchor_iterations,
        residual_modes_per_anchor=args.residual_modes_per_anchor,
        feature_set=args.feature_set,
        seed=args.seed,
        device=args.device,
    )
    model.save(args.output)
    report = {
        **metadata,
        "model": str(args.output),
        "hidden_dim": args.hidden_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "anchor_iterations": args.anchor_iterations,
        "residual_modes_per_anchor": args.residual_modes_per_anchor,
        "seed": args.seed,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
