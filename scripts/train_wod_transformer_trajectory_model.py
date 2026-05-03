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

from minimal_shot_av.model.transformer_trajectory_model import (  # noqa: E402
    fit_transformer_trajectory_proposal_model,
    load_transformer_training_frame_cache,
)
from minimal_shot_av.model.wod_e2e import load_preference_frames  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a transformer WOD-E2E trajectory proposal model.")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "train",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_transformer_trajectory_model.pt")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "artifacts" / "wod_transformer_trajectory_train_report.json",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--confidence-weight", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--frame-cache",
        type=Path,
        help="JSONL cache from export_wod_neural_training_frames.py; avoids TensorFlow in training.",
    )
    args = parser.parse_args()

    if args.frame_cache is not None:
        frames = load_transformer_training_frame_cache(args.frame_cache)
    else:
        frames = load_preference_frames(
            args.train_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=False,
            require_preferences=False,
        )

    model, metadata = fit_transformer_trajectory_proposal_model(
        frames,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        modes=args.modes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        smoothness_weight=args.smoothness_weight,
        confidence_weight=args.confidence_weight,
        seed=args.seed,
        device=args.device,
    )
    model.save(args.output)
    report = {
        **metadata,
        "model": str(args.output),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "smoothness_weight": args.smoothness_weight,
        "confidence_weight": args.confidence_weight,
        "seed": args.seed,
        "train_dir": str(args.train_dir),
        "frame_cache": str(args.frame_cache) if args.frame_cache is not None else None,
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
