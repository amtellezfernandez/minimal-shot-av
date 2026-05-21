#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.v20_planner import (  # noqa: E402
    fit_neural_system2_planner,
    load_neural_planner_frame_cache,
)
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames  # noqa: E402
from minimal_shot_av.model.world_model import (  # noqa: E402
    attach_external_embedding_cache,
    load_external_embedding_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the v20 neural system-2 WOD planner in the unified .venv-v20 environment."
    )
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "train",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_v20_system2_planner.pt")
    parser.add_argument(
        "--report-output",
        type=Path,
        default=ROOT / "artifacts" / "wod_v20_system2_planner_train_report.json",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--modes", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--confidence-weight", type=float, default=0.2)
    parser.add_argument("--critic-weight", type=float, default=1.0)
    parser.add_argument("--smoothness-weight", type=float, default=0.02)
    parser.add_argument("--generator-utility-weight", type=float, default=0.04)
    parser.add_argument("--diversity-weight", type=float, default=0.0)
    parser.add_argument("--decoder-type", choices=("linear", "mode_query"), default="mode_query")
    parser.add_argument("--max-reference-candidates", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument(
        "--frame-cache",
        type=Path,
        action="append",
        default=[],
        help="Planner/preference frame cache. May be passed multiple times; avoids TensorFlow parsing.",
    )
    parser.add_argument(
        "--external-embedding-cache",
        type=Path,
        help="Optional world/InternVLA embedding cache attached before v20 training.",
    )
    parser.add_argument(
        "--allow-missing-external-embeddings",
        action="store_true",
        help="Keep frames without external embeddings instead of failing.",
    )
    args = parser.parse_args()

    frames = _load_frames(args)
    if args.external_embedding_cache is not None:
        frames = attach_external_embedding_cache(
            frames,
            load_external_embedding_cache(args.external_embedding_cache),
            require_all=not args.allow_missing_external_embeddings,
        )

    model, metadata = fit_neural_system2_planner(
        frames,
        hidden_dim=args.hidden_dim,
        layers=args.layers,
        heads=args.heads,
        modes=args.modes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_reference_candidates=args.max_reference_candidates,
        confidence_weight=args.confidence_weight,
        critic_weight=args.critic_weight,
        smoothness_weight=args.smoothness_weight,
        generator_utility_weight=args.generator_utility_weight,
        diversity_weight=args.diversity_weight,
        decoder_type=args.decoder_type,
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
        "confidence_weight": args.confidence_weight,
        "critic_weight": args.critic_weight,
        "smoothness_weight": args.smoothness_weight,
        "generator_utility_weight": args.generator_utility_weight,
        "diversity_weight": args.diversity_weight,
        "decoder_type": args.decoder_type,
        "max_reference_candidates": args.max_reference_candidates,
        "seed": args.seed,
        "train_dir": str(args.train_dir),
        "frame_cache": [str(path) for path in args.frame_cache],
        "external_embedding_cache": (
            str(args.external_embedding_cache) if args.external_embedding_cache is not None else None
        ),
        "allow_missing_external_embeddings": bool(args.allow_missing_external_embeddings),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _load_frames(args: argparse.Namespace) -> list[WodE2EPreferenceFrame]:
    if args.frame_cache:
        frames: list[WodE2EPreferenceFrame] = []
        for path in args.frame_cache:
            frames.extend(load_neural_planner_frame_cache(path))
        if not frames:
            raise ValueError("frame cache inputs did not contain any v20 planner frames")
        return frames
    return list(
        load_preference_frames(
            args.train_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=False,
            require_preferences=True,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
