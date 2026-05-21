#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import islice
import json
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.v20_planner import NeuralSystem2Planner, load_neural_planner_frame_cache  # noqa: E402
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames  # noqa: E402
from minimal_shot_av.model.wod_submission import load_frame_names  # noqa: E402
from minimal_shot_av.model.world_model import (  # noqa: E402
    attach_external_embedding_cache,
    load_external_embedding_cache,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate WOD-E2E candidates from a trained v20 planner.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--val-dir", type=Path, help="Deprecated alias for --data-dir.")
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "wod_v20_system2_planner.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_v20_candidates.jsonl")
    parser.add_argument("--frame-list", type=Path, help="Optional JSON list of frame names to emit.")
    parser.add_argument("--frame-cache", type=Path, help="Planner/preference frame cache; avoids TensorFlow parsing.")
    parser.add_argument("--external-embedding-cache", type=Path)
    parser.add_argument("--allow-missing-external-embeddings", action="store_true")
    parser.add_argument("--include-unlabeled", action="store_true")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--refine-step-size", type=float, default=0.03)
    parser.add_argument("--smoothness-weight", type=float, default=0.03)
    parser.add_argument("--trust-region-weight", type=float, default=0.03)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    args = parser.parse_args()

    model = NeuralSystem2Planner.load(args.model)
    frames = _load_frames(args)
    if args.external_embedding_cache is not None:
        frames = attach_external_embedding_cache(
            list(frames),
            load_external_embedding_cache(args.external_embedding_cache),
            require_all=not args.allow_missing_external_embeddings,
        )
    if args.frame_list is not None:
        frame_names = load_frame_names(args.frame_list)
        frames = (frame for frame in frames if frame.frame_name in frame_names)
    if args.max_preference_frames is not None:
        frames = islice(frames, args.max_preference_frames)

    count = write_candidate_jsonl(
        frames,
        args.output,
        model,
        top_k=args.top_k,
        refine_steps=args.refine_steps,
        refine_step_size=args.refine_step_size,
        smoothness_weight=args.smoothness_weight,
        trust_region_weight=args.trust_region_weight,
    )
    print(
        json.dumps(
            {
                "schema": "wod_v20_candidate_generation_report_v1",
                "wrote_candidates": count,
                "path": str(args.output),
                "model": str(args.model),
                "top_k": int(args.top_k),
                "refine_steps": int(args.refine_steps),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def write_candidate_jsonl(
    frames: Iterable[WodE2EPreferenceFrame],
    path: Path,
    model: NeuralSystem2Planner,
    *,
    top_k: int,
    refine_steps: int = 0,
    refine_step_size: float = 0.03,
    smoothness_weight: float = 0.03,
    trust_region_weight: float = 0.03,
) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            for candidate_index, (name, trajectory, confidence) in enumerate(
                model.candidate_trajectories_for_frame(
                    frame,
                    top_k=top_k,
                    refine_steps=refine_steps,
                    refine_step_size=refine_step_size,
                    smoothness_weight=smoothness_weight,
                    trust_region_weight=trust_region_weight,
                )
            ):
                stream.write(_candidate_payload(frame.frame_name, name, candidate_index, trajectory, confidence) + "\n")
                count += 1
    return count


def _candidate_payload(
    frame_name: str,
    candidate_name: str,
    candidate_index: int,
    trajectory,
    confidence: float,
) -> str:
    payload = {
        "frame_name": frame_name,
        "source": "wod_v20_neural_system2_planner",
        "candidate_name": candidate_name,
        "candidate_index": int(candidate_index),
        "confidence": float(confidence),
        "trajectory_20wp_4hz": [[float(x), float(y)] for x, y in trajectory],
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True)
    return encoded


def _load_frames(args: argparse.Namespace) -> Iterable[WodE2EPreferenceFrame]:
    if args.frame_cache is not None:
        return load_neural_planner_frame_cache(args.frame_cache)
    data_dir = args.val_dir if args.val_dir is not None else args.data_dir
    return load_preference_frames(
        data_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=False,
        require_preferences=not args.include_unlabeled,
    )


if __name__ == "__main__":
    raise SystemExit(main())
