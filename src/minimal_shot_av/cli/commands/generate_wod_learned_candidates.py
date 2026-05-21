#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.learned_trajectory_model import RidgeTrajectoryModel
from minimal_shot_av.model.wod_e2e import load_preference_frames
from minimal_shot_av.model.wod_submission import load_frame_names


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate non-text WOD-E2E learned trajectory candidates.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
        help="Directory containing WOD-E2E TFRecord shards.",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        help="Deprecated alias for --data-dir.",
    )
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts" / "wod_ridge_trajectory_model.json")
    parser.add_argument(
        "--aux-model",
        type=Path,
        help="Optional second trajectory model emitted as temporal-prefixed auxiliary candidates.",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_learned_candidates.jsonl")
    parser.add_argument("--frame-list", type=Path, help="Optional JSON list of frame names to emit.")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Allow train/test frames without validation preference labels.",
    )
    parser.add_argument("--max-residual-modes", type=int, default=3)
    parser.add_argument("--pairwise-residuals", action="store_true")
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    args = parser.parse_args()

    model = RidgeTrajectoryModel.load(args.model)
    aux_model = RidgeTrajectoryModel.load(args.aux_model) if args.aux_model is not None else None
    data_dir = args.val_dir if args.val_dir is not None else args.data_dir
    frames = load_preference_frames(
        data_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=False,
        require_preferences=not args.include_unlabeled,
    )
    if args.frame_list is not None:
        frame_names = load_frame_names(args.frame_list)
        frames = (frame for frame in frames if frame.frame_name in frame_names)
    if args.max_preference_frames is not None:
        frames = _take(frames, args.max_preference_frames)
    count = write_candidate_jsonl(
        frames,
        args.output,
        model,
        aux_model=aux_model,
        max_residual_modes=args.max_residual_modes,
        include_pairwise_residuals=args.pairwise_residuals,
    )
    print(
        json.dumps(
            {
                "wrote_candidates": count,
                "path": str(args.output),
                "model": str(args.model),
                "aux_model": str(args.aux_model) if args.aux_model is not None else None,
            },
            indent=2,
        )
    )
    return 0


def write_candidate_jsonl(
    frames,
    path: Path,
    model: RidgeTrajectoryModel,
    *,
    aux_model: RidgeTrajectoryModel | None = None,
    max_residual_modes: int = 3,
    include_pairwise_residuals: bool = False,
) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            candidate_index = 0
            for name, trajectory in model.candidate_trajectories(
                frame.past_trajectory,
                intent=frame.intent,
                init_speed_mps=frame.init_speed_mps,
                max_residual_modes=max_residual_modes,
                include_pairwise_residuals=include_pairwise_residuals,
            ):
                stream.write(_candidate_payload(frame.frame_name, name, candidate_index, trajectory) + "\n")
                candidate_index += 1
                count += 1
            if aux_model is not None:
                for name, trajectory in aux_model.candidate_trajectories(
                    frame.past_trajectory,
                    intent=frame.intent,
                    init_speed_mps=frame.init_speed_mps,
                    max_residual_modes=max_residual_modes,
                    include_pairwise_residuals=include_pairwise_residuals,
                ):
                    stream.write(
                        _candidate_payload(
                            frame.frame_name,
                            f"temporal_{name}",
                            candidate_index,
                            trajectory,
                            source="wod_temporal_summary_trajectory_non_text",
                        )
                        + "\n"
                    )
                    candidate_index += 1
                    count += 1
    return count


def _candidate_payload(
    frame_name: str,
    candidate_name: str,
    candidate_index: int,
    trajectory,
    *,
    source: str = "wod_ridge_trajectory_non_text",
) -> str:
    payload = {
        "frame_name": frame_name,
        "source": source,
        "candidate_name": candidate_name,
        "candidate_index": candidate_index,
        "trajectory_20wp_4hz": [[float(x), float(y)] for x, y in trajectory],
    }
    encoded = json.dumps(payload, separators=(",", ":"))
    return encoded


def _take(frames, limit: int):
    for index, frame in enumerate(frames):
        if index >= limit:
            break
        yield frame


if __name__ == "__main__":
    raise SystemExit(main())
