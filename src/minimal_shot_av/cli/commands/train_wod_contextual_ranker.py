#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from itertools import islice
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames
from minimal_shot_av.model.wod_ranker import WodPreferenceRanker, candidate_ranker_row, raw_features
from minimal_shot_av.model.wod_ranker import selector_numeric_features
from minimal_shot_av.model.zero_shot_eval import CandidateRecord, load_candidate_record_groups, load_official_rfs_scorer
from minimal_shot_av.model.zero_shot_eval import local_rfs_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the contextual WOD-E2E candidate ranker.")
    parser.add_argument("--val-dir", type=Path, default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_contextual_ranker.json")
    parser.add_argument("--selector-ridge", type=float, default=100.0)
    parser.add_argument(
        "--selector-target",
        choices=("absolute", "frame_delta", "frame_zscore", "oracle_binary"),
        default="frame_delta",
    )
    parser.add_argument(
        "--selector-features",
        choices=(
            "linear",
            "squared",
            "contextual",
            "geometry_contextual",
            "camera_contextual",
            "image_contextual",
        ),
        default="contextual",
    )
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument("--rfs-backend", choices=("official", "local"), default="official")
    parser.add_argument("--waymo-src", type=Path, default=ROOT / "workspace" / "waymo-open-dataset" / "src")
    args = parser.parse_args()

    candidates_by_frame, invalid = load_candidate_record_groups(args.candidates)
    if invalid:
        raise ValueError(f"{args.candidates}: invalid candidate records: {invalid[:5]}")
    candidate_frame_names = {frame_name for frame_name, records in candidates_by_frame.items() if records}
    frame_iter = load_preference_frames(
        args.val_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=args.selector_features in {"camera_contextual", "image_contextual"},
    )
    if args.max_preference_frames is not None:
        frame_iter = islice(frame_iter, args.max_preference_frames)
    frames = [frame for frame in frame_iter if frame.frame_name in candidate_frame_names]
    if not frames:
        raise ValueError("no validation preference frames matched candidate records")
    scorer = local_rfs_score if args.rfs_backend == "local" else load_official_rfs_scorer(args.waymo_src)
    rows = scored_ranker_rows(frames, candidates_by_frame, scorer=scorer)
    ranker = fit_contextual_ranker(
        rows,
        ridge=args.selector_ridge,
        target_mode=args.selector_target,
        feature_mode=args.selector_features,
    )
    payload = {
        "model_type": "wod_contextual_ranker_v1",
        "numeric_features": ranker.numeric_features,
        "candidate_names": ranker.candidate_names,
        "candidate_families": ranker.candidate_families,
        "feature_mean": list(ranker.feature_mean),
        "feature_scale": list(ranker.feature_scale),
        "weights": list(ranker.weights),
        "bias": ranker.bias,
        "selector_ridge": float(args.selector_ridge),
        "selector_target": args.selector_target,
        "selector_features": args.selector_features,
        "score_backend": "local_rfs_metric" if args.rfs_backend == "local" else "official_waymo_rfs",
        "frames": len({row["frame_name"] for row in rows}),
        "candidate_rows": len(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"weights"}}, indent=2))
    print(f"saved={args.output}")
    return 0


def scored_ranker_rows(
    frames: list[WodE2EPreferenceFrame],
    candidates_by_frame: dict[str, list[CandidateRecord]],
    *,
    scorer,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for frame in frames:
        for candidate in candidates_by_frame.get(frame.frame_name, []):
            row = candidate_ranker_row(
                frame=frame,
                trajectory=candidate.trajectory,
                candidate_name=candidate.candidate_name,
                candidate_index=candidate.candidate_index,
                source=candidate.source,
            )
            row["rfs_score"] = float(scorer(candidate, frame))
            rows.append(row)
    if not rows:
        raise ValueError("no ranker rows were scored")
    return rows


def fit_contextual_ranker(
    rows: list[dict[str, Any]],
    *,
    ridge: float,
    target_mode: str = "frame_delta",
    feature_mode: str = "contextual",
) -> WodPreferenceRanker:
    numeric_features = selector_numeric_features(feature_mode)
    candidate_names = sorted({str(row["candidate_name"]) for row in rows})
    candidate_families = sorted({str(row["features"].get("candidate_family", row["candidate_name"])) for row in rows})
    x = np.asarray(
        [raw_features(row, numeric_features, candidate_names, candidate_families) for row in rows],
        dtype=np.float64,
    )
    y = selector_targets(rows, target_mode)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean) / scale
    design = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return WodPreferenceRanker(
        numeric_features=numeric_features,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        weights=weights[1:].tolist(),
        bias=float(weights[0]),
    )


def selector_targets(rows: list[dict[str, Any]], target_mode: str) -> np.ndarray:
    scores = np.asarray([float(row["rfs_score"]) for row in rows], dtype=np.float64)
    if target_mode == "absolute":
        return scores
    if target_mode == "oracle_binary":
        best_by_frame: dict[str, float] = {}
        for row, score in zip(rows, scores):
            frame_name = str(row["frame_name"])
            best_by_frame[frame_name] = max(float(score), best_by_frame.get(frame_name, float("-inf")))
        return np.asarray(
            [1.0 if float(score) == best_by_frame[str(row["frame_name"])] else 0.0 for row, score in zip(rows, scores)],
            dtype=np.float64,
        )
    if target_mode not in {"frame_delta", "frame_zscore"}:
        raise ValueError(f"unsupported selector target mode: {target_mode}")
    scores_by_frame: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        scores_by_frame[str(row["frame_name"])].append(float(score))
    mean_by_frame = {
        frame_name: sum(frame_scores) / len(frame_scores)
        for frame_name, frame_scores in scores_by_frame.items()
    }
    scale_by_frame = {
        frame_name: max(float(np.std(np.asarray(frame_scores, dtype=np.float64))), 1.0e-8)
        for frame_name, frame_scores in scores_by_frame.items()
    }
    targets = [float(score) - mean_by_frame[str(row["frame_name"])] for row, score in zip(rows, scores)]
    if target_mode == "frame_zscore":
        targets = [target / scale_by_frame[str(row["frame_name"])] for row, target in zip(rows, targets)]
    return np.asarray(targets, dtype=np.float64)


if __name__ == "__main__":
    raise SystemExit(main())
