#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.anchor_trajectory_model import fit_anchor_residual_trajectory_model
from minimal_shot_av.model.rfs_metric import ManeuverCandidate, score_candidate
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames
from minimal_shot_av.model.wod_preference import trajectory_features
from minimal_shot_av.model.wod_ranker import DEFAULT_NUMERIC_FEATURES, WodPreferenceRanker, raw_features


ANCHOR_NUMERIC_FEATURES = [*DEFAULT_NUMERIC_FEATURES, "model_confidence"]


def main() -> int:
    parser = argparse.ArgumentParser(description="Segment-grouped CV for WOD-E2E anchor-residual proposals.")
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_anchor_trajectory_cv_local.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--anchors", type=int, required=True)
    parser.add_argument("--top-k", type=int, required=True)
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--anchor-iterations", type=int, default=25)
    parser.add_argument("--residual-modes-per-anchor", type=int, default=0)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    args = parser.parse_args()

    frames = list(
        load_preference_frames(
            args.val_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=False,
        )
    )
    if args.max_preference_frames is not None:
        frames = frames[: args.max_preference_frames]
    report = cross_validate_anchor_model(
        frames,
        folds=args.folds,
        seed=args.seed,
        anchor_count=args.anchors,
        top_k=args.top_k,
        ridge=args.ridge,
        anchor_iterations=args.anchor_iterations,
        residual_modes_per_anchor=args.residual_modes_per_anchor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "folds_detail"}, indent=2, sort_keys=True))
    print(f"wrote {args.output}")
    return 0


def cross_validate_anchor_model(
    frames: list[WodE2EPreferenceFrame],
    *,
    folds: int,
    seed: int,
    anchor_count: int,
    top_k: int,
    ridge: float,
    anchor_iterations: int,
    residual_modes_per_anchor: int,
) -> dict[str, object]:
    fold_frame_sets = _split_frame_names_by_segment(frames, folds=folds, seed=seed)
    fold_reports: list[dict[str, object]] = []
    for fold_index, test_names in enumerate(fold_frame_sets):
        train_frames = [frame for frame in frames if frame.frame_name not in test_names]
        test_frames = [frame for frame in frames if frame.frame_name in test_names]
        model = fit_anchor_residual_trajectory_model(
            train_frames,
            anchor_count=anchor_count,
            ridge=ridge,
            anchor_iterations=anchor_iterations,
            seed=seed + fold_index,
            residual_modes_per_anchor=residual_modes_per_anchor,
        )
        confidence_head = _fit_confidence_head(
            _candidate_rows(
                train_frames,
                model,
                top_k=top_k,
                residual_modes_per_anchor=residual_modes_per_anchor,
            ),
            ridge=ridge,
        )
        fold_reports.append(
            _evaluate_fold(
                test_frames,
                model,
                confidence_head,
                top_k=top_k,
                residual_modes_per_anchor=residual_modes_per_anchor,
                fold_index=fold_index,
            )
        )
    return {
        "benchmark_type": "segment_grouped_cross_validation",
        "score_backend": "local_rfs_metric",
        "frames": sum(int(fold["frames"]) for fold in fold_reports),
        "fold_count": folds,
        "anchors": anchor_count,
        "top_k": top_k,
        "ridge": float(ridge),
        "anchor_iterations": int(anchor_iterations),
        "residual_modes_per_anchor": int(residual_modes_per_anchor),
        "confidence_selected_mean_rfs": _weighted_mean(fold_reports, "confidence_selected_mean_rfs"),
        "rfs_head_selected_mean_rfs": _weighted_mean(fold_reports, "rfs_head_selected_mean_rfs"),
        "anchor_oracle_mean_rfs": _weighted_mean(fold_reports, "anchor_oracle_mean_rfs"),
        "confidence_regret_to_anchor_oracle": _weighted_delta(
            fold_reports,
            "anchor_oracle_mean_rfs",
            "confidence_selected_mean_rfs",
        ),
        "rfs_head_regret_to_anchor_oracle": _weighted_delta(
            fold_reports,
            "anchor_oracle_mean_rfs",
            "rfs_head_selected_mean_rfs",
        ),
        "confidence_top1_oracle_match_rate": _weighted_mean(fold_reports, "confidence_top1_oracle_match_rate"),
        "rfs_head_top1_oracle_match_rate": _weighted_mean(fold_reports, "rfs_head_top1_oracle_match_rate"),
        "folds_detail": fold_reports,
    }


def _evaluate_fold(
    frames: list[WodE2EPreferenceFrame],
    model,
    confidence_head: WodPreferenceRanker,
    *,
    top_k: int,
    residual_modes_per_anchor: int,
    fold_index: int,
) -> dict[str, object]:
    selected_scores: list[float] = []
    rfs_head_scores: list[float] = []
    oracle_scores: list[float] = []
    top1_matches = 0
    rfs_head_top1_matches = 0
    for frame in frames:
        rows = _candidate_rows(
            [frame],
            model,
            top_k=top_k,
            residual_modes_per_anchor=residual_modes_per_anchor,
        )
        scores = [float(row["rfs_score"]) for row in rows]
        selected_scores.append(float(rows[0]["rfs_score"]))
        rfs_head_selected = confidence_head.select_row(rows)
        rfs_head_scores.append(float(rfs_head_selected["rfs_score"]))
        oracle = max(float(row["rfs_score"]) for row in rows)
        oracle_scores.append(oracle)
        if float(rows[0]["rfs_score"]) == oracle:
            top1_matches += 1
        if float(rfs_head_selected["rfs_score"]) == oracle:
            rfs_head_top1_matches += 1
    return {
        "fold_index": fold_index,
        "frames": len(frames),
        "confidence_selected_mean_rfs": _mean(selected_scores),
        "rfs_head_selected_mean_rfs": _mean(rfs_head_scores),
        "anchor_oracle_mean_rfs": _mean(oracle_scores),
        "confidence_top1_oracle_match_rate": float(top1_matches / len(frames)),
        "rfs_head_top1_oracle_match_rate": float(rfs_head_top1_matches / len(frames)),
    }


def _candidate_rows(
    frames: list[WodE2EPreferenceFrame],
    model,
    *,
    top_k: int,
    residual_modes_per_anchor: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        candidates = model.candidate_trajectories(
            frame.past_trajectory,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
            top_k=top_k,
            residual_modes_per_anchor=residual_modes_per_anchor,
        )
        for candidate_index, (candidate_name, trajectory, confidence) in enumerate(candidates):
            features = trajectory_features(
                trajectory,
                candidate_name=candidate_name,
                intent=frame.intent,
                init_speed_mps=frame.init_speed_mps,
            )
            features["model_confidence"] = float(confidence)
            rows.append(
                {
                    "frame_name": frame.frame_name,
                    "candidate_name": candidate_name,
                    "candidate_index": candidate_index,
                    "rfs_score": float(
                        score_candidate(
                            ManeuverCandidate(candidate_name, trajectory),
                            frame.references,
                            frame.init_speed_mps,
                        ).combined_score
                    ),
                    "features": features,
                }
            )
    return rows


def _fit_confidence_head(rows: list[dict[str, object]], *, ridge: float) -> WodPreferenceRanker:
    candidate_names = sorted({str(row["candidate_name"]) for row in rows})
    candidate_families = sorted({str(row["features"].get("candidate_family", row["candidate_name"])) for row in rows})
    x = np.asarray(
        [raw_features(row, ANCHOR_NUMERIC_FEATURES, candidate_names, candidate_families) for row in rows],
        dtype=np.float64,
    )
    y = _frame_delta_targets(rows)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean) / scale
    design = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return WodPreferenceRanker(
        numeric_features=ANCHOR_NUMERIC_FEATURES,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        weights=weights[1:].tolist(),
        bias=float(weights[0]),
    )


def _frame_delta_targets(rows: list[dict[str, object]]) -> np.ndarray:
    scores_by_frame: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        scores_by_frame[str(row["frame_name"])].append(float(row["rfs_score"]))
    mean_by_frame = {
        frame_name: sum(scores) / len(scores)
        for frame_name, scores in scores_by_frame.items()
    }
    return np.asarray(
        [float(row["rfs_score"]) - mean_by_frame[str(row["frame_name"])] for row in rows],
        dtype=np.float64,
    )


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(sum(values) / len(values))


def _weighted_mean(fold_reports: list[dict[str, object]], key: str) -> float:
    total_frames = sum(int(fold["frames"]) for fold in fold_reports)
    return float(sum(float(fold[key]) * int(fold["frames"]) for fold in fold_reports) / total_frames)


def _weighted_delta(fold_reports: list[dict[str, object]], left_key: str, right_key: str) -> float:
    return _weighted_mean(fold_reports, left_key) - _weighted_mean(fold_reports, right_key)


def _split_frame_names_by_segment(
    frames: list[WodE2EPreferenceFrame],
    *,
    folds: int,
    seed: int,
) -> list[set[str]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    frames_by_segment: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        frames_by_segment[_segment_id(frame.frame_name)].add(frame.frame_name)
    segments = sorted(frames_by_segment, key=lambda segment: _stable_hash_key(segment, seed))
    if folds > len(segments):
        raise ValueError(f"--folds={folds} exceeds segment count {len(segments)}")
    return [
        {frame for segment in segments[fold_index::folds] for frame in frames_by_segment[segment]}
        for fold_index in range(folds)
    ]


def _segment_id(frame_name: str) -> str:
    head, separator, tail = frame_name.rpartition("-")
    if separator and tail.isdigit():
        return head
    return frame_name


def _stable_hash_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
