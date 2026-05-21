#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.wod_ranker import DEFAULT_NUMERIC_FEATURES, WodPreferenceRanker, raw_features


NUMERIC_FEATURES = DEFAULT_NUMERIC_FEATURES


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a lightweight WOD-E2E preference ranker from RFS-labeled JSONL rows."
    )
    parser.add_argument("--input", type=Path, default=Path("artifacts/wod_preference_candidates.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/wod_preference_ranker.json"))
    parser.add_argument("--ridge", type=float, default=1.0)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--folds",
        type=int,
        default=1,
        help="Run segment-grouped K-fold evaluation before saving the final model.",
    )
    parser.add_argument(
        "--baseline-candidate",
        default="logged_future",
        help="Candidate name used as the non-ranker baseline; falls back to candidate_index=0 when absent.",
    )
    args = parser.parse_args()

    rows = _load_rows(args.input)
    candidate_names = sorted({str(row["candidate_name"]) for row in rows})
    candidate_families = sorted({str(row["features"].get("candidate_family", row["candidate_name"])) for row in rows})
    cv_metrics = None
    if args.folds > 1:
        cv_metrics = _cross_validate(
            rows,
            candidate_names,
            candidate_families,
            folds=args.folds,
            seed=args.seed,
            ridge=args.ridge,
            baseline_candidate_name=args.baseline_candidate,
        )
        print(
            "cv "
            f"folds={cv_metrics['folds']} frames={cv_metrics['frames']} "
            f"selected_mean={cv_metrics['selected_mean_rfs']:.3f} "
            f"baseline_mean={cv_metrics['baseline_mean_rfs']:.3f} "
            f"oracle_mean={cv_metrics['oracle_mean_rfs']:.3f} "
            f"top1={cv_metrics['top1_oracle_match_rate']:.3f}"
        )

    train_frames, test_frames = _split_frames(rows, args.test_fraction, seed=args.seed)
    train_rows = [row for row in rows if row["frame_name"] in train_frames]
    test_rows = [row for row in rows if row["frame_name"] in test_frames]
    if not train_rows or not test_rows:
        raise ValueError("need at least one train and one test frame")

    split_model = _fit_ridge(train_rows, candidate_names, candidate_families, ridge=args.ridge)
    train_metrics = _evaluate(
        train_rows,
        split_model,
        candidate_names,
        candidate_families,
        baseline_candidate_name=args.baseline_candidate,
    )
    test_metrics = _evaluate(
        test_rows,
        split_model,
        candidate_names,
        candidate_families,
        baseline_candidate_name=args.baseline_candidate,
    )
    final_model = _fit_ridge(rows, candidate_names, candidate_families, ridge=args.ridge)

    payload = {
        "model_type": "ridge_regression_rfs_ranker_v1",
        "numeric_features": NUMERIC_FEATURES,
        "candidate_names": candidate_names,
        "candidate_families": candidate_families,
        "ridge": args.ridge,
        "seed": args.seed,
        "folds": args.folds,
        "baseline_candidate_name": args.baseline_candidate,
        "feature_mean": final_model["mean"].tolist(),
        "feature_scale": final_model["scale"].tolist(),
        "weights": final_model["weights"].tolist(),
        "bias": float(final_model["bias"]),
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "cv_metrics": cv_metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(
        "train "
        f"frames={train_metrics['frames']} selected_mean={train_metrics['selected_mean_rfs']:.3f} "
        f"baseline_mean={train_metrics['baseline_mean_rfs']:.3f} "
        f"oracle_mean={train_metrics['oracle_mean_rfs']:.3f} "
        f"top1={train_metrics['top1_oracle_match_rate']:.3f}"
    )
    print(
        "test "
        f"frames={test_metrics['frames']} selected_mean={test_metrics['selected_mean_rfs']:.3f} "
        f"baseline_mean={test_metrics['baseline_mean_rfs']:.3f} "
        f"oracle_mean={test_metrics['oracle_mean_rfs']:.3f} "
        f"top1={test_metrics['top1_oracle_match_rate']:.3f}"
    )
    print(f"saved={args.output}")
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def _split_frames(rows: list[dict[str, Any]], test_fraction: float, *, seed: int) -> tuple[set[str], set[str]]:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("--test-fraction must be between 0 and 1")
    frames_by_segment: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        frame_name = str(row["frame_name"])
        frames_by_segment[_segment_id(frame_name)].add(frame_name)

    segments = sorted(
        frames_by_segment,
        key=lambda segment: _stable_hash_key(segment, seed),
    )
    test_segment_count = max(1, int(round(len(segments) * test_fraction)))
    if test_segment_count >= len(segments):
        test_segment_count = len(segments) - 1

    test_segments = set(segments[:test_segment_count])
    test_frames = {
        frame
        for segment in test_segments
        for frame in frames_by_segment[segment]
    }
    train_frames = {
        frame
        for segment, segment_frames in frames_by_segment.items()
        if segment not in test_segments
        for frame in segment_frames
    }
    return train_frames, test_frames


def _cross_validate(
    rows: list[dict[str, Any]],
    candidate_names: list[str],
    candidate_families: list[str],
    *,
    folds: int,
    seed: int,
    ridge: float,
    baseline_candidate_name: str = "logged_future",
) -> dict[str, float | int]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    frames_by_segment: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        frame_name = str(row["frame_name"])
        frames_by_segment[_segment_id(frame_name)].add(frame_name)
    segments = sorted(frames_by_segment, key=lambda segment: _stable_hash_key(segment, seed))
    if folds > len(segments):
        raise ValueError(f"--folds={folds} exceeds segment count {len(segments)}")

    fold_metrics: list[dict[str, float | int]] = []
    for fold_index in range(folds):
        test_segments = set(segments[fold_index::folds])
        test_frames = {
            frame
            for segment in test_segments
            for frame in frames_by_segment[segment]
        }
        train_rows = [row for row in rows if row["frame_name"] not in test_frames]
        test_rows = [row for row in rows if row["frame_name"] in test_frames]
        model = _fit_ridge(train_rows, candidate_names, candidate_families, ridge=ridge)
        metrics = _evaluate(
            test_rows,
            model,
            candidate_names,
            candidate_families,
            baseline_candidate_name=baseline_candidate_name,
        )
        fold_metrics.append(metrics)

    total_frames = sum(int(metrics["frames"]) for metrics in fold_metrics)
    return {
        "folds": folds,
        "frames": total_frames,
        "selected_mean_rfs": _weighted_mean(fold_metrics, "selected_mean_rfs"),
        "baseline_mean_rfs": _weighted_mean(fold_metrics, "baseline_mean_rfs"),
        "logged_mean_rfs": _weighted_mean(fold_metrics, "logged_mean_rfs"),
        "oracle_mean_rfs": _weighted_mean(fold_metrics, "oracle_mean_rfs"),
        "mean_regret": _weighted_mean(fold_metrics, "mean_regret"),
        "top1_oracle_match_rate": _weighted_mean(fold_metrics, "top1_oracle_match_rate"),
    }


def _fit_ridge(
    rows: list[dict[str, Any]],
    candidate_names: list[str],
    candidate_families: list[str],
    *,
    ridge: float,
) -> dict[str, Any]:
    x = np.array([_raw_features(row, candidate_names, candidate_families) for row in rows], dtype=np.float64)
    y = np.array([float(row["rfs_score"]) for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean) / scale
    design = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "bias": weights[0],
        "weights": weights[1:],
        "mean": mean,
        "scale": scale,
    }


def _evaluate(
    rows: list[dict[str, Any]],
    model: dict[str, Any],
    candidate_names: list[str],
    candidate_families: list[str],
    *,
    baseline_candidate_name: str = "logged_future",
) -> dict[str, float | int]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[str(row["frame_name"])].append(row)

    selected_scores: list[float] = []
    baseline_scores: list[float] = []
    oracle_scores: list[float] = []
    top1_matches = 0

    for frame_rows in by_frame.values():
        ranker = _ranker_from_model(model, candidate_names, candidate_families)
        selected = ranker.select_row(frame_rows)
        baseline = _baseline_row(frame_rows, baseline_candidate_name)
        oracle = max(frame_rows, key=lambda row: float(row["rfs_score"]))

        selected_score = float(selected["rfs_score"])
        oracle_score = float(oracle["rfs_score"])
        selected_scores.append(selected_score)
        baseline_scores.append(float(baseline["rfs_score"]))
        oracle_scores.append(oracle_score)
        if selected_score == oracle_score:
            top1_matches += 1

    count = len(selected_scores)
    return {
        "frames": count,
        "selected_mean_rfs": float(sum(selected_scores) / count),
        "baseline_mean_rfs": float(sum(baseline_scores) / count),
        "logged_mean_rfs": float(sum(baseline_scores) / count),
        "oracle_mean_rfs": float(sum(oracle_scores) / count),
        "mean_regret": float(sum(o - s for o, s in zip(oracle_scores, selected_scores)) / count),
        "top1_oracle_match_rate": float(top1_matches / count),
    }


def _baseline_row(frame_rows: list[dict[str, Any]], baseline_candidate_name: str) -> dict[str, Any]:
    for row in frame_rows:
        if str(row["candidate_name"]) == baseline_candidate_name:
            return row
    return min(frame_rows, key=lambda row: int(row.get("candidate_index", 0)))


def predict(
    row: dict[str, Any],
    model: dict[str, Any],
    candidate_names: list[str],
    candidate_families: list[str] | None = None,
) -> float:
    return _ranker_from_model(model, candidate_names, candidate_families or []).predict_row(row)


def _raw_features(
    row: dict[str, Any],
    candidate_names: list[str],
    candidate_families: list[str] | None = None,
) -> list[float]:
    return raw_features(row, NUMERIC_FEATURES, candidate_names, candidate_families or [])


def _ranker_from_model(
    model: dict[str, Any],
    candidate_names: list[str],
    candidate_families: list[str],
) -> WodPreferenceRanker:
    return WodPreferenceRanker(
        numeric_features=NUMERIC_FEATURES,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=model["mean"],
        feature_scale=model["scale"],
        weights=model["weights"],
        bias=float(model["bias"]),
    )


def _segment_id(frame_name: str) -> str:
    head, separator, tail = frame_name.rpartition("-")
    if separator and tail.isdigit():
        return head
    return frame_name


def _stable_hash_key(value: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
    return digest


def _weighted_mean(metrics: list[dict[str, float | int]], key: str) -> float:
    total_frames = sum(int(metric["frames"]) for metric in metrics)
    return float(sum(float(metric[key]) * int(metric["frames"]) for metric in metrics) / total_frames)


if __name__ == "__main__":
    raise SystemExit(main())
