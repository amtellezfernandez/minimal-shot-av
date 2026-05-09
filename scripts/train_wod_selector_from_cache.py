#!/usr/bin/env python3
"""Train a ridge preference selector on val479 frame cache and save as WodPreferenceRanker.

Uses the same candidate_ranker_row feature extraction as evaluate_wod_trajectory_model_cv.py,
and the same contextual features + frame_delta target as the 7.88 fastkin-gate policy.
Saves a WodPreferenceRanker JSON loadable by score_wod_candidates_with_ranker.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.learned_trajectory_model import RidgeTrajectoryModel
from minimal_shot_av.model.rfs_metric import ManeuverCandidate, RfsReference, score_candidate
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame
from minimal_shot_av.model.wod_ranker import (
    CONTEXTUAL_NUMERIC_FEATURES,
    WodPreferenceRanker,
    candidate_ranker_row,
    selector_numeric_features,
)


def _cv_traj_for_frame(past: list) -> list[tuple[float, float]]:
    if len(past) < 2:
        return [(0.0, 0.0)] * 20
    dx = past[-1][0] - past[-2][0]
    dy = past[-1][1] - past[-2][1]
    return [(dx * (i + 1), dy * (i + 1)) for i in range(20)]


def build_training_rows(
    val_frames: list[dict[str, Any]],
    base_model: RidgeTrajectoryModel,
    temporal_model: RidgeTrajectoryModel,
    feature_mode: str = "contextual",
) -> tuple[list[dict], list[float]]:
    numeric_features = selector_numeric_features(feature_mode)
    rows: list[dict] = []
    labels: list[float] = []

    for fd in val_frames:
        refs_raw = fd.get("references", [])
        if not refs_raw:
            continue

        past = [tuple(pt) for pt in fd["past_trajectory"]]
        intent = int(fd["intent"])
        init_speed = float(fd["init_speed_mps"])
        refs = [
            RfsReference(label=r["label"], trajectory=[tuple(pt) for pt in r["trajectory"]], score=float(r["score"]))
            for r in refs_raw
        ]

        frame = WodE2EPreferenceFrame(
            frame_name=fd["frame_name"],
            past_trajectory=past,
            future_trajectory=[tuple(pt) for pt in fd.get("future_trajectory", [])],
            intent=intent,
            init_speed_mps=init_speed,
            references=refs,
            camera_images=[],
        )

        # Constant velocity for delta target
        cv_traj = _cv_traj_for_frame(past)
        cv_rfs = float(score_candidate(ManeuverCandidate("kinematic", cv_traj), refs, init_speed).combined_score)

        # Generate kinematic + learned + temporal candidates
        candidates: list[tuple[str, str, int, list]] = []
        cindex = 0
        for src, cname, traj in [("kinematic", "constant_velocity", cv_traj)]:
            candidates.append((src, cname, cindex, traj))
            cindex += 1
        for cname, traj in base_model.candidate_trajectories(past, intent=intent, init_speed_mps=init_speed, max_residual_modes=3):
            candidates.append(("learned", cname, cindex, list(traj)))
            cindex += 1
        for cname, traj in temporal_model.candidate_trajectories(past, intent=intent, init_speed_mps=init_speed, max_residual_modes=3):
            candidates.append(("temporal", f"temporal_{cname}", cindex, list(traj)))
            cindex += 1

        for src, cname, cidx, traj in candidates:
            rfs = float(score_candidate(ManeuverCandidate(src, traj), refs, init_speed).combined_score)
            gain = rfs - cv_rfs  # frame_delta target
            row = candidate_ranker_row(
                frame=frame,
                trajectory=traj,
                candidate_name=cname,
                candidate_index=cidx,
                source=src,
            )
            rows.append(row)
            labels.append(gain)

    return rows, labels


def _build_X(rows: list[dict], numeric_features: list[str]) -> np.ndarray:
    X = []
    for row in rows:
        feats = row["features"]
        X.append([float(feats.get(k, 0.0)) for k in numeric_features])
    return np.array(X, dtype=np.float64)


def train_ridge_ranker(
    rows: list[dict],
    labels: list[float],
    numeric_features: list[str],
    ridge: float,
) -> WodPreferenceRanker:
    candidate_names = sorted({row["candidate_name"] for row in rows})
    candidate_families: list[str] = []  # keep empty for simplicity

    # Build full feature matrix (numeric + candidate one-hots)
    X_numeric = _build_X(rows, numeric_features)
    # candidate one-hots
    cname_to_idx = {cn: i for i, cn in enumerate(candidate_names)}
    X_onehot = np.zeros((len(rows), len(candidate_names)), dtype=np.float64)
    for i, row in enumerate(rows):
        cidx = cname_to_idx.get(row["candidate_name"])
        if cidx is not None:
            X_onehot[i, cidx] = 1.0
    X = np.concatenate([X_numeric, X_onehot], axis=1)
    y = np.array(labels, dtype=np.float64)

    # Standardize
    feature_mean = X.mean(axis=0)
    feature_scale = X.std(axis=0)
    feature_scale[feature_scale < 1e-12] = 1.0
    X_norm = (X - feature_mean) / feature_scale

    # Ridge regression
    n_feat = X_norm.shape[1]
    A = X_norm.T @ X_norm + ridge * np.eye(n_feat)
    b = X_norm.T @ y
    weights = np.linalg.solve(A, b)

    # Bias = mean(y) - mean(X_norm) @ weights (should be ~mean(y) since X_norm is centered)
    bias = float(np.mean(y) - feature_mean @ (weights / feature_scale))

    return WodPreferenceRanker(
        numeric_features=numeric_features,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=feature_mean.tolist(),
        feature_scale=feature_scale.tolist(),
        weights=weights.tolist(),
        bias=bias,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Train ridge preference selector on val479 and save ranker.")
    parser.add_argument("--val-cache", type=Path,
                        default=ROOT / "artifacts" / "wod_preference_frames_val479.json")
    parser.add_argument("--base-model", type=Path,
                        default=ROOT / "artifacts" / "wod_ridge_trajectory_model.json")
    parser.add_argument("--temporal-model", type=Path,
                        default=ROOT / "artifacts" / "wod_ridge_temporal_model.json")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts" / "wod_selector_ranker_val479.json")
    parser.add_argument("--feature-mode", default="contextual")
    parser.add_argument("--ridge", type=float, default=175.0)
    args = parser.parse_args()

    print("Loading val479 frame cache...", flush=True)
    cache = json.loads(args.val_cache.read_text(encoding="utf-8"))
    val_frames = cache["frames"]
    print(f"  {len(val_frames)} frames", flush=True)

    base_model = RidgeTrajectoryModel.load(args.base_model)
    temporal_model = RidgeTrajectoryModel.load(args.temporal_model)

    print("Building training rows...", flush=True)
    rows, labels = build_training_rows(val_frames, base_model, temporal_model, feature_mode=args.feature_mode)
    print(f"  {len(rows)} rows", flush=True)

    numeric_features = selector_numeric_features(args.feature_mode)
    print(f"Training ridge ranker (ridge={args.ridge}, features={len(numeric_features)} + one-hots)...", flush=True)
    ranker = train_ridge_ranker(rows, labels, numeric_features, args.ridge)

    # Save as WodPreferenceRanker JSON
    payload = {
        "numeric_features": ranker.numeric_features,
        "candidate_names": ranker.candidate_names,
        "candidate_families": ranker.candidate_families,
        "feature_mean": list(ranker.feature_mean),
        "feature_scale": list(ranker.feature_scale),
        "weights": list(ranker.weights),
        "bias": ranker.bias,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved ranker → {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
