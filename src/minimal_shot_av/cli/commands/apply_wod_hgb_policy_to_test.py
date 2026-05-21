#!/usr/bin/env python3
"""Train a direct HGB selector on val479 preferences, apply to test candidates.

Pipeline:
  1. Load val479 frame cache (has preferences/labels).
  2. Generate kinematic + learned candidates for each val frame using saved models.
  3. Score each candidate vs val preferences using local RFS.
  4. Build training rows: label = gain vs constant_velocity (> 0 → positive).
  5. Train sklearn HistGradientBoosting classifier.
  6. Apply to test merged candidates JSONL to output scores.
  7. Use write_wod_e2e_submission.py --score-field hgb_score to select best per frame.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.learned_trajectory_model import RidgeTrajectoryModel
from minimal_shot_av.model.rfs_metric import ManeuverCandidate, RfsReference, score_candidate
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame


# ── feature extraction ────────────────────────────────────────────────────────

def _traj_features(traj: list[tuple[float, float]]) -> dict[str, float]:
    xs = [x for x, y in traj]
    ys = [y for x, y in traj]
    n = len(traj)
    if n == 0:
        return {}
    dt = 0.25  # 4 Hz

    # distances per step
    step_dists = [
        ((xs[i] - xs[i - 1]) ** 2 + (ys[i] - ys[i - 1]) ** 2) ** 0.5
        for i in range(1, n)
    ]
    total_dist = sum(step_dists) if step_dists else 0.0
    mean_speed = total_dist / (n * dt) if n > 0 else 0.0
    endpoint_dist = (xs[-1] ** 2 + ys[-1] ** 2) ** 0.5
    # Lateral displacement at step 3 (0.75s proxy) and step 12 (3s)
    lat_3s = ys[min(11, n - 1)]
    lat_end = ys[-1]
    forward_end = xs[-1]
    max_lat = max(abs(y) for y in ys)

    return {
        "endpoint_dist": endpoint_dist,
        "forward_end": forward_end,
        "lat_end": lat_end,
        "lat_3s": lat_3s,
        "max_lat": max_lat,
        "total_dist": total_dist,
        "mean_speed": mean_speed,
        "final_speed": step_dists[-1] / dt if step_dists else 0.0,
        "mean_step_dist": total_dist / len(step_dists) if step_dists else 0.0,
    }


def _candidate_feats(
    traj: list[tuple[float, float]],
    source: str,
    candidate_name: str,
    baseline_feats: dict[str, float],
) -> list[float]:
    tf = _traj_features(traj)
    # Source one-hots
    sources = ["kinematic", "learned", "temporal", "anchor", "scene"]
    src_lower = source.lower()
    cname_lower = candidate_name.lower()
    if "temporal" in src_lower or cname_lower.startswith("temporal_"):
        src_label = "temporal"
    elif "kinematic" in src_lower or cname_lower in {"constant_velocity", "constant_acceleration", "hold_position"}:
        src_label = "kinematic"
    elif "scene" in src_lower or cname_lower.startswith("scene_"):
        src_label = "scene"
    elif "anchor" in src_lower or cname_lower.startswith("anchor_"):
        src_label = "anchor"
    else:
        src_label = "learned"
    src_vec = [1.0 if src_label == s else 0.0 for s in sources]

    tf_keys = ["endpoint_dist", "forward_end", "lat_end", "lat_3s", "max_lat",
               "total_dist", "mean_speed", "final_speed", "mean_step_dist"]
    abs_feats = [tf.get(k, 0.0) for k in tf_keys]
    # Delta vs constant_velocity baseline
    delta_feats = [tf.get(k, 0.0) - baseline_feats.get(k, 0.0) for k in tf_keys]

    return src_vec + abs_feats + delta_feats


def _cv_traj(past_trajectory: list[tuple[float, float]], init_speed_mps: float) -> list[tuple[float, float]]:
    if not past_trajectory:
        dx, dy = 0.0, 0.0
    else:
        dx = past_trajectory[-1][0] - past_trajectory[-2][0] if len(past_trajectory) >= 2 else 0.0
        dy = past_trajectory[-1][1] - past_trajectory[-2][1] if len(past_trajectory) >= 2 else 0.0
    # constant velocity: step forward by (dx, dy) each time step
    pts = []
    x, y = 0.0, 0.0
    for _ in range(20):
        x += dx
        y += dy
        pts.append((x, y))
    return pts


# ── build val training data ────────────────────────────────────────────────────

def build_val_training_data(
    val_frames: list[dict[str, Any]],
    base_model: RidgeTrajectoryModel,
    temporal_model: RidgeTrajectoryModel,
) -> tuple[np.ndarray, np.ndarray]:
    X_rows: list[list[float]] = []
    y_rows: list[float] = []

    for fd in val_frames:
        refs = fd.get("references", [])
        if not refs:
            continue
        past = [tuple(pt) for pt in fd["past_trajectory"]]
        intent = int(fd["intent"])
        init_speed = float(fd["init_speed_mps"])

        wod_refs = [
            RfsReference(
                label=r["label"],
                trajectory=[tuple(pt) for pt in r["trajectory"]],
                score=float(r["score"]),
            )
            for r in refs
        ]

        frame = WodE2EPreferenceFrame(
            frame_name=fd["frame_name"],
            past_trajectory=[tuple(pt) for pt in fd["past_trajectory"]],
            future_trajectory=[tuple(pt) for pt in fd.get("future_trajectory", [])],
            intent=intent,
            init_speed_mps=init_speed,
            references=wod_refs,
            camera_images=[],
        )

        # Constant velocity trajectory and score
        cv_traj = _cv_traj(past, init_speed)
        cv_record = ManeuverCandidate("kinematic", cv_traj)
        cv_rfs = float(score_candidate(cv_record, frame.references, frame.init_speed_mps).combined_score)
        cv_baseline_feats = _traj_features(cv_traj)

        # Generate base + temporal candidates
        all_candidates: list[tuple[str, str, list[tuple[float, float]]]] = []
        for name, traj in base_model.candidate_trajectories(
            past, intent=intent, init_speed_mps=init_speed, max_residual_modes=3
        ):
            all_candidates.append(("learned", name, list(traj)))
        for name, traj in temporal_model.candidate_trajectories(
            past, intent=intent, init_speed_mps=init_speed, max_residual_modes=3
        ):
            all_candidates.append(("temporal", f"temporal_{name}", list(traj)))
        # Also include kinematic constant_velocity as a candidate
        all_candidates.append(("kinematic", "constant_velocity", cv_traj))

        for src, cname, traj in all_candidates:
            record = ManeuverCandidate(src, traj)
            rfs = float(score_candidate(record, frame.references, frame.init_speed_mps).combined_score)
            gain = rfs - cv_rfs
            feats = _candidate_feats(traj, src, cname, cv_baseline_feats)
            X_rows.append(feats)
            y_rows.append(gain)

    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.float32)


# ── score test candidates ──────────────────────────────────────────────────────

def score_test_candidates(
    test_jsonl: Path,
    hgb,
    output: Path,
    *,
    chunk_frames: int = 5000,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0

    # Stream through JSONL, accumulate per-frame, score in chunks.
    by_frame: dict[str, list[dict[str, Any]]] = {}
    cv_feats_cache: dict[str, dict[str, float]] = {}

    def _flush_chunk(by_frame_chunk: dict[str, list[dict[str, Any]]]) -> tuple[list[dict], np.ndarray]:
        all_rows: list[dict[str, Any]] = []
        all_feats: list[list[float]] = []
        for fn, rows in by_frame_chunk.items():
            cv_f = cv_feats_cache.get(fn, {})
            for row in rows:
                traj = [(pt[0], pt[1]) for pt in row["trajectory_20wp_4hz"]]
                feats = _candidate_feats(traj, str(row.get("source", "")),
                                         str(row.get("candidate_name", "")), cv_f)
                all_rows.append(row)
                all_feats.append(feats)
        scores = hgb.predict(np.array(all_feats, dtype=np.float32)) if all_feats else np.array([])
        return all_rows, scores

    with output.open("w", encoding="utf-8") as f_out:
        for line in test_jsonl.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            fn = str(row["frame_name"])
            by_frame.setdefault(fn, []).append(row)
            # Cache constant_velocity baseline features per frame
            if row.get("candidate_name") == "constant_velocity" and fn not in cv_feats_cache:
                cv_traj = [(pt[0], pt[1]) for pt in row["trajectory_20wp_4hz"]]
                cv_feats_cache[fn] = _traj_features(cv_traj)

            if len(by_frame) >= chunk_frames:
                rows_out, scores = _flush_chunk(by_frame)
                for r, s in zip(rows_out, scores):
                    f_out.write(json.dumps({**r, "hgb_score": float(s)}, separators=(",", ":")) + "\n")
                count += len(rows_out)
                by_frame = {}

        # Final chunk
        if by_frame:
            rows_out, scores = _flush_chunk(by_frame)
            for r, s in zip(rows_out, scores):
                f_out.write(json.dumps({**r, "hgb_score": float(s)}, separators=(",", ":")) + "\n")
            count += len(rows_out)

    return count


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Train HGB selector on val479, apply to test candidates.")
    parser.add_argument("--val-cache", type=Path,
                        default=ROOT / "artifacts" / "wod_preference_frames_val479.json")
    parser.add_argument("--base-model", type=Path,
                        default=ROOT / "artifacts" / "wod_ridge_trajectory_model.json")
    parser.add_argument("--temporal-model", type=Path,
                        default=ROOT / "artifacts" / "wod_ridge_temporal_model.json")
    parser.add_argument("--test-candidates", type=Path, required=True,
                        help="Merged test JSONL (kinematic + learned).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL with hgb_score field.")
    parser.add_argument("--hgb-max-iter", type=int, default=200)
    parser.add_argument("--hgb-learning-rate", type=float, default=0.08)
    parser.add_argument("--hgb-max-leaf-nodes", type=int, default=24)
    args = parser.parse_args()

    print("Loading val479 frame cache...", flush=True)
    cache = json.loads(args.val_cache.read_text(encoding="utf-8"))
    val_frames = cache["frames"]
    print(f"  {len(val_frames)} val frames", flush=True)

    print("Loading ridge models...", flush=True)
    base_model = RidgeTrajectoryModel.load(args.base_model)
    temporal_model = RidgeTrajectoryModel.load(args.temporal_model)

    print("Building val training data...", flush=True)
    X_train, y_train = build_val_training_data(val_frames, base_model, temporal_model)
    pos = int((y_train > 0).sum())
    print(f"  {len(X_train)} examples, {pos} positive ({100*pos/len(X_train):.1f}%)", flush=True)

    print("Training HistGradientBoosting regressor...", flush=True)
    from sklearn.ensemble import HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(
        max_iter=args.hgb_max_iter,
        learning_rate=args.hgb_learning_rate,
        max_leaf_nodes=args.hgb_max_leaf_nodes,
        random_state=42,
        verbose=0,
    )
    hgb.fit(X_train, y_train)
    print("  done", flush=True)

    print("Scoring test candidates...", flush=True)
    count = score_test_candidates(args.test_candidates, hgb, args.output)
    print(json.dumps({"scored_rows": count, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
