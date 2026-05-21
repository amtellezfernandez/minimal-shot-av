"""Collect trajectory-scoring training data from expert or on-policy rollouts.

For each visited state, this logs:
  - state_features: 10-D geometric world-state vector
  - candidate_traj: all token candidate futures, flattened as relative (x, y, heading)
  - expert_idx: Spotlight Reflex selected maneuver index
  - rollout_policy_idx: maneuver index chosen by the rollout policy at that state

This is intended for trajectory-informed scorer training of the form
f(s, tau_a, interaction_a) -> score.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]

SEED_START = 1
SEED_END = 80
N_WORKERS = 20
TRAIN_SUITES = [("compositional", 5), ("adversarial", 2)]
TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]
TOKEN_INDEX = {name: idx for idx, name in enumerate(TOKEN_ORDER)}
ESCAPE_ENCODING = {"left": -1.0, "right": 1.0, "balanced": 0.0}

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class CollectTask:
    suite: str
    case_idx: int
    seed: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-policy", choices=("spotlight", "token_bc"), default="spotlight")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "bc_models")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "score_data")
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seed-end", type=int, default=SEED_END)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--horizon-seconds", type=float, default=2.0)
    return parser.parse_args()


def _init_worker(model_dir_str: str) -> None:
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)

    class GeomMLP(nn.Module):
        def __init__(self, out_dim: int, n_features: int = 10, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    model_dir = Path(model_dir_str)
    ckpt = torch.load(model_dir / "token_bc.pt", map_location="cpu", weights_only=False)
    model = GeomMLP(len(TOKEN_ORDER))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _WORKER["torch"] = torch
    _WORKER["token_bc_model"] = model
    _WORKER["token_bc_feat_mean"] = np.array(ckpt["feat_mean"], dtype=np.float32)
    _WORKER["token_bc_feat_std"] = np.array(ckpt["feat_std"], dtype=np.float32)


def _extract_features(world_state, perception, speed_mps: float) -> np.ndarray:
    escape = ESCAPE_ENCODING.get(str(getattr(world_state, "preferred_escape_side", "balanced")).lower(), 0.0)
    return np.array(
        [
            float(getattr(world_state, "obstacle_pressure", 0.0)),
            float(getattr(world_state, "route_blockage", 0.0)),
            float(getattr(world_state, "corridor_blocked", False)),
            min(float(getattr(world_state, "left_clearance", 20.0)), 20.0),
            min(float(getattr(world_state, "right_clearance", 20.0)), 20.0),
            escape,
            float(speed_mps),
            float(getattr(perception, "lane_error", 0.0)),
            float(max(getattr(world_state, "uncertainty", 0.0), getattr(perception, "uncertainty", 0.0))),
            min(
                (float(getattr(obstacle, "signed_distance", 20.0)) for obstacle in getattr(perception, "visible_obstacles", [])),
                default=20.0,
            ),
        ],
        dtype=np.float32,
    )


def _trajectory_to_xyh(
    position: tuple[float, float],
    trajectory: list[tuple[float, float]],
    *,
    point_limit: int,
) -> np.ndarray:
    rows = np.zeros((point_limit, 3), dtype=np.float32)
    if not trajectory:
        return rows

    capped = trajectory[:point_limit]
    for idx, point in enumerate(capped):
        prev = position if idx == 0 else capped[idx - 1]
        dx = float(point[0] - position[0])
        dy = float(point[1] - position[1])
        heading = math.atan2(point[1] - prev[1], point[0] - prev[0])
        rows[idx] = (dx, dy, heading)
    if len(capped) < point_limit:
        rows[len(capped):] = rows[len(capped) - 1]
    return rows


def _token_bc_choice(features: np.ndarray) -> tuple[int, str]:
    torch = _WORKER["torch"]
    model = _WORKER["token_bc_model"]
    feat_mean = _WORKER["token_bc_feat_mean"]
    feat_std = _WORKER["token_bc_feat_std"]
    x = torch.from_numpy((features - feat_mean) / feat_std).unsqueeze(0)
    with torch.no_grad():
        chosen_idx = int(model(x).argmax(1).item())
    return chosen_idx, TOKEN_ORDER[chosen_idx]


def _collect_rollout(task: CollectTask, rollout_policy: str, horizon_seconds: float) -> list[dict[str, Any]]:
    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.interaction_features import candidate_interaction_features
    from minimal_shot_av.simulator.planner import PlannedAction
    from minimal_shot_av.simulator.policy import RolloutConfig, _run_rollout
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG,
        _planning_heading,
        generate_maneuver_candidates,
        select_maneuver,
    )

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, task.suite)

    traj_cfg = DEFAULT_SPOTLIGHT_CONFIG.trajectory
    point_horizon = max(1, min(traj_cfg.point_count, int(round(horizon_seconds / traj_cfg.horizon_seconds * traj_cfg.point_count))))
    records: list[dict[str, Any]] = []
    rollout_id = f"{task.suite}:{task.case_idx}:{task.seed}"

    def planner(active_scenario, position, world_state, perception, config):
        speed_mps = config.step_size
        features = _extract_features(world_state, perception, speed_mps)
        selection = select_maneuver(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            config=DEFAULT_SPOTLIGHT_CONFIG,
        )
        heading = _planning_heading(position, world_state, perception, active_scenario, DEFAULT_SPOTLIGHT_CONFIG)
        candidates = generate_maneuver_candidates(position, heading, speed_mps, DEFAULT_SPOTLIGHT_CONFIG)
        by_name = {candidate.name: candidate for candidate in candidates}
        candidate_traj = np.stack(
            [_trajectory_to_xyh(position, by_name[name].trajectory, point_limit=point_horizon) for name in TOKEN_ORDER],
            axis=0,
        )
        candidate_interaction = np.stack(
            [
                candidate_interaction_features(
                    active_scenario,
                    position,
                    by_name[name].trajectory,
                    point_limit=point_horizon,
                    horizon_seconds=horizon_seconds,
                )
                for name in TOKEN_ORDER
            ],
            axis=0,
        )

        expert_idx = TOKEN_INDEX[selection.candidate.name]
        if rollout_policy == "spotlight":
            chosen_name = selection.candidate.name
            rollout_policy_idx = expert_idx
        else:
            rollout_policy_idx, chosen_name = _token_bc_choice(features)

        chosen_candidate = by_name[chosen_name]
        next_point = chosen_candidate.trajectory[DEFAULT_SPOTLIGHT_CONFIG.trajectory.action_index]
        vec = (next_point[0] - position[0], next_point[1] - position[1])
        dist = math.hypot(*vec)
        direction = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)

        records.append(
            {
                "rollout_id": rollout_id,
                "step_idx": len(records),
                "state_features": features,
                "candidate_traj": candidate_traj,
                "candidate_interaction": candidate_interaction,
                "expert_idx": expert_idx,
                "rollout_policy_idx": rollout_policy_idx,
            }
        )
        return PlannedAction(direction=direction, speed=dist, mode=f"{rollout_policy}:{chosen_name}", score=0.0), {}

    cfg = RolloutConfig(spotlight=DEFAULT_SPOTLIGHT_CONFIG)
    _run_rollout(scenario, planner, cfg)
    return records


def main() -> None:
    args = _parse_args()
    tasks = [
        CollectTask(suite=suite, case_idx=case_idx, seed=seed)
        for suite, n_cases in TRAIN_SUITES
        for case_idx in range(n_cases)
        for seed in range(args.seed_start, args.seed_end + 1)
    ]
    print(f"Collecting scorer data from {len(tasks)} rollouts using {args.rollout_policy} policy ({args.workers} workers)...")

    with multiprocessing.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.model_dir),),
    ) as pool:
        rollouts = pool.starmap(_collect_rollout, [(task, args.rollout_policy, args.horizon_seconds) for task in tasks])

    transitions = [row for rollout in rollouts for row in rollout]
    if not transitions:
        raise RuntimeError("no scorer transitions collected")

    state_features = np.stack([row["state_features"] for row in transitions], axis=0).astype(np.float32)
    candidate_traj = np.stack([row["candidate_traj"] for row in transitions], axis=0).astype(np.float32)
    candidate_interaction = np.stack([row["candidate_interaction"] for row in transitions], axis=0).astype(np.float32)
    expert_idx = np.array([row["expert_idx"] for row in transitions], dtype=np.int64)
    rollout_policy_idx = np.array([row["rollout_policy_idx"] for row in transitions], dtype=np.int64)
    rollout_ids_text = [str(row["rollout_id"]) for row in transitions]
    rollout_vocab = {name: idx for idx, name in enumerate(dict.fromkeys(rollout_ids_text))}
    rollout_id = np.array([rollout_vocab[name] for name in rollout_ids_text], dtype=np.int64)
    step_idx = np.array([row["step_idx"] for row in transitions], dtype=np.int64)

    feat_mean = state_features.mean(axis=0)
    feat_std = state_features.std(axis=0) + 1e-6

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "state_features.npy", state_features)
    np.save(args.output_dir / "candidate_traj.npy", candidate_traj)
    np.save(args.output_dir / "candidate_interaction.npy", candidate_interaction)
    np.save(args.output_dir / "expert_idx.npy", expert_idx)
    np.save(args.output_dir / "rollout_policy_idx.npy", rollout_policy_idx)
    np.save(args.output_dir / "rollout_id.npy", rollout_id)
    np.save(args.output_dir / "step_idx.npy", step_idx)
    np.save(args.output_dir / "feat_mean.npy", feat_mean)
    np.save(args.output_dir / "feat_std.npy", feat_std)

    counts = np.bincount(expert_idx, minlength=len(TOKEN_ORDER))
    policy_counts = np.bincount(rollout_policy_idx, minlength=len(TOKEN_ORDER))
    meta = {
        "schema": "score_data_v2",
        "rollout_policy": args.rollout_policy,
        "n_rollouts": len(tasks),
        "n_transitions": len(transitions),
        "n_features": int(state_features.shape[1]),
        "n_tokens": len(TOKEN_ORDER),
        "trajectory_shape": list(candidate_traj.shape[1:]),
        "trajectory_point_horizon": int(candidate_traj.shape[2]),
        "trajectory_horizon_seconds": float(args.horizon_seconds),
        "trajectory_channels": ["rel_x", "rel_y", "heading_rad"],
        "interaction_shape": list(candidate_interaction.shape[1:]),
        "interaction_feature_names": [
            "min_dynamic_clearance_m",
            "time_to_closest_approach_s",
            "closest_longitudinal_m",
            "closest_lateral_m",
            "relative_speed_mps",
            "crossing_relation",
            "dynamic_actor_present",
        ],
        "train_suites": [suite for suite, _ in TRAIN_SUITES],
        "seeds": f"{args.seed_start}-{args.seed_end}",
        "token_names": TOKEN_ORDER,
        "expert_distribution": {name: int(count) for name, count in zip(TOKEN_ORDER, counts)},
        "rollout_policy_distribution": {name: int(count) for name, count in zip(TOKEN_ORDER, policy_counts)},
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Collected {len(transitions):,} scorer transitions")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
