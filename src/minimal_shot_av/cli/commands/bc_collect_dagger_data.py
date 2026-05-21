"""Collect one-iteration DAgger-style on-policy relabel data for Token-BC.

The learned Token-BC policy is rolled out on the training suites. At every visited
state, Spotlight Reflex is queried as the expert and the resulting token label is
saved. This addresses the basic covariate-shift objection for the BC baseline:
training is no longer limited to expert-visited states.

Output:
  artifacts/bc_dagger_data/features.npy
  artifacts/bc_dagger_data/token_idx.npy
  artifacts/bc_dagger_data/speed_scale.npy
  artifacts/bc_dagger_data/lat_offset.npy
  artifacts/bc_dagger_data/rollout_id.npy
  artifacts/bc_dagger_data/step_idx.npy
  artifacts/bc_dagger_data/meta.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DEFAULT_MODEL_DIR = ROOT / "artifacts" / "bc_models"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "bc_dagger_data"
TRAIN_SUITES = [("compositional", 5), ("adversarial", 2)]
TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]
TOKEN_INDEX = {name: i for i, name in enumerate(TOKEN_ORDER)}
ESCAPE_ENCODING = {"left": -1.0, "right": 1.0, "balanced": 0.0}

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class CollectTask:
    suite: str
    case_idx: int
    seed: int


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--policy-ckpt",
        type=str,
        default="token_bc.pt",
        help="Checkpoint within --model-dir to roll out for DAgger state collection.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=80)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    tasks = [
        CollectTask(suite=suite, case_idx=case_idx, seed=seed)
        for suite, n_cases in TRAIN_SUITES
        for case_idx in range(n_cases)
        for seed in range(args.seed_start, args.seed_end + 1)
    ]
    print(f"Collecting DAgger-style states from {len(tasks)} rollouts ({args.workers} workers)...")

    with multiprocessing.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.model_dir), args.policy_ckpt),
    ) as pool:
        rollouts = pool.map(_collect_rollout, tasks)

    transitions = [transition for rollout in rollouts for transition in rollout]
    if not transitions:
        raise RuntimeError("no DAgger transitions collected")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features = np.array([row["features"] for row in transitions], dtype=np.float32)
    token_idx = np.array([row["token_idx"] for row in transitions], dtype=np.int64)
    speed_scale = np.array([row["speed_scale"] for row in transitions], dtype=np.float32)
    lat_offset = np.array([row["lat_offset"] for row in transitions], dtype=np.float32)
    rollout_ids_text = [str(row["rollout_id"]) for row in transitions]
    rollout_vocab = {name: idx for idx, name in enumerate(dict.fromkeys(rollout_ids_text))}
    rollout_id = np.array([rollout_vocab[name] for name in rollout_ids_text], dtype=np.int64)
    step_idx = np.array([row["step_idx"] for row in transitions], dtype=np.int64)
    np.save(args.output_dir / "features.npy", features)
    np.save(args.output_dir / "token_idx.npy", token_idx)
    np.save(args.output_dir / "speed_scale.npy", speed_scale)
    np.save(args.output_dir / "lat_offset.npy", lat_offset)
    np.save(args.output_dir / "rollout_id.npy", rollout_id)
    np.save(args.output_dir / "step_idx.npy", step_idx)

    counts = np.bincount(token_idx, minlength=len(TOKEN_ORDER))
    meta = {
        "schema": "bc_dagger_data_v1",
        "source_policy": Path(args.policy_ckpt).stem,
        "expert": "spotlight_reflex",
        "n_rollouts": len(tasks),
        "n_unique_rollout_ids": len(rollout_vocab),
        "n_transitions": len(transitions),
        "has_sequence_index": True,
        "train_suites": [suite for suite, _ in TRAIN_SUITES],
        "seeds": f"{args.seed_start}-{args.seed_end}",
        "token_names": TOKEN_ORDER,
        "token_distribution": {name: int(count) for name, count in zip(TOKEN_ORDER, counts)},
    }
    (args.output_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Collected {len(transitions):,} DAgger transitions")
    print(f"Wrote {args.output_dir}")


def _init_worker(model_dir_str: str, policy_ckpt_name: str) -> None:
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    model_dir = Path(model_dir_str)

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

        def forward(self, x):
            return self.net(x)

    ckpt = torch.load(model_dir / policy_ckpt_name, map_location="cpu", weights_only=False)
    model = GeomMLP(len(TOKEN_ORDER))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    _WORKER["torch"] = torch
    _WORKER["model"] = model
    _WORKER["feat_mean"] = np.array(ckpt["feat_mean"], dtype=np.float32)
    _WORKER["feat_std"] = np.array(ckpt["feat_std"], dtype=np.float32)
    _WORKER["policy_name"] = Path(policy_ckpt_name).stem


def _collect_rollout(task: CollectTask) -> list[dict[str, Any]]:
    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import RolloutConfig, _run_rollout
    from minimal_shot_av.simulator.planner import PlannedAction
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG,
        generate_maneuver_candidates,
        plan_spotlight_reflex_action,
        _planning_heading,
    )

    torch = _WORKER["torch"]
    model = _WORKER["model"]
    feat_mean = _WORKER["feat_mean"]
    feat_std = _WORKER["feat_std"]
    policy_name = str(_WORKER["policy_name"])
    token_params = {m.name: (m.speed_scale, m.lateral_offset_m) for m in DEFAULT_SPOTLIGHT_CONFIG.maneuvers}
    transitions: list[dict[str, Any]] = []

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, task.suite)
    rollout_name = f"{task.suite}:{task.case_idx}:{task.seed}"
    step_idx = 0

    def token_bc_planner(active_scenario, position, world_state, perception, config):
        nonlocal step_idx
        expert_action, expert_selection = plan_spotlight_reflex_action(
            active_scenario,
            position,
            world_state,
            perception,
            nominal_step_size=config.step_size,
            config=DEFAULT_SPOTLIGHT_CONFIG,
        )
        del expert_action
        expert_token = expert_selection.candidate.name
        if expert_token in TOKEN_INDEX:
            features = _extract_features(world_state, perception, config.step_size)
            speed_scale, lat_offset = token_params.get(expert_token, (0.75, 0.0))
            transitions.append(
                {
                    "features": features.tolist(),
                    "token_idx": TOKEN_INDEX[expert_token],
                    "speed_scale": float(speed_scale),
                    "lat_offset": float(lat_offset),
                    "rollout_id": rollout_name,
                    "step_idx": step_idx,
                }
            )

        features = _extract_features(world_state, perception, config.step_size)
        x = torch.from_numpy((features - feat_mean) / feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(model(x).argmax(1).item())
        chosen_name = TOKEN_ORDER[chosen_idx]

        heading = _planning_heading(position, world_state, perception, active_scenario, DEFAULT_SPOTLIGHT_CONFIG)
        candidates = {
            candidate.name: candidate
            for candidate in generate_maneuver_candidates(
                position,
                heading,
                config.step_size,
                DEFAULT_SPOTLIGHT_CONFIG,
            )
        }
        candidate = candidates.get(chosen_name) or candidates["maintain"]
        next_point = candidate.trajectory[DEFAULT_SPOTLIGHT_CONFIG.trajectory.action_index]
        vec = (next_point[0] - position[0], next_point[1] - position[1])
        dist = math.hypot(*vec)
        direction = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)
        return PlannedAction(
            direction=direction,
            speed=dist,
            mode=f"{policy_name}_dagger_collect:{chosen_name}",
            score=0.0,
        ), {}


    def counting_planner(active_scenario, position, world_state, perception, config):
        nonlocal step_idx
        action, meta = token_bc_planner(active_scenario, position, world_state, perception, config)
        step_idx += 1
        return action, meta

    cfg = RolloutConfig(spotlight=DEFAULT_SPOTLIGHT_CONFIG)
    _run_rollout(scenario, counting_planner, cfg)
    return transitions


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
                (
                    float(getattr(obstacle, "signed_distance", 20.0))
                    for obstacle in getattr(perception, "visible_obstacles", [])
                ),
                default=20.0,
            ),
        ],
        dtype=np.float32,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
