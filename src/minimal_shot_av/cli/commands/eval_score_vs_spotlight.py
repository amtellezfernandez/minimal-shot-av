"""Evaluate trajectory scorer against Token-DAgger-BC and Spotlight Reflex.

This plugs a trained trajectory-informed scorer into the same maneuver candidate
library used by Spotlight Reflex. The comparison is intentionally focused on the
strongest learned baseline frontier rather than rerunning every weaker baseline.

Output:
  artifacts/score_vs_spotlight.json
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eval_heldout_latin_hypercube import build_lhs_profiles

SEED_START = 1
SEED_END = 80
N_WORKERS = 20
COLLISION_RISK = 0.90
SUITES = [
    ("gauntlet", 6),
    ("adversarial", 2),
    ("compositional", 5),
]
SUITE_CASES = {
    "gauntlet": 6,
    "adversarial": 2,
    "compositional": 5,
    "hidden": 1,
}
TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]
TOKEN_INDEX = {name: idx for idx, name in enumerate(TOKEN_ORDER)}
ESCAPE_ENCODING = {"left": -1.0, "right": 1.0, "balanced": 0.0}

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class EvalTask:
    profile_idx: int
    suite: str
    case_idx: int
    seed: int


@dataclass(frozen=True)
class EvalResult:
    profile_idx: int
    suite: str
    score_passed: bool
    score_collision: bool
    dagger_passed: bool
    dagger_collision: bool
    spotlight_passed: bool
    spotlight_collision: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score-model", type=Path, default=ROOT / "artifacts" / "score_models_merged" / "trajectory_scorer.pt")
    parser.add_argument("--bc-model-dir", type=Path, default=ROOT / "artifacts" / "bc_models")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "score_vs_spotlight.json")
    parser.add_argument("--seed-start", type=int, default=SEED_START)
    parser.add_argument("--seed-end", type=int, default=SEED_END)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument("--profiles", type=int, default=0, help="Held-out LHS profile count. Use 0 for default fixed suites.")
    parser.add_argument("--profile-seed", type=int, default=2027)
    parser.add_argument("--suites", type=str, default="gauntlet,adversarial,compositional")
    return parser.parse_args()


def _wrap_angle_torch(x):
    import torch
    return torch.atan2(torch.sin(x), torch.cos(x))


def _extract_features(world_state, perception, speed_mps: float):
    import numpy as np
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


def _trajectory_to_xyh(position: tuple[float, float], trajectory: list[tuple[float, float]], *, point_limit: int):
    import numpy as np

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


def _init_worker(score_model_path: str, bc_model_dir_str: str, profiles_payload: list[Any]) -> None:
    import numpy as np
    import torch
    import torch.nn as nn
    from minimal_shot_av.simulator.interaction_features import INTERACTION_FEATURE_NAMES

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

        def forward(self, x):
            return self.net(x)

    class KinematicTrajectoryEncoder(nn.Module):
        def __init__(self, point_horizon: int, traj_hidden: int = 96, traj_dim: int = 3) -> None:
            super().__init__()
            self.point_horizon = point_horizon
            raw_dim = point_horizon * traj_dim
            segment_count = max(point_horizon - 1, 1)
            curvature_count = max(point_horizon - 2, 1)
            invariant_dim = (
                segment_count +
                segment_count +
                curvature_count +
                curvature_count +
                max(curvature_count - 1, 1) +
                max(segment_count - 1, 1) +
                4
            )
            self.net = nn.Sequential(
                nn.Linear(raw_dim + invariant_dim, traj_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(traj_hidden, traj_hidden),
                nn.GELU(),
            )

        def forward(self, traj, speed_mps, horizon_seconds: float):
            eps = 1e-6
            dx = traj[:, 1:, 0] - traj[:, :-1, 0]
            dy = traj[:, 1:, 1] - traj[:, :-1, 1]
            dtheta = _wrap_angle_torch(traj[:, 1:, 2] - traj[:, :-1, 2])
            ds = torch.sqrt(dx.pow(2) + dy.pow(2) + eps)
            if dtheta.shape[1] > 1:
                curvature = dtheta[:, 1:] / (ds[:, 1:] + eps)
            else:
                curvature = dtheta / (ds + eps)
            speed = speed_mps.unsqueeze(1)
            lateral_accel = speed.pow(2) * curvature
            if curvature.shape[1] > 1:
                curvature_delta = curvature[:, 1:] - curvature[:, :-1]
            else:
                curvature_delta = curvature
            dt = max(horizon_seconds / max(self.point_horizon, 1), 1e-3)
            segment_speed = ds / dt
            if segment_speed.shape[1] > 1:
                jerk_proxy = (segment_speed[:, 1:] - segment_speed[:, :-1]) / dt
            else:
                jerk_proxy = segment_speed / dt
            path_length = ds.sum(dim=1, keepdim=True)
            final_offset = torch.sqrt(traj[:, -1, 0].pow(2) + traj[:, -1, 1].pow(2) + eps).unsqueeze(1)
            final_heading = traj[:, -1, 2].unsqueeze(1)
            heading_span = _wrap_angle_torch(traj[:, -1, 2] - traj[:, 0, 2]).unsqueeze(1)
            invariants = torch.cat(
                [ds, dtheta, curvature, lateral_accel, curvature_delta, jerk_proxy, path_length, final_offset, final_heading, heading_span],
                dim=1,
            )
            x = torch.cat([traj.reshape(traj.shape[0], -1), invariants], dim=-1)
            return self.net(x)

    class InteractionFeatureEncoder(nn.Module):
        def __init__(self, interaction_dim: int, interaction_hidden: int = 48) -> None:
            super().__init__()
            self.interaction_dim = interaction_dim
            if interaction_dim > 0:
                self.net = nn.Sequential(
                    nn.LayerNorm(interaction_dim),
                    nn.Linear(interaction_dim, interaction_hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(interaction_hidden, interaction_hidden),
                    nn.GELU(),
                )
            else:
                self.net = None

        def forward(self, interaction):
            if self.interaction_dim <= 0 or self.net is None:
                return interaction.new_zeros((interaction.shape[0], 0))
            return self.net(interaction)

    class TrajectoryScorer(nn.Module):
        def __init__(
            self,
            point_horizon: int,
            state_hidden: int,
            traj_hidden: int,
            fusion_hidden: int,
            interaction_dim: int = 0,
            interaction_hidden: int = 48,
        ) -> None:
            super().__init__()
            self.interaction_dim = interaction_dim
            self.state_encoder = nn.Sequential(
                nn.LayerNorm(10),
                nn.Linear(10, state_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(state_hidden, state_hidden),
                nn.GELU(),
            )
            self.traj_encoder = KinematicTrajectoryEncoder(point_horizon=point_horizon, traj_hidden=traj_hidden)
            self.interaction_encoder = InteractionFeatureEncoder(interaction_dim=interaction_dim, interaction_hidden=interaction_hidden)
            self.fusion = nn.Sequential(
                nn.Linear(state_hidden + traj_hidden + (interaction_hidden if interaction_dim > 0 else 0), fusion_hidden),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(fusion_hidden, 1),
            )
            self.log_temperature = nn.Parameter(torch.tensor(0.0))

        def forward(self, state, trajectories, horizon_seconds: float, interaction=None):
            batch, n_tokens, _, _ = trajectories.shape
            state_enc = self.state_encoder(state).unsqueeze(1).expand(-1, n_tokens, -1)
            traj_flat = trajectories.reshape(batch * n_tokens, trajectories.shape[2], trajectories.shape[3])
            speed_mps = state[:, 6].unsqueeze(1).expand(-1, n_tokens).reshape(batch * n_tokens)
            traj_enc = self.traj_encoder(traj_flat, speed_mps, horizon_seconds).reshape(batch, n_tokens, -1)
            parts = [state_enc, traj_enc]
            if self.interaction_dim > 0:
                if interaction is None:
                    interaction = trajectories.new_zeros((batch, n_tokens, self.interaction_dim))
                interaction_flat = interaction.reshape(batch * n_tokens, interaction.shape[2])
                interaction_enc = self.interaction_encoder(interaction_flat).reshape(batch, n_tokens, -1)
                parts.append(interaction_enc)
            fused = torch.cat(parts, dim=-1)
            scores = self.fusion(fused.reshape(batch * n_tokens, -1)).reshape(batch, n_tokens)
            temperature = self.log_temperature.exp().clamp(min=0.05, max=10.0)
            return scores / temperature

    score_ckpt = torch.load(score_model_path, map_location="cpu", weights_only=False)
    score_model = TrajectoryScorer(
        point_horizon=int(score_ckpt["point_horizon"]),
        state_hidden=int(score_ckpt["state_hidden"]),
        traj_hidden=int(score_ckpt["traj_hidden"]),
        fusion_hidden=int(score_ckpt["fusion_hidden"]),
        interaction_dim=int(score_ckpt.get("interaction_dim", 0)),
        interaction_hidden=int(score_ckpt.get("interaction_hidden", 48)),
    )
    score_model.load_state_dict(score_ckpt["state_dict"])
    score_model.eval()

    bc_model_dir = Path(bc_model_dir_str)
    dagger_ckpt = torch.load(bc_model_dir / "token_dagger_bc.pt", map_location="cpu", weights_only=False)
    dagger_model = GeomMLP(9)
    dagger_model.load_state_dict(dagger_ckpt["state_dict"])
    dagger_model.eval()

    _WORKER["torch"] = torch
    _WORKER["score_model"] = score_model
    _WORKER["score_feat_mean"] = np.array(score_ckpt["feat_mean"], dtype=np.float32)
    _WORKER["score_feat_std"] = np.array(score_ckpt["feat_std"], dtype=np.float32)
    _WORKER["score_point_horizon"] = int(score_ckpt["point_horizon"])
    _WORKER["score_horizon_seconds"] = float(score_ckpt["horizon_seconds"])
    _WORKER["score_interaction_dim"] = int(score_ckpt.get("interaction_dim", 0))
    _WORKER["score_interaction_feature_names"] = tuple(
        score_ckpt.get("source_meta", {}).get("interaction_feature_names", list(INTERACTION_FEATURE_NAMES))
    )
    _WORKER["dagger_model"] = dagger_model
    _WORKER["dagger_feat_mean"] = np.array(dagger_ckpt["feat_mean"], dtype=np.float32)
    _WORKER["dagger_feat_std"] = np.array(dagger_ckpt["feat_std"], dtype=np.float32)
    _WORKER["profiles"] = profiles_payload


def _is_collision(rollout) -> bool:
    return (
        not rollout.success
        and rollout.steps is not None
        and any(getattr(st, "collision_risk", 0) > COLLISION_RISK for st in rollout.steps)
    )


def _run_task(task: EvalTask) -> EvalResult:
    import numpy as np
    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.interaction_features import candidate_interaction_features
    from minimal_shot_av.simulator.planner import PlannedAction
    from minimal_shot_av.simulator.policy import RolloutConfig, _run_rollout, run_spotlight_reflex_policy
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG,
        _planning_heading,
        generate_maneuver_candidates,
    )

    torch = _WORKER["torch"]
    score_model = _WORKER["score_model"]
    dagger_model = _WORKER["dagger_model"]
    score_feat_mean = _WORKER["score_feat_mean"]
    score_feat_std = _WORKER["score_feat_std"]
    score_point_horizon = int(_WORKER["score_point_horizon"])
    score_horizon_seconds = float(_WORKER["score_horizon_seconds"])
    score_interaction_dim = int(_WORKER["score_interaction_dim"])
    dagger_feat_mean = _WORKER["dagger_feat_mean"]
    dagger_feat_std = _WORKER["dagger_feat_std"]

    profile = None
    if _WORKER["profiles"]:
        profile = _WORKER["profiles"][task.profile_idx]
    effective_seed = task.seed + task.case_idx * 1_000 + task.profile_idx * 100_000
    if profile is None:
        scenario = generate_compositional_scenario(effective_seed, task.suite)
    else:
        scenario = generate_compositional_scenario(effective_seed, task.suite, profile=profile)

    def token_candidate_action(active_scenario, position, world_state, perception, speed_mps, chosen_name: str, mode_prefix: str):
        cfg = DEFAULT_SPOTLIGHT_CONFIG
        heading = _planning_heading(position, world_state, perception, active_scenario, cfg)
        candidates = {c.name: c for c in generate_maneuver_candidates(position, heading, speed_mps, cfg)}
        candidate = candidates.get(chosen_name, candidates.get("maintain"))
        next_pt = candidate.trajectory[cfg.trajectory.action_index]
        vec = (next_pt[0] - position[0], next_pt[1] - position[1])
        dist = math.hypot(*vec)
        norm = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)
        return PlannedAction(direction=norm, speed=dist, mode=f"{mode_prefix}:{chosen_name}", score=0.0), {}

    def score_planner(active_scenario, position, world_state, perception, speed_mps, config):
        feats = _extract_features(world_state, perception, speed_mps)
        heading = _planning_heading(position, world_state, perception, active_scenario, DEFAULT_SPOTLIGHT_CONFIG)
        candidates = generate_maneuver_candidates(position, heading, speed_mps, DEFAULT_SPOTLIGHT_CONFIG)
        by_name = {c.name: c for c in candidates}
        candidate_traj = np.stack(
            [_trajectory_to_xyh(position, by_name[name].trajectory, point_limit=score_point_horizon) for name in TOKEN_ORDER],
            axis=0,
        ).astype(np.float32)
        candidate_interaction = None
        if score_interaction_dim > 0:
            candidate_interaction = np.stack(
                [
                    candidate_interaction_features(
                        active_scenario,
                        position,
                        by_name[name].trajectory,
                        point_limit=score_point_horizon,
                        horizon_seconds=score_horizon_seconds,
                    )
                    for name in TOKEN_ORDER
                ],
                axis=0,
            ).astype(np.float32)
        state = torch.from_numpy(((feats - score_feat_mean) / score_feat_std).astype(np.float32)).unsqueeze(0)
        traj = torch.from_numpy(candidate_traj).unsqueeze(0)
        interaction = None if candidate_interaction is None else torch.from_numpy(candidate_interaction).unsqueeze(0)
        with torch.no_grad():
            scores = score_model(state, traj, score_horizon_seconds, interaction).squeeze(0).cpu().numpy()
        chosen_name = TOKEN_ORDER[int(scores.argmax())]
        return token_candidate_action(active_scenario, position, world_state, perception, speed_mps, chosen_name, "score_model")

    def dagger_planner(active_scenario, position, world_state, perception, speed_mps, config):
        feats = _extract_features(world_state, perception, speed_mps)
        x = torch.from_numpy((feats - dagger_feat_mean) / dagger_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(dagger_model(x).argmax(1).item())
        return token_candidate_action(active_scenario, position, world_state, perception, speed_mps, TOKEN_ORDER[chosen_idx], "token_dagger_bc")

    def run_with_planner(scenario_obj, planner_fn):
        def wrapped(active_scenario, position, world_state, perception, config):
            return planner_fn(active_scenario, position, world_state, perception, config.step_size, config)
        cfg = RolloutConfig(spotlight=DEFAULT_SPOTLIGHT_CONFIG)
        return _run_rollout(scenario_obj, wrapped, cfg)

    rs = run_spotlight_reflex_policy(scenario)
    rscore = run_with_planner(scenario, score_planner)
    rdagger = run_with_planner(scenario, dagger_planner)
    return EvalResult(
        profile_idx=task.profile_idx,
        suite=task.suite,
        score_passed=rscore.success,
        score_collision=_is_collision(rscore),
        dagger_passed=rdagger.success,
        dagger_collision=_is_collision(rdagger),
        spotlight_passed=rs.success,
        spotlight_collision=_is_collision(rs),
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    args = _parse_args()
    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    invalid = [suite for suite in suites if suite not in SUITE_CASES]
    if invalid:
        raise ValueError(f"invalid suites: {invalid}")
    profile_count = max(1, args.profiles) if args.profiles else 1
    profiles = build_lhs_profiles(args.profiles, seed=args.profile_seed) if args.profiles else []
    tasks = [
        EvalTask(profile_idx=profile_idx, suite=suite, case_idx=case_idx, seed=seed)
        for profile_idx in range(profile_count)
        for suite in suites
        for case_idx in range(SUITE_CASES[suite])
        for seed in range(args.seed_start, args.seed_end + 1)
    ]
    print(f"Evaluating scorer vs DAgger vs Spotlight on {len(tasks)} scenarios ({args.workers} workers)...")
    with multiprocessing.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.score_model), str(args.bc_model_dir), profiles),
    ) as pool:
        results = pool.map(_run_task, tasks)

    output: dict[str, Any] = {}
    output["config"] = {
        "profiles": args.profiles,
        "profile_seed": args.profile_seed,
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "suites": list(suites),
    }
    for suite in suites:
        sr = [result for result in results if result.suite == suite]
        n = len(sr)
        rows = {
            "Trajectory-Scorer": (sum(r.score_passed for r in sr), sum(r.score_collision for r in sr)),
            "Token-DAgger-BC": (sum(r.dagger_passed for r in sr), sum(r.dagger_collision for r in sr)),
            "Spotlight Reflex": (sum(r.spotlight_passed for r in sr), sum(r.spotlight_collision for r in sr)),
        }
        print(f"\n-- {suite.upper()} (n={n}) --")
        for name, (passes, collisions) in rows.items():
            p_lo, p_hi = _wilson_ci(passes, n)
            c_lo, c_hi = _wilson_ci(collisions, n)
            print(f"  {name:<18} pass={passes/n*100:5.1f}% [{p_lo*100:.1f},{p_hi*100:.1f}] coll={collisions/n*100:5.1f}% [{c_lo*100:.1f},{c_hi*100:.1f}]")
            output.setdefault("by_suite", {}).setdefault(suite, {})[name] = {
                "passes": passes,
                "collisions": collisions,
                "total": n,
                "pass_rate": round(passes / n * 100, 2),
                "collision_rate": round(collisions / n * 100, 2),
                "pass_ci95": [round(p_lo * 100, 2), round(p_hi * 100, 2)],
                "coll_ci95": [round(c_lo * 100, 2), round(c_hi * 100, 2)],
            }

    if profiles:
        profile_rows = []
        for profile_idx, profile in enumerate(profiles):
            pr = [result for result in results if result.profile_idx == profile_idx]
            row = {"profile_idx": profile_idx, "name": profile.name}
            for agent_name, pass_attr, coll_attr in [
                ("Trajectory-Scorer", "score_passed", "score_collision"),
                ("Token-DAgger-BC", "dagger_passed", "dagger_collision"),
                ("Spotlight Reflex", "spotlight_passed", "spotlight_collision"),
            ]:
                passes = sum(getattr(r, pass_attr) for r in pr)
                collisions = sum(getattr(r, coll_attr) for r in pr)
                row[agent_name] = {
                    "pass_rate": round(passes / len(pr) * 100, 2),
                    "collision_rate": round(collisions / len(pr) * 100, 2),
                }
            profile_rows.append(row)
        output["by_profile"] = profile_rows

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
