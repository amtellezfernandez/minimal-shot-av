#!/usr/bin/env python3
"""Evaluate internal proxy-state transfer failure modes and an oracle ladder.

This script keeps simulator physics fixed while perturbing only the planner-visible
proxy state (lane geometry, actor freshness, and noisy scalar cues). It supports a
ladder of selectors over the same maneuver library:

raw_iter2 -> clamped_iter2 -> axis_constrained_iter2 -> hybrid_veto_iter2 -> oracle_token -> spotlight

The goal is to test whether the same multi-axis conflicts observed in AlpaSim
reappear inside the controlled simulator once the planner sees a corrupted proxy.

`oracle_token` is a privileged token scorer, not the full simulator oracle. It uses the
same clamped maneuver library as the other token selectors but enables short-horizon
future-actor clearance forecasts inside the Spotlight-style scorer.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
import json
import math
import multiprocessing
from pathlib import Path
import random
import statistics
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.alpasim_token_bc import (  # noqa: E402
    _adapter_spotlight_config,
    _extract_features,
    _generate_adapter_candidates,
    _select_token_with_mode,
)
from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario  # noqa: E402
from minimal_shot_av.simulator.perception import PerceivedObstacle, ScenePerception, perceive_scene  # noqa: E402
from minimal_shot_av.simulator.policy import (  # noqa: E402
    DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG,
    EgoState,
    Rollout,
    RolloutConfig,
    _commit_wod_intersection_crossing,
    _initial_heading,
    _recenter_route_drift,
    _recover_stalled_spotlight_progress,
    _step_record,
    advance_ego_state,
)
from minimal_shot_av.simulator.safety import apply_safety_filter  # noqa: E402
from minimal_shot_av.simulator.spotlight_reflex import (  # noqa: E402
    DEFAULT_SPOTLIGHT_CONFIG,
    SpotlightReflexConfig,
    _planning_heading,
    evaluate_maneuver_candidates,
    plan_spotlight_reflex_action,
)
from minimal_shot_av.simulator.world_model import WorldState, update_world_state  # noqa: E402

from scripts.eval_heldout_latin_hypercube import (  # noqa: E402
    DEFAULT_SUITES,
    SUITE_CASES,
    TOKEN_ORDER,
    build_lhs_profiles,
)

DEFAULT_OUTPUT = ROOT / "artifacts" / "internal_proxy_transfer.json"
DEFAULT_MODEL_PATH = ROOT / "artifacts" / "bc_models_iter2" / "token_dagger_bc.pt"
DEFAULT_WORKERS = 8

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class ProxyPerturbation:
    name: str
    lane_error_scale: float = 1.0
    lane_heading_bias_deg: float = 0.0
    actor_latency_ticks: int = 0
    route_offset_m: float = 0.0
    feature_noise_std: float = 0.0


@dataclass(frozen=True)
class EvalTask:
    profile_idx: int
    suite: str
    case_idx: int
    seed: int
    perturbation: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=6)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--profile-seed", type=int, default=2027)
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suites from: compositional, adversarial, gauntlet, hidden",
    )
    parser.add_argument(
        "--perturbations",
        type=str,
        default="clean,lane_error_x1.5,heading_bias_p12,latency_3,route_offset_p1.5,feature_noise_0.15",
    )
    parser.add_argument(
        "--execution-observer",
        choices=("proxy_state", "true_state"),
        default="proxy_state",
        help="Which observation stream the safety/execution stack consumes after planning.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def _named_perturbations() -> dict[str, ProxyPerturbation]:
    return {
        "clean": ProxyPerturbation("clean"),
        "lane_error_x1.5": ProxyPerturbation("lane_error_x1.5", lane_error_scale=1.5),
        "heading_bias_p12": ProxyPerturbation("heading_bias_p12", lane_heading_bias_deg=12.0),
        "latency_3": ProxyPerturbation("latency_3", actor_latency_ticks=3),
        "route_offset_p1.5": ProxyPerturbation("route_offset_p1.5", route_offset_m=1.5),
        "feature_noise_0.15": ProxyPerturbation("feature_noise_0.15", feature_noise_std=0.15),
    }


def _load_torch_model(model_path: Path):
    import torch
    import torch.nn as nn

    class GeomMLP(nn.Module):
        def __init__(self, out_dim: int, n_features: int = 10, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Linear(n_features, hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2),
                nn.GELU(),
                nn.Linear(hidden // 2, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = GeomMLP(len(TOKEN_ORDER))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
    feat_std = np.maximum(np.asarray(ckpt["feat_std"], dtype=np.float32), 1e-6)
    return torch, model, feat_mean, feat_std


def _init_worker(
    model_path_str: str,
    profiles,
    perturbations: dict[str, ProxyPerturbation],
    execution_observer: str,
) -> None:
    torch, model, feat_mean, feat_std = _load_torch_model(Path(model_path_str))
    torch.set_num_threads(1)
    clamped = _adapter_spotlight_config(trajectory_mode="clamped_lateral", max_lateral_offset_m=2.0)
    oracle_token_cfg = replace(clamped, scoring=replace(clamped.scoring, use_privileged_actor_forecast=True))
    _WORKER["torch"] = torch
    _WORKER["model"] = model
    _WORKER["feat_mean"] = feat_mean
    _WORKER["feat_std"] = feat_std
    _WORKER["profiles"] = profiles
    _WORKER["perturbations"] = perturbations
    _WORKER["spotlight_cfg"] = DEFAULT_SPOTLIGHT_CONFIG
    _WORKER["clamped_cfg"] = clamped
    _WORKER["oracle_token_cfg"] = oracle_token_cfg
    _WORKER["rollout_cfg"] = DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG
    _WORKER["execution_observer"] = execution_observer


def _rotate(vector: tuple[float, float], degrees: float) -> tuple[float, float]:
    if abs(degrees) < 1e-9:
        return vector
    angle = math.radians(degrees)
    c = math.cos(angle)
    s = math.sin(angle)
    return (vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c)


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _copy_perception(perception: ScenePerception) -> ScenePerception:
    return ScenePerception(
        lane_index=int(perception.lane_index),
        lane_point=(float(perception.lane_point[0]), float(perception.lane_point[1])),
        lane_error=float(perception.lane_error),
        lane_heading=(float(perception.lane_heading[0]), float(perception.lane_heading[1])),
        corridor_margin=float(perception.corridor_margin),
        free_space_confidence=float(perception.free_space_confidence),
        uncertainty=float(perception.uncertainty),
        visible_obstacles=[
            PerceivedObstacle(
                x=float(obstacle.x),
                y=float(obstacle.y),
                radius=float(obstacle.radius),
                length=None if obstacle.length is None else float(obstacle.length),
                heading=float(obstacle.heading),
                signed_distance=float(obstacle.signed_distance),
            )
            for obstacle in perception.visible_obstacles
        ],
    )


def _apply_proxy_perturbation(
    scenario,
    position: tuple[float, float],
    perception: ScenePerception,
    world_state: WorldState,
    history: list[tuple[ScenePerception, WorldState]],
    perturbation: ProxyPerturbation,
    rng: random.Random,
) -> tuple[ScenePerception, WorldState]:
    if perturbation.actor_latency_ticks > 0 and len(history) > perturbation.actor_latency_ticks:
        stale_perception, stale_world = history[-1 - perturbation.actor_latency_ticks]
        perception = _copy_perception(stale_perception)
        world_state = replace(stale_world, position=position, goal_distance=math.dist(position, scenario.goal))
    else:
        perception = _copy_perception(perception)

    lane_heading = _normalize(_rotate(perception.lane_heading, perturbation.lane_heading_bias_deg))
    if lane_heading == (0.0, 0.0):
        lane_heading = perception.lane_heading
    left = (-lane_heading[1], lane_heading[0])
    lane_point = (
        perception.lane_point[0] + left[0] * perturbation.route_offset_m,
        perception.lane_point[1] + left[1] * perturbation.route_offset_m,
    )
    lane_error = max(0.0, float(perception.lane_error) * perturbation.lane_error_scale + abs(perturbation.route_offset_m))

    noisy_obstacles = []
    for obstacle in perception.visible_obstacles:
        signed_distance = float(obstacle.signed_distance)
        if perturbation.feature_noise_std > 0.0:
            signed_distance += rng.gauss(0.0, perturbation.feature_noise_std * 2.0)
        noisy_obstacles.append(
            PerceivedObstacle(
                x=obstacle.x,
                y=obstacle.y,
                radius=obstacle.radius,
                length=obstacle.length,
                heading=obstacle.heading,
                signed_distance=signed_distance,
            )
        )

    uncertainty = float(perception.uncertainty)
    if perturbation.feature_noise_std > 0.0:
        lane_error = max(0.0, lane_error + rng.gauss(0.0, perturbation.feature_noise_std * 2.0))
        uncertainty = min(1.0, max(0.0, uncertainty + rng.gauss(0.0, perturbation.feature_noise_std)))

    perturbed = replace(
        perception,
        lane_heading=lane_heading,
        lane_point=lane_point,
        lane_error=lane_error,
        corridor_margin=max(0.0, scenario.lane_half_width - lane_error),
        uncertainty=uncertainty,
        free_space_confidence=max(0.0, 1.0 - uncertainty),
        visible_obstacles=noisy_obstacles,
    )
    derived_world = update_world_state(scenario, position, perturbed)
    return perturbed, derived_world


def _token_action(position, heading, speed_mps: float, chosen_name: str, config: SpotlightReflexConfig):
    from minimal_shot_av.simulator.planner import PlannedAction

    by_name = _generate_adapter_candidates(position, heading, speed_mps, config=config)
    candidate = by_name.get(chosen_name) or by_name["maintain"]
    next_point = candidate.trajectory[config.trajectory.action_index]
    vec = (next_point[0] - position[0], next_point[1] - position[1])
    dist = math.hypot(*vec)
    direction = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)
    return PlannedAction(direction=direction, speed=dist, mode=chosen_name, score=0.0)


def _selection_metadata_from_evaluations(
    evaluations: list[Any],
    reference_count: int,
    chosen_token: str,
) -> dict[str, Any]:
    by_name = {evaluation.candidate.name: evaluation for evaluation in evaluations}
    chosen = by_name[chosen_token]
    ordered = sorted(evaluations, key=lambda item: item.explanation.effective_score, reverse=True)
    return {
        "candidate_count": len(evaluations),
        "reference_count": reference_count,
        "selected_maneuver": chosen_token,
        "selector_score": float(chosen.score.combined_score),
        "selector_effective_score": float(chosen.explanation.effective_score),
        "selector_3s_score": float(chosen.score.score_3s),
        "selector_5s_score": float(chosen.score.score_5s),
        "selector_3s_reference": str(chosen.score.reference_3s_label),
        "selector_5s_reference": str(chosen.score.reference_5s_label),
        "selector_3s_inside_region": bool(chosen.score.inside_3s_region),
        "selector_5s_inside_region": bool(chosen.score.inside_5s_region),
        "decision_reason": "; ".join(chosen.explanation.reasons[:6]),
        "decision_reasons": list(chosen.explanation.reasons),
        "top_candidate_summaries": [evaluation.explanation.to_summary() for evaluation in ordered[:3]],
    }


def _step_rng(base_seed: int, tick: int, channel: int) -> random.Random:
    return random.Random(base_seed * 1_000_003 + tick * 97 + channel * 7_919)


def _run_rollout_with_proxy(scenario, planner_variant: str, perturbation: ProxyPerturbation, rng_seed: int) -> Rollout:
    torch = _WORKER["torch"]
    model = _WORKER["model"]
    feat_mean = _WORKER["feat_mean"]
    feat_std = _WORKER["feat_std"]
    spotlight_cfg = _WORKER["spotlight_cfg"]
    clamped_cfg = _WORKER["clamped_cfg"]
    oracle_token_cfg = _WORKER["oracle_token_cfg"]
    config: RolloutConfig = _WORKER["rollout_cfg"]
    execution_observer = _WORKER["execution_observer"]

    ego_state = EgoState(
        x=scenario.start[0],
        y=scenario.start[1],
        heading_rad=_initial_heading(scenario),
        speed_mps=0.0,
        steering_rad=0.0,
    )
    steps = []
    history: list[tuple[ScenePerception, WorldState]] = []
    collision = False
    reached_goal = False
    for tick in range(config.max_steps):
        from minimal_shot_av.simulator.environment import DEFAULT_EGO_RADIUS_M, min_time_swept_clearance, scenario_at_tick

        active_scenario = scenario_at_tick(scenario, tick)
        position = (ego_state.x, ego_state.y)
        previous_position = position
        perception = perceive_scene(active_scenario, position)
        world_state = update_world_state(active_scenario, position, perception)
        planner_perception, planner_world = _apply_proxy_perturbation(
            active_scenario,
            position,
            perception,
            world_state,
            history,
            perturbation,
            _step_rng(rng_seed, tick, 0),
        )
        history.append((_copy_perception(perception), world_state))

        if planner_variant == "spotlight":
            planned_action, selection = plan_spotlight_reflex_action(
                active_scenario,
                position,
                planner_world,
                planner_perception,
                nominal_step_size=config.step_size,
                config=spotlight_cfg,
            )
            metadata = selection.to_metadata()
        else:
            speed_mps = max(0.25, float(config.step_size))
            features = _extract_features(planner_world, planner_perception, speed_mps)
            if perturbation.feature_noise_std > 0.0:
                feature_rng = _step_rng(rng_seed, tick, 1)
                features = features + np.asarray(
                    [feature_rng.gauss(0.0, perturbation.feature_noise_std) for _ in range(features.shape[0])],
                    dtype=np.float32,
                )
            x = torch.from_numpy(((features - feat_mean) / feat_std).astype(np.float32)).unsqueeze(0)
            with torch.no_grad():
                logits = model(x).squeeze(0).cpu().numpy()
            if planner_variant == "raw_iter2":
                chosen_token = TOKEN_ORDER[int(np.argmax(logits))]
                evaluations, reference_count = evaluate_maneuver_candidates(
                    active_scenario,
                    position,
                    planner_world,
                    planner_perception,
                    speed_mps=speed_mps,
                    config=spotlight_cfg,
                )
                heading = _planning_heading(position, planner_world, planner_perception, active_scenario, spotlight_cfg)
                planned_action = _token_action(position, heading, speed_mps, chosen_token, spotlight_cfg)
                metadata = _selection_metadata_from_evaluations(evaluations, reference_count, chosen_token)
            elif planner_variant == "clamped_iter2":
                chosen_token = TOKEN_ORDER[int(np.argmax(logits))]
                evaluations, reference_count = evaluate_maneuver_candidates(
                    active_scenario,
                    position,
                    planner_world,
                    planner_perception,
                    speed_mps=speed_mps,
                    config=clamped_cfg,
                )
                heading = _planning_heading(position, planner_world, planner_perception, active_scenario, clamped_cfg)
                planned_action = _token_action(position, heading, speed_mps, chosen_token, clamped_cfg)
                metadata = _selection_metadata_from_evaluations(evaluations, reference_count, chosen_token)
            elif planner_variant == "hybrid_veto_iter2":
                evaluations, reference_count = evaluate_maneuver_candidates(
                    active_scenario,
                    position,
                    planner_world,
                    planner_perception,
                    speed_mps=speed_mps,
                    config=clamped_cfg,
                )
                selection = _select_token_with_mode(
                    logits=logits,
                    token_order=tuple(TOKEN_ORDER),
                    evaluations=evaluations,
                    selection_mode="hybrid_veto",
                    hybrid_top_k=3,
                    hybrid_geometric_weight=0.75,
                    hybrid_policy_temperature=1.0,
                    hybrid_veto_margin=8.0,
                    hybrid_max_geometric_rank=2,
                )
                chosen_token = str(selection["hybrid_token"])
                heading = _planning_heading(position, planner_world, planner_perception, active_scenario, clamped_cfg)
                planned_action = _token_action(position, heading, speed_mps, chosen_token, clamped_cfg)
                metadata = _selection_metadata_from_evaluations(evaluations, reference_count, chosen_token)
                metadata["dagger_argmax_vetoed"] = selection["dagger_argmax_vetoed"]
                metadata["veto_reason"] = selection["veto_reason"]
            elif planner_variant == "axis_constrained_iter2":
                evaluations, reference_count = evaluate_maneuver_candidates(
                    active_scenario,
                    position,
                    planner_world,
                    planner_perception,
                    speed_mps=speed_mps,
                    config=clamped_cfg,
                )
                selection = _select_token_with_mode(
                    logits=logits,
                    token_order=tuple(TOKEN_ORDER),
                    evaluations=evaluations,
                    selection_mode="axis_constrained",
                    hybrid_top_k=3,
                    hybrid_geometric_weight=0.0,
                    hybrid_policy_temperature=1.0,
                    hybrid_veto_margin=8.0,
                    hybrid_max_geometric_rank=2,
                )
                chosen_token = str(selection["hybrid_token"])
                heading = _planning_heading(position, planner_world, planner_perception, active_scenario, clamped_cfg)
                planned_action = _token_action(position, heading, speed_mps, chosen_token, clamped_cfg)
                metadata = _selection_metadata_from_evaluations(evaluations, reference_count, chosen_token)
                metadata["dagger_argmax_vetoed"] = selection["dagger_argmax_vetoed"]
                metadata["veto_reason"] = selection["veto_reason"]
            elif planner_variant == "oracle_token":
                planned_action, selection = plan_spotlight_reflex_action(
                    active_scenario,
                    position,
                    planner_world,
                    planner_perception,
                    nominal_step_size=config.step_size,
                    config=oracle_token_cfg,
                )
                metadata = selection.to_metadata()
            else:
                raise ValueError(f"unknown planner_variant={planner_variant}")

        exec_world = planner_world if execution_observer == "proxy_state" else world_state
        exec_perception = planner_perception if execution_observer == "proxy_state" else perception
        safe_action = apply_safety_filter(planned_action, exec_world, exec_perception)
        safe_action = _recenter_route_drift(active_scenario, safe_action, exec_world, exec_perception)
        safe_action = _recover_stalled_spotlight_progress(safe_action, planned_action, exec_world, exec_perception, steps)
        safe_action = _commit_wod_intersection_crossing(active_scenario, safe_action, exec_world, exec_perception, steps)
        if (
            config.preserve_planned_direction
            and safe_action.direction == (0.0, 0.0)
            and safe_action.speed > 0.0
            and planned_action.direction != (0.0, 0.0)
        ):
            safe_action.direction = planned_action.direction

        ego_state = advance_ego_state(ego_state, safe_action.direction, safe_action.speed, config)
        position = (ego_state.x, ego_state.y)
        min_obstacle_distance = min_time_swept_clearance(
            scenario,
            previous_position,
            position,
            tick,
            tick + 1,
            ego_radius=DEFAULT_EGO_RADIUS_M,
        )
        collision = min_obstacle_distance <= 0.0
        previous_goal_distance = math.dist(previous_position, scenario.goal)
        goal_distance = math.dist(position, scenario.goal)
        steps.append(
            _step_record(
                tick,
                position,
                perception,
                world_state,
                safe_action,
                goal_distance,
                previous_goal_distance,
                min_obstacle_distance,
                len(active_scenario.actors),
                steps[-1].speed if steps else None,
                metadata,
                config,
                ego_state,
            )
        )
        if collision:
            break
        if goal_distance < config.goal_tolerance_m:
            reached_goal = True
            break

    return Rollout(success=reached_goal and not collision, collision=collision, reached_goal=reached_goal, steps=steps)


def _summarize_rollout(rollout: Rollout, lane_half_width: float) -> dict[str, Any]:
    steps = rollout.steps or []
    clearances = np.asarray([float(step.min_obstacle_distance) for step in steps], dtype=np.float32)
    speeds = np.asarray([float(step.speed) for step in steps], dtype=np.float32)
    lane_errors = np.asarray([float(step.lane_error) for step in steps], dtype=np.float32)
    progress = np.asarray([float(step.progress or 0.0) for step in steps], dtype=np.float32)

    accel = np.diff(speeds) if len(speeds) >= 2 else np.zeros(0, dtype=np.float32)
    jerk = np.diff(accel) if len(accel) >= 2 else np.zeros(0, dtype=np.float32)
    lane_violation = bool(np.any(lane_errors > lane_half_width))
    offroad_proxy = bool(np.any(lane_errors > lane_half_width * 1.5))
    return {
        "pass": bool(rollout.success),
        "collision": bool(rollout.collision),
        "lane_violation_any": lane_violation,
        "offroad_proxy_any": offroad_proxy,
        "min_clearance_m": float(clearances.min()) if len(clearances) else math.inf,
        "total_progress_m": float(progress.sum()) if len(progress) else 0.0,
        "mean_progress_m": float(progress.mean()) if len(progress) else 0.0,
        "mean_abs_jerk": float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
        "max_lane_error_m": float(lane_errors.max()) if len(lane_errors) else 0.0,
        "steps": len(steps),
    }


def _run_task(task: EvalTask) -> dict[str, Any]:
    profile = _WORKER["profiles"][task.profile_idx]
    perturbation: ProxyPerturbation = _WORKER["perturbations"][task.perturbation]
    scenario_seed = task.seed + task.case_idx * 1_000 + task.profile_idx * 100_000
    scenario = generate_compositional_scenario(scenario_seed, task.suite, profile=profile)
    variants = ("raw_iter2", "clamped_iter2", "axis_constrained_iter2", "hybrid_veto_iter2", "oracle_token", "spotlight")
    agents = {}
    for idx, variant in enumerate(variants):
        rollout = _run_rollout_with_proxy(scenario, variant, perturbation, rng_seed=scenario_seed)
        agents[variant] = _summarize_rollout(rollout, lane_half_width=scenario.lane_half_width)
    return {
        "profile_idx": task.profile_idx,
        "suite": task.suite,
        "case_idx": task.case_idx,
        "seed": task.seed,
        "perturbation": task.perturbation,
        "agents": agents,
    }


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _safe_rate(hits: int, total: int) -> float | None:
    return None if total <= 0 else hits / total


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_binary = ("pass", "collision", "lane_violation_any", "offroad_proxy_any")
    metrics_cont = ("min_clearance_m", "total_progress_m", "mean_abs_jerk", "max_lane_error_m")
    by_perturbation: dict[str, Any] = {}
    for perturbation in sorted({row["perturbation"] for row in rows}):
        subset = [row for row in rows if row["perturbation"] == perturbation]
        agent_names = sorted(subset[0]["agents"])
        summary = {}
        comparisons = {}
        for agent in agent_names:
            agent_rows = [row["agents"][agent] for row in subset]
            summary[agent] = {
                **{
                    metric: _safe_rate(sum(1 for r in agent_rows if r[metric]), len(agent_rows))
                    for metric in metrics_binary
                },
                **{
                    metric: _mean([float(r[metric]) for r in agent_rows])
                    for metric in metrics_cont
                },
            }
        baseline = "raw_iter2"
        for agent in agent_names:
            if agent == baseline:
                continue
            progress_up_collision_up = 0
            progress_up_lane_worse = 0
            clearance_up_progress_down = 0
            deltas = defaultdict(list)
            for row in subset:
                base = row["agents"][baseline]
                cand = row["agents"][agent]
                if cand["total_progress_m"] > base["total_progress_m"] and cand["collision"] > base["collision"]:
                    progress_up_collision_up += 1
                if cand["total_progress_m"] > base["total_progress_m"] and cand["lane_violation_any"] > base["lane_violation_any"]:
                    progress_up_lane_worse += 1
                if cand["min_clearance_m"] > base["min_clearance_m"] and cand["total_progress_m"] < base["total_progress_m"]:
                    clearance_up_progress_down += 1
                for metric in metrics_cont:
                    deltas[metric].append(float(cand[metric]) - float(base[metric]))
            comparisons[agent] = {
                "mean_deltas": {metric: _mean(deltas[metric]) for metric in metrics_cont},
                "axis_conflicts": {
                    "progress_up_collision_up": progress_up_collision_up,
                    "progress_up_lane_worse": progress_up_lane_worse,
                    "clearance_up_progress_down": clearance_up_progress_down,
                },
            }
        by_perturbation[perturbation] = {
            "task_count": len(subset),
            "summary": summary,
            "comparisons_vs_raw_iter2": comparisons,
        }
    return by_perturbation


def main() -> None:
    args = _parse_args()
    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    invalid = [suite for suite in suites if suite not in SUITE_CASES]
    if invalid:
        raise SystemExit(f"invalid suites: {invalid}")
    perturbations = _named_perturbations()
    selected_perturbations = [name.strip() for name in args.perturbations.split(",") if name.strip()]
    missing = [name for name in selected_perturbations if name not in perturbations]
    if missing:
        raise SystemExit(f"unknown perturbations: {missing}")
    profiles = build_lhs_profiles(args.profiles, seed=args.profile_seed)
    tasks = [
        EvalTask(profile_idx, suite, case_idx, seed, perturbation)
        for perturbation in selected_perturbations
        for profile_idx in range(len(profiles))
        for suite in suites
        for case_idx in range(SUITE_CASES[suite])
        for seed in range(args.seed_start, args.seed_end + 1)
    ]
    with multiprocessing.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(
            str(args.model_path),
            profiles,
            {name: perturbations[name] for name in selected_perturbations},
            args.execution_observer,
        ),
    ) as pool:
        rows = pool.map(_run_task, tasks)
    report = {
        "schema": "internal_proxy_transfer_v1",
        "config": {
            "profiles": args.profiles,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "workers": args.workers,
            "profile_seed": args.profile_seed,
            "suites": list(suites),
            "perturbations": selected_perturbations,
            "tasks": len(tasks),
            "model_path": str(args.model_path),
            "execution_observer": args.execution_observer,
        },
        "by_perturbation": _aggregate(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["by_perturbation"], indent=2))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
