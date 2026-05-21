#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.simulator.compositional_scenarios import (  # noqa: E402
    COMPOSITIONAL_SUITES,
    generate_compositional_scenario,
)
from minimal_shot_av.simulator.environment import Scenario  # noqa: E402
from minimal_shot_av.simulator.wod_scenarios import WOD_E2E_CLUSTERS, generate_wod_scenario  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "rlvr_curriculum_v1.json"

OBSERVATION_MODES = (
    "privileged_geometry_debug",
    "noisy_geometry",
    "topdown_raster_spec",
    "camera_scene_embedding_required",
)

ACTION_SPACE = {
    "type": "bounded_waypoint_velocity_control",
    "control_hz": 4,
    "action": {
        "target_dx_m": {"min": -4.0, "max": 6.0},
        "target_dy_m": {"min": -4.0, "max": 4.0},
        "speed_m_per_step": {"min": 0.0, "max": 1.35},
        "mode": ["nominal", "yield", "crawl", "stop", "escape"],
    },
    "forbidden_inputs": [
        "scenario.tags",
        "scenario.cluster",
        "future actor trajectories",
        "grader thresholds",
        "heldout split label",
    ],
}

GRADER_LIBRARY = {
    "reach_goal": "final_goal_distance <= goal_tolerance and reached_goal is true",
    "no_collision": "rollout.collision is false",
    "trajectory_safety": "no low-clearance, terminal-collision, or extreme-risk-with-tight-clearance event",
    "progress_floor": "average progress stays above suite-specific floor",
    "intervention_budget": "intervention rate stays below suite-specific ceiling",
    "benchmark_pass": "reach_goal and no_collision and trajectory_safety and progress/intervention gates",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an explicit RLVR-style autonomy curriculum manifest.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=20)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = build_curriculum(seed_start=args.seed_start, seed_end=args.seed_end)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


def build_curriculum(*, seed_start: int = 1, seed_end: int = 20) -> dict[str, Any]:
    if seed_end < seed_start:
        raise ValueError("--seed-end must be >= --seed-start")

    tasks: list[dict[str, Any]] = []
    for seed in range(seed_start, seed_end + 1):
        for cluster in WOD_E2E_CLUSTERS:
            scenario = generate_wod_scenario(cluster, seed)
            tasks.append(_task_record(scenario, suite="wod", seed=seed, source="wod_procedural"))
        for suite in COMPOSITIONAL_SUITES:
            for case_index in range(1, 7):
                scenario = generate_compositional_scenario(seed + (case_index - 1) * 1_000, suite=suite)
                tasks.append(
                    _task_record(
                        scenario,
                        suite=suite,
                        seed=seed,
                        source="compositional_procedural",
                        case_index=case_index,
                    )
                )

    return {
        "schema": "minimal_shot_av_rlvr_curriculum_v1",
        "gym": {
            "environment": "minimal_shot_av.closed_loop_simulator",
            "observation_modes": list(OBSERVATION_MODES),
            "action_space": ACTION_SPACE,
            "agent_loop": "obs_t -> policy.act(obs_t, memory) -> env.step(action_t) -> grader(trajectory)",
        },
        "grader_library": GRADER_LIBRARY,
        "split_policy": {
            "train_curriculum": "development scenarios and composition keys not reserved for holdout",
            "dev_holdout": "deterministic held-out WOD seeds and non-hidden composition keys",
            "blind_holdout": "hidden suite and reserved composition keys",
            "stress_holdout": "gauntlet suite",
        },
        "tasks": tasks,
        "coverage": _coverage(tasks),
        "difficulty_buckets": _difficulty_buckets(tasks),
        "search_objective": {
            "calibrated_band_target_pass_rate": [0.25, 0.75],
            "primary_signal": "benchmark_pass",
            "secondary_signals": ["trajectory_safety", "progress_floor", "intervention_budget"],
            "curriculum_engineering_goal": (
                "Find verifiable tasks that are neither saturated nor impossible for the target model."
            ),
        },
    }


def _task_record(
    scenario: Scenario,
    *,
    suite: str,
    seed: int,
    source: str,
    case_index: int | None = None,
) -> dict[str, Any]:
    tags = scenario.tags
    axes = _axes(scenario, suite)
    composition_key = _composition_key(axes)
    split = _split_for(suite=suite, seed=seed, composition_key=composition_key)
    difficulty = float(tags.get("difficulty", tags.get("severity", 0.0)) or 0.0)
    task_id_parts = [source, suite, str(seed)]
    if case_index is not None:
        task_id_parts.append(f"case{case_index}")
    task_id_parts.append(_short_hash(composition_key))
    return {
        "task_id": ":".join(task_id_parts),
        "prompt": _task_prompt(scenario, axes),
        "source": source,
        "split": split,
        "suite": suite,
        "seed": seed,
        "cluster": scenario.cluster,
        "difficulty": round(difficulty, 3),
        "difficulty_bucket": _difficulty_bucket(difficulty),
        "composition_key": composition_key,
        "axes": axes,
        "observation_contract": {
            "default_mode": "noisy_geometry",
            "available_modes": list(OBSERVATION_MODES),
            "privileged_state_allowed": False,
            "camera_or_embedding_required_for_top_lab_gate": True,
        },
        "action_space_id": "bounded_waypoint_velocity_control",
        "grader_ids": list(GRADER_LIBRARY),
        "success_criteria": {
            "must_reach_goal": True,
            "must_avoid_collision": True,
            "must_pass_trajectory_safety": True,
            "must_respect_progress_and_intervention_budget": True,
        },
    }


def _axes(scenario: Scenario, suite: str) -> dict[str, str | int | float]:
    tags = scenario.tags
    environment = scenario.environment
    hazards = str(tags.get("hazard_composition") or tags.get("primary_hazard_type") or scenario.cluster)
    condition = str(tags.get("condition") or environment.get("weather") or "unknown")
    topology = str(tags.get("topology") or scenario.cluster)
    return {
        "suite": suite,
        "topology": topology,
        "hazards": hazards,
        "primary_hazard_type": str(tags.get("primary_hazard_type") or scenario.cluster),
        "condition": condition,
        "visibility": float(environment.get("visibility", 1.0) or 1.0),
        "hazard_count": int(tags.get("hazard_count", 1) or 1),
        "blocking_hazards": int(tags.get("blocking_hazards", 0) or 0),
        "ood_axes": str(tags.get("ood_axes", "")),
    }


def _composition_key(axes: dict[str, str | int | float]) -> str:
    parts = [
        str(axes["suite"]),
        str(axes["topology"]),
        str(axes["hazards"]),
        str(axes["condition"]),
    ]
    return "|".join(parts)


def _split_for(*, suite: str, seed: int, composition_key: str) -> str:
    if suite == "gauntlet":
        return "stress_holdout"
    if suite == "hidden":
        return "blind_holdout"
    bucket = int(hashlib.sha256(composition_key.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "blind_holdout"
    if seed % 5 == 0 or bucket in {1, 2}:
        return "dev_holdout"
    return "train_curriculum"


def _difficulty_bucket(difficulty: float) -> str:
    if difficulty < 0.45:
        return "l0_foundation"
    if difficulty < 0.65:
        return "l1_single_factor"
    if difficulty < 0.8:
        return "l2_compound"
    if difficulty < 0.95:
        return "l3_adversarial"
    return "l4_gauntlet"


def _task_prompt(scenario: Scenario, axes: dict[str, str | int | float]) -> str:
    return (
        "Drive from start to goal without collision while preserving clearance and progress. "
        f"Scenario: {scenario.cluster}; hazard: {axes['primary_hazard_type']}; "
        f"condition: {axes['condition']}."
    )


def _coverage(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_count": len(tasks),
        "split_counts": _counts(task["split"] for task in tasks),
        "suite_counts": _counts(task["suite"] for task in tasks),
        "topology_count": len({task["axes"]["topology"] for task in tasks}),
        "hazard_count": len({task["axes"]["primary_hazard_type"] for task in tasks}),
        "composition_count": len({task["composition_key"] for task in tasks}),
        "observation_mode_count": len(OBSERVATION_MODES),
        "grader_count": len(GRADER_LIBRARY),
    }


def _difficulty_buckets(tasks: list[dict[str, Any]]) -> dict[str, int]:
    return _counts(task["difficulty_bucket"] for task in tasks)


def _counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


if __name__ == "__main__":
    raise SystemExit(main())
