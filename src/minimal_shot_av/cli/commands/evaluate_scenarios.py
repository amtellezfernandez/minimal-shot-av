from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.simulator.compositional_scenarios import (
    COMPOSITIONAL_SUITES,
    generate_compositional_scenario,
)
from minimal_shot_av.simulator.environment import Scenario
from minimal_shot_av.simulator.policy import Rollout, run_policy, run_spotlight_reflex_policy
from minimal_shot_av.simulator.wod_scenarios import WOD_E2E_CLUSTERS, generate_wod_scenario


ScenarioGenerator = Callable[[str, int], Scenario]


def summarize_rollout(
    suite: str,
    cluster: str,
    seed: int,
    policy: str,
    rollout: Rollout,
    goal: tuple[float, float],
    tags: dict,
) -> dict[str, float | int | str | bool]:
    safety_events = _trajectory_safety_events(suite, rollout)
    if rollout.steps:
        last = rollout.steps[-1]
        clearances = sorted(step.min_obstacle_distance for step in rollout.steps)
        min_clearance = clearances[0]
        p05_clearance = clearances[max(0, int(len(clearances) * 0.05) - 1)]
        max_collision_risk = max(step.collision_risk for step in rollout.steps)
        max_lane_error = max(abs(step.lane_error) for step in rollout.steps)
        avg_progress = sum((step.progress or 0.0) for step in rollout.steps) / len(rollout.steps)
        avg_comfort = sum((step.comfort_cost or 0.0) for step in rollout.steps) / len(rollout.steps)
        interventions = sum(1 for step in rollout.steps if step.intervention)
        final_goal_distance = math.dist((last.x, last.y), goal)
        stall = any(bool(step.stall) for step in rollout.steps)
        last_mode = last.action_mode
        decision_modes = len({step.action_mode for step in rollout.steps})
        intervention_rate = interventions / max(1, len(rollout.steps))
        avg_speed = sum(step.speed for step in rollout.steps) / len(rollout.steps)
    else:
        min_clearance = math.inf
        p05_clearance = math.inf
        max_collision_risk = 0.0
        max_lane_error = 0.0
        avg_progress = 0.0
        avg_comfort = 0.0
        interventions = 0
        final_goal_distance = math.inf
        stall = False
        last_mode = "none"
        decision_modes = 0
        intervention_rate = 0.0
        avg_speed = 0.0

    near_miss = min_clearance < _near_miss_clearance(suite)
    excessive_intervention = intervention_rate > _max_intervention_rate(suite)
    slow_crawl = avg_progress < _min_avg_progress(suite)
    trajectory_safety_pass = rollout.success and not safety_events
    benchmark_pass = (
        trajectory_safety_pass
        and not near_miss
        and not excessive_intervention
        and not slow_crawl
    )

    return {
        "suite": suite,
        "cluster": cluster,
        "topology": str(tags.get("topology", "")),
        "primary_hazard_type": str(tags.get("primary_hazard_type", "")),
        "difficulty": float(tags.get("difficulty", tags.get("severity", 0.0))),
        "ood_axes": str(tags.get("ood_axes", "")),
        "seed": seed,
        "policy": policy,
        "success": rollout.success,
        "collision": rollout.collision,
        "reached_goal": rollout.reached_goal,
        "safe_stall": (not rollout.success) and (not rollout.collision) and stall,
        "steps": len(rollout.steps),
        "final_goal_distance": round(final_goal_distance, 3),
        "min_clearance": round(min_clearance, 3),
        "p05_clearance": round(p05_clearance, 3),
        "max_collision_risk": round(max_collision_risk, 4),
        "max_lane_error": round(max_lane_error, 3),
        "avg_progress": round(avg_progress, 4),
        "avg_speed": round(avg_speed, 4),
        "avg_comfort_cost": round(avg_comfort, 4),
        "intervention_rate": round(intervention_rate, 4),
        "decision_mode_count": decision_modes,
        "near_miss": near_miss,
        "trajectory_safety_pass": trajectory_safety_pass,
        "trajectory_safety_event_count": len(safety_events),
        "trajectory_safety_events": ";".join(safety_events[:8]),
        "excessive_intervention": excessive_intervention,
        "slow_crawl": slow_crawl,
        "benchmark_pass": benchmark_pass,
        "last_mode": last_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate policies across WOD-style and compositional OOD scenario suites."
    )
    parser.add_argument("--seed-start", type=int, default=1, help="First seed to evaluate, inclusive.")
    parser.add_argument("--seed-end", type=int, default=20, help="Last seed to evaluate, inclusive.")
    parser.add_argument(
        "--policy",
        choices=("baseline", "spotlight-reflex", "both"),
        default="spotlight-reflex",
        help="Policy set to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/minor_eval"),
        help="Evaluation output directory.",
    )
    parser.add_argument(
        "--suite",
        choices=("wod",) + COMPOSITIONAL_SUITES + ("all",),
        default="wod",
        help="Scenario suite to evaluate.",
    )
    args = parser.parse_args()

    policies = [("spotlight-reflex", run_spotlight_reflex_policy)]
    if args.policy == "baseline":
        policies = [("baseline", run_policy)]
    elif args.policy == "both":
        policies = [("baseline", run_policy), ("spotlight-reflex", run_spotlight_reflex_policy)]

    rows: list[dict[str, float | int | str | bool]] = []
    for suite_name, cluster_names, generator in _scenario_plan(args.suite):
        for cluster in cluster_names:
            for seed in range(args.seed_start, args.seed_end + 1):
                scenario = generator(cluster, seed)
                for policy_name, policy_fn in policies:
                    rollout = policy_fn(scenario)
                    rows.append(
                        summarize_rollout(
                            suite_name,
                            scenario.cluster,
                            seed,
                            policy_name,
                            rollout,
                            scenario.goal,
                            scenario.tags,
                        )
                    )

    if not rows:
        raise RuntimeError("no evaluation rows generated")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "scenario_eval.csv"
    json_path = args.output_dir / "scenario_eval.json"

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path.write_text(
        json.dumps(
            {
                "runs": rows,
                "summary": _aggregate(rows),
                "curriculum": _curriculum_manifest(rows),
                "statistics": _statistics_manifest(rows),
            },
            indent=2,
        )
    )
    print(f"Wrote {csv_path} and {json_path}")


def _scenario_plan(suite: str) -> list[tuple[str, tuple[str, ...], ScenarioGenerator]]:
    plan: list[tuple[str, tuple[str, ...], ScenarioGenerator]] = []
    if suite in {"wod", "all"}:
        plan.append(("wod", WOD_E2E_CLUSTERS, lambda cluster, seed: generate_wod_scenario(cluster, seed)))
    selected_compositional = COMPOSITIONAL_SUITES if suite == "all" else (suite,)
    for compositional_suite in selected_compositional:
        if compositional_suite not in COMPOSITIONAL_SUITES:
            continue
        clusters = tuple(f"{compositional_suite}_case_{index}" for index in range(1, 7))
        plan.append(
            (
                compositional_suite,
                clusters,
                lambda cluster, seed, selected_suite=compositional_suite: generate_compositional_scenario(
                    seed + _case_offset(cluster),
                    selected_suite,
                ),
            )
        )
    return plan


def _case_offset(cluster: str) -> int:
    try:
        return (int(cluster.rsplit("_", 1)[1]) - 1) * 1_000
    except (IndexError, ValueError):
        return 0


def _aggregate(rows: list[dict[str, float | int | str | bool]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, str, str], list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        groups.setdefault((str(row["policy"]), str(row["suite"]), str(row["cluster"])), []).append(row)

    summary: list[dict[str, float | int | str]] = []
    for (policy, suite, cluster), group in sorted(groups.items()):
        count = len(group)
        benchmark_passes = sum(1 for row in group if row["benchmark_pass"])
        success_passes = sum(1 for row in group if row["success"])
        safety_passes = sum(1 for row in group if row["trajectory_safety_pass"])
        benchmark_ci = _wilson_interval(benchmark_passes, count)
        summary.append(
            {
                "policy": policy,
                "suite": suite,
                "cluster": cluster,
                "runs": count,
                "success_rate": round(success_passes / count, 3),
                "success_rate_ci95_low": round(_wilson_interval(success_passes, count)[0], 3),
                "success_rate_ci95_high": round(_wilson_interval(success_passes, count)[1], 3),
                "collision_rate": round(sum(1 for row in group if row["collision"]) / count, 3),
                "safe_stall_rate": round(sum(1 for row in group if row["safe_stall"]) / count, 3),
                "avg_steps": round(sum(int(row["steps"]) for row in group) / count, 2),
                "avg_min_clearance": round(sum(float(row["min_clearance"]) for row in group) / count, 3),
                "avg_p05_clearance": round(sum(float(row["p05_clearance"]) for row in group) / count, 3),
                "avg_max_collision_risk": round(sum(float(row["max_collision_risk"]) for row in group) / count, 4),
                "avg_trajectory_safety_events": round(
                    sum(int(row["trajectory_safety_event_count"]) for row in group) / count,
                    3,
                ),
                "trajectory_safety_pass_rate": round(safety_passes / count, 3),
                "avg_intervention_rate": round(sum(float(row["intervention_rate"]) for row in group) / count, 4),
                "avg_difficulty": round(sum(float(row["difficulty"]) for row in group) / count, 3),
                "near_miss_rate": round(sum(1 for row in group if row["near_miss"]) / count, 3),
                "slow_crawl_rate": round(sum(1 for row in group if row["slow_crawl"]) / count, 3),
                "benchmark_pass_rate": round(benchmark_passes / count, 3),
                "benchmark_pass_rate_ci95_low": round(benchmark_ci[0], 3),
                "benchmark_pass_rate_ci95_high": round(benchmark_ci[1], 3),
            }
        )
    return summary


def _trajectory_safety_events(suite: str, rollout: Rollout) -> list[str]:
    events: list[str] = []
    clearance_threshold = _near_miss_clearance(suite)
    high_risk_threshold = 0.90 if suite == "gauntlet" else 0.97
    high_risk_clearance_threshold = 1.25 if suite == "gauntlet" else 1.0
    for step in rollout.steps:
        if step.min_obstacle_distance < clearance_threshold:
            events.append(f"t{step.t}:clearance<{clearance_threshold:.2f}")
        if (
            step.collision_risk >= high_risk_threshold
            and step.min_obstacle_distance < high_risk_clearance_threshold
        ):
            events.append(f"t{step.t}:risk>={high_risk_threshold:.2f}")
    if rollout.collision:
        events.append("terminal:collision")
    return events


def _curriculum_manifest(rows: list[dict[str, float | int | str | bool]]) -> dict[str, object]:
    suites = sorted({str(row["suite"]) for row in rows})
    clusters = sorted({str(row["cluster"]) for row in rows})
    topologies = sorted({str(row["topology"]) for row in rows if str(row["topology"])})
    hazards = sorted(
        {str(row["primary_hazard_type"]) for row in rows if str(row["primary_hazard_type"])}
    )
    difficulties = [float(row["difficulty"]) for row in rows]
    return {
        "generator": "closed_loop_procedural_curriculum",
        "suite_count": len(suites),
        "cluster_count": len(clusters),
        "topology_count": len(topologies),
        "primary_hazard_type_count": len(hazards),
        "difficulty_min": round(min(difficulties), 3) if difficulties else 0.0,
        "difficulty_max": round(max(difficulties), 3) if difficulties else 0.0,
        "suites": suites,
        "clusters": clusters,
        "topologies": topologies,
        "primary_hazard_types": hazards,
    }


def _statistics_manifest(rows: list[dict[str, float | int | str | bool]]) -> dict[str, object]:
    groups: dict[tuple[str, str], list[dict[str, float | int | str | bool]]] = {}
    for row in rows:
        groups.setdefault((str(row["policy"]), str(row["suite"])), []).append(row)
    by_policy_suite: list[dict[str, float | int | str]] = []
    for (policy, suite), group in sorted(groups.items()):
        count = len(group)
        benchmark_passes = sum(1 for row in group if row["benchmark_pass"])
        safety_passes = sum(1 for row in group if row["trajectory_safety_pass"])
        benchmark_ci = _wilson_interval(benchmark_passes, count)
        safety_ci = _wilson_interval(safety_passes, count)
        by_policy_suite.append(
            {
                "policy": policy,
                "suite": suite,
                "runs": count,
                "pass_at_1": round(benchmark_passes / count, 3),
                "pass_at_1_ci95_low": round(benchmark_ci[0], 3),
                "pass_at_1_ci95_high": round(benchmark_ci[1], 3),
                "trajectory_safety_pass_rate": round(safety_passes / count, 3),
                "trajectory_safety_pass_ci95_low": round(safety_ci[0], 3),
                "trajectory_safety_pass_ci95_high": round(safety_ci[1], 3),
            }
        )
    return {
        "confidence_interval": "Wilson score interval, 95%",
        "unit": "closed-loop rollout",
        "by_policy_suite": by_policy_suite,
    }


def _wilson_interval(successes: int, count: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 0.0)
    phat = successes / count
    denominator = 1.0 + z * z / count
    center = (phat + z * z / (2.0 * count)) / denominator
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * count)) / count) / denominator
    return (max(0.0, center - spread), min(1.0, center + spread))


def _near_miss_clearance(suite: str) -> float:
    return 1.0 if suite == "gauntlet" else 0.55


def _max_intervention_rate(suite: str) -> float:
    return 0.28 if suite == "gauntlet" else 0.65


def _min_avg_progress(suite: str) -> float:
    return 0.55 if suite == "gauntlet" else 0.25


if __name__ == "__main__":
    main()
