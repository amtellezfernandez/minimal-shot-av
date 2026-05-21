from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_scenarios import (
    _max_intervention_rate,
    _min_avg_progress,
    _near_miss_clearance,
    summarize_rollout,
)
from minimal_shot_av.simulator.compositional_scenarios import (  # noqa: E402
    COMPOSITIONAL_TOPOLOGIES,
    generate_compositional_scenario,
)
from minimal_shot_av.simulator.environment import Obstacle, Scenario, interpolate_lane
from minimal_shot_av.simulator.policy import run_spotlight_reflex_policy


DEFAULT_BUDGETS = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.9, 1.15, 1.5, 2.0)
SWEEP_FAMILIES = ("coupled_stealth", "radius_only", "count_only", "placement_only")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep seizure-like perception perturbation budgets by route topology."
    )
    parser.add_argument("--seeds-per-topology", type=int, default=4)
    parser.add_argument("--suite", choices=("compositional", "hidden", "adversarial"), default="hidden")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument(
        "--match-primary-hazard",
        default="",
        help="If set, sample only scenarios with this primary hazard on every topology.",
    )
    parser.add_argument(
        "--match-condition",
        default="",
        help="If set, sample only scenarios with this condition on every topology.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/seizure_topology_susceptibility"))
    parser.add_argument(
        "--budgets",
        type=float,
        nargs="*",
        default=list(DEFAULT_BUDGETS),
        help="Perturbation budgets to sweep. Zero is the clean condition.",
    )
    parser.add_argument(
        "--sweep-family",
        choices=(*SWEEP_FAMILIES, "all"),
        default="all",
        help="Attack control to sweep. 'all' separates radius, count, and placement controls.",
    )
    args = parser.parse_args()

    clean_rows: list[dict[str, Any]] = []
    sweep_rows: list[dict[str, Any]] = []
    match_primary_hazard = args.match_primary_hazard.strip()
    match_condition = args.match_condition.strip()
    selected = _select_scenarios(
        args.suite,
        args.seed_start,
        args.seeds_per_topology,
        match_primary_hazard=match_primary_hazard,
        match_condition=match_condition,
    )

    for topology, scenarios in selected.items():
        for scenario_index, scenario in enumerate(scenarios):
            clean_rollout = run_spotlight_reflex_policy(scenario)
            clean_summary = summarize_rollout(
                args.suite,
                scenario.cluster,
                scenario.seed,
                "spotlight-reflex",
                clean_rollout,
                scenario.goal,
                scenario.tags,
            )
            clean_rows.append({**clean_summary, **_topology_metrics(scenario), "scenario_index": scenario_index})

            clean_progress = _total_progress(clean_rollout.steps)
            clean_pass = bool(clean_summary["benchmark_pass"])
            for family in _selected_families(args.sweep_family):
                for budget in args.budgets:
                    attack_parameters = _attack_parameters(family, budget)
                    attacked = _with_seizure_phantoms(scenario, attack_parameters)
                    rollout = run_spotlight_reflex_policy(attacked)
                    summary = summarize_rollout(
                        args.suite,
                        attacked.cluster,
                        attacked.seed,
                        "spotlight-reflex+seizure-phantoms",
                        rollout,
                        attacked.goal,
                        attacked.tags,
                    )
                    progress_ratio = _total_progress(rollout.steps) / max(clean_progress, 1e-9)
                    collapse = _collapse(clean_pass, clean_summary, summary, progress_ratio)
                    collapse_reason = _collapse_reason(args.suite, clean_pass, summary, progress_ratio)
                    sweep_rows.append(
                        {
                            **summary,
                            **_topology_metrics(scenario),
                            "scenario_index": scenario_index,
                            "sweep_family": family,
                            "attack_budget": budget,
                            "phantom_count": attack_parameters["phantom_count"],
                            "phantom_radius_m": attack_parameters["phantom_radius_m"],
                            "phantom_lateral_scale": attack_parameters["phantom_lateral_scale"],
                            "placement_strategy": attack_parameters["placement_strategy"],
                            "stealth_bounded": attack_parameters["stealth_bounded"],
                            "clean_benchmark_pass": clean_pass,
                            "clean_progress": round(clean_progress, 4),
                            "attacked_progress": round(_total_progress(rollout.steps), 4),
                            "progress_ratio": round(progress_ratio, 4),
                            "collapse": collapse,
                            "collapse_reason": collapse_reason,
                            "collapse_reason_family": collapse_reason.split(":", maxsplit=1)[0],
                            "attack_model": "phase_locked_phantom_guard_obstacles",
                        }
                    )

    topology_summary = _summarize_by_topology(sweep_rows)
    stratified_summary = _summarize_strata(sweep_rows)
    confound_audit = _confound_audit(sweep_rows)
    payload = {
        "attack_model": {
            "name": "phase_locked_phantom_guard_obstacles",
            "interpretation": (
                "A bounded perception/control seizure proxy: false obstacle guards are injected at repeated "
                "route phases. The default evidence separates radius-only, count-only, placement-only, and "
                "coupled stealth controls; it is not a claim that the simulator contains a real neural seizure."
            ),
            "collapse_definition": (
                "A clean-passing scenario collapses if the attacked rollout collides, fails a decomposed "
                "benchmark criterion, or keeps less than 65% of clean progress."
            ),
            "sweep_families": list(_selected_families(args.sweep_family)),
            "matched_design": bool(match_primary_hazard or match_condition),
            "match_primary_hazard": match_primary_hazard,
            "match_condition": match_condition,
        },
        "clean_runs": clean_rows,
        "sweep_runs": sweep_rows,
        "topology_summary": topology_summary,
        "stratified_summary": stratified_summary,
        "confound_audit": confound_audit,
        "ranking": sorted(
            topology_summary,
            key=lambda row: (
                float(row["critical_attack_budget"]) if row["critical_attack_budget"] is not None else math.inf,
                float(row["first_any_collapse_budget"]) if row["first_any_collapse_budget"] is not None else math.inf,
                -float(row["collapse_rate_at_max_budget"]),
                -float(row["collapse_rate_at_0_35"]),
                -float(row["mean_curvature_per_100m"]),
            ),
        ),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "topology_susceptibility_sweep.csv", sweep_rows)
    _write_csv(args.output_dir / "topology_susceptibility_summary.csv", topology_summary)
    _write_csv(args.output_dir / "topology_susceptibility_strata.csv", stratified_summary)
    (args.output_dir / "topology_susceptibility.json").write_text(json.dumps(payload, indent=2))
    print(f"Wrote topology susceptibility audit to {args.output_dir}")


def _select_scenarios(
    suite: str,
    seed_start: int,
    seeds_per_topology: int,
    match_primary_hazard: str = "",
    match_condition: str = "",
) -> dict[str, list[Scenario]]:
    selected = {topology: [] for topology in COMPOSITIONAL_TOPOLOGIES}
    seed = seed_start
    max_seed = seed_start + 120_000
    while seed < max_seed and any(len(rows) < seeds_per_topology for rows in selected.values()):
        scenario = generate_compositional_scenario(seed, suite=suite)
        topology = str(scenario.tags.get("topology", ""))
        if (
            topology in selected
            and len(selected[topology]) < seeds_per_topology
            and _matches_scenario(scenario, match_primary_hazard, match_condition)
        ):
            selected[topology].append(scenario)
        seed += 1
    missing = [topology for topology, rows in selected.items() if len(rows) < seeds_per_topology]
    if missing:
        filters = []
        if match_primary_hazard:
            filters.append(f"primary_hazard={match_primary_hazard}")
        if match_condition:
            filters.append(f"condition={match_condition}")
        suffix = f" with filters {' '.join(filters)}" if filters else ""
        raise RuntimeError(f"could not sample enough scenarios for: {', '.join(missing)}{suffix}")
    return selected


def _matches_scenario(scenario: Scenario, match_primary_hazard: str, match_condition: str) -> bool:
    if match_primary_hazard and str(scenario.tags.get("primary_hazard_type", "")) != match_primary_hazard:
        return False
    if match_condition and str(scenario.tags.get("condition", "")) != match_condition:
        return False
    return True


def _selected_families(requested: str) -> tuple[str, ...]:
    if requested == "all":
        return SWEEP_FAMILIES
    return (requested,)


def _attack_parameters(family: str, budget: float) -> dict[str, Any]:
    budget = max(0.0, float(budget))
    if family == "radius_only":
        phantom_count = 4
        radius = 0.45 + budget * 0.95
        lateral_scale = 0.08
        placement = "curvature_guard"
    elif family == "count_only":
        phantom_count = max(1, int(round(2 + budget * 7)))
        radius = 0.95
        lateral_scale = 0.08
        placement = "curvature_guard"
    elif family == "placement_only":
        phantom_count = 4
        radius = 0.95
        lateral_scale = max(0.02, 0.14 - budget * 0.10)
        placement = "curvature_guard"
    elif family == "coupled_stealth":
        phantom_count = max(1, int(round(2 + budget * 4)))
        radius = 0.55 + budget * 0.80
        lateral_scale = max(0.035, 0.11 - budget * 0.065)
        placement = "curvature_guard"
    else:
        raise ValueError(f"unknown sweep family {family!r}")
    return {
        "sweep_family": family,
        "attack_budget": round(budget, 4),
        "phantom_count": phantom_count,
        "phantom_radius_m": round(min(radius, 1.45), 4),
        "phantom_lateral_scale": round(lateral_scale, 4),
        "placement_strategy": placement,
        "stealth_bounded": True,
    }


def _with_seizure_phantoms(scenario: Scenario, attack: dict[str, Any]) -> Scenario:
    budget = float(attack["attack_budget"])
    if budget <= 0.0:
        return scenario
    lane = interpolate_lane(scenario.lane_center, samples_per_segment=12)
    count = int(attack["phantom_count"])
    radius = float(attack["phantom_radius_m"])
    lateral_scale = float(attack["phantom_lateral_scale"])
    phase_indices = _guard_phase_indices(lane, count)
    phantoms: list[Obstacle] = []
    for index, lane_index in enumerate(phase_indices):
        x, y = lane[lane_index]
        previous_point = lane[max(0, lane_index - 1)]
        next_point = lane[min(len(lane) - 1, lane_index + 1)]
        tangent = (next_point[0] - previous_point[0], next_point[1] - previous_point[1])
        norm = math.hypot(tangent[0], tangent[1]) or 1.0
        normal = (-tangent[1] / norm, tangent[0] / norm)
        side = -1.0 if index % 2 else 1.0
        offset = side * scenario.lane_half_width * lateral_scale
        phantoms.append(
            Obstacle(
                x=x + normal[0] * offset,
                y=y + normal[1] * offset,
                radius=radius,
                kind="seizure_phantom",
                label=f"seizure_budget_{budget:.2f}_phase_{index}",
            )
        )
    tags = {
        **scenario.tags,
        "seizure_attack_budget": round(budget, 4),
        "seizure_attack_model": "phase_locked_phantom_guard_obstacles",
        "seizure_sweep_family": attack["sweep_family"],
        "seizure_phantom_count": count,
        "seizure_phantom_radius_m": radius,
        "seizure_phantom_lateral_scale": lateral_scale,
    }
    return replace(scenario, obstacles=[*scenario.obstacles, *phantoms], tags=tags)


def _guard_phase_indices(lane: list[tuple[float, float]], count: int) -> list[int]:
    if count <= 0:
        return []
    curvature_scores: list[tuple[float, int]] = []
    for index in range(1, len(lane) - 1):
        first, second, third = lane[index - 1], lane[index], lane[index + 1]
        h1 = math.atan2(second[1] - first[1], second[0] - first[0])
        h2 = math.atan2(third[1] - second[1], third[0] - second[0])
        curvature_scores.append((abs(_angle_delta(h2, h1)), index))
    high_curvature = [
        index
        for _, index in sorted(curvature_scores, reverse=True)
        if len(lane) * 0.12 <= index <= len(lane) * 0.88
    ][: max(1, count // 2)]
    evenly_spaced = [
        min(len(lane) - 1, max(0, int(((slot + 1) / (count + 1)) * (len(lane) - 1)))) for slot in range(count)
    ]
    selected: list[int] = []
    for index in [*high_curvature, *evenly_spaced]:
        if index not in selected:
            selected.append(index)
        if len(selected) == count:
            break
    return selected


def _topology_metrics(scenario: Scenario) -> dict[str, Any]:
    lane = scenario.lane_center
    length = sum(math.dist(a, b) for a, b in zip(lane, lane[1:]))
    heading_changes: list[float] = []
    for first, second, third in zip(lane, lane[1:], lane[2:]):
        h1 = math.atan2(second[1] - first[1], second[0] - first[0])
        h2 = math.atan2(third[1] - second[1], third[0] - second[0])
        heading_changes.append(abs(_angle_delta(h2, h1)))
    curvature = sum(heading_changes) / max(length, 1e-9)
    feature_kinds = {str(feature.get("kind", "")) for feature in scenario.map_features}
    return {
        "route_length_m": round(length, 3),
        "lane_half_width_m": round(scenario.lane_half_width, 3),
        "curvature_per_100m": round(curvature * 100.0, 4),
        "topology_feature_count": len(feature_kinds),
        "has_speed_limit_feature": "speed_limit" in feature_kinds,
        "has_temporary_alignment_feature": "temporary_alignment" in feature_kinds,
        "has_roundabout_feature": "roundabout_entry" in feature_kinds,
        "has_pullout_feature": "single_track_pullout" in feature_kinds,
    }


def _angle_delta(a: float, b: float) -> float:
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


def _total_progress(steps: list[Any]) -> float:
    return sum(max(0.0, float(step.progress or 0.0)) for step in steps)


def _collapse(
    clean_pass: bool,
    clean_summary: dict[str, Any],
    attacked_summary: dict[str, Any],
    progress_ratio: float,
) -> bool:
    if not clean_pass:
        return False
    if bool(attacked_summary["collision"]):
        return True
    if not bool(attacked_summary["benchmark_pass"]):
        return True
    return progress_ratio < 0.65 and float(clean_summary["avg_progress"]) > 0.0


def _collapse_reason(suite: str, clean_pass: bool, attacked_summary: dict[str, Any], progress_ratio: float) -> str:
    if not clean_pass:
        return "clean_failed"
    if bool(attacked_summary["collision"]):
        return "collision"
    if bool(attacked_summary["near_miss"]):
        return f"near_miss:min_clearance<{_near_miss_clearance(suite):.2f}"
    if bool(attacked_summary["trajectory_safety_event_count"]):
        return "trajectory_safety_event"
    if bool(attacked_summary["excessive_intervention"]):
        return f"excessive_intervention:rate>{_max_intervention_rate(suite):.2f}"
    if bool(attacked_summary["slow_crawl"]):
        return f"slow_crawl:avg_progress<{_min_avg_progress(suite):.2f}"
    if not bool(attacked_summary["reached_goal"]):
        return "goal_failure"
    if progress_ratio < 0.65:
        return "progress_collapse"
    if not bool(attacked_summary["benchmark_pass"]):
        return "benchmark_failure:other"
    return "no_collapse"


def _summarize_by_topology(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    groups = sorted({(str(row["topology"]), str(row["sweep_family"])) for row in rows})
    for topology, family in groups:
        scoped = [row for row in rows if row["topology"] == topology and row["sweep_family"] == family]
        budgets = sorted({float(row["attack_budget"]) for row in scoped})
        scenario_count = len({int(row["scenario_index"]) for row in scoped})
        clean_passes = {
            int(row["scenario_index"])
            for row in scoped
            if float(row["attack_budget"]) == 0.0 and bool(row["clean_benchmark_pass"])
        }
        critical: float | None = None
        first_any: float | None = None
        for budget in budgets:
            at_budget = [
                row
                for row in scoped
                if float(row["attack_budget"]) == budget and int(row["scenario_index"]) in clean_passes
            ]
            if first_any is None and any(bool(row["collapse"]) for row in at_budget):
                first_any = budget
            if at_budget and sum(1 for row in at_budget if bool(row["collapse"])) / len(at_budget) >= 0.5:
                critical = budget
                break
        at_035 = [
            row
            for row in scoped
            if abs(float(row["attack_budget"]) - 0.35) < 1e-9 and int(row["scenario_index"]) in clean_passes
        ]
        collapse_rate_035 = (sum(1 for row in at_035 if bool(row["collapse"])) / len(at_035)) if at_035 else 0.0
        max_budget = max(budgets)
        at_max = [
            row
            for row in scoped
            if abs(float(row["attack_budget"]) - max_budget) < 1e-9 and int(row["scenario_index"]) in clean_passes
        ]
        collapse_rate_max = (sum(1 for row in at_max if bool(row["collapse"])) / len(at_max)) if at_max else 0.0
        clean_rows = [row for row in scoped if float(row["attack_budget"]) == 0.0]
        summary.append(
            {
                "topology": topology,
                "sweep_family": family,
                "scenario_count": scenario_count,
                "clean_pass_count": len(clean_passes),
                "critical_attack_budget": critical,
                "first_any_collapse_budget": first_any,
                "collapse_rate_at_0_35": round(collapse_rate_035, 4),
                "collapse_rate_at_max_budget": round(collapse_rate_max, 4),
                "max_attack_budget": max_budget,
                "mean_lane_half_width_m": round(_mean(clean_rows, "lane_half_width_m"), 3),
                "mean_curvature_per_100m": round(_mean(clean_rows, "curvature_per_100m"), 4),
                "mean_route_length_m": round(_mean(clean_rows, "route_length_m"), 3),
            }
        )
    return summary


def _summarize_strata(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    keys = sorted(
        {
            (
                str(row["topology"]),
                str(row["sweep_family"]),
                str(row["primary_hazard_type"]),
                _condition(row),
            )
            for row in rows
            if bool(row["clean_benchmark_pass"])
        }
    )
    for topology, family, hazard, condition in keys:
        scoped = [
            row
            for row in rows
            if row["topology"] == topology
            and row["sweep_family"] == family
            and row["primary_hazard_type"] == hazard
            and _condition(row) == condition
            and bool(row["clean_benchmark_pass"])
        ]
        if not scoped:
            continue
        scenarios = {int(row["scenario_index"]) for row in scoped}
        collapse_rows = [row for row in scoped if bool(row["collapse"])]
        first_any = min((float(row["attack_budget"]) for row in collapse_rows), default=None)
        max_budget = max(float(row["attack_budget"]) for row in scoped)
        max_budget_rows = [row for row in scoped if abs(float(row["attack_budget"]) - max_budget) < 1e-9]
        summary.append(
            {
                "topology": topology,
                "sweep_family": family,
                "primary_hazard_type": hazard,
                "condition": condition,
                "scenario_count": len(scenarios),
                "sweep_rows": len(scoped),
                "collapse_rows": len(collapse_rows),
                "collapse_row_rate": round(len(collapse_rows) / len(scoped), 4),
                "first_any_collapse_budget": first_any,
                "collapse_rate_at_max_budget": round(
                    sum(1 for row in max_budget_rows if bool(row["collapse"])) / len(max_budget_rows),
                    4,
                ),
                "dominant_collapse_reason": _dominant_reason(collapse_rows),
            }
        )
    return summary


def _confound_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean_rows = [row for row in rows if bool(row["clean_benchmark_pass"])]
    collapse_rows = [row for row in clean_rows if bool(row["collapse"])]
    collapse_count = len(collapse_rows)
    hazards = _counts(collapse_rows, "primary_hazard_type")
    topologies = _counts(collapse_rows, "topology")
    families = _counts(collapse_rows, "sweep_family")
    conditions = _condition_counts(collapse_rows)
    top_hazard_count = max(hazards.values(), default=0)
    top_hazard_share = top_hazard_count / collapse_count if collapse_count else 0.0
    affected_topology_count = len(topologies)
    affected_hazard_count = len(hazards)
    selected_hazards = {str(row["primary_hazard_type"]) for row in clean_rows}
    selected_conditions = {_condition(row) for row in clean_rows}
    matched_primary_hazard = len(selected_hazards) == 1
    matched_condition = len(selected_conditions) == 1
    interpretation = "no_clean_passing_collapses"
    if collapse_count:
        if matched_primary_hazard:
            interpretation = "matched_hazard"
        else:
            interpretation = "hazard_concentrated" if top_hazard_share >= 0.70 else "multi_hazard"
        if affected_topology_count <= 2:
            interpretation += "_narrow_topology"
        else:
            interpretation += "_multi_topology"
    return {
        "clean_passing_sweep_rows": len(clean_rows),
        "collapse_rows": collapse_count,
        "collapse_row_rate": round(collapse_count / len(clean_rows), 4) if clean_rows else 0.0,
        "affected_topology_count": affected_topology_count,
        "affected_hazard_count": affected_hazard_count,
        "matched_primary_hazard": matched_primary_hazard,
        "matched_condition": matched_condition,
        "selected_hazards": sorted(selected_hazards),
        "selected_conditions": sorted(selected_conditions),
        "top_hazard_share": round(top_hazard_share, 4),
        "hazard_counts": hazards,
        "topology_counts": topologies,
        "sweep_family_counts": families,
        "condition_counts": conditions,
        "interpretation": interpretation,
        "claim_boundary": (
            "Use this as a topology-by-scenario-family susceptibility audit. Do not claim a pure topology "
            "effect unless collapse rows remain distributed across hazards or a matched-hazard sweep is run."
        ),
    }


def _condition(row: dict[str, Any]) -> str:
    for axis in str(row.get("ood_axes", "")).split(","):
        if axis.startswith("condition:"):
            return axis.split(":", maxsplit=1)[1]
    return "unknown"


def _counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _condition_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        condition = _condition(row)
        counts[condition] = counts.get(condition, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _dominant_reason(rows: list[dict[str, Any]]) -> str:
    counts = _counts(rows, "collapse_reason_family")
    return next(iter(counts), "none")


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    values = [float(row[key]) for row in rows]
    return sum(values) / max(1, len(values))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
