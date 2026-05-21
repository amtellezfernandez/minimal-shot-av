from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.simulator.compositional_scenarios import COMPOSITIONAL_SUITES, generate_compositional_scenario
from minimal_shot_av.simulator.environment import generate_scenario, scenario_to_dict, write_rollout
from minimal_shot_av.simulator.policy import (
    DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG,
    run_policy,
    run_spotlight_reflex_policy,
)
from minimal_shot_av.simulator.render import render_svg
from minimal_shot_av.simulator.wod_scenarios import WOD_E2E_CLUSTERS, generate_wod_scenario


SHOWCASE_PRESETS: dict[str, Callable[[], object | None]] = {
    "default": lambda: None,
    "smooth-showcase": lambda: _spotlight_rollout_variant(1.8, 2.5, 0.8, 1.1),
    "showcase-spotlight": lambda: _spotlight_rollout_variant(1.8, 2.5, 0.65, 1.1),
    "showcase-intersection": lambda: _spotlight_rollout_variant(1.8, 2.5, 0.65, 1.1),
    "showcase-construction": lambda: _spotlight_rollout_variant(2.2, 3.0, 0.65, 1.4),
    "showcase-fod": lambda: _spotlight_rollout_variant(1.8, 2.5, 0.8, 1.1),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a closed-loop simulator demo.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for scenario generation.")
    parser.add_argument(
        "--scenario-cluster",
        choices=("baseline",) + WOD_E2E_CLUSTERS + COMPOSITIONAL_SUITES,
        default="baseline",
        help="Procedural scenario family to generate.",
    )
    parser.add_argument(
        "--policy",
        choices=("baseline", "spotlight-reflex"),
        default="baseline",
        help="Policy implementation to run.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--rollout-preset",
        choices=tuple(SHOWCASE_PRESETS),
        default="default",
        help="Optional rollout tuning preset. Showcase presets are for presentation demos, not audited defaults.",
    )
    args = parser.parse_args()

    if args.scenario_cluster == "baseline":
        scenario = generate_scenario(seed=args.seed)
        scenario_source = "baseline random lane-and-obstacle simulator"
    elif args.scenario_cluster in COMPOSITIONAL_SUITES:
        scenario = generate_compositional_scenario(seed=args.seed, suite=args.scenario_cluster)
        scenario_source = "compositional OOD scenario generator"
    else:
        scenario = generate_wod_scenario(args.scenario_cluster, seed=args.seed)
        scenario_source = "WOD-E2E procedural cluster generator"
    if args.policy == "spotlight-reflex":
        rollout_config = _spotlight_rollout_config(args.rollout_preset, parser)
        rollout = run_spotlight_reflex_policy(scenario, config=rollout_config)
        planner_description = (
            "deterministic maneuver library with simulator-native trajectory selector reference selection"
        )
        safety_description = "selector-chosen action with existing uncertainty-aware safety filter"
        decision_explainability = (
            "per-step selector references, score terms, and top candidate summaries"
        )
    else:
        if args.rollout_preset != "default":
            parser.error("--rollout-preset only applies to --policy spotlight-reflex")
        rollout = run_policy(scenario)
        planner_description = "repo-internal potential-field reactive planner toward the target waypoint"
        safety_description = "uncertainty-aware slowdown and emergency stop"
        decision_explainability = "repo-internal reactive baseline rollout metrics only"

    payload = {
        "scenario": scenario_to_dict(scenario),
        "architecture": {
            "policy": args.policy,
            "rollout_preset": args.rollout_preset,
            "scenario_source": scenario_source,
            "perception": "lane and obstacle perception with confidence estimation",
            "world_model": "short-horizon progress and collision-risk state",
            "planner": planner_description,
            "safety_filter": safety_description,
            "decision_explainability": decision_explainability,
        },
        "rollout": {
            "success": rollout.success,
            "collision": rollout.collision,
            "reached_goal": rollout.reached_goal,
            "steps": [step.__dict__ for step in rollout.steps],
        },
    }

    json_path = args.artifacts_dir / "latest_rollout.json"
    svg_path = args.artifacts_dir / "latest_rollout.svg"
    write_rollout(json_path, payload)
    render_svg(svg_path, scenario, rollout)

    status = "success" if rollout.success else "failure"
    print(f"Generated {json_path} and {svg_path} ({status}).")


def _spotlight_rollout_config(preset: str, parser: argparse.ArgumentParser):
    factory = SHOWCASE_PRESETS.get(preset)
    if factory is None:
        parser.error(f"Unsupported rollout preset: {preset}")
    return factory()


def _spotlight_rollout_variant(
    max_accel_mps2: float,
    max_decel_mps2: float,
    max_abs_steering_rad: float,
    max_steering_rate_rad_s: float,
):
    return replace(
        DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG,
        max_accel_mps2=max_accel_mps2,
        max_decel_mps2=max_decel_mps2,
        max_abs_steering_rad=max_abs_steering_rad,
        max_steering_rate_rad_s=max_steering_rate_rad_s,
    )


if __name__ == "__main__":
    main()
