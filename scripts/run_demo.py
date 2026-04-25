from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.compositional_scenarios import COMPOSITIONAL_SUITES, generate_compositional_scenario
from minimal_shot_av.environment import generate_scenario, scenario_to_dict, write_rollout
from minimal_shot_av.policy import run_policy, run_spotlight_reflex_policy
from minimal_shot_av.render import render_svg
from minimal_shot_av.wod_scenarios import WOD_E2E_CLUSTERS, generate_wod_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline minimal-shot autonomy demo.")
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
        rollout = run_spotlight_reflex_policy(scenario)
        planner_description = "deterministic maneuver library with exact RFS trust-region reference selection"
        safety_description = "RFS-selected action with existing uncertainty-aware safety filter"
    else:
        rollout = run_policy(scenario)
        planner_description = "candidate trajectory search toward target waypoint"
        safety_description = "uncertainty-aware slowdown and emergency stop"

    payload = {
        "scenario": scenario_to_dict(scenario),
        "architecture": {
            "policy": args.policy,
            "scenario_source": scenario_source,
            "perception": "lane and obstacle perception with confidence estimation",
            "world_model": "short-horizon progress and collision-risk state",
            "planner": planner_description,
            "safety_filter": safety_description,
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


if __name__ == "__main__":
    main()
