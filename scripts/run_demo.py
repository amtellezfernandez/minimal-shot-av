from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.environment import generate_scenario, scenario_to_dict, write_rollout
from minimal_shot_av.policy import run_policy
from minimal_shot_av.render import render_svg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline minimal-shot autonomy demo.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for scenario generation.")
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Output directory for generated artifacts.",
    )
    args = parser.parse_args()

    scenario = generate_scenario(seed=args.seed)
    rollout = run_policy(scenario)

    payload = {
        "scenario": scenario_to_dict(scenario),
        "architecture": {
            "perception": "lane and obstacle perception with confidence estimation",
            "world_model": "short-horizon progress and collision-risk state",
            "planner": "candidate trajectory search toward target waypoint",
            "safety_filter": "uncertainty-aware slowdown and emergency stop",
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
