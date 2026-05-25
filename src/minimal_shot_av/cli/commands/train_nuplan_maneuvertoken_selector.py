#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_shot_av.model.nuplan_devkit_integration import NuPlanSceneFilter
from minimal_shot_av.model.nuplan_devkit_integration import load_nuplan_scenes
from minimal_shot_av.model.nuplan_maneuver_token_selector import build_training_examples
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_selector_mlp


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_maneuvertoken_selector.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a small ManeuverToken selector MLP from real nuPlan DB scenarios "
            "or scene summaries with expert trajectories."
        )
    )
    parser.add_argument("--input", type=Path, help="JSON list of scenes with expert_trajectory.")
    parser.add_argument(
        "--nuplan-data-root",
        type=Path,
        help="nuPlan dataset root or split directory containing DB files.",
    )
    parser.add_argument(
        "--nuplan-db-file",
        action="append",
        default=[],
        help="Specific nuPlan DB file or DB directory. Pass multiple times if needed.",
    )
    parser.add_argument("--scenario-type", action="append", default=[], help="Optional nuPlan scenario type filter.")
    parser.add_argument("--scenario-token", action="append", default=[], help="Optional nuPlan token filter.")
    parser.add_argument("--log-name", action="append", default=[], help="Optional nuPlan log filename filter.")
    parser.add_argument("--map-name", action="append", default=[], help="Optional nuPlan map-name filter.")
    parser.add_argument("--limit", type=int, help="Limit the number of loaded nuPlan scenes.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    scenes = _load_scenes(args)
    examples = build_training_examples(scenes)
    selector, metrics = fit_selector_mlp(
        examples,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    payload = {
        **selector.to_payload(),
        "training_metrics": metrics,
        "token_order": list({str(example["token"]) for example in examples}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "training_metrics": metrics}, indent=2, sort_keys=True))
    return 0


def _load_scenes(args: argparse.Namespace) -> list[dict]:
    if args.input is not None:
        return json.loads(args.input.read_text(encoding="utf-8"))
    if args.nuplan_data_root is None and not args.nuplan_db_file:
        raise ValueError("pass either --input or --nuplan-data-root/--nuplan-db-file")
    data_root = args.nuplan_data_root or Path(".")
    return load_nuplan_scenes(
        data_root=data_root,
        scene_filter=NuPlanSceneFilter(
            db_files=tuple(str(value) for value in args.nuplan_db_file),
            log_names=tuple(str(value) for value in args.log_name),
            map_names=tuple(str(value) for value in args.map_name),
            scenario_tokens=tuple(str(value) for value in args.scenario_token),
            scenario_types=tuple(str(value) for value in args.scenario_type),
            limit=args.limit,
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
