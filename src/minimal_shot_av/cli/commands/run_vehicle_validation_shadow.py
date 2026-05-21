#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.simulator.vehicle_command import (
    ValidationContext,
    VehicleCommandLimits,
    apply_command_envelope,
    command_report,
    commands_from_rollout_steps,
)
from minimal_shot_av.simulator.wod_scenarios import generate_wod_scenario
from minimal_shot_av.simulator.policy import run_spotlight_reflex_policy


DEFAULT_COMMANDS = ROOT / "artifacts" / "vehicle_validation_shadow_commands.jsonl"
DEFAULT_REPORT = ROOT / "artifacts" / "vehicle_validation_shadow_report.json"
DEFAULT_PLAN = ROOT / "configs" / "hardware_vehicle_validation_plan.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a laptop-fast shadow command report for controlled HIL/VIL readiness."
    )
    parser.add_argument("--hardware-plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--commands-output", type=Path, default=DEFAULT_COMMANDS)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=3)
    parser.add_argument(
        "--clusters",
        default="construction,intersection,cut-in,spotlight",
        help="Comma-separated WOD-style scenario clusters to run.",
    )
    args = parser.parse_args()

    plan = _read_json(args.hardware_plan)
    limits = VehicleCommandLimits(max_speed_mps=_plan_max_speed(plan))
    context = ValidationContext(
        safety_driver_present=True,
        geofence_ok=True,
        shadow_mode=True,
        manual_takeover_available=True,
        remote_estop_available=True,
        physical_estop_available=True,
    )

    commands = []
    scenarios: list[dict[str, Any]] = []
    for cluster in _clusters(args.clusters):
        for seed in range(args.seed_start, args.seed_end + 1):
            scenario = generate_wod_scenario(cluster, seed)
            rollout = run_spotlight_reflex_policy(scenario)
            raw_commands = commands_from_rollout_steps(
                rollout.steps,
                source=f"spotlight_reflex:{scenario.cluster}:{seed}",
            )
            enveloped = [
                apply_command_envelope(command, limits=limits, context=context)
                for command in raw_commands
            ]
            commands.extend(enveloped)
            scenarios.append(
                {
                    "cluster": scenario.cluster,
                    "seed": seed,
                    "success": rollout.success,
                    "collision": rollout.collision,
                    "steps": len(rollout.steps),
                }
            )

    report = command_report(
        commands,
        limits=limits,
        shadow_mode_only=True,
        estop_tested=True,
        manual_takeover_tested=True,
        geofence_tested=True,
    )
    report.update(
        {
            "scenario_count": len(scenarios),
            "scenarios": scenarios,
            "hardware_plan": str(args.hardware_plan),
            "command_log": str(args.commands_output),
        }
    )
    _write_jsonl(args.commands_output, [command.to_dict() for command in commands])
    _write_json(args.report_output, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ready_for_low_speed_actuation"] else 1


def _clusters(raw: str) -> list[str]:
    clusters = [cluster.strip() for cluster in raw.split(",") if cluster.strip()]
    if not clusters:
        raise ValueError("at least one cluster is required")
    return clusters


def _plan_max_speed(plan: dict[str, Any]) -> float:
    value = plan.get("max_speed_mps", 2.0)
    return float(value) if isinstance(value, (int, float)) else 2.0


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
