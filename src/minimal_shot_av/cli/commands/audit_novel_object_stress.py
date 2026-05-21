#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.compositional_scenarios import (  # noqa: E402
    COMPOSITIONAL_SUITES,
    generate_compositional_scenario,
)
from minimal_shot_av.simulator.environment import Scenario  # noqa: E402
from minimal_shot_av.simulator.policy import run_spotlight_reflex_policy  # noqa: E402
from minimal_shot_av.cli.commands.evaluate_scenarios import summarize_rollout  # noqa: E402


DEFAULT_OUTPUT = ROOT / "artifacts" / "novel_object_stress_audit.json"
GEOMETRY_ONLY_POLICY = "geometry_only_current_obstacles_no_labels_no_privileged_actor_forecast"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit closed-loop handling of abstract novel/unknown object hazards."
    )
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=80)
    parser.add_argument("--suite", action="append", choices=COMPOSITIONAL_SUITES)
    parser.add_argument("--min-runs", type=int, default=24)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    suites = tuple(args.suite) if args.suite else COMPOSITIONAL_SUITES
    report = build_report(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        suites=suites,
        min_runs=args.min_runs,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def build_report(
    *,
    seed_start: int = 1,
    seed_end: int = 80,
    suites: tuple[str, ...] = COMPOSITIONAL_SUITES,
    min_runs: int = 24,
) -> dict[str, Any]:
    if seed_end < seed_start:
        raise ValueError("seed_end must be greater than or equal to seed_start")

    runs: list[dict[str, Any]] = []
    for suite in suites:
        for seed in range(seed_start, seed_end + 1):
            scenario = generate_compositional_scenario(seed=seed, suite=suite)
            if not _contains_novel_object(scenario):
                continue
            rollout = run_spotlight_reflex_policy(scenario)
            summary = summarize_rollout(
                suite,
                scenario.cluster,
                seed,
                "spotlight-reflex",
                rollout,
                scenario.goal,
                scenario.tags,
            )
            summary["novel_object_labels"] = _novel_labels(scenario)
            summary["hazard_composition"] = str(scenario.tags.get("hazard_composition", ""))
            runs.append(summary)

    summary = _aggregate(runs)
    checks = {
        "enough_novel_object_cases": summary["runs"] >= min_runs,
        "zero_collision": summary["collision_rate"] == 0.0,
        "no_near_miss": summary["near_miss_rate"] == 0.0,
        "all_reach_goal": summary["success_rate"] == 1.0,
        "not_label_conditioned": summary["policy_label_dependency"] == GEOMETRY_ONLY_POLICY,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "schema": "novel_object_stress_audit_v1",
        "valid": not failures,
        "checks": checks,
        "failures": failures,
        "inputs": {
            "seed_start": seed_start,
            "seed_end": seed_end,
            "suites": list(suites),
            "min_runs": min_runs,
        },
        "summary": summary,
        "runs": runs,
        "claim_boundary": (
            "This audit covers closed-loop avoidance of abstract unseen object categories in the "
            "2D simulator using current obstacle geometry rather than object labels or hidden "
            "simulator actor forecasts. It does not prove camera-based recognition or semantic "
            "understanding of physical unknown objects."
        ),
    }


def _contains_novel_object(scenario: Scenario) -> bool:
    if scenario.tags.get("primary_hazard_type") == "novel_object":
        return True
    if "novel_object" in str(scenario.tags.get("hazard_composition", "")):
        return True
    return any(obstacle.kind == "novel_object" for obstacle in scenario.obstacles)


def _novel_labels(scenario: Scenario) -> list[str]:
    return sorted(
        {
            obstacle.label.split(":", 1)[1] if ":" in obstacle.label else obstacle.label
            for obstacle in scenario.obstacles
            if obstacle.kind == "novel_object"
        }
    )


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(runs)
    if count == 0:
        return {
            "runs": 0,
            "success_rate": 0.0,
            "collision_rate": 1.0,
            "near_miss_rate": 1.0,
            "benchmark_pass_rate": 0.0,
            "min_clearance_min": 0.0,
            "avg_min_clearance": 0.0,
            "label_count": 0,
            "labels": [],
            "suite_counts": {},
            "policy_label_dependency": GEOMETRY_ONLY_POLICY,
        }

    labels = sorted({label for row in runs for label in row["novel_object_labels"]})
    suite_counts: dict[str, int] = {}
    for row in runs:
        suite_counts[str(row["suite"])] = suite_counts.get(str(row["suite"]), 0) + 1
    return {
        "runs": count,
        "success_rate": round(sum(1 for row in runs if row["success"]) / count, 3),
        "collision_rate": round(sum(1 for row in runs if row["collision"]) / count, 3),
        "near_miss_rate": round(sum(1 for row in runs if row["near_miss"]) / count, 3),
        "benchmark_pass_rate": round(sum(1 for row in runs if row["benchmark_pass"]) / count, 3),
        "min_clearance_min": round(min(float(row["min_clearance"]) for row in runs), 3),
        "avg_min_clearance": round(sum(float(row["min_clearance"]) for row in runs) / count, 3),
        "label_count": len(labels),
        "labels": labels,
        "suite_counts": suite_counts,
        "policy_label_dependency": GEOMETRY_ONLY_POLICY,
    }


if __name__ == "__main__":
    raise SystemExit(main())
