#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
from minimal_shot_av.simulator.policy import run_spotlight_reflex_policy
from minimal_shot_av.simulator.wod_scenarios import WOD_E2E_CLUSTERS, generate_wod_scenario


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Minor submission closed-loop runtime constraints.")
    parser.add_argument("--seed-start", type=int, default=1)
    parser.add_argument("--seed-end", type=int, default=5)
    parser.add_argument("--target-step-ms", type=float, default=50.0)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "minor_runtime_constraints.json")
    args = parser.parse_args()

    report = build_report(
        seed_start=args.seed_start,
        seed_end=args.seed_end,
        target_step_ms=args.target_step_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


def build_report(
    *,
    seed_start: int = 1,
    seed_end: int = 5,
    target_step_ms: float = 50.0,
) -> dict[str, Any]:
    if seed_end < seed_start:
        raise ValueError("seed_end must be >= seed_start")
    if target_step_ms <= 0.0:
        raise ValueError("target_step_ms must be positive")

    rows: list[dict[str, Any]] = []
    for cluster in WOD_E2E_CLUSTERS:
        for seed in range(seed_start, seed_end + 1):
            rows.append(_measure_rollout("wod", cluster, seed, generate_wod_scenario(cluster, seed)))
    for suite in ("compositional", "adversarial", "gauntlet", "hidden"):
        for seed in range(seed_start, seed_end + 1):
            rows.append(_measure_rollout(suite, f"{suite}_case", seed, generate_compositional_scenario(seed, suite)))

    step_latencies = [row["mean_step_ms"] for row in rows]
    rollout_latencies = [row["rollout_ms"] for row in rows]
    steps = [row["steps"] for row in rows]
    p95_step_ms = _percentile(step_latencies, 95.0)
    p95_rollout_ms = _percentile(rollout_latencies, 95.0)
    total_steps = sum(steps)
    total_runtime_s = sum(rollout_latencies) / 1000.0
    gates = {
        "all_rollouts_successful": all(row["success"] for row in rows),
        "zero_collisions": not any(row["collision"] for row in rows),
        "p95_step_latency_within_budget": p95_step_ms <= target_step_ms,
        "all_step_latencies_finite": all(row["mean_step_ms"] > 0.0 for row in rows),
    }
    return {
        "schema": "minor_runtime_constraints_audit_v1",
        "valid": all(gates.values()),
        "policy": "spotlight-reflex",
        "contract": "closed_loop_simulator_control_runtime",
        "target_step_ms": float(target_step_ms),
        "run_count": len(rows),
        "total_steps": total_steps,
        "p95_step_latency_ms": p95_step_ms,
        "mean_step_latency_ms": statistics.mean(step_latencies),
        "p95_rollout_latency_ms": p95_rollout_ms,
        "throughput_steps_per_second": total_steps / total_runtime_s if total_runtime_s > 0.0 else None,
        "success_count": sum(1 for row in rows if row["success"]),
        "collision_count": sum(1 for row in rows if row["collision"]),
        "gates": gates,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
        "notes": [
            (
                "Measures the closed-loop simulator policy path, including perception, world-state update, "
                "maneuver selection, safety filtering, and rollout bookkeeping."
            ),
            "Excludes SVG rendering, JSON serialization, archive packaging, WOD parsing, and model training.",
            (
                "The 50 ms target corresponds to a conservative 20 Hz control-loop budget for this "
                "abstract simulator tier."
            ),
        ],
        "runs": rows,
    }


def _measure_rollout(suite: str, cluster: str, seed: int, scenario) -> dict[str, Any]:
    started = time.perf_counter()
    rollout = run_spotlight_reflex_policy(scenario)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    steps = max(1, len(rollout.steps))
    return {
        "suite": suite,
        "cluster": cluster,
        "seed": seed,
        "success": rollout.success,
        "collision": rollout.collision,
        "reached_goal": rollout.reached_goal,
        "steps": len(rollout.steps),
        "rollout_ms": elapsed_ms,
        "mean_step_ms": elapsed_ms / steps,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


if __name__ == "__main__":
    raise SystemExit(main())
