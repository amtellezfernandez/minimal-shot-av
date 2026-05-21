"""Token-count ablation across gauntlet, adversarial, and compositional suites.

Runs 4 nested token sets on each suite. N_CASES and seeds per suite:
  gauntlet:      6 cases × 80 seeds = 480 rollouts / config
  adversarial:   2 cases × 80 seeds = 160 rollouts / config
  compositional: 5 cases × 80 seeds = 400 rollouts / config

Reports pass rate and collision rate with Wilson 95% CI.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

SEED_START = 1
SEED_END = 80
N_WORKERS = 20
COLLISION_RISK_THRESHOLD = 0.90

SUITES = [
    ("gauntlet",      6),
    ("adversarial",   2),
    ("compositional", 5),
]

TOKEN_LABELS = [
    "4-token (stop+maintain+evasive×2)",
    "5-token (+slow_yield)",
    "7-token (+crawl+lane_recover)",
    "9-token (full Spotlight Reflex)",
]


@dataclass
class AblationTask:
    suite: str
    cfg_idx: int
    case_idx: int
    seed: int


@dataclass
class AblationResult:
    suite: str
    cfg_idx: int
    passed: bool
    collision: bool


def _run_one(task: AblationTask) -> AblationResult:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import RolloutConfig, run_spotlight_reflex_policy
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG, ManeuverSpec, SpotlightReflexConfig,
    )

    maneuver_sets = [
        (
            ManeuverSpec("stop",          0.0,  0.0,   0.0),
            ManeuverSpec("maintain",      0.75, 1.0,   0.0),
            ManeuverSpec("evasive_left",  0.55, 0.70,  8.0, "early"),
            ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
        ),
        (
            ManeuverSpec("stop",          0.0,  0.0,   0.0),
            ManeuverSpec("maintain",      0.75, 1.0,   0.0),
            ManeuverSpec("slow_yield",    0.45, 0.55,  0.0),
            ManeuverSpec("evasive_left",  0.55, 0.70,  8.0, "early"),
            ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
        ),
        (
            ManeuverSpec("stop",          0.0,  0.0,   0.0),
            ManeuverSpec("crawl",         0.35, 0.25,  0.0),
            ManeuverSpec("maintain",      0.75, 1.0,   0.0),
            ManeuverSpec("slow_yield",    0.45, 0.55,  0.0),
            ManeuverSpec("evasive_left",  0.55, 0.70,  8.0, "early"),
            ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
            ManeuverSpec("lane_recover",  0.50, 0.65,  0.0),
        ),
        DEFAULT_SPOTLIGHT_CONFIG.maneuvers,
    ]

    maneuvers = maneuver_sets[task.cfg_idx]
    config = SpotlightReflexConfig(
        selector=DEFAULT_SPOTLIGHT_CONFIG.selector,
        references=DEFAULT_SPOTLIGHT_CONFIG.references,
        scoring=DEFAULT_SPOTLIGHT_CONFIG.scoring,
        trajectory=DEFAULT_SPOTLIGHT_CONFIG.trajectory,
        maneuvers=maneuvers,
    )
    rollout_cfg = RolloutConfig(spotlight=config)
    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, task.suite)
    rollout = run_spotlight_reflex_policy(scenario, config=rollout_cfg)

    collision = (
        not rollout.success
        and rollout.steps is not None
        and any(
            getattr(st, "collision_risk", 0) > COLLISION_RISK_THRESHOLD
            for st in rollout.steps
        )
    )
    return AblationResult(
        suite=task.suite,
        cfg_idx=task.cfg_idx,
        passed=rollout.success,
        collision=collision,
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    tasks = [
        AblationTask(suite=suite, cfg_idx=cfg_idx, case_idx=case_idx, seed=seed)
        for suite, n_cases in SUITES
        for cfg_idx in range(4)
        for case_idx in range(n_cases)
        for seed in range(SEED_START, SEED_END + 1)
    ]
    total = len(tasks)
    print(f"Multi-suite token ablation: {total} rollouts total, {N_WORKERS} workers")

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(_run_one, tasks)

    output = {}
    print()
    for suite, n_cases in SUITES:
        suite_results = [r for r in results if r.suite == suite]
        n_per_cfg = n_cases * (SEED_END - SEED_START + 1)
        print(f"\n── {suite.upper()} ({n_per_cfg} rollouts per config) ──")
        print(f"{'Token set':<42} {'N':>4}  {'Pass%':>6} {'95% CI':>13}  {'Coll%':>6} {'95% CI':>11}")
        print("─" * 90)

        suite_records = []
        for cfg_idx, label in enumerate(TOKEN_LABELS):
            cfg_results = [r for r in suite_results if r.cfg_idx == cfg_idx]
            n = len(cfg_results)
            k_p = sum(r.passed for r in cfg_results)
            k_c = sum(r.collision for r in cfg_results)
            p_lo, p_hi = wilson_ci(k_p, n)
            c_lo, c_hi = wilson_ci(k_c, n)
            print(
                f"  {label:<40} {n:>4}  {k_p/n*100:>5.1f}% [{p_lo*100:.1f},{p_hi*100:.1f}]  "
                f"{k_c/n*100:>5.1f}% [{c_lo*100:.1f},{c_hi*100:.1f}]"
            )
            suite_records.append({
                "label": label, "n_tokens": [4, 5, 7, 9][cfg_idx],
                "total": n, "passes": k_p, "collisions": k_c,
                "pass_rate": round(k_p / n * 100, 1),
                "collision_rate": round(k_c / n * 100, 1),
                "pass_ci95": [round(p_lo * 100, 1), round(p_hi * 100, 1)],
                "coll_ci95": [round(c_lo * 100, 1), round(c_hi * 100, 1)],
            })
        output[suite] = suite_records

    out_path = ROOT / "artifacts" / "ablation_token_count" / "results_multisuite.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
