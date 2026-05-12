"""Comprehensive cross-suite evaluation: baseline vs Spotlight-9 across all suites.

Runs matched seeds on compositional, adversarial, gauntlet, and hidden suites
and reports pass rate + collision rate with Wilson 95% CI for both policies.
Used to identify where the system shows the most meaningful differentiation.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED_START = 1
SEED_END = 80
N_WORKERS = 20
COLLISION_RISK_THRESHOLD = 0.90

SUITES = [
    ("compositional", 5),  # ~5 cases (COMPOSITIONAL_TOPOLOGIES)
    ("adversarial",   2),  # 2-hazard, 3-hazard
    ("gauntlet",      6),  # 6 gauntlet cases
    ("hidden",        1),  # 1 holdout case
]


@dataclass
class EvalTask:
    suite: str
    case_idx: int
    seed: int


@dataclass
class EvalResult:
    suite: str
    baseline_passed: bool
    baseline_collision: bool
    spotlight_passed: bool
    spotlight_collision: bool


def _run_pair(task: EvalTask) -> EvalResult:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import run_policy, run_spotlight_reflex_policy

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, task.suite)

    rb = run_policy(scenario)
    rs = run_spotlight_reflex_policy(scenario)

    def _collision(rollout) -> bool:
        return (
            not rollout.success
            and rollout.steps is not None
            and any(
                getattr(st, "collision_risk", 0) > COLLISION_RISK_THRESHOLD
                for st in rollout.steps
            )
        )

    return EvalResult(
        suite=task.suite,
        baseline_passed=rb.success,
        baseline_collision=_collision(rb),
        spotlight_passed=rs.success,
        spotlight_collision=_collision(rs),
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
    # Build all tasks
    all_tasks: list[EvalTask] = []
    for suite, n_cases in SUITES:
        for case_idx in range(n_cases):
            for seed in range(SEED_START, SEED_END + 1):
                all_tasks.append(EvalTask(suite=suite, case_idx=case_idx, seed=seed))

    print(f"Total rollout pairs: {len(all_tasks)}")
    print(f"Workers: {N_WORKERS}")

    with multiprocessing.Pool(N_WORKERS) as pool:
        all_results = pool.map(_run_pair, all_tasks)

    # Aggregate per suite
    suite_records = []
    print()
    print(f"{'Suite':<16} {'N':>5}  {'Base pass%':>10} {'95% CI':>14}  {'SR pass%':>9} {'95% CI':>14}  {'Ratio':>6}  {'Base coll%':>10}  {'SR coll%':>9}")
    print("─" * 110)

    output = {}
    for suite, n_cases in SUITES:
        results = [r for r in all_results if r.suite == suite]
        n = len(results)
        b_pass = sum(r.baseline_passed for r in results)
        b_coll = sum(r.baseline_collision for r in results)
        s_pass = sum(r.spotlight_passed for r in results)
        s_coll = sum(r.spotlight_collision for r in results)

        bp_lo, bp_hi = wilson_ci(b_pass, n)
        sp_lo, sp_hi = wilson_ci(s_pass, n)

        ratio = (s_pass / n) / max(b_pass / n, 1e-6)

        ci_b = f"[{bp_lo*100:.1f},{bp_hi*100:.1f}]"
        ci_s = f"[{sp_lo*100:.1f},{sp_hi*100:.1f}]"
        print(
            f"{suite:<16} {n:>5}  {b_pass/n*100:>9.1f}% {ci_b:>14}  {s_pass/n*100:>8.1f}% {ci_s:>14}  {ratio:>5.2f}×  "
            f"{b_coll/n*100:>9.1f}%    {s_coll/n*100:>8.1f}%"
        )

        discordant = sum(1 for r in results if not r.baseline_passed and r.spotlight_passed)
        output[suite] = {
            "n": n, "n_cases": n_cases,
            "baseline": {
                "passes": b_pass, "collisions": b_coll,
                "pass_rate": round(b_pass / n * 100, 1),
                "collision_rate": round(b_coll / n * 100, 1),
                "pass_ci95": [round(bp_lo * 100, 1), round(bp_hi * 100, 1)],
            },
            "spotlight_9token": {
                "passes": s_pass, "collisions": s_coll,
                "pass_rate": round(s_pass / n * 100, 1),
                "collision_rate": round(s_coll / n * 100, 1),
                "pass_ci95": [round(sp_lo * 100, 1), round(sp_hi * 100, 1)],
            },
            "improvement_ratio": round(ratio, 2),
            "discordant_base_fail_sr_pass": discordant,
        }

    out_path = ROOT / "artifacts" / "eval_all_suites.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
