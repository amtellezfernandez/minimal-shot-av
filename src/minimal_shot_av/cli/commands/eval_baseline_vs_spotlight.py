"""Head-to-head evaluation: potential-field baseline vs 9-token Spotlight Reflex.

Runs matched seeds on the Gauntlet suite and reports pass rate + collision rate
with Wilson 95% CI for both policies. Used to verify the improvement claim in
the CoRL 2027 paper and to audit the current simulator state.
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
N_CASES = 6
N_WORKERS = 20
COLLISION_RISK_THRESHOLD = 0.90
SUITE = "gauntlet"


@dataclass
class EvalTask:
    case_idx: int
    seed: int


@dataclass
class EvalResult:
    baseline_passed: bool
    baseline_collision: bool
    spotlight_passed: bool
    spotlight_collision: bool


def _run_pair(task: EvalTask) -> EvalResult:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import run_policy, run_spotlight_reflex_policy

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, SUITE)

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
    tasks = [
        EvalTask(case_idx=c, seed=s)
        for c in range(N_CASES)
        for s in range(SEED_START, SEED_END + 1)
    ]
    n = len(tasks)
    print(f"Gauntlet evaluation: {n} matched rollouts ({N_CASES} cases × {SEED_END} seeds)")
    print(f"Workers: {N_WORKERS}")

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(_run_pair, tasks)

    b_pass = sum(r.baseline_passed for r in results)
    b_coll = sum(r.baseline_collision for r in results)
    s_pass = sum(r.spotlight_passed for r in results)
    s_coll = sum(r.spotlight_collision for r in results)

    bp_lo, bp_hi = wilson_ci(b_pass, n)
    sp_lo, sp_hi = wilson_ci(s_pass, n)
    bc_lo, bc_hi = wilson_ci(b_coll, n)
    sc_lo, sc_hi = wilson_ci(s_coll, n)

    improvement = (s_pass / n) / max(b_pass / n, 1e-6)

    print()
    print(f"Baseline (potential-field reactive):")
    print(f"  pass:      {b_pass}/{n} = {b_pass/n*100:.1f}%  95% CI [{bp_lo*100:.1f}, {bp_hi*100:.1f}]")
    print(f"  collision: {b_coll}/{n} = {b_coll/n*100:.1f}%  95% CI [{bc_lo*100:.1f}, {bc_hi*100:.1f}]")
    print()
    print(f"Spotlight Reflex (9-token, calibrated):")
    print(f"  pass:      {s_pass}/{n} = {s_pass/n*100:.1f}%  95% CI [{sp_lo*100:.1f}, {sp_hi*100:.1f}]")
    print(f"  collision: {s_coll}/{n} = {s_coll/n*100:.1f}%  95% CI [{sc_lo*100:.1f}, {sc_hi*100:.1f}]")
    print()
    print(f"Pass-rate improvement: {improvement:.1f}×")

    # Seeds where baseline fails but spotlight passes
    discordant = [(i, results[i]) for i in range(n) if not results[i].baseline_passed and results[i].spotlight_passed]
    print(f"Baseline-fail / Spotlight-pass (discordant): {len(discordant)}/{n}")

    out = {
        "suite": SUITE,
        "seeds": f"{SEED_START}-{SEED_END}",
        "n_cases": N_CASES,
        "n_total": n,
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
        "improvement_ratio": round(improvement, 2),
    }
    out_path = ROOT / "artifacts" / "baseline_vs_spotlight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
