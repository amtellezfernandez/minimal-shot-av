"""Token-count ablation for the CoRL 2027 paper.

Experimental design
-------------------
- Suite: Gauntlet (4 simultaneous hazards, 3.5 m corridor)
- Seeds: 1-80 per case × 6 gauntlet cases = 480 matched rollouts per config
- Configs: 4, 5, 7, 9 tokens (nested: each set is a strict subset of the next)
- Controls: identical scenario seeds, scoring, trajectory, selector parameters
- Parallelism: multiprocessing.Pool over (config, case, seed) triples
- Statistics: Wilson 95% CI for pass rate and collision rate (binomial proportions)
- Metric: pass = episode completes without collision; collision = max step risk > 0.90

The 9-token result must reproduce the paper's Gauntlet 57.6% (seeds 1-10).
If it deviates by > 5 pp, investigate before including results in the paper.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
from minimal_shot_av.simulator.policy import RolloutConfig, run_spotlight_reflex_policy
from minimal_shot_av.simulator.spotlight_reflex import (
    DEFAULT_SPOTLIGHT_CONFIG,
    ManeuverSpec,
    SpotlightReflexConfig,
)

# ── Experiment parameters ─────────────────────────────────────────────────────

SEED_START = 1
SEED_END = 80          # 80 seeds × 6 cases = 480 rollouts per config
SUITE = "gauntlet"
N_CASES = 6            # gauntlet_case_1 … _6, offset seed by (k-1)*1000
N_WORKERS = 20         # leave 4 cores for OS; tune if needed
COLLISION_RISK_THRESHOLD = 0.90

# ── Token sets (nested; each is a strict subset of the next) ─────────────────

_ALL_9: tuple[ManeuverSpec, ...] = DEFAULT_SPOTLIGHT_CONFIG.maneuvers

_TOKENS_4 = (
    ManeuverSpec("stop",          0.0,  0.0,   0.0),
    ManeuverSpec("maintain",      0.75, 1.0,   0.0),
    ManeuverSpec("evasive_left",  0.55, 0.70,  8.0, "early"),
    ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
)

_TOKENS_5 = (
    ManeuverSpec("stop",          0.0,  0.0,   0.0),
    ManeuverSpec("maintain",      0.75, 1.0,   0.0),
    ManeuverSpec("slow_yield",    0.45, 0.55,  0.0),
    ManeuverSpec("evasive_left",  0.55, 0.70,  8.0, "early"),
    ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
)

_TOKENS_7 = (
    ManeuverSpec("stop",          0.0,  0.0,   0.0),
    ManeuverSpec("crawl",         0.35, 0.25,  0.0),
    ManeuverSpec("maintain",      0.75, 1.0,   0.0),
    ManeuverSpec("slow_yield",    0.45, 0.55,  0.0),
    ManeuverSpec("evasive_left",  0.55, 0.70,  8.0, "early"),
    ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
    ManeuverSpec("lane_recover",  0.50, 0.65,  0.0),
)

TOKEN_SETS: list[tuple[str, tuple[ManeuverSpec, ...]]] = [
    ("4-token (stop+maintain+evasive×2)", _TOKENS_4),
    ("5-token (+slow_yield)",             _TOKENS_5),
    ("7-token (+crawl+lane_recover)",     _TOKENS_7),
    ("9-token (full Spotlight Reflex)",   _ALL_9),
]

# ── Wilson 95% CI for a binomial proportion ───────────────────────────────────

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lo, hi) Wilson 95% CI in [0,1]."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, centre - half), min(1.0, centre + half))


# ── Per-rollout worker (runs in a subprocess) ─────────────────────────────────

@dataclass
class RolloutTask:
    cfg_idx: int
    case_idx: int
    seed: int


@dataclass
class RolloutResult:
    cfg_idx: int
    passed: bool
    collision: bool


def _run_one(task: RolloutTask) -> RolloutResult:
    """Execute one rollout. Runs in a worker process."""
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
    scenario = generate_compositional_scenario(effective_seed, "gauntlet")
    rollout = run_spotlight_reflex_policy(scenario, config=rollout_cfg)

    collision = (
        not rollout.success
        and rollout.steps is not None
        and any(step.collision_risk > 0.90 for step in rollout.steps)
    )
    return RolloutResult(
        cfg_idx=task.cfg_idx,
        passed=rollout.success,
        collision=collision,
    )


# ── Sanity check: 9-token result should match paper (seeds 1-10) ─────────────

def _sanity_check() -> float:
    """Quick 9-token check on seeds 1-10 to verify 57.6% ± 5 pp."""
    tasks = [
        RolloutTask(cfg_idx=3, case_idx=c, seed=s)
        for c in range(N_CASES)
        for s in range(1, 11)
    ]
    with multiprocessing.Pool(min(N_WORKERS, len(tasks))) as pool:
        results = pool.map(_run_one, tasks)
    passes = sum(r.passed for r in results)
    total = len(results)
    rate = passes / total * 100
    print(f"  Sanity: 9-token seeds 1-10 → {passes}/{total} = {rate:.1f}%  (paper: 57.6%)")
    if abs(rate - 57.6) > 10.0:
        print(f"  WARNING: deviation > 10 pp from paper value. Investigate before using results.")
    return rate


# ── Main ablation ─────────────────────────────────────────────────────────────

def run_ablation() -> list[dict]:
    tasks = [
        RolloutTask(cfg_idx=cfg_idx, case_idx=case_idx, seed=seed)
        for cfg_idx in range(len(TOKEN_SETS))
        for case_idx in range(N_CASES)
        for seed in range(SEED_START, SEED_END + 1)
    ]
    total_tasks = len(tasks)
    print(f"Total rollouts: {total_tasks}  ({len(TOKEN_SETS)} configs × {N_CASES} cases × {SEED_END - SEED_START + 1} seeds)")
    print(f"Workers: {N_WORKERS}")

    with multiprocessing.Pool(N_WORKERS) as pool:
        results = pool.map(_run_one, tasks)

    # Aggregate per config
    passes_per   = [0] * len(TOKEN_SETS)
    collisions_per = [0] * len(TOKEN_SETS)
    totals_per   = [0] * len(TOKEN_SETS)
    for r in results:
        totals_per[r.cfg_idx]     += 1
        passes_per[r.cfg_idx]     += int(r.passed)
        collisions_per[r.cfg_idx] += int(r.collision)

    records = []
    for cfg_idx, (label, maneuvers) in enumerate(TOKEN_SETS):
        n   = totals_per[cfg_idx]
        k_p = passes_per[cfg_idx]
        k_c = collisions_per[cfg_idx]
        pass_rate = k_p / n if n > 0 else 0.0
        coll_rate = k_c / n if n > 0 else 0.0
        p_lo, p_hi = wilson_ci(k_p, n)
        c_lo, c_hi = wilson_ci(k_c, n)
        record = {
            "label":        label,
            "n_tokens":     len(maneuvers),
            "token_names":  [m.name for m in maneuvers],
            "total":        n,
            "passes":       k_p,
            "collisions":   k_c,
            "pass_rate":    round(pass_rate * 100, 1),
            "collision_rate": round(coll_rate * 100, 1),
            "pass_ci95_lo": round(p_lo * 100, 1),
            "pass_ci95_hi": round(p_hi * 100, 1),
            "coll_ci95_lo": round(c_lo * 100, 1),
            "coll_ci95_hi": round(c_hi * 100, 1),
        }
        records.append(record)
        print(
            f"  [{cfg_idx+1}/4] {label:<42} "
            f"pass={pass_rate*100:.1f}% [{p_lo*100:.1f},{p_hi*100:.1f}]  "
            f"coll={coll_rate*100:.1f}% [{c_lo*100:.1f},{c_hi*100:.1f}]  "
            f"({k_p}/{n})"
        )
    return records


def main() -> None:
    print(f"Token-count ablation — {SUITE} suite, seeds {SEED_START}–{SEED_END}")
    print(f"Running sanity check (9-token, seeds 1–10)…")
    _sanity_check()

    print(f"\nRunning full ablation…")
    results = run_ablation()

    out_dir = ROOT / "artifacts" / "ablation_token_count"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps({"ablation": results}, indent=2))

    print("\n═══ FINAL RESULTS ═══")
    print(f"{'Token set':<45} {'N':>3}  {'Pass%':>6} {'95% CI':>12}  {'Coll%':>6} {'95% CI':>12}")
    print("─" * 95)
    for r in results:
        ci_p = f"[{r['pass_ci95_lo']},{r['pass_ci95_hi']}]"
        ci_c = f"[{r['coll_ci95_lo']},{r['coll_ci95_hi']}]"
        print(f"{r['label']:<45} {r['n_tokens']:>3}  {r['pass_rate']:>5.1f}% {ci_p:>12}  {r['collision_rate']:>5.1f}% {ci_c:>12}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
