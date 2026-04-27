# COMPASS Benchmark

COMPASS means **Compositional Out-of-Distribution Policy Assessment for
Minimal-Shot Autonomous Systems**.

It is the benchmark layer above the abstract simulator. It adds:

- paired in-distribution and OOD scenarios
- independent oracle feasibility checks
- oracle reference reasoning traces
- diagnostic reasoning-trace overlap scoring
- recovery scoring after near misses
- explicit generalisation-gap scoring
- one composite `COMPASS Score`

## Run

Official calibrated ladder:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.compass ladder \
  --policy spotlight-reflex \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_ladder.json
```

Single-suite probe:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.compass run \
  --policy spotlight-reflex \
  --suite gauntlet \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_gauntlet.json
```

If installed as a package, the console command is also available:

```bash
uv run compass run --policy spotlight-reflex --suite gauntlet --seed-start 1 --seed-end 10
```

Generate the SOTIF-aligned evidence package:

```bash
PYTHONPATH=src uv run --no-sync python -m minimal_shot_av.simulator.certification \
  --policy spotlight-reflex \
  --profile sotif-v0 \
  --seed-start 1 \
  --seed-end 10 \
  --output artifacts/compass_evidence_report.json
```

This evidence package is intentionally not a legal certification. It records a
machine-readable ODD, simulator tier, COMPASS results, Wilson confidence
intervals for success and collision rates, per-level driving-quality scores,
SOTIF-aligned evidence status, and remaining gaps.

The evidence layer is stricter than the COMPASS smoke benchmark. With the
default `1%` collision upper-bound threshold at `95%` confidence, even a level
with zero observed collisions needs `381` ranked runs before the per-level
sample-size gate can pass. Small runs are useful for debugging, not evidence
claims.

Profiles make the certification assumptions explicit:

- `sotif-v0`: conservative default, abstract simulator, `90%` success lower
  confidence bound, `1%` collision upper confidence bound, and minimum
  per-level COMPASS/safety/route-quality/comfort scores.
- `smoke`: development-only profile for checking report plumbing.

Use `--odd-spec odd.json` and `--thresholds thresholds.json` to evaluate a
different ODD or evidence standard without changing source code. Use
`minimal_shot_av.simulator.compass ladder --profile-json compass_profile.json`, or
`minimal_shot_av.simulator.certification --compass-profile-json compass_profile.json`, to
override benchmark levels, official weights, trajectory scoring constants,
suite penalties, coverage gates, minimum ranked runs, and scenario-generation
settings. The `scenario_generation` block controls suite seed offsets, suite
pressure, hazard-count distributions, ambient-object density, gauntlet corridor
width caps, corridor clearance, topology geometry, hazard geometry, environment
visibility/latency ranges, and difficulty scoring. The same profile can also
override the oracle feasibility-check horizon, action lattice, scoring weights,
lookahead, and clearance gates. Unknown JSON fields are rejected so stale
configuration does not silently pass.

The repository includes `configs/compass_stress.json` as an example non-default
benchmark profile. It makes scenario generation and oracle settings harder
without changing policy code, which is the intended workflow for discovering
policy failures instead of making the benchmark fit one model.

## Score

The official COMPASS score is computed over levels 0-3:

```text
official = 0.10 * level_0_sanity
         + 0.25 * level_1_wod_aligned
         + 0.40 * level_2_compositional
         + 0.25 * level_3_adversarial
```

Level 4 is reported separately as `frontier_probe`. It is intentionally hard
and should not collapse the ranked score.

The current dependency-free score is:

```text
COMPASS = 0.30 * safety
        + 0.25 * route_quality
        + 0.15 * comfort
        + 0.15 * recovery_rate
        + 0.15 * generalization_score
```

This intentionally prevents a policy from getting a top score by merely
surviving. `safety` includes collision and clearance, `route_quality` includes
progress and lane discipline, and `comfort` includes intervention burden,
stalling, and speed-change cost.

The oracle has privileged access to scenario state and future actor projection.
It is used as an independent feasibility check and to produce a reference trace.
The first implementation is not a proof-grade optimal solver; it uses
deterministic simulator state rather than an external LLM judge, so it runs
without network or GPU access.

Cases where both the oracle and submitted policy fail are excluded from the
official level score and counted as `excluded_unsolvable`. This prevents traps
from polluting the benchmark. If a policy succeeds where the oracle fails, the
run remains ranked and is counted as `oracle_gap_runs`; that is evidence the
feasibility checker is incomplete, not evidence the scenario should disappear.
The official report also includes `official_coverage_weight` and `score_valid`;
the score should only be treated as ranked if enough official-level weight is
accepted by the feasibility check or solved by the submitted policy.

The current `reasoning_quality` field is explicitly marked
`diagnostic_rollout_trace_overlap_v0_not_official`. It is a dependency-free
placeholder based on rollout trace terms, not a VLA/LLM reasoning judge, and it
does not contribute to the official COMPASS score.

Small runs are not valid ranked reports. The report includes
`minimum_runs_per_official_level`, `sample_size_valid`, and `score_valid` so a
smoke test cannot be mistaken for a benchmark claim.

Level summaries are macro-averaged: each official level receives its configured
weight regardless of how many scenarios it contains. Level 1 expands each seed
across all 11 WOD-style clusters, then averages within that level.

Because the oracle is intentionally independent and still limited, the report
separates `oracle_failed_policy_succeeded` from `oracle_failed_policy_failed`.
The former means the policy found a route the feasibility checker did not prove;
it is included in ranked results as an oracle gap.

## Interpretation

Raw route completion is not the benchmark. A policy can reach the goal and still
score poorly if it crawls, has near misses, relies on many interventions, fails
to recover after close calls, or collapses from the paired in-distribution case
to the OOD case.
