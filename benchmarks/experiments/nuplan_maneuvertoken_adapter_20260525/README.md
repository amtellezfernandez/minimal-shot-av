# nuPlan ManeuverToken Adapter

This folder defines the next public closed-loop experiment package after the
AlpaSim diagnostic ladder.

Goal: move the main evidence surface from private/reactive AlpaSim to public
nuPlan closed-loop evaluation while preserving the same failure-localization
logic.

## Main Claim Shift

The paper should no longer stop at:

- "collision survives actor completion"
- "collision survives selector removal"

It should terminate at:

- "a public closed-loop diagnostic ladder localizes residual failure to a
  specific rung"

## Benchmark Choice

Primary surface:

- nuPlan fixed hard split such as `Test14-Hard`, or the closest public fixed
  hard split with persistent closed-loop failures.

Reason:

- easy aggregate surfaces are not useful here;
- the method needs collisions, near-misses, and erosion-before-impact cases;
- public reproducibility matters more than private simulator headroom.

If `Test14-Hard` is unavailable in the installed devkit, substitute a fixed
public hard split and keep the split name recorded in every artifact.

## Experiment Ladder

### Experiment 1: nuPlan ManeuverToken adapter

Target pipeline:

```text
nuPlan scene
-> ego state
-> surrounding actors
-> map / route
-> six-scalar state
-> ManeuverToken candidates
-> learned token ranker
-> selected token
-> controller rollout
-> per-frame diagnostics
```

Deliverables:

- nuPlan adapter for six-scalar state extraction
- ManeuverToken candidate generator on nuPlan route geometry
- per-frame artifact export for selected token, proxy safety, realized safety,
  and clearance traces

Minimum acceptance:

- one fixed hard split runs end to end
- collision, offroad, progress, TTC, and route metrics export per scene
- per-frame diagnostic traces are saved for all collision and near-miss scenes

### Experiment 2: simple learned selector

Policy family:

```text
pi_phi(a | x)
```

Where:

- `a` is one ManeuverToken
- `x` is six-scalar state + route features + actor summary

Supervision:

- generate all ManeuverToken short-horizon rollouts
- compare each token to the logged expert trajectory
- score by imitation distance with a safety-feasibility penalty
- train a small MLP to rank tokens

Constraints:

- do not build a large planner
- do not let learning replace the paper's diagnostic method
- keep the learned component small enough that failure localization remains
  interpretable

Minimum acceptance:

- selector training data cached to disk
- one compact MLP baseline trained reproducibly
- selector beats a fixed baseline on token imitation quality or safety-weighted
  ranking accuracy

### Experiment 3: diagnostic ladder

For each collision or near-miss, classify:

1. Was the actor visible?
2. Did any safe token exist?
3. Did the selector choose a safe token?
4. Did the proxy predict the chosen token as safe?
5. Did the executed rollout still collide?

Mapped rungs:

- actor not visible -> state / visibility failure
- no safe token -> candidate-feasibility failure
- safe token exists but not selected -> selector-ranking failure
- selected token predicted unsafe -> proxy-risk-estimation failure
- selected token predicted safe but rollout collides -> controller-proxy mismatch

Minimum acceptance:

- every failure scene assigned to one ladder rung
- rung counts exported as scene-level and frame-level summaries
- paired examples saved for appendix figures

### Experiment 4: proxy vs realized clearance

Primary measurement:

```text
Delta c(t) = c_proxy(t) - c_realized(t)
```

Report:

- proxy-safe / realized-near rate
- erosion-before-impact rate
- time from first erosion to impact
- collision-scene fraction by ladder rung
- bootstrap confidence intervals
- threshold sensitivity

This becomes the public replacement for the private AlpaSim `16/18` and `13/18`
motivation numbers.

## Required Tables

### Table 1: benchmark surfaces

| Surface | Use in paper | Why |
| --- | --- | --- |
| AlpaSim | Motivation / appendix | Origin of the hypothesis, but not public enough |
| nuPlan | Main result | Public closed-loop planning benchmark |
| nuPlan-R | Optional future / ablation | Ideal if code access is clean |
| NAVSIM v2 | Optional appendix | Public and scalable, but not reactive enough |
| Fail2Drive | Related work | Paired scenario diagnosis, not rung-level localization |

## Artifact Contract

Expected artifact directories:

- `artifacts/nuplan_maneuvertoken/<split>/adapter/`
- `artifacts/nuplan_maneuvertoken/<split>/selector/`
- `artifacts/nuplan_maneuvertoken/<split>/ladder/`
- `artifacts/nuplan_maneuvertoken/<split>/clearance/`

Each directory should contain:

- scene-level CSV / JSON summaries
- per-frame traces for failure scenes
- markdown report for paper drafting
- exact split name and thresholds in metadata

## Code Targets

Likely next implementation surfaces:

- `src/minimal_shot_av/model/nuplan_maneuver_token_adapter.py`
- `src/minimal_shot_av/cli/commands/run_nuplan_maneuvertoken_rollout.py`
- `src/minimal_shot_av/cli/commands/train_nuplan_maneuvertoken_selector.py`
- `src/minimal_shot_av/cli/commands/audit_nuplan_diagnostic_ladder.py`
- `src/minimal_shot_av/cli/commands/audit_nuplan_proxy_clearance.py`

## Decision Rule

This experiment package is successful only if:

- the main public nuPlan result reproduces nontrivial closed-loop failures,
- the learned selector is small and replaceable,
- the diagnostic ladder produces positive localization rather than a list of
  unresolved hypotheses,
- proxy-vs-realized clearance erosion is reported with uncertainty.
