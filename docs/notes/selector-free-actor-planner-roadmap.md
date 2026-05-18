# Selector-Free Actor-Aware Planner Roadmap

This branch starts after the CoRL-2027 selector audit was finalized and published.
The previous branch should be treated as a completed diagnostic artifact, not as a
positive collision-repair method.

## Starting Point

The matched 30-scene AlpaSim world-frame oracle audit established:

- Raw collisions are unchanged: `18/30 -> 18/30`.
- Actor completion changes selector awareness and selected actions.
- A selector-side counterfactual finds `0/18` actor-axis-safe first-impact frames.
- Only `5/25` actionable actor-axis-safe frames are baseline selector misses.
- `12/18` collision scenes have actor-axis-safe selected tokens earlier, then still collide.

Conclusion: selector miss is real but not dominant. The remaining collision surface is
more likely caused by the bounded token/candidate interface, low-level controller
execution, traffic interaction, or a mix of these.

## New Project Question

Can a selector-free, actor-aware planner avoid the AlpaSim collision surface when given
the same world-frame actor proxy and route context?

This is a different project from the token-selector paper. It tests whether the explicit
ManeuverToken interface is the bottleneck.

## Minimum Viable Experiment

1. Reuse the existing 30-scene world-frame oracle actor proxy.
2. Implement a direct planner that outputs a 5-second trajectory instead of selecting a
   token.
3. Optimize a small continuous trajectory family against:
   - actor clearance,
   - rear-flow time-to-collision,
   - route tracking,
   - lane margin,
   - progress,
   - smoothness and controller feasibility.
4. Run only the 18 baseline collision scenes first.
5. Compare against:
   - axis-constrained clamped baseline,
   - world-frame oracle actor selector,
   - selector-side counterfactual audit.

## Decision Logic

- If direct planning reduces collision, the token/candidate interface is the bottleneck.
- If direct planning still collides, the likely bottleneck is controller execution,
  AlpaSim traffic response, or insufficient actor dynamics.
- If direct planning reduces collision but worsens offroad/wrong-lane, the planner needs
  stronger route/lane constraints before it becomes a paper method.

## First Implementation Target

Create an AlpaSim model preset that bypasses token logits and returns a direct optimized
trajectory:

- `src/minimal_shot_av/simulator/alpasim_direct_actor_planner.py`
- `scripts/run_alpasim_direct_actor_planner_collision18.sh`
- `artifacts/alpasim_direct_actor_planner_collision18_analysis.{json,md}`

Do not update the CoRL paper until this branch has a clean 18-scene collision result.
