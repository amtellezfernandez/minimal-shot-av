# NAVSIM intervention matrix

NAVSIM public non-reactive/pseudo-closed-loop metric surface. This supports public verification of per-axis non-co-monotonicity, not controller-proxy mismatch.

Common scenes: `1`
Non-co-monotone interventions: `0`

## Per-Model Metrics

| Model | Scenes | Collision | Offroad | Progress | TTC | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clamped | 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| hard_veto | 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| raw | 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| source_decay | 1 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |

## Comparisons Vs Baseline

- `clamped`: `unchanged`; improved=[]; worsened=[]; conflicts={'progress_up_collision_up': 0, 'progress_up_offroad_up': 0, 'offroad_down_collision_up': 0, 'score_up_collision_up': 0}
- `hard_veto`: `unchanged`; improved=[]; worsened=[]; conflicts={'progress_up_collision_up': 0, 'progress_up_offroad_up': 0, 'offroad_down_collision_up': 0, 'score_up_collision_up': 0}
- `source_decay`: `unchanged`; improved=[]; worsened=[]; conflicts={'progress_up_collision_up': 0, 'progress_up_offroad_up': 0, 'offroad_down_collision_up': 0, 'score_up_collision_up': 0}

## Limitation

- NAVSIM is not a fully reactive closed-loop controller-proxy surface.
- Use this report for public per-axis intervention validation; keep AlpaSim for controller-proxy localization.
