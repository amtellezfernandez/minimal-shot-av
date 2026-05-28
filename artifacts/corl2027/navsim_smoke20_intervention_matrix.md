# NAVSIM intervention matrix

NAVSIM public non-reactive/pseudo-closed-loop metric surface. This supports public verification of per-axis non-co-monotonicity, not controller-proxy mismatch.

Common scenes: `20`
Non-co-monotone interventions: `3`

## Per-Model Metrics

| Model | Scenes | Collision | Offroad | Progress | TTC | Score |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clamped | 20 | 0.0000 | 0.0000 | 0.9870 | 1.0000 | 0.8710 |
| hard_veto | 20 | 0.0000 | 0.0000 | 0.9561 | 1.0000 | 0.8613 |
| raw | 20 | 0.0000 | 0.0500 | 0.9922 | 1.0000 | 0.8288 |
| source_decay | 20 | 0.0000 | 0.0000 | 0.9736 | 1.0000 | 0.8668 |

## Comparisons Vs Baseline

- `clamped`: `tradeoff`; improved=['offroad', 'score']; worsened=['progress']; conflicts={'progress_up_collision_up': 0, 'progress_up_offroad_up': 0, 'offroad_down_collision_up': 0, 'score_up_collision_up': 0}
- `hard_veto`: `tradeoff`; improved=['offroad', 'score']; worsened=['progress']; conflicts={'progress_up_collision_up': 0, 'progress_up_offroad_up': 0, 'offroad_down_collision_up': 0, 'score_up_collision_up': 0}
- `source_decay`: `tradeoff`; improved=['offroad', 'score']; worsened=['progress']; conflicts={'progress_up_collision_up': 0, 'progress_up_offroad_up': 0, 'offroad_down_collision_up': 0, 'score_up_collision_up': 0}

## Limitation

- NAVSIM is not a fully reactive closed-loop controller-proxy surface.
- Use this report for public per-axis intervention validation; keep AlpaSim for controller-proxy localization.
