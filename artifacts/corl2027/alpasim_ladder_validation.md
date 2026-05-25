# AlpaSim ladder validation

Uses committed per-frame audit artifacts. Realized current clearance is the transformed ego-frame closest actor-proxy clearance at the rollout frame; proxy clearance is the selected direct-grid future minimum clearance stored by the open-loop audit.

## Clearance Divergence

| Realized near threshold (m) | Scenes | Rate | Median onset (s before impact) |
| ---: | ---: | ---: | ---: |
| 1.0 | 7 | 0.389 | 3.500 |
| 2.0 | 13 | 0.722 | 3.200 |
| 3.0 | 14 | 0.778 | 3.250 |
| 4.0 | 15 | 0.833 | 3.400 |
| 5.0 | 16 | 0.889 | 4.050 |
| 6.0 | 18 | 1.000 | 4.350 |

## Proxy Clearance Erosion

| Proxy threshold (m) | Scenes | Rate | Median first unsafe (s before impact) |
| ---: | ---: | ---: | ---: |
| 0.0 | 16 | 0.889 | 0.850 |
| 0.55 | 16 | 0.889 | 0.950 |
| 1.0 | 18 | 1.000 | 1.000 |

## Failure Subsets

- `candidate_feasibility_or_actor_axis_definition`: 2 scenes; median hazards=24.000; median realized current clearance=7.716 m; median selected proxy clearance=0.925 m.
- `controller_proxy_residual`: 12 scenes; median hazards=22.750; median realized current clearance=6.345 m; median selected proxy clearance=2.272 m.
- `selector_ranking_secondary`: 4 scenes; median hazards=21.500; median realized current clearance=3.434 m; median selected proxy clearance=1.850 m.
