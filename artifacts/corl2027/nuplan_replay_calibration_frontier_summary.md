# nuPlan Replay Calibration Frontier Summary

- Replay surface: `1000` interaction-selected public nuPlan scenes.
- Split: DB-level train/holdout, seed 0; holdout has `300` scenes.
- Bridge: heldout `40` base scenes; `120` selector executions per lambda run.
- Bridge limitation: absolute nuPlan box-clearance fails all selectors; report aligned replay-equivalent logged-actor clearance separately.

## Frontier Table

| Selector | Replay-infeasible | Rate | Progress | ADE3 | Recovery | Bridge aligned failures | Mean aligned clearance | Token entropy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| proxy | 209/300 | 0.697 | 29.383 | 2.251 | 0/160 | 20/40 | 0.827 | 2.240 |
| lambda20 | 115/300 | 0.383 | 25.429 | 3.563 | 94/160 | 20/40 | 0.913 | 2.531 |
| lambda40 | 94/300 | 0.313 | 24.295 | 3.971 | 115/160 | 20/40 | 1.042 | 2.457 |
| lambda80 | 49/300 | 0.163 | 19.474 | 5.933 | 160/160 | 0/40 | 1.584 | 2.179 |
| oracle | 49/300 | 0.163 | 10.889 | 9.329 | 160/160 | 0/40 | 1.639 | 2.344 |

## Token Distribution Shift

| Selector | maintain | slow_yield | evasive | nudge | lane_recover | stop | crawl |
|---|---:|---:|---:|---:|---:|---:|---:|
| proxy | 98 | 38 | 83 | 61 | 20 | 0 | 0 |
| lambda20 | 80 | 86 | 56 | 45 | 7 | 0 | 26 |
| lambda40 | 70 | 94 | 58 | 41 | 5 | 0 | 32 |
| lambda80 | 48 | 94 | 69 | 0 | 3 | 0 | 86 |
| oracle | 10 | 54 | 53 | 3 | 4 | 94 | 82 |

## Interpretation

- `lambda20` is the balanced replay point: large replay-risk reduction with moderate utility loss, but no aligned bridge pass/fail improvement on this bridge subset.
- `lambda40` improves mean aligned clearance but still does not reduce aligned bridge failures.
- `lambda80` is safety-heavy: it matches oracle on replay holdout and aligned bridge failures, but with substantial progress/ADE cost and a shift toward crawl/slow_yield.
- The main paper should present the frontier, not a single operating point.
