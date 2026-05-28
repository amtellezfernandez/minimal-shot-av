# nuPlan Replay Utility-Barrier Audit

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expert/replay.json`
- Scene count: `1000`
- Recoverable proxy failures: `346`
- Mean progress retention of best replay-safe token: `0.500`
- Median progress retention: `0.700`
- Intervention-like recovery rate `<0.25 retention`: `0.341`

## Retention Thresholds

| Min retention | Recoverable | Rate | Blocked | Blocked rate | Mean safe progress |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 228 | 0.659 | 118 | 0.341 | 20.488 |
| 0.50 | 208 | 0.601 | 138 | 0.399 | 22.032 |
| 0.75 | 70 | 0.202 | 276 | 0.798 | 24.385 |
| 0.90 | 10 | 0.029 | 336 | 0.971 | 13.414 |

## Retention Bins

| Retention bin | Count | Rate | Best replay-safe tokens |
|---|---:|---:|---|
| lt_0.25 | 118 | 0.341 | `{"crawl": 72, "stop": 46}` |
| 0.25_to_0.50 | 20 | 0.058 | `{"crawl": 20}` |
| 0.50_to_0.75 | 138 | 0.399 | `{"evasive_left": 10, "evasive_right": 33, "slow_yield": 95}` |
| gte_0.75 | 70 | 0.202 | `{"evasive_left": 3, "evasive_right": 32, "lane_recover": 3, "nudge_left": 7, "nudge_right": 25}` |

## Token Histograms

- Proxy tokens: `{"evasive_left": 63, "evasive_right": 34, "lane_recover": 33, "maintain": 121, "nudge_left": 23, "nudge_right": 52, "slow_yield": 20}`
- Best replay-safe tokens: `{"crawl": 92, "evasive_left": 13, "evasive_right": 65, "lane_recover": 3, "nudge_left": 7, "nudge_right": 25, "slow_yield": 95, "stop": 46}`

## Interpretation

- This audit measures progress loss from the failing proxy token to the best replay-safe token.
- Low retention means recovery is intervention-like, not a marginal progress-preserving correction.
