# CoRL Replay-Value Repeated-Split Summary

- Input replay JSON: `artifacts/corl2027/nuplan_public_replay_study_interaction1000_expanded/replay.json`
- Split count: `5`
- Seed start: `0`
- Holdout fraction: `0.30`

| Selector | Replay fail | Progress (m) | ADE3 (m) | FDE3 (m) | Entropy |
|---|---:|---:|---:|---:|---:|
| Proxy top-1 | 0.602 ± 0.130 | 25.792 ± 4.552 | 3.847 ± 0.563 | 6.660 ± 0.732 | 2.100 ± 0.345 |
| Scene-token student | 0.590 ± 0.110 | 25.210 ± 5.348 | 3.940 ± 0.605 | 6.752 ± 0.528 | 2.161 ± 0.294 |
| Replay-value, safety-heavy | 0.218 ± 0.177 | 15.856 ± 4.442 | 7.052 ± 1.112 | 11.432 ± 1.682 | 3.248 ± 0.514 |
| Replay-value, balanced | 0.289 ± 0.164 | 19.512 ± 4.513 | 5.790 ± 1.357 | 9.373 ± 2.236 | 3.185 ± 0.421 |
| Replay-value oracle | 0.208 ± 0.175 | 12.197 ± 2.352 | 8.360 ± 0.982 | 13.677 ± 1.631 | 3.302 ± 0.398 |

## Per-split Seeds

- seed `0`: proxy `0.483`, scene-token `0.460`, balanced `0.097`
- seed `1`: proxy `0.471`, scene-token `0.510`, balanced `0.129`
- seed `2`: proxy `0.667`, scene-token `0.670`, balanced `0.390`
- seed `3`: proxy `0.820`, scene-token `0.760`, balanced `0.537`
- seed `4`: proxy `0.568`, scene-token `0.548`, balanced `0.290`
