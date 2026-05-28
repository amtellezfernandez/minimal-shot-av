# NAVSIM v2 mapped smoke K-scaling summary

Scope: `navhard_two_stage` mapped smoke group `g1` with 24 scenes: 2 original NAVSIM v2 tokens plus 22 reactive synthetic scenarios. This is a smoke-scale artifact for verifying the K-scaling pipeline, not a full validation claim.

| K | Scenes | Top-1 EPDMS | Best fixed candidate | Best fixed EPDMS | Token-oracle proxy | Proxy gap |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24 | 0.000000 | 0 | 0.000000 | 0.603267 | 0.603267 |
| 2 | 24 | 0.000000 | 0 | 0.000000 | 0.604058 | 0.604058 |
| 4 | 24 | 0.000000 | 3 | 0.362018 | 0.661465 | 0.661465 |
| 8 | 24 | 0.000000 | 2 | 0.800662 | 0.695776 | 0.695776 |
| 16 | 24 | 0.000000 | 7 | 0.627946 | 0.697256 | 0.697256 |
| 32 | 24 | 0.000000 | 13 | 0.832070 | 0.707940 | 0.707940 |

Readout: the proxy oracle rises monotonically from 0.6033 at K=1 to 0.7079 at K=32, while the best fixed branch jumps sharply at K=8 and only modestly improves by K=32. On this smoke group, the useful headroom appears by K=8 and then begins to saturate rather than scaling linearly with decoding budget.
