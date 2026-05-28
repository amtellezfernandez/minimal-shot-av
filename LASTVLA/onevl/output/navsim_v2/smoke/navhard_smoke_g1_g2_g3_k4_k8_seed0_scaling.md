# NAVSIM v2 mapped smoke scaling across g1/g2/g3

Scope: seed-0 mapped `navhard_two_stage` smoke groups. Groups are cumulative mapped subsets, not full validation splits.

| Group | K | Scenes | Official top-1 | Best fixed candidate | Best fixed EPDMS | Token-oracle proxy | Zero-perception LOSO proxy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| g1 | 4 | 24 | 0.000000 | 3 | 0.362018 | 0.661465 |  |
| g1 | 8 | 24 | 0.000000 | 2 | 0.800662 | 0.695776 | 0.608252 |
| g2 | 4 | 48 | 0.045811 | 3 | 0.278660 | 0.602334 | 0.550518 |
| g2 | 8 | 48 | 0.045811 | 2 | 0.418909 | 0.614206 | 0.515941 |
| g3 | 4 | 64 | 0.064128 | 1 | 0.291094 | 0.554995 | 0.472268 |
| g3 | 8 | 64 | 0.064128 | 2 | 0.436662 | 0.601387 | 0.461974 |

## Matched K=4 to K=8 deltas

| Group | Scenes | Best fixed delta | Proxy delta |
|---:|---:|---:|---:|
| g1 | 24 | 0.438645 | 0.034310 |
| g2 | 48 | 0.140249 | 0.011873 |
| g3 | 64 | 0.145568 | 0.046392 |

Readout: across matched seed-0 groups, K=8 consistently improves best-fixed official EPDMS over K=4. Proxy-oracle gains are positive for g2/g3 and largest on g3. The zero-perception LOSO probe remains below the proxy top-1 baseline where measured, so the paper should frame this as a candidate-bank recoverability result plus an unresolved selector-learning problem, not as a solved linear-probe reranking result.
