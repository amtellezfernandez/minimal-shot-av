# NAVSIM v2 mapped smoke K=4/K=8 seed variance

Scope: `navhard_two_stage` mapped smoke group `g1` with 24 scenes: 2 original NAVSIM v2 tokens plus 22 reactive synthetic scenarios. This is a smoke-scale variance artifact, not a full validation claim.

## Per-seed results

| K | Seed | Top-1 EPDMS | Best fixed candidate | Best fixed EPDMS | Token-oracle proxy | Oracle winner not top-1 |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 0 | 0.000000 | 3 | 0.362018 | 0.661465 | 41.7% |
| 4 | 1 | 0.000000 | 1 | 0.437500 | 0.684874 | 37.5% |
| 4 | 2 | 0.000000 | 0 | 0.000000 | 0.629534 | 37.5% |
| 4 | 3 | 0.000000 | 0 | 0.000000 | 0.613060 | 33.3% |
| 8 | 0 | 0.000000 | 2 | 0.800662 | 0.695776 | 50.0% |
| 8 | 1 | 0.000000 | 5 | 0.438276 | 0.701722 | 50.0% |
| 8 | 2 | 0.000000 | 7 | 0.437500 | 0.659757 | 33.3% |
| 8 | 3 | 0.000000 | 7 | 0.737777 | 0.698339 | 50.0% |

## Aggregate readout

| K | Seeds | Best fixed mean ± std | Proxy mean ± std | Winner-not-top1 mean |
|---:|---:|---:|---:|---:|
| 4 | 4 | 0.199879 ± 0.201653 | 0.647233 ± 0.027841 | 37.5% |
| 8 | 4 | 0.603554 ± 0.167151 | 0.688898 ± 0.016956 | 45.8% |

Readout: On this mapped smoke group, K=8 improves the mean proxy oracle over K=4 and reduces proxy variance across seeds. The best fixed branch also becomes much stronger at K=8, driven by seeds 0 and 3, but fixed-branch identity remains unstable, so the result supports a candidate-bank density story more than a fixed-index branch story.
