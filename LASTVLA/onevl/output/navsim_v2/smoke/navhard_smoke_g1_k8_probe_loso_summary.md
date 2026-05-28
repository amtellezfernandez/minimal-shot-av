# NAVSIM v2 mapped smoke K=8 ridge probe LOSO

Scope: mapped `navhard_two_stage` smoke group `g1`, K=8, seeds 0-3. Scores here are per-token proxy selection scores from NAVSIM rows, not the official combined EPDMS aggregation. This is a diagnostic artifact, not a final validation result.

| Mode | Seed | Features | Top-1 proxy | Oracle proxy | LOSO ridge proxy | Gap closed | Alpha |
|---|---:|---:|---:|---:|---:|---:|---:|
| zero-perception | 0 | 35 | 0.603267 | 0.695776 | 0.608252 | 5.4% | 0.01 |
| full probe | 0 | 37 | 0.603267 | 0.695776 | 0.608650 | 5.8% | 0.1 |
| zero-perception | 1 | 35 | 0.603267 | 0.701722 | 0.605569 | 2.3% | 0.1 |
| full probe | 1 | 37 | 0.603267 | 0.701722 | 0.604420 | 1.2% | 0.1 |
| zero-perception | 2 | 35 | 0.603267 | 0.659757 | 0.584892 | -32.5% | 1.0 |
| full probe | 2 | 37 | 0.603267 | 0.659757 | 0.589406 | -24.5% | 1.0 |
| zero-perception | 3 | 35 | 0.603267 | 0.698339 | 0.576550 | -28.1% | 1.0 |
| full probe | 3 | 37 | 0.603267 | 0.698339 | 0.572857 | -32.0% | 0.1 |

| Mode | Seeds | Mean LOSO proxy ± std | Mean gap closed ± std |
|---|---:|---:|---:|
| zero-perception | 4 | 0.593816 ± 0.013456 | -13.2% ± 17.2% |
| full probe | 4 | 0.593833 ± 0.014064 | -12.4% ± 16.2% |

Readout: On the 24-scene g1 smoke group, the zero-perception LOSO probe does not reliably close the K=8 oracle-proxy gap. This weak result is useful because it prevents overclaiming from the v1 NAVSIM probe and motivates the larger mapped-group run now queued on Spark.
