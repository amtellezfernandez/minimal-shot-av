# Failure-Gate Progress-Retention Ablation

Seed-0 DB split. Retention requires fallback progress to be at least a fraction of proxy progress before switching.

| Config | Replay fail rate | Replay progress | Replay ADE3 | Bridge gate fails | Bridge proxy fails | Bridge gate progress |
|---|---:|---:|---:|---:|---:|---:|
| no_guard | 0.220 | 20.805 | 5.450 | 0/40 | 20/40 | 9.988 |
| retention_0.25 | 0.457 | 26.856 | 3.152 | 20/40 | 20/40 | 21.694 |
| retention_0.50 | 0.493 | 27.386 | 2.973 | 20/40 | 20/40 | 22.189 |
| retention_0.75 | 0.673 | 29.212 | 2.334 | n/a | n/a | n/a |

## Interpretation

- Progress retention improves replay utility but blocks the low-progress switches that were responsible for the aligned bridge gain.
- On the seed-0 bridge, retention `0.25` and `0.50` both leave aligned failures at `20/40`, the same as proxy.
- Therefore the progress-retention guard is an appendix ablation, not the main method.
