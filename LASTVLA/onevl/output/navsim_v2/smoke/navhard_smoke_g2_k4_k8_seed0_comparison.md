# NAVSIM v2 mapped smoke g2 K=4 vs K=8

Scope: matched `g2` seed-0 runs with 48 scenes: 4 original NAVSIM v2 tokens plus 44 reactive synthetic scenarios. This is a smoke-scale comparison.

| K | Scenes | Official top-1 | Best fixed candidate | Best fixed EPDMS | Token-oracle proxy | Zero-perception LOSO proxy |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 48 | 0.045811 | 3 | 0.278660 | 0.602334 | 0.550518 |
| 8 | 48 | 0.045811 | 2 | 0.418909 | 0.614206 | 0.515941 |

Readout: increasing from K=4 to K=8 on the same 48-scene mapped group improves official best-fixed EPDMS from 0.2787 to 0.4189 and token-oracle proxy from 0.6023 to 0.6142. The gain is real but modest at this scale; the current zero-perception linear LOSO probe still fails to recover the headroom.
