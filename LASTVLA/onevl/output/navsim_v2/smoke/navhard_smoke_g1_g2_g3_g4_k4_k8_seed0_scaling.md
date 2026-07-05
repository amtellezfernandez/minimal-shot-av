# NAVSIM v2 mapped smoke scaling

Seed-specific corrected hybrid decoding over matched NAVSIM v2 smoke groups. Official EPDMS uses fixed-candidate selection over each mapped subset. The token-oracle proxy is per-scene candidate-bank headroom and is not an official aggregate.

| Group | Scenes | K | Official top-1 | Best fixed candidate | Best fixed EPDMS | Token-oracle proxy | Zero-perception LOSO proxy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| g1 | 24 | 4 | 0.000000 | 3 | 0.362018 | 0.661465 | - |
| g1 | 24 | 8 | 0.000000 | 2 | 0.800662 | 0.695776 | 0.608252 |
| g2 | 48 | 4 | 0.045811 | 3 | 0.278660 | 0.602334 | 0.550518 |
| g2 | 48 | 8 | 0.045811 | 2 | 0.418909 | 0.614206 | 0.515941 |
| g3 | 64 | 4 | 0.064128 | 1 | 0.291094 | 0.554995 | 0.472268 |
| g3 | 64 | 8 | 0.064128 | 2 | 0.436662 | 0.601387 | 0.461974 |
| g4 | 92 | 4 | 0.157245 | 2 | 0.378455 | 0.663590 | 0.557979 |
| g4 | 92 | 8 | 0.157245 | 2 | 0.379370 | 0.696682 | 0.540848 |

## Delta readout

| Group | Scenes | Best fixed delta | Proxy delta | Zero-perception delta |
| --- | ---: | ---: | ---: | ---: |
| g1 | 24 | 0.438645 | 0.034310 | - |
| g2 | 48 | 0.140249 | 0.011873 | -0.034577 |
| g3 | 64 | 0.145568 | 0.046392 | -0.010295 |
| g4 | 92 | 0.000915 | 0.033092 | -0.017131 |
