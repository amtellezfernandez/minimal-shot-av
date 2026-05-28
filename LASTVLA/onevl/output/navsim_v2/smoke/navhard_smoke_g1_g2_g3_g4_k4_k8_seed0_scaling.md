# NAVSIM v2 mapped K4/K8 scaling, groups g1-g4

Seed 0, corrected hybrid decoding. Official EPDMS uses fixed candidate selection over the mapped navhard smoke subset. The proxy oracle is per-scene and should be read only as candidate-bank headroom, not an official aggregate.

| group | scenes | K | top1 EPDMS | best fixed candidate | best fixed EPDMS | proxy oracle |
|---|---:|---:|---:|---:|---:|---:|
| g1 | 24 | 4 | 0.0000 | 3 | 0.3620 | 0.6615 |
| g1 | 24 | 8 | 0.0000 | 2 | 0.8007 | 0.6958 |
| g2 | 48 | 4 | 0.0458 | 3 | 0.2787 | 0.6023 |
| g2 | 48 | 8 | 0.0458 | 2 | 0.4189 | 0.6142 |
| g3 | 64 | 4 | 0.0641 | 1 | 0.2911 | 0.5550 |
| g3 | 64 | 8 | 0.0641 | 2 | 0.4367 | 0.6014 |
| g4 | 92 | 4 | 0.1572 | 2 | 0.3785 | 0.6636 |
| g4 | 92 | 8 | 0.1572 | 2 | 0.3794 | 0.6967 |

Key readout: K8 consistently improves the token-level oracle proxy over K4, but official fixed-candidate gains vary with scale. On g4, K8 exposes higher proxy headroom while the best fixed official candidate stays effectively tied with K4, which argues for studying selector quality rather than claiming monotonic official gains from K alone.
