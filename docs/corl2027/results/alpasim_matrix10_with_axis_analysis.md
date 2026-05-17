## Metric Provenance Audit

The earlier `0.67 -> 0.33` oracle-actor collision reduction was **aggregate-filtered** and
only covered the first six completed scenes. It should not be the headline result.

| Subset / metric source | Axis-constrained clamped collision | Oracle-actor collision | Interpretation |
| --- | ---: | ---: | --- |
| First 6 scenes, aggregate-filtered `metrics_results.txt` | 0.667 | 0.333 | Diagnostic live read; filtered after incident/cutoff events. |
| First 6 scenes, raw `metrics_unprocessed.parquet` | 0.667 | 0.500 | Same scene subset without AlpaSim post-filtering. |
| Full 10 scenes, aggregate-filtered `metrics_results.txt` | 0.700 | 0.400 | Useful secondary view, but filtered/truncated. |
| Full 10 scenes, raw `metrics_unprocessed.parquet` | 0.700 | 0.600 | Headline result for paper tables. |

Use raw per-scene outcomes for main claims. Aggregate-filtered metrics can be reported as
a secondary AlpaSim scoring view, but they should not carry the causal claim.

## Scene-3 Mechanism First

Scene 3 is the cleanest diagnostic example because actor-aware ranking changes the
selected tokens while producing almost no actor-veto pressure. Wrong-lane failure
therefore persists **inside the admissible set**, not because the actor veto is too
conservative.

| Variant | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT | Veto frames | Decision summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Axis-constrained clamped | 0 | 0 | 0 | 0.323 | 55.225 | 2.144 | n/a | baseline proxy, no oracle actors |
| Oracle actor + axis selector | 0 | 0 | 1 | 0.350 | 59.947 | 2.522 | 2/199 | 157 agreement, 40 DAgger wins, 2 fallback |
| Oracle actor + lexicographic selector | 0 | 0 | 1 | 0.491 | 84.393 | 2.031 | 3/199 | 162 Spotlight wins, 17 DAgger wins, 3 fallback |

The mechanism is sharper than the aggregate table: privileged actors are present, but the
lane error remains with only 2-3 actor-veto frames out of 199. This localizes the next
bottleneck to missing lane/offroad observability in the ranking features rather than to
over-conservative actor suppression.

## Oracle Actor Proxy, Full 10 Raw Scenes

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| axis_constrained_clamped | 10 | 0.700 | 0.200 | 0.200 | 0.358 | 54.802 | 2.252 |
| oracle_actor_axis | 10 | 0.600 | 0.300 | 0.500 | 0.289 | 46.233 | 4.596 |
| oracle_actor_lexicographic | 10 | 0.600 | 0.200 | 0.500 | 0.352 | 54.965 | 3.241 |

Oracle actors modestly reduce raw collision (`0.700 -> 0.600`), which is one fewer
collision in the 10 paired clips or a 0.10 absolute raw reduction. They also worsen
wrong-lane behavior (`0.200 -> 0.500`). Lexicographic reordering restores offroad and
most progress relative to the axis oracle, but it does not solve wrong-lane. The correct
claim is therefore not "oracle actors solve AlpaSim transfer"; it is that actor-complete
proxy state affects a measured subset of the collision surface while exposing a separate
lane-ranking bottleneck.

## Pre-Oracle Learned-Policy Matrix

Paired summary table uses the 10 scenes shared by the 5 started models.

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_iter2 | 10 | 0.600 | 0.900 | 0.700 | 0.034 | 61.979 | 13.858 |
| axis_constrained_clamped | 10 | 0.700 | 0.200 | 0.200 | 0.358 | 54.802 | 2.252 |
| clamped_iter2 | 10 | 0.700 | 0.500 | 0.200 | 0.384 | 59.850 | 5.207 |
| hybrid_clamped | 10 | 0.800 | 0.200 | 0.700 | 0.837 | 166.063 | 22.713 |
| srcdecay | 10 | 0.600 | 0.900 | 0.400 | 0.175 | 62.924 | 27.156 |

### axis_constrained_clamped vs raw_iter2
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.100 | n=10; McNemar p=1.0000; better=1, worse=2 |
| raw_offroad | -0.700 | n=10; McNemar p=0.0156; better=7, worse=0 |
| raw_wrong_lane | -0.500 | n=10; McNemar p=0.0625; better=5, worse=0 |
| raw_progress | +0.324 | bootstrap95=[0.252, 0.399]; sign p=0.0020 |
| raw_dist_traveled_m | -7.177 | bootstrap95=[-9.114, -5.185]; sign p=0.0020 |
| raw_dist_to_gt_trajectory | -11.605 | bootstrap95=[-13.417, -9.634]; sign p=0.0020 |

### clamped_iter2 vs raw_iter2
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.100 | n=10; McNemar p=1.0000; better=1, worse=2 |
| raw_offroad | -0.400 | n=10; McNemar p=0.1250; better=4, worse=0 |
| raw_wrong_lane | -0.500 | n=10; McNemar p=0.0625; better=5, worse=0 |
| raw_progress | +0.350 | bootstrap95=[0.275, 0.433]; sign p=0.0020 |
| raw_dist_traveled_m | -2.129 | bootstrap95=[-2.452, -1.839]; sign p=0.0020 |
| raw_dist_to_gt_trajectory | -8.651 | bootstrap95=[-10.764, -6.642]; sign p=0.0020 |

### hybrid_clamped vs raw_iter2
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.200 | n=10; McNemar p=0.5000; better=0, worse=2 |
| raw_offroad | -0.700 | n=10; McNemar p=0.0156; better=7, worse=0 |
| raw_wrong_lane | +0.000 | n=10; McNemar p=1.0000; better=3, worse=3 |
| raw_progress | +0.802 | bootstrap95=[0.685, 0.907]; sign p=0.0020 |
| raw_dist_traveled_m | +104.083 | bootstrap95=[69.785, 136.176]; sign p=0.0020 |
| raw_dist_to_gt_trajectory | +8.855 | bootstrap95=[-9.231, 34.895]; sign p=0.7539 |

### srcdecay vs raw_iter2
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.000 | n=10; McNemar p=n/a; better=0, worse=0 |
| raw_offroad | +0.000 | n=10; McNemar p=n/a; better=0, worse=0 |
| raw_wrong_lane | -0.300 | n=10; McNemar p=0.2500; better=3, worse=0 |
| raw_progress | +0.140 | bootstrap95=[0.075, 0.215]; sign p=0.0020 |
| raw_dist_traveled_m | +0.945 | bootstrap95=[0.445, 1.406]; sign p=0.3438 |
| raw_dist_to_gt_trajectory | +13.298 | bootstrap95=[6.575, 17.835]; sign p=0.0215 |
