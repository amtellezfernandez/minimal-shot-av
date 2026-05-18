## Matched Rear-Risk Actor-Axis Rerun

The stale-frame-relative oracle result is superseded. The current actor proxy stores
actors in world coordinates and transforms them into the current rollout ego frame at
inference time. The newest matched rerun uses the same 10 scenes and same launch path for
the baseline and actor-axis model:

`runs/alpasim_rear_flow_actor_axis_10scene`

Evidence hierarchy:

| Number | Probe / method | Metric view | Allowed claim |
| --- | --- | --- | --- |
| `0.700 -> 0.400` | World-frame oracle actor probe | Score cutoff | Actor visibility affects cutoff-era collision accounting; diagnostic only. |
| `0.700 -> 0.600` | World-frame oracle actor probe | Raw full rollout | The oracle effect weakens to one paired raw improvement; inconclusive at `n=10`. |
| `0.700 -> 0.700` | Rear-risk actor-axis method | Same-pass raw full rollout | Deployable actor-axis proxy is active but not collision-positive. |

These rows answer different questions and must not be compared as repeated measurements
of one intervention. The first two rows are oracle-probe provenance. The third row is the
only deployable actor-axis method result.

Raw full-rollout summaries from `aggregate/metrics_unprocessed.parquet`:

| Variant | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT | Paired collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Axis-constrained clamped baseline | 0.700 | 0.100 | 0.300 | 0.216 | 33.340 | 1.315 | -- |
| Rear-risk actor-axis oracle proxy | 0.700 | 0.100 | 0.400 | 0.471 | 90.914 | 8.555 | better=0, worse=0 |

Paired tests:

| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.000 | n=10; McNemar p=n/a; better=0, worse=0 |
| raw_offroad | +0.000 | n=10; McNemar p=n/a; better=0, worse=0 |
| raw_wrong_lane | +0.100 | n=10; McNemar p=1.0000; better=1, worse=2 |
| raw_progress | +0.255 | bootstrap95=[0.083, 0.444]; sign p=0.1094 |
| raw_dist_traveled_m | +57.574 | bootstrap95=[15.491, 106.777]; sign p=0.1094 |
| raw_dist_to_gt_trajectory | +7.240 | bootstrap95=[-0.881, 22.222]; sign p=0.7539 |

Selector-log audit for the rear-risk actor-axis model:

| Signal | Count |
| --- | ---: |
| Frames | 1990 |
| Oracle proxy enabled | 1990 |
| Oracle proxy hit | 1990 |
| Frames with rear actor signal | 519 |
| Frames with rear-flow risk | 312 |
| Actor route guard applied | 878 |

Interpretation: the actor-aware proxy is instrumented and active, but it is not a
positive external method yet. It increases progress and distance travelled, but raw
collision and offroad are unchanged and wrong-lane regresses. The deployable method claim
should therefore be "inconclusive actor-axis proxy," not "actor-aware success."

## Metric Provenance Audit

The earlier `0.67 -> 0.33` oracle-actor collision reduction was aggregate-filtered and
only covered the first six completed scenes. The previous score-cutoff world-frame oracle
table also remains diagnostic, not a method result: score-cutoff aggregation can hide
late raw incidents after AlpaSim applies collision/offroad or distance-to-ground-truth
cutoffs. The paper now cites raw full-rollout metrics for the matched actor-axis method
claim and keeps score-cutoff summaries only as provenance.

## Scene-3 Mechanism Vignette

Scene 3 remains a clean diagnostic example: actor-aware signals improve route tracking
and progress, but introduce wrong-lane behavior.

| Variant | Collision | Offroad | Wrong lane | Progress | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Axis-constrained clamped | 0 | 0 | 0 | 0.185 | 0.791 |
| Rear-risk actor-axis oracle proxy | 0 | 0 | 1 | 0.754 | 0.219 |

## Superseded Relative-Frame Oracle Result

A still earlier relative-frame oracle table also reported `0.700 -> 0.600` raw collision,
but it is not part of the hierarchy above: actor hazards were projected into the
source-rollout ego frame and then reused under different ego motion. The matched
rear-risk actor-axis rerun above is the recoverable method result to cite.

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
