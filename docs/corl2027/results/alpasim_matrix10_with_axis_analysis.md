## World-Frame Actor Probe and Actor-Axis Rerun

The stale-frame-relative oracle result is superseded. The current paper uses a
world-frame actor proxy: actors are stored in global coordinates and transformed into the
current rollout ego frame at inference time.

Score-cutoff AlpaSim summaries on the matched 10-scene rerun:

| Variant | Collision | At-fault | Offroad | Wrong lane | Progress | Dist.-GT | Paired collision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Axis-constrained clamped baseline | 0.700 | 0.000 | 0.100 | 0.200 | 0.218 | 0.904 | -- |
| World-frame oracle + axis | 0.400 | 0.000 | 0.200 | 0.400 | 0.221 | 1.739 | better=3, worse=0, p=0.25 |
| World-frame oracle + lexicographic | 0.500 | 0.000 | 0.200 | 0.400 | 0.282 | 1.966 | better=2, worse=0, p=0.50 |
| Route-aware actor-axis proxy | 0.700 | 0.200 | 0.000 | 0.300 | 0.464 | 1.457 | better=1, worse=1, p=1.00 |
| Time-swept actor-axis proxy | 0.600 | 0.000 | 0.000 | 0.300 | 0.445 | 1.074 | better=1, worse=0, p=1.00 |

Interpretation: world-frame oracle actors are a useful diagnostic for actor-incomplete
proxy state, but not a population-level causal claim at n=10 and not a deployable method.
The time-swept actor-axis proxy is an engineering improvement over the first actor-axis
proxy, but collision improves by only one paired clip and wrong-lane regresses. Do not
headline it as a positive external method before adding rear-risk and lane guards.

## Metric Provenance Audit

The earlier `0.67 -> 0.33` oracle-actor collision reduction was aggregate-filtered and
only covered the first six completed scenes. It should not be the headline result.
Full unprocessed traces are still useful for debugging, but paper claims now use
score-cutoff summaries plus paired counts from the matched world-frame rerun.

## Scene-3 Mechanism Vignette

Scene 3 is the cleanest diagnostic example because actor-aware signals improve
progress/route tracking while wrong-lane behavior remains unresolved.

| Variant | Collision | At-fault | Offroad | Wrong lane | Progress | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Axis-constrained clamped | 0 | 0 | 0 | 0 | 0.323 | 2.144 |
| World-frame oracle + axis | 0 | 0 | 0 | 1 | 0.351 | 3.165 |
| World-frame oracle + lexicographic | 0 | 0 | 0 | 1 | 0.554 | 3.609 |
| Route-aware actor-axis proxy | 1 | 1 | 0 | 1 | 0.646 | 0.220 |
| Time-swept actor-axis proxy | 0 | 0 | 0 | 1 | 0.754 | 0.219 |

The route-aware actor-axis proxy turns this vignette into an at-fault collision; the
time-swept proxy removes that collision but not the wrong-lane failure. This localizes
the next method work to lane/offroad ranking plus rear-risk handling.

## Superseded Relative-Frame Oracle Result

The previous relative-frame oracle table (`0.700 -> 0.600` raw collision) should be
treated as superseded because actor hazards were projected into the source-rollout ego
frame and then reused under different ego motion. The world-frame rerun above is the
recoverable result to cite.

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
