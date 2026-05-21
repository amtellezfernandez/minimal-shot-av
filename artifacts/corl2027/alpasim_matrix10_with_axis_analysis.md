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
