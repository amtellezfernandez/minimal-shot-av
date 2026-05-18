Paired summary table uses the 10 scenes shared by the 2 started models.

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| actor_axis_oracle_actor_clamped | 10 | 0.700 | 0.100 | 0.400 | 0.471 | 90.914 | 8.555 |
| axis_constrained_clamped | 10 | 0.700 | 0.100 | 0.300 | 0.216 | 33.340 | 1.315 |

### actor_axis_oracle_actor_clamped vs axis_constrained_clamped
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.000 | n=10; McNemar p=n/a; better=0, worse=0 |
| raw_offroad | +0.000 | n=10; McNemar p=n/a; better=0, worse=0 |
| raw_wrong_lane | +0.100 | n=10; McNemar p=1.0000; better=1, worse=2 |
| raw_progress | +0.255 | bootstrap95=[0.083, 0.444]; sign p=0.1094 |
| raw_dist_traveled_m | +57.574 | bootstrap95=[15.491, 106.777]; sign p=0.1094 |
| raw_dist_to_gt_trajectory | +7.240 | bootstrap95=[-0.881, 22.222]; sign p=0.7539 |
