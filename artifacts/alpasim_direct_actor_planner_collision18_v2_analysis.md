Paired summary table uses the 18 scenes shared by the 2 started models.

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selector_free_direct_actor_planner | 18 | 0.889 | 0.125 (16 valid, 2 invalid) | 0.312 (16 valid, 2 invalid) | 0.246 | 39.878 | 1.020 |
| axis_constrained_clamped | 18 | 1.000 | 0.118 (17 valid, 1 invalid) | 0.235 (17 valid, 1 invalid) | 0.167 | 25.492 | 1.409 |

### selector_free_direct_actor_planner vs axis_constrained_clamped
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | -0.111 | n=18; McNemar p=0.5000; better=2, worse=0 |
| raw_offroad | +0.000 | n=16, invalid=2; McNemar p=n/a; better=0, worse=0 |
| raw_wrong_lane | +0.062 | n=16, invalid=2; McNemar p=1.0000; better=0, worse=1 |
| raw_progress | +0.079 | bootstrap95=[0.044, 0.115]; sign p=0.0309 |
| raw_dist_traveled_m | +14.386 | bootstrap95=[7.542, 21.946]; sign p=0.0309 |
| raw_dist_to_gt_trajectory | -0.388 | bootstrap95=[-1.000, 0.186]; sign p=0.8145 |
