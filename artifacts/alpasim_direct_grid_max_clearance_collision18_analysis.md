Paired summary table uses the 18 scenes shared by the 2 started models.

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| selector_free_direct_grid_max_clearance_oracle | 18 | 0.889 | 0.059 (17 valid, 1 invalid) | 0.176 (17 valid, 1 invalid) | 0.174 | 26.776 | 0.741 |
| axis_constrained_clamped | 18 | 1.000 | 0.118 (17 valid, 1 invalid) | 0.235 (17 valid, 1 invalid) | 0.167 | 25.492 | 1.409 |

### selector_free_direct_grid_max_clearance_oracle vs axis_constrained_clamped
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | -0.111 | n=18; McNemar p=0.5000; better=2, worse=0 |
| raw_offroad | -0.059 | n=17, invalid=1; McNemar p=1.0000; better=1, worse=0 |
| raw_wrong_lane | -0.059 | n=17, invalid=1; McNemar p=1.0000; better=2, worse=1 |
| raw_progress | +0.007 | bootstrap95=[-0.007, 0.020]; sign p=0.4807 |
| raw_dist_traveled_m | +1.285 | bootstrap95=[-0.630, 3.242]; sign p=0.4807 |
| raw_dist_to_gt_trajectory | -0.667 | bootstrap95=[-1.224, -0.189]; sign p=0.0075 |
