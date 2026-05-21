Paired summary table uses the 30 scenes shared by the 2 started models.

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| axis_constrained_clamped | 30 | 0.600 | 0.207 (29 valid, 1 invalid) | 0.241 (29 valid, 1 invalid) | 0.170 | 27.230 | 1.536 |
| oracle_actor_axis_constrained_clamped | 30 | 0.600 | 0.310 (29 valid, 1 invalid) | 0.172 (29 valid, 1 invalid) | 0.184 | 29.891 | 1.417 |

### oracle_actor_axis_constrained_clamped vs axis_constrained_clamped
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.000 | n=30; McNemar p=n/a; better=0, worse=0 |
| raw_offroad | +0.103 | n=29, invalid=1; McNemar p=0.2500; better=0, worse=3 |
| raw_wrong_lane | -0.069 | n=29, invalid=1; McNemar p=0.5000; better=2, worse=0 |
| raw_progress | +0.014 | bootstrap95=[0.004, 0.024]; sign p=0.0428 |
| raw_dist_traveled_m | +2.661 | bootstrap95=[1.062, 4.411]; sign p=0.0987 |
| raw_dist_to_gt_trajectory | -0.118 | bootstrap95=[-0.408, 0.134]; sign p=1.0000 |
