Paired summary table uses the 13 scenes shared by the 2 started models.

| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw_iter2 | 13 | 0.615 | 0.923 | 0.692 | 0.030 | 60.202 | 14.161 |
| clamped_iter2 | 13 | 0.692 | 0.500 (12 valid, 1 invalid) | 0.333 (12 valid, 1 invalid) | 0.369 | 58.004 | 4.992 |

### clamped_iter2 vs raw_iter2
| Axis | Delta | Paired test |
| --- | ---: | --- |
| raw_collision_any | +0.077 | n=13; McNemar p=1.0000; better=1, worse=2 |
| raw_offroad | -0.417 | n=12, invalid=1; McNemar p=0.0625; better=5, worse=0 |
| raw_wrong_lane | -0.333 | n=12, invalid=1; McNemar p=0.2188; better=5, worse=1 |
| raw_progress | +0.339 | bootstrap95=[0.264, 0.417]; sign p=0.0002 |
| raw_dist_traveled_m | -2.198 | bootstrap95=[-2.487, -1.937]; sign p=0.0002 |
| raw_dist_to_gt_trajectory | -9.169 | bootstrap95=[-11.550, -7.057]; sign p=0.0002 |
