# nuPlan Closed-Loop Bridge Calibration

- Scene count: `120`
- Threshold: `1.000 m`

## Replay Class Summary

| Replay class | Count | Executed fail | Median replay clearance | Median executed clearance |
|---|---:|---:|---:|---:|
| replay_safe | 60 | 60 | 1.390 | -3.838 |
| replay_infeasible | 60 | 60 | 0.365 | -1.545 |

## Metric Variants

| Surface | Metric variant | Replay-safe controls failed | Replay-infeasible failed |
|---|---|---:|---:|
| executed_observation | center distance, no footprint | 6 | 20 |
| executed_observation | box clearance, no inflation | 60 | 60 |
| executed_observation | box clearance + 0.0m buffer | 60 | 60 |
| executed_observation | box clearance + 0.5m buffer | 60 | 60 |
| executed_observation | box clearance + 1.0m buffer | 60 | 60 |
| aligned_logged_actors | center distance, no footprint | 0 | 0 |
| aligned_logged_actors | box clearance, no inflation | 0 | 40 |
| aligned_logged_actors | box clearance + 0.0m buffer | 0 | 40 |
| aligned_logged_actors | box clearance + 0.5m buffer | 31 | 60 |
| aligned_logged_actors | box clearance + 1.0m buffer | 53 | 60 |

## Nearest Actor Agreement

| Agreement | Count |
|---|---:|
| same_actor | 6 |
| different_actor | 114 |
| missing_actor_id | 0 |

## Horizon Alignment

| Metric | Value |
|---|---:|
| median_replay_horizon_s | 4.000 |
| median_executed_horizon_s | 3.950 |
| median_dt_replay_s | 0.500 |
| median_dt_executed_s | 0.050 |

## Trajectory Agreement

| Metric | Value |
|---|---:|
| median_mean_trajectory_deviation_m | 0.560 |
| median_max_trajectory_deviation_m | 1.101 |
| median_final_pose_deviation_m | 0.268 |
