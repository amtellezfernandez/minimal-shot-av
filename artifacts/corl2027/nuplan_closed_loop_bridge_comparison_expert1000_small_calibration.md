# nuPlan Closed-Loop Bridge Calibration

- Scene count: `30`
- Threshold: `1.000 m`

## Replay Class Summary

| Replay class | Count | Executed fail | Median replay clearance | Median executed clearance |
|---|---:|---:|---:|---:|
| replay_safe | 15 | 15 | 1.727 | -3.979 |
| replay_infeasible | 15 | 15 | 0.338 | -1.519 |

## Metric Variants

| Surface | Metric variant | Replay-safe controls failed | Replay-infeasible failed |
|---|---|---:|---:|
| executed_observation | center distance, no footprint | 0 | 5 |
| executed_observation | box clearance, no inflation | 15 | 15 |
| executed_observation | box clearance + 0.0m buffer | 15 | 15 |
| executed_observation | box clearance + 0.5m buffer | 15 | 15 |
| executed_observation | box clearance + 1.0m buffer | 15 | 15 |
| aligned_logged_actors | center distance, no footprint | 0 | 0 |
| aligned_logged_actors | box clearance, no inflation | 15 | 15 |
| aligned_logged_actors | box clearance + 0.0m buffer | 15 | 15 |
| aligned_logged_actors | box clearance + 0.5m buffer | 15 | 15 |
| aligned_logged_actors | box clearance + 1.0m buffer | 15 | 15 |

## Nearest Actor Agreement

| Agreement | Count |
|---|---:|
| same_actor | 0 |
| different_actor | 30 |
| missing_actor_id | 0 |

## Horizon Alignment

| Metric | Value |
|---|---:|
| median_replay_horizon_s | 4.000 |
| median_executed_horizon_s | 3.949 |
| median_dt_replay_s | 0.500 |
| median_dt_executed_s | 0.050 |

## Trajectory Agreement

| Metric | Value |
|---|---:|
| median_mean_trajectory_deviation_m | 0.466 |
| median_max_trajectory_deviation_m | 0.990 |
| median_final_pose_deviation_m | 0.184 |
