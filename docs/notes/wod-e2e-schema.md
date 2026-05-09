# WOD-E2E Schema and Metric Reference

This is the compact implementation reference for WOD-E2E. Use it when building loaders, notebook checks, evaluators, or submission writers.

Local source files:

- `waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_data.proto`
- `waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_submission.proto`
- `waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_metrics.proto`
- `waymo-open-dataset/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py`
- `waymo-open-dataset/tutorial/tutorial_vision_based_e2e_driving.ipynb`

## `E2EDFrame`

`E2EDFrame` is the record parsed from TFRecord bytes.

Fields used by this project:

| Field | Meaning | Implementation note |
| --- | --- | --- |
| `frame` | Waymo `Frame` containing image/calibration metadata | Only context, timestamp, images, and camera calibrations are expected. |
| `frame.context.name` | Unique frame identifier | Must be copied into submission `frame_name`. |
| `frame.timestamp_micros` | Current frame timestamp | Useful for sanity checks and sequence ordering. |
| `frame.images` | Camera JPEG payloads | Decode with TensorFlow image utilities or equivalent. |
| `frame.context.camera_calibrations` | Intrinsics/extrinsics | Required for image projection and camera geometry. |
| `past_states` | Ego trajectory over `(-4s, 0]` | 4 Hz history, includes position, velocity, acceleration. |
| `future_states` | Ego future over `(0, 5s]` | 4 Hz target in train/validation. |
| `intent` | Coarse route command | `UNKNOWN`, `GO_STRAIGHT`, `GO_LEFT`, `GO_RIGHT`. |
| `preference_trajectories` | Rater-scored future trajectories | Valid for selected validation frames. |

Do not assume the E2E proto contains map data, lidar, object tracks, or dense annotations.

## `EgoTrajectoryStates`

Trajectory fields:

| Field | Unit | Used for |
| --- | --- | --- |
| `pos_x` | meters | past/future x trajectory, forward positive |
| `pos_y` | meters | past/future y trajectory, left positive |
| `pos_z` | meters | visualization only for future states |
| `vel_x` | m/s | current/previous ego velocity |
| `vel_y` | m/s | current/previous ego velocity |
| `accel_x` | m/s^2 | current/previous ego acceleration |
| `accel_y` | m/s^2 | current/previous ego acceleration |
| `preference_score` | `[0, 10]` | only populated for rated trajectories |

Timing:

- `past_states`: `(-4s, 0]`, 4 Hz, usually 16 samples.
- `future_states`: `(0, 5s]`, 4 Hz, usually 20 samples.
- Submission trajectory: 20 samples, same future timing, no current-time duplicate.

Coordinate convention:

- `+x` is forward.
- `+y` is left.
- `+z` is up.
- Proto states the origin is the middle of the ego rear axle.
- Public prose may describe vehicle center; treat this as a version/documentation mismatch to verify with actual data.

## Intent Enum

| Value | Meaning |
| --- | --- |
| `0` | `UNKNOWN` |
| `1` | `GO_STRAIGHT` |
| `2` | `GO_LEFT` |
| `3` | `GO_RIGHT` |

Use intent as a weak route prior, not a safety override.

## Submission Proto

`TrajectoryPrediction`:

- `pos_x`: repeated float, length 20.
- `pos_y`: repeated float, length 20.
- Units are meters.
- Coordinates are vehicle-frame at `t=0`.
- First point is `t+0.25s`.
- Last point is `t+5.0s`.

`FrameTrajectoryPredictions`:

- `frame_name`: copied from `E2EDFrame.frame.context.name`.
- `trajectory`: exactly one `TrajectoryPrediction`.

`E2EDChallengeSubmission`:

- `predictions`: one per requested test frame.
- `submission_type`: must be `E2ED_SUBMISSION`.
- `account_name`: Waymo account email.
- `unique_method_name`: short method name.
- `authors`, `affiliation`, `description`, `method_link`.
- `uses_public_model_pretraining`: required.
- `public_model_names`: required when public models are used.
- `num_model_parameters`: required, string with suffix such as `200M` or `7B`.

Challenge page packaging requirements:

- The test-set required frames are specified by a challenge-provided JSON.
- Submit serialized `E2EDChallengeSubmission` proto file(s) compressed into `.tar.gz`.
- Sharded proto files are allowed if a single proto is too large.
- Each shard must contain a subset of the required `FrameTrajectoryPredictions`.
- The current timestep must not be included in the 20 predicted points.
- Test submissions are limited to 6 every 30 days, excluding errored submissions.

## RFS Utility Contract

Local function:

```python
rater_feedback_utils.get_rater_feedback_score(
    inference_trajectories,       # [B, I, T, 2]
    inference_probs,              # [B, I]
    rater_specified_trajectories, # List[List[np.ndarray]]
    rater_feedback_labels,        # List[np.ndarray]
    init_speed,                   # [B]
    frequency=4,
    length_seconds=5,
)
```

Defaults:

- `frequency = 4`
- `length_seconds = 5`
- `default_num_of_rater_specified_trajectories = 3`
- `lat_lng_threshold_multipliers = (1.0, 4.0)`
- `decay_factor = 0.1`
- `minimum_score_outside_trust_region = 4.0`

Metric-critical timestamps:

- 3 seconds: index `3 * 4 - 1 = 11`
- 5 seconds: index `5 * 4 - 1 = 19`

Base thresholds:

| Time | Lateral | Longitudinal |
| --- | ---: | ---: |
| 3s | `1.0m` | `4.0m` |
| 5s | `1.8m` | `7.2m` |

Speed scaling:

```text
scale(v) = 0.5                                  if v < 1.4 m/s
scale(v) = 0.5 + 0.5 * (v - 1.4) / (11 - 1.4)  if 1.4 <= v < 11 m/s
scale(v) = 1.0                                  if v >= 11 m/s
```

`init_speed` should be computed from the last past-state velocity:

```python
init_speed = sqrt(past_states.vel_x[-1] ** 2 + past_states.vel_y[-1] ** 2)
```

## Loader Sanity Checks

Every loader should assert:

- `frame.context.name` is non-empty.
- Required image names are present or missing views are reported.
- Past state arrays have equal lengths.
- Future state arrays have equal lengths where labels exist.
- Future target length is 20 before training/evaluation.
- Submitted `pos_x` and `pos_y` lengths are exactly 20.
- All trajectory values are finite.
- No current-time point is prepended.
- Rater labels with score `-1` are ignored.

## Evaluation Sanity Checks

Before reporting results:

- Compare constant stop, constant velocity, and curvature extrapolation.
- Report ADE separately from RFS.
- Report RFS only on validation frames with valid preference trajectories.
- Break down failures by scenario cluster when tags are available.
- Visualize 3s and 5s points because they dominate RFS matching.
