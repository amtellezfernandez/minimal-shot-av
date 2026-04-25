# Waymo Data Access and Local Data Check

This file records where to access the Waymo Open Dataset, what is currently available in this workspace, and what must be verified after downloading WOD-E2E records.

## Official Access Links

- Waymo Open Dataset home and access entry point: <https://waymo.com/open/>
- Waymo Open Dataset download page: <https://waymo.com/open/download/>
- WOD-E2E dataset page: <https://waymo.com/open/data/e2e/>
- 2025 Vision-based E2E Driving Challenge: <https://waymo.com/open/challenges/2025/e2e-driving/>
- Official codebase: <https://github.com/waymo-research/waymo-open-dataset>
- Official tutorial notebook: <https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_vision_based_e2e_driving.ipynb>

The download page requires Google sign-in through Waymo. Dataset files cannot be fetched anonymously from this environment.

## Current Local Status

Checked from repo root on 2026-04-25.

Actual WOD-E2E TFRecords:

- Validation split is present under
  `waymo_open_dataset_end_to_end_camera_v_1_0_0/val`.
- Local validation shard count: `93`.
- Parsed preference-labeled validation frames: `479`.
- Parsed valid human-rated reference trajectories: `1437`.
- Train split is not present in this workspace.
- Test split is not present in this workspace.

Files currently present:

- Official Waymo code checkout: `waymo-open-dataset/`
- E2E proto definitions:
  - `waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_data.proto`
  - `waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_submission.proto`
  - `waymo-open-dataset/src/waymo_open_dataset/protos/end_to_end_driving_metrics.proto`
- Official E2E tutorial:
  - `waymo-open-dataset/tutorial/tutorial_vision_based_e2e_driving.ipynb`
- Official RFS utility:
  - `waymo-open-dataset/src/waymo_open_dataset/metrics/python/rater_feedback_utils.py`

Do not treat the separate `waymo-open-dataset/` checkout as downloaded
WOD-E2E data. It contains code, protos, tutorials, and non-submission examples,
not the local train/test challenge splits.

## Download Procedure

Manual steps required:

1. Open <https://waymo.com/open/>.
2. Use **Access Waymo Open Dataset** or go directly to <https://waymo.com/open/download/>.
3. Sign in with the Google account that has accepted the Waymo Open Dataset terms.
4. Select the End-to-End Driving dataset.
5. Download any missing train/test TFRecords plus any challenge frame-list JSON provided for test submission. The validation split is already present locally.
6. Place files outside git-tracked source, for example:
   - `data/waymo/e2e/train/`
   - `data/waymo/e2e/validation/`
   - `data/waymo/e2e/test/`
   - `data/waymo/e2e/submission_frames/`
7. Add those paths to local config or notebook parameters. Do not commit downloaded dataset files.

Recommended `.gitignore` coverage:

- `data/`
- `*.tfrecord`
- `*.tf_record`
- `*.record`
- `*.tar.gz`

## Post-Download Integrity Checks

Run these checks before any modeling:

- Count TFRecord files by split.
- Parse at least one `E2EDFrame` from each split.
- Verify `frame.context.name` is non-empty.
- Verify 8 camera images are present or report missing views.
- Verify camera calibrations are present.
- Verify `past_states` arrays have consistent lengths.
- Verify train/validation `future_states.pos_x` and `future_states.pos_y` have 20 points where populated.
- Verify validation records include frames with valid `preference_trajectories`.
- Verify invalid rater labels with score `-1` are skipped.
- Verify test records provide 12 seconds of camera context and no future labels.
- Verify the challenge-provided JSON frame list, if present, matches frames in the test records.

## Challenge-Specific Rules To Preserve

The 2025 challenge page states:

- The dataset contains 4,021 run segments, each 20 seconds long.
- Train split: 2,037 segments.
- Validation split: 479 segments.
- Test split: remaining segments, with only the first 12 seconds provided.
- Input camera data contains 8 cameras with 360-degree view.
- Participants predict future 5-second BEV waypoints from camera history, historical pose, and routing information.
- Submission is serialized `E2EDChallengeSubmission` proto, compressed into `.tar.gz`.
- Large submissions may be sharded across multiple proto files before tar/gzip.
- A challenge-provided JSON specifies which test frames must be covered.
- Each prediction is `(20, 2)`, sampled at 4 Hz.
- First predicted point is `t+0.25s`; the current timestep must not be included.
- Test submissions are limited to 6 every 30 days, excluding submissions that error out.
- Automated labeling methods such as MLLMs may be used to augment challenge training data.
- Leaderboard ranking is RFS at 3s and 5s, averaged over 11 scenario types.
- ADE against the highest-rater-scored trajectory is the tie-breaker/secondary metric.

This project can choose a stricter rule than the challenge allows. The current submission thesis is **no AV-dataset fine-tuning**, even though the challenge itself allows additional public research datasets and automated labeling.

## Local Data Status

Current known values:

```text
WOD_E2E_ROOT=waymo_open_dataset_end_to_end_camera_v_1_0_0
TRAIN_GLOB=
VALIDATION_GLOB=waymo_open_dataset_end_to_end_camera_v_1_0_0/val/val_*.tfrecord-*
TEST_GLOB=
SUBMISSION_FRAME_JSON=

train_tfrecord_count=
validation_tfrecord_count=93
test_tfrecord_count=

sample_train_frame_name=
sample_validation_frame_name=
sample_test_frame_name=

validation_frames_with_rater_labels=479
validation_valid_reference_trajectories=1437
test_required_frame_count=
notes=validation split present locally; train/test missing as of 2026-04-25
```
