# WOD-E2E Analysis Notebook Plan

Add the WOD-E2E exploration notebook here:

- `wod_e2e_analysis.ipynb`

The notebook should show the thinking process, not just final metrics. It should be executable once WOD-E2E TFRecords are available locally or through a declared data path.

## 1. Dataset Access and Split Declaration

Document:

- Waymo access path, starting from <https://waymo.com/open/> or <https://waymo.com/open/download/>.
- Local TFRecord glob after download.
- Dataset version, expected `v1.0.0`.
- Split used: train, validation, test, or sample-only.
- Whether scenario tags are available.
- Whether rater feedback labels are available.
- Whether the challenge-provided required-frame JSON is available.
- Waymo terms and non-commercial constraints.

Do not mix validation and test claims. Test future logs and rater labels are withheld.

## 2. Proto Parse Smoke Test

Parse one record:

```python
dataset = tf.data.TFRecordDataset(filenames, compression_type="")
raw = next(dataset.as_numpy_iterator())
frame = wod_e2ed_pb2.E2EDFrame()
frame.ParseFromString(raw)
```

Print and assert:

- `frame.frame.context.name`
- `frame.frame.timestamp_micros`
- count of `frame.frame.images`
- count of `frame.frame.context.camera_calibrations`
- `frame.intent`
- lengths of `past_states.pos_x`, `past_states.vel_x`, and `past_states.accel_x`
- lengths of `future_states.pos_x` where populated
- number of `preference_trajectories`

## 3. Camera Inspection

For each camera:

- decode JPEG image
- record camera enum/name
- record image shape
- attach matching calibration
- render an 8-camera montage with view labels

Minimum visualizations:

- front-left/front/front-right strip
- full 8-camera montage
- one frame with projected future trajectory if projection code is available

## 4. Ego Trajectory Inspection

Plot in vehicle coordinates:

- past ego trajectory over `(-4s, 0]`
- future log trajectory over `(0, 5s]`
- velocity magnitude over past states
- acceleration magnitude over past states

Sanity checks:

- `+x` should represent forward motion.
- `+y` should represent leftward offset.
- future target should contain 20 points for labeled train/validation records.
- no current-time duplicate should be included in prediction targets.
- note origin mismatch risk: proto states rear axle, public prose may say vehicle center.

## 5. Scenario Cluster Analysis

When scenario tags are available, summarize counts and examples for:

- construction
- intersection
- pedestrian
- cyclist
- multi-lane maneuver
- single-lane maneuver
- cut-in
- foreign object debris
- special vehicle
- spotlight
- others

For each inspected cluster, record:

- visual cues
- likely maneuver families
- why constant velocity fails
- why log imitation may be insufficient
- which camera views carry the key evidence

## 6. Rater Feedback Exploration

Find validation frames with valid `preference_trajectories`:

- skip empty preference trajectories
- skip trajectories with `preference_score == -1`
- stack rater trajectories as `[P, 20, 2]`
- collect scores as `[P]`
- compute initial speed from the last past-state velocity

Use the official utility:

```python
rater_feedback_utils.get_rater_feedback_score(
    prediction_trajectories,      # [B, I, 20, 2]
    prediction_probabilities,     # [B, I]
    rater_specified_trajectories,
    rater_scores,
    initial_speed,
    frequency=4,
    length_seconds=5,
)
```

Compare:

- RFS versus ADE at 3s
- RFS versus ADE at 5s
- highest-rated rater trajectory versus logged future
- predictions inside versus outside trust regions

Visualize the 3s and 5s points because these are metric-critical.

## 7. Baseline Policies

Implement and evaluate these before architecture claims:

- constant stop: all zeros
- constant velocity: extrapolate from final past velocity
- curvature extrapolation: fit recent ego path trend
- intent template: straight/left/right kinematic templates
- learned residual candidate model
- source-aware RFS selector

Report:

- valid trajectory rate
- ADE at 3 seconds
- ADE at 5 seconds
- RFS on frames with valid rater labels
- cluster-level RFS where tags are available
- runtime per target frame

## 8. Candidate and Selector Experiments

For each model-side experiment:

- record feature set and training split
- record candidate sources enabled
- record selector features enabled
- record validation RFS, ADE at 3s, ADE at 5s, and invalid rate
- record runtime per frame
- record negative experiments and remove regressing features from the default path

The notebook should make it clear which behavior came from candidate generation and which behavior came from RFS-based selection.

## 9. Failure Cases

Document at least:

- one visual interpretation failure
- one trajectory-selection failure
- one metric mismatch, such as low ADE but poor RFS or safe-looking stop with poor rater score

For each failure:

- frame name
- scenario cluster
- camera montage
- past trajectory
- predicted trajectory
- rater trajectories and scores, if available
- failed component: candidate generator, temporal state, source ranker, trajectory projector, or selector

## 10. Final System Rationale

End the notebook with:

- what survived ablation
- what was rejected
- why RFS changed the design
- why the final system respects the no-AV-finetune constraint
- what would be submitted to leaderboard versus what is only validation analysis
