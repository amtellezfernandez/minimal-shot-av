# WOD-E2E Dataset Dossier

This document is the dataset-first reference for the Minimal-Shot AV submission. It is intentionally more concrete than the architecture pitch: every model decision should trace back to a WOD-E2E field, evaluation rule, or scenario-cluster failure mode.

Primary sources:

- Waymo Open Dataset access page: <https://waymo.com/open/>
- Waymo Open Dataset download page: <https://waymo.com/open/download/>
- WOD-E2E dataset page: <https://waymo.com/open/data/e2e/>
- 2025 Vision-based E2E challenge page: <https://waymo.com/open/challenges/2025/e2e-driving/>
- WOD-E2E paper: <https://arxiv.org/abs/2510.26125>
- Official codebase: <https://github.com/waymo-research/waymo-open-dataset>
- Local proto and metric references in `workspace/waymo-open-dataset/src/waymo_open_dataset/`

Related local references:

- Data access and local status: `docs/waymo-data-access.md`
- Schema and metric reference: `docs/wod-e2e-schema.md`
- User-provided leaderboard snapshot: `docs/leaderboard.md`
- Official winner report notes: `docs/winning-methods.md`
- Competitive synthesis: `docs/competitive-analysis.md`
- Zero-shot architecture proposal: `docs/spotlight-reflex.md`

## Dataset Thesis

WOD-E2E is not a normal driving benchmark. It is a curated long-tail benchmark built to expose failures in end-to-end driving systems on rare but safety-critical events. The submission should therefore optimize for **rare-scene reasoning and rater-preferred trajectory choice**, not just average imitation.

The key facts:

- 4,021 total driving segments, approximately 12 hours.
- 2,037 training segments.
- 479 validation segments with rater feedback labels and scenario tags.
- 1,505 testing segments with future driving logs and rater labels withheld.
- Each train/validation segment is 20 seconds.
- Test inputs provide 12 seconds before the decision point.
- The subsequent 8 seconds in test are reserved for evaluation.
- Each segment includes 8-camera 360-degree imagery, high-level routing command, and ego state.
- Required output is one 5-second ego trajectory at 4 Hz: 20 `(x, y)` waypoints.
- The challenge page states the long-tail events were mined at less than `0.003%` frequency in daily driving.

Current local data status is tracked in `docs/waymo-data-access.md`. At the
time of this repo check, the Waymo code/protos/tutorial are present locally and
the WOD-E2E validation split is present under
`workspace/waymo_open_dataset_end_to_end_camera_v_1_0_0/val`. Train and test TFRecords
are not present in this workspace. The local parser currently finds 479
preference-labeled validation frames and 1,437 valid human-rated reference
trajectories.

## Data Record Contract

The unit of data is `E2EDFrame`.

Important fields:

- `frame`: a Waymo `Frame` object with camera images, camera calibrations, context metadata, and timestamp populated.
- `frame.context.name`: unique frame identifier and the key used in submission predictions.
- `frame.context.camera_calibrations`: intrinsics/extrinsics for all cameras.
- `frame.images`: JPEG images for the available camera views.
- `future_states`: ego future state over `(0, 5s]` at 4 Hz. Position fields are prediction targets in train/validation.
- `past_states`: ego history over `(-4s, 0]` at 4 Hz.
- `intent`: route-level command as `UNKNOWN`, `GO_STRAIGHT`, `GO_LEFT`, or `GO_RIGHT`.
- `preference_trajectories`: up to 3 rater-scored future trajectories on selected validation frames.

Only a subset of the regular Waymo `Frame` is populated for WOD-E2E. Do not assume lidar, object labels, map features, traffic lights, or full perception annotations exist in this proto.

## Coordinate Systems

The public dataset page describes vehicle coordinates with `+x` forward, `+y` left, and `+z` up. The official proto uses the same axis convention but states that the origin is at the middle of the ego vehicle's rear axle.

This is a critical implementation hazard:

- Treat all trajectory fields as vehicle-frame coordinates.
- Do not mix them with camera-frame coordinates without applying calibration.
- Do not assume the origin is the vehicle center unless verified against the exact downloaded proto/version.
- The submitted trajectory is relative to the ego vehicle at `t=0`.

Camera frames are separate right-handed sensor frames. Camera calibration is required for projecting trajectories into images.

## Camera Data

WOD-E2E provides 8 camera views around the ego vehicle:

- front
- front left
- front right
- side left
- side right
- rear
- rear left
- rear right

Dataset page facts:

- Camera video is 10 Hz.
- Each camera image is a JPEG.
- Intrinsics and extrinsics are provided for projection.
- The challenge input uses 360-degree camera context around the driving agent.

Architecture implication:

- A single front camera is not enough for this benchmark. Cut-ins, debris, pedestrians, special vehicles, and rear/side interactions can depend on non-front views.
- Any future visual encoder should consume either all 8 views or a declared camera montage that preserves view identity.
- Any visualization should label camera names explicitly to avoid confusing left/right evidence.

## Ego State

`EgoTrajectoryStates` contains:

- `pos_x`, `pos_y`, `pos_z`
- `vel_x`, `vel_y`
- `accel_x`, `accel_y`
- optional `preference_score` for rater trajectories

Past ego state:

- Time interval: `(-4s, 0]`.
- Frequency: 4 Hz.
- Expected history length is 16 points when complete.
- Last velocity sample is the current speed used by the RFS utility.

Future ego state:

- Time interval: `(0, 5s]`.
- Frequency: 4 Hz.
- Expected target length is 20 points.
- First target point is at `t+0.25s`.
- The current-time point must not be included.

Architecture implication:

- Baselines should include constant-velocity extrapolation from the final past velocity and curvature extrapolation from recent past positions.
- Temporal models should not waste capacity rediscovering the output clock; the 20-step horizon is fixed.
- The trajectory decoder should always produce exactly 20 finite points.

## Intent / Route Command

`EgoIntent.Intent` is coarse:

- `UNKNOWN = 0`
- `GO_STRAIGHT = 1`
- `GO_LEFT = 2`
- `GO_RIGHT = 3`

This is routing context, not a detailed route or map. It should constrain candidate trajectories but not override visible safety evidence.

Architecture implication:

- The maneuver library should treat intent as a prior over trajectory families.
- Route-command violations should be counted as diagnostics.
- `UNKNOWN` must fall back to history and scene evidence.

## Rater Feedback Labels

`preference_trajectories` contains future trajectories with human-labeled scores:

- Up to 3 rated trajectories for selected frames.
- Valid scores are `[0, 10]`.
- Invalid/empty labels may have `-1` or no trajectory.
- Validation includes these labels for analysis and local evaluation.
- Test labels are withheld.

The dataset intentionally captures that multiple futures can be acceptable. This is the main reason ADE-only training is misaligned: the logged future is not necessarily the only good driving decision.

Architecture implication:

- Candidate generation matters. A single deterministic imitation head can miss alternate rater-approved futures.
- The selector should rank candidate futures by rater-style acceptability, not just distance to logs.
- Failure analysis should distinguish "physically smooth but rater-poor" from "invalid trajectory."

## Submission Contract

The challenge submission uses `E2EDChallengeSubmission`.

Required prediction shape:

- One `FrameTrajectoryPredictions` entry per requested test frame.
- `frame_name` must match `E2EDFrame.frame.context.name`.
- `trajectory` must contain `pos_x` and `pos_y`.
- Each of `pos_x` and `pos_y` must have exactly 20 values.
- Waypoints cover `(0, 5s]` at 4 Hz.
- First waypoint is `t+0.25s`; last is `t+5s`.
- No `z` coordinate is submitted.

Submission metadata fields:

- `submission_type = E2ED_SUBMISSION`
- `account_name`
- `unique_method_name`
- `authors`
- `affiliation`
- `description`
- `method_link`
- `uses_public_model_pretraining`
- `public_model_names`
- `num_model_parameters`

Submission packaging:

- Upload a serialized `E2EDChallengeSubmission` proto compressed into `.tar.gz`.
- If a single proto is too large, shard predictions across multiple proto files, then tar/gzip the shards.
- The challenge provides a JSON specifying which test frames must be covered.
- Test submissions are limited to 6 every 30 days; submissions that error out do not count against this limit.

Challenge training allowance:

- The challenge allows participants to use public research/academic datasets in addition to the provided training set.
- The challenge allows automated labeling methods such as MLLMs to augment challenge training data.
- This repo's current thesis is stricter: no AV-dataset fine-tuning for the submitted policy unless explicitly revised in `models/DECLARATION.md`.

Architecture implication:

- The implementation should have a strict validator before writing submission protos.
- Any frozen VLM/LLM use must be reflected in `uses_public_model_pretraining`, `public_model_names`, and `models/DECLARATION.md`.

## RFS Metric Mechanics

Leaderboard ranking is by Rater Feedback Score averaged across 11 scenario clusters. ADE at 3 and 5 seconds is secondary.

RFS behavior from the official utility and challenge page:

- RFS compares predicted trajectories against rater-specified trajectories.
- The metric evaluates trust regions at 3 seconds and 5 seconds.
- Lateral base thresholds are `1.0m` at 3s and `1.8m` at 5s.
- Longitudinal thresholds are 4x larger: `4.0m` at 3s and `7.2m` at 5s.
- Thresholds are scaled by initial speed.
- The speed scale is clipped from `0.5` to `1.0`.
- If a prediction is inside a trust region, it receives the corresponding rater trajectory score.
- Outside trust regions, score decays exponentially from the closest rater score.
- The outside-trust-region score is floored at `4.0`.

Important implementation detail:

- The local `rater_feedback_utils.get_rater_feedback_score` accepts inference trajectories shaped `[B, I, T, 2]` and probabilities shaped `[B, I]`.
- The challenge submission allows a single trajectory, but local analysis can score multiple candidates before selecting one.
- Initial speed is computed from the last past-state velocity sample.

Architecture implication:

- Optimize candidate endpoints at 3s and 5s. These are metric-critical.
- Longitudinal error is tolerated more than lateral error because camera depth uncertainty is explicitly accounted for.
- Lateral lane/obstacle decisions matter heavily.
- A stop trajectory can receive a nonzero floor but may still be poor if raters preferred progress.
- RFS rewards matching an acceptable rater future, not necessarily the logged future.

## Scenario Clusters

The metric reports cluster scores for:

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

Cluster-specific implications:

- Construction: expect cones, lane shifts, blocked lanes, workers, and unusual signal context. Candidate library needs slow-progress, merge, and stop options.
- Intersection: expect conflicting agents and route ambiguity. Temporal history and intent are both critical.
- Pedestrian: expect occlusion, erratic motion, and low-speed yielding. Conservative fallback must avoid becoming a default stop everywhere.
- Cyclist: expect lateral clearance and speed-matching. Lateral error at 3s/5s is especially costly.
- Multi-lane maneuver: route command and side cameras matter. Candidate library needs lane-change and abort-lane-change options.
- Single-lane maneuver: progress versus caution dominates. Constant stop may be safe-looking but rater-poor.
- Cut-in: side/front temporal evidence matters. A frame-only policy can miss the setup.
- Foreign object debris: recognition may depend on rare object semantics. This is a priority case for any future visual encoder.
- Special vehicle: unusual actor identity matters, including emergency, service, or oversized vehicles.
- Spotlight: manually selected hard cases should be treated as failure-analysis targets.
- Others: avoid overfitting cluster-specific heuristics; report separately.

## Minimal-Shot Architecture Implications

The dataset pushes the architecture toward this contract:

- **Structured candidate generators:** enumerate plausible 20-point futures from ego history and route intent.
- **Temporal state:** compress the 4-second ego state history and any declared context features.
- **Residual/anchor proposal models:** broaden the candidate set while staying benchmark-gated.
- **Trajectory projector:** emit exactly 20 future vehicle-frame waypoints with smooth dynamics.
- **RFS-aware selector:** choose the single submitted trajectory from candidates using validation-time RFS analysis.

The important claim is not "real-time self-driving." WOD-E2E is open-loop. The credible claim is:

> fast candidate generation plus RFS-calibrated selection can produce rater-plausible long-tail trajectories with a fully declared WOD-E2E training boundary.

## Baselines

Minimum baselines before claiming architecture value:

- constant stop: all zeros
- constant velocity: extrapolate from final past velocity
- curvature extrapolation: fit recent ego path trend
- intent template: straight/left/right kinematic templates
- learned residual candidate model
- source-aware RFS selector

Report for each:

- valid trajectory rate
- ADE at 3s and 5s where future labels exist
- RFS on validation frames with preference labels
- cluster-level RFS where scenario tags are available
- qualitative failure examples

## Known Dataset Risks

- Public page and proto differ on origin wording: vehicle center versus rear axle. Treat proto as implementation source and document any version mismatch.
- The challenge is open-loop. A high RFS score is not closed-loop safety.
- WOD-E2E does not provide dense agent tracks or map labels in `E2EDFrame`; do not design around unavailable annotations.
- Validation rater labels are available, but test rater labels are withheld.
- The test set limits submissions; use validation analysis heavily before leaderboard attempts.

## Acceptance Criteria For This Repo

Minimum credible submission preparation:

- Dataset contract is documented at proto-field level.
- Local tooling can parse validation `E2EDFrame` records from the downloaded
  validation split.
- Local tooling can find valid `preference_trajectories`.
- Official RFS utility or the audited local trust-region implementation is used
  for validation frames with rater labels.
- At least three baselines are evaluated before the architecture claim.
- One success and one failure case are documented with camera evidence, ego history, chosen trajectory, and component-level diagnosis.

Strong submission preparation:

- Cluster-level validation table.
- RFS/ADE comparison showing where ADE misleads.
- Candidate trajectory visualizations at 3s and 5s trust regions.
- Submission proto writer with strict validation.
- Model declaration aligned with submission metadata.
