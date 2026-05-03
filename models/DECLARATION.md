# Base Model and Architecture Declaration

This declaration describes the current WOD-E2E model-track submission path in
this repository. It intentionally excludes simulator components.

## Submission Proto Metadata

- `uses_public_model_pretraining`: `false`
- `public_model_names`: empty
- `num_model_parameters`: `under 1M`
- `unique_method_name`: `ridge_non_text_rfs_selector`
- `description`: Fast non-text WOD-E2E trajectory baseline using ego history,
  route intent, residual trajectory candidates, experimental lightweight
  world-model candidates, and a validation-calibrated structured selector.
- `method_link`: project repository link, to be filled at packaging time.

## Base Models

No frozen VLM, LLM, prompt model, or hosted model is used in the active
submission path.

### Scene Critic

- Model name: none
- Provider: none
- Input modalities used: none
- Usage: not used
- Fine-tuned on AV-specific data: no
- Scaffolded with WOD-E2E examples: no

### Visual Encoder

- Model name: none in the active path
- Provider: none
- Usage: not used for the current submitter
- Fine-tuned on AV-specific data: no

### Trajectory Model

- Implementation: `minimal_shot_av.model.learned_trajectory_model.RidgeTrajectoryModel`
- Source: local NumPy ridge regression
- Input: past ego trajectory, initial speed, WOD-E2E high-level route intent
- Output: one 20-waypoint mean trajectory plus residual-mode candidates
- Training source: WOD-E2E frames with `future_states`, when a final train split
  model is produced
- AV-specific training: yes for the leaderboard variant if trained on WOD-E2E
  train labels; no public foundation model is used

### Selector

- Implementation: structured ridge ranker over candidate trajectory features
- Input: candidate kinematic features and candidate identity/family
- Target: validation RFS-derived candidate score during development
- Validation labels: WOD-E2E validation `preference_trajectories`
- Test access: test frames are used only for final prediction generation

### Lightweight World Model

- Implementation: `minimal_shot_av.model.world_model.LearnedWorldModel`
- Source: local NumPy latent model and ridge heads
- Input: ego temporal features by default; scene-token mode is implemented but
  not yet part of the active submitter
- Output: an imagined 20-waypoint trajectory candidate and optional memory
  neighbor candidates
- Current evidence: the best confirmed official validation-CV run improves
  selected RFS from `7.602814916652251` to `7.606198495114426`, which is too
  small to claim a meaningful solution
- Status: experimental candidate source, not a demonstrated scene-understanding
  model

## Datasets

### WOD-E2E

- Dataset version: `waymo_open_dataset_end_to_end_camera_v_1_0_0`
- Local status at audit time: validation split present locally; train/test and
  challenge frame-list JSON still required for final leaderboard packaging
- Training split purpose: fit final ridge trajectory model from `future_states`
- Validation split purpose: official RFS evaluation, ablations, selector
  calibration, and failure analysis
- Test split purpose: final prediction only; no future states or rater labels are
  available or used
- Scenario tags used: only for reporting where available
- Rater feedback labels used: yes, validation only
- Was any training performed on WOD-E2E: yes for the learned trajectory baseline
  and validation-calibrated selector; this must not be described as strict
  zero-shot

Required fields:

- `E2EDFrame.frame.context.name` used as submission key: yes
- 8-camera images used: no in the current active path
- `past_states` used: yes
- `intent` used: yes
- `future_states` used for training: yes for final train-split model
- `preference_trajectories` used for validation RFS: yes

### Other Datasets

No other dataset is used in the active WOD-E2E model path.

## External Services

No external API or hosted model is used in the active WOD-E2E model path.
WOD-E2E images and labels are not sent to third-party services.

## Architecture Summary

- WOD-E2E adapter: official TFRecord/proto parser loading `past_states`,
  `future_states` when available, route `intent`, frame names, and optional
  camera JPEGs.
- Trajectory decoder: ridge regression predicts a 20 waypoint, 4 Hz, 5 second
  trajectory in the WOD-E2E vehicle coordinate frame.
- Candidate generator: residual principal components around the base ridge mean
  plus auxiliary temporal-summary ridge candidates produce a small candidate set.
- Rater-aware selector: structured ridge ranker chooses one candidate per frame.
- Submission packaging: generated candidates are scored with the contextual
  ranker and packaged with `ranker_score` selection, rather than relying on
  candidate emission order.
- Submission writer: packages one `TrajectoryPrediction` per required frame into
  `E2EDChallengeSubmission` `.tar.gz`.

Dataset contract:

- Input cameras: not used in current active path
- Ego history window: latest 16 past waypoints, with temporal-summary motion
  features used only for auxiliary proposal generation
- Route intent handling: one-hot encoded WOD-E2E high-level command
- Output coordinate frame: WOD-E2E vehicle coordinates
- Output horizon: 20 waypoints, 4 Hz, 5 seconds
- Number of submitted trajectories per frame: one
- Rear-axle vs vehicle-center origin wording: handled by preserving WOD-E2E proto
  coordinates without extra origin transforms

Metric contract:

- RFS evaluated only on validation frames with valid rater labels
- Current promoted full-validation official-RFS benchmark report:
  `artifacts/wod_fastkin_gate_ridge175_scene020_cv_official.json`
- Current promoted full-validation CV score: `7.65941846208851` RFS on
  `479` preference-labeled validation frames
- Current promoted full-validation candidate oracle:
  `9.098014272661512` official RFS
- Neural anchor-residual proposal ensembles are tracked as secondary WOD
  development evidence. The best official held-out subset run reaches
  `7.737680847131364` RFS on `159` frames with oracle `9.178972912996967`;
  this is not a full-validation, hidden-test, or strict zero-shot result.
- Earlier ridge-only official-RFS benchmark report:
  `benchmarks/current/wod_ridge_trajectory_cv_official.json`
- Earlier ridge-only validation CV score: `7.602814916652251` RFS
- Current local-RFS candidate report:
  `benchmarks/current/wod_contextual_r175_speed_router_local_cv.json`
- Current local-RFS candidate score: `7.695139906515457` RFS with
  speed-routed train-margin fallback
- Current local-RFS baseline comparison:
  `benchmarks/current/comparisons/wod_contextual_r175_speed_router_vs_r100_local_cv.json`
- Current experimental world-model validation CV score:
  `7.606198495114426` official RFS in
  `artifacts/world_sweeps/wod_cv_479_world_ego_l8_m0_official.json`
- Current world-model oracle headroom:
  `9.046125057713054` official RFS, versus the previous `9.015857402777051`
- Current online runtime report: `benchmarks/current/wod_online_runtime.json`
- Current online runtime p95: `1.40214185 ms` for base plus temporal
  candidates with contextual selector features on the audited CPU environment,
  excluding TFRecord parsing, image decoding, official RFS scoring, and offline
  training
- Current bias audit: `benchmarks/current/wod_model_bias_audit.json`, status
  `warn`
- Current local-best bias audit:
  `benchmarks/current/wod_model_bias_audit_local_best.json`, status `warn`
- Current architecture audit: `docs/wod-architecture-audit.md`
- This is not a leaderboard/test score
- Simulator metrics are not used for model selection

## Safety and Uncertainty

- Uncertainty is represented by residual trajectory candidates around the ridge
  mean and by disagreement between base and temporal-summary proposal sources.
- Invalid trajectories are rejected by the WOD 20-waypoint validator before
  packaging.
- The current model is open-loop and does not enforce closed-loop collision,
  traffic-rule, or route feasibility guarantees.

## Generalization Claim

This system is a fast, structured, non-text WOD-E2E baseline and measurement
harness. Its claim is not strict zero-shot autonomy and not Alpamayo-class
learned visual reasoning. The lightweight world-model result is a small signal,
not a solution, and the neural held-out subset result is secondary evidence
rather than the promoted WOD claim. A meaningful leaderboard solution claim
requires hidden-test evidence, and a meaningful validation solution claim
requires at least a `+0.1` official RFS gain on the same full 479-frame
segment-grouped validation contract plus reduced worst-slice regret.

## Known Limitations

- Current active path does not use camera images, so it cannot reason about
  pedestrians, debris, signals, occlusions, or cross-traffic from pixels.
- Scene-token world-model mode is currently uncached and too slow for practical
  full validation sweeps.
- Memory-neighbor world candidates increased oracle headroom in local sweeps but
  degraded selected RFS because the current selector over-picked weak memory
  candidates.
- WOD-E2E validation RFS is an internal development estimate, not leaderboard
  performance.
- Validation preference labels must not be represented as test labels.
- The selector can overfit validation preferences if used without grouped
  cross-validation and a clear declaration.
- Current bias audit flags validation-preference bias, visual-blindness bias,
  selector oracle gap, and slice-level regret bias. The worst current slice is
  `intent:3`, where selected RFS is `6.3061590342335325` against oracle
  `8.030133377574096` with mean regret `1.7239743433405623`. These must be
  reduced before making a stronger generalization claim.
- The system is not a closed-loop AV controller.
