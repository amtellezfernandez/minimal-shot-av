# Base Model and Architecture Declaration

Complete this before submission. The declaration should be specific enough that a judge can tell exactly which intelligence came from frozen base models, which parts were hand-designed, and whether any AV-specific fine-tuning occurred.

## Base Models

Submission proto alignment:

- `uses_public_model_pretraining`:
- `public_model_names`:
- `num_model_parameters`:
- `unique_method_name`:
- `description`:
- `method_link`:

These fields must match the actual `E2EDChallengeSubmission` metadata if a leaderboard file is produced.

### Scene Critic

- Model name:
- Provider:
- Version/date:
- Input modalities used:
- Usage:
- Was it fine-tuned on AV-specific data for this project? `No`
- Was it prompted/scaffolded with WOD-E2E examples?
- Prompt files or notebook cells:

### Visual Encoder

- Model name:
- Provider:
- Version/date:
- Usage:
- Was it fine-tuned on AV-specific data for this project? `No`

### Temporal / SSM Component

- Model name or implementation:
- Provider/source:
- Version/date:
- Training source:
- Was it trained or fine-tuned on AV-specific data for this project?
- If not trained, describe initialization and use:

## Datasets

### WOD-E2E

- Dataset version: `v1.0.0` unless a newer official release is explicitly used.
- Split used:
- Purpose:
- Scenario tags used?
- Rater feedback labels used?
- Was any training performed on WOD-E2E?
- Were validation rater labels used only for analysis/calibration, or for model fitting?
- Were test frames used only for final prediction?
- Local source path or GCS path:

Required declaration:

- `E2EDFrame.frame.context.name` used as submission key? `Yes/No`
- 8-camera images used? `Yes/No`
- `past_states` used? `Yes/No`
- `intent` used? `Yes/No`
- `future_states` used for training? `Yes/No`
- `preference_trajectories` used for validation RFS? `Yes/No`

### Other Datasets

- Dataset:
- Split used:
- Purpose:
- AV-specific? `Yes/No`
- If AV-specific, explain why this does not violate the submission constraint:

## External Services

- API or hosted service:
- Provider:
- Purpose:
- Data sent:
- Were WOD-E2E images or labels sent to the service?
- Retention/privacy setting, if applicable:

## Architecture Summary

Describe the end-to-end system in plain language:

- WOD-E2E adapter:
- Frozen scene critic:
- Predictive state-space pilot:
- Latent maneuver library:
- Trajectory decoder:
- Safety projector:
- Rater-aware selector:

Dataset contract:

- Input cameras:
- Camera montage or per-view encoding:
- Ego history window:
- Route intent handling:
- Output coordinate frame:
- Output horizon:
- Number of submitted trajectories per frame:
- Handling of the rear-axle versus vehicle-center origin wording:

Metric contract:

- RFS evaluated on validation frames with valid rater labels?
- ADE at 3s and 5s reported separately?
- Cluster-level scores reported?
- Constant-stop, constant-velocity, and curvature baselines included?

## Safety and Uncertainty

- How does the system estimate uncertainty?
- What happens when confidence is low?
- What hard constraints exist outside the learned components?
- How are invalid, non-smooth, or route-inconsistent trajectories rejected?

## Generalization Claim

State the exact claim the submission asks the judges to believe.

Recommended form:

> This system uses frozen general-purpose visual reasoning and structured maneuver generation to predict WOD-E2E long-tail trajectories without AV-specific model fine-tuning. Its generalization claim is minimal-shot scenario transfer, not route memorization or closed-loop deployment readiness.

## Known Limitations

- WOD-E2E is open-loop; the system is not validated as a closed-loop AV controller.
- RFS measures rater preference over 5-second trajectory candidates, not full safety certification.
- Frozen models may hallucinate scene semantics, especially under occlusion or low visibility.
- The maneuver library may be too coarse for rare cases that require multi-agent negotiation.
