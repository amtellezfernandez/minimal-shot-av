# 2025 WOD-E2E Winning Methods

This document tracks official Waymo technical reports for the 2025 Vision-based End-to-End Driving Challenge winners and special mention.

## Official Winner Links

| Recognition | Method | Authors | Affiliation | Technical report |
|---|---|---|---|---|
| First Place | UniPlan | Lan Feng, Alexandre Alahi | EPFL | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%201st%20Place%20-%20UniPlan.pdf) |
| Second Place | DiffusionLTF | Long Nguyen, Micha Fauth, Bernhard Jaeger, Daniel Dauner, Maximilian Igl, Andreas Geiger, Kashyap Chitta | University of Tubingen, Tubingen AI Center, NVIDIA Research | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%202nd%20Place%20-%20DiffusionLTF.pdf) |
| Third Place | Swin-Trajectory | Sungjin Park, Gwangik Shin, Jaeha Song, Sumin Lee, Hyukju Shon, Byounggun Park, Jinhee Na, Hawook Jeong, Soonmin Hwang | Hanyang University, RideFlux Inc. | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%203rd%20Place%20-%20Swin-Trajectory.pdf) |
| Special Mention | Poutine | Luke Rowe, Rodrigue de Schaetzen, Roger Girgis, Christopher Pal, Liam Paull | Mila, Universite de Montreal, Polytechnique Montreal, CIFAR AI Chair | [PDF](https://storage.googleapis.com/waymo-uploads/files/research/2025%20Technical%20Reports/2025%20WOD%20E2E%20Driving%20Challenge%20-%20Special%20Mention%20-%20Poutine.pdf) |

Official challenge overview source: <https://waymo.com/intl/es/open/challenges/>

Important distinction:

- These are official prize/report results.
- They are not the same as the later leaderboard ordering captured in `docs/leaderboard.md`.
- Poutine appears as special mention rather than official prize placement because it was not prize-eligible under the competition's Quebec-residency exclusion rule.
- Later leaderboard entries such as RAP, TTVLM, and NTR should be analyzed separately from official prize placement.

## Method Notes

### UniPlan

UniPlan is an EPFL system built around a DiffusionDrive-style anchored diffusion planner. It samples trajectory anchors, applies a small number of denoising steps, and selects from candidate futures using confidence scores. The report describes training with WOD-E2E plus nuPlan, front-three-camera concatenation, WOD-E2E-specific planner anchors, four seeded models, and 80 total inference candidates.

Implementation signals:

- Diffusion decoder with anchored Gaussian trajectory proposals.
- K-means anchors adapted to WOD-E2E 5-second prediction.
- Front-left/front/front-right camera concatenation.
- Ensemble/candidate selection materially improves RFS.
- Joint training with nuPlan improves some long-tail categories but not uniformly.

### DiffusionLTF

DiffusionLTF is the second-place method from the Open X-AV report. It argues that fragmented AV datasets limit generalization and proposes a multi-dataset workflow. The reported OXAV setup combines CARLA, NAVSIM, WOD-P, and WOD-E2E. DiffusionLTF extends a Latent TransFuser-style baseline with diffusion trajectory generation and proposal ensembling.

Implementation signals:

- Small-model lesson: the project synthesis identifies a ResNet34-scale backbone and roughly one day of A100 training, making data strategy the main differentiator rather than model scale.
- Uses three front-facing RGB cameras and vehicle status.
- Replaces LiDAR input with learnable/constant latent structure for vision-only use.
- Adds past speeds and past positions as status tokens.
- Uses a discrete trajectory-pattern vocabulary and denoising proposals.
- Finds WOD-P pretraining especially useful, while WOD-E2E-only post-training remains important.

### Swin-Trajectory

Swin-Trajectory is the third-place method from Hanyang University and RideFlux. It is intentionally minimalist: a single front-facing camera, ego-vehicle status, a Swin Transformer visual backbone, and structured waypoint queries. The report emphasizes deployment efficiency and reports 14 ms inference on an RTX 4090.

Implementation signals:

- Single front camera rather than all 8 cameras.
- Swin Transformer feature extractor.
- Ego-info encoder uses history, velocity, acceleration, yaw-rate, speed, curvature, vehicle size, and intent.
- Trajectory decoder uses cross-attention between image features and waypoint queries.
- Focuses on avoiding shortcut learning while keeping the model lightweight.

### Poutine

Poutine is the special-mention method and leaderboard-top system in the supplied leaderboard snapshot. It is a 3B-parameter VLM trained with vision-language-trajectory next-token prediction, then preference-tuned with GRPO on fewer than 500 WOD-E2E validation preference-labeled frames. The report also describes Poutine-Base as the pre-RL variant.

Implementation signals:

- Uses Qwen2.5-VL 3B Instruct as the driving VLM.
- Auto-generates language annotations with a 72B VLM.
- Pretrains on CoVLA nominal driving plus WOD-E2E long-tail driving.
- Predicts sparse 1 Hz waypoints and upsamples to 4 Hz with cubic splines.
- Uses RFS as a reward during GRPO post-training.
- Structured prompts audit critical objects, explain the trajectory, select meta-behavior, then predict waypoints.

## Architecture Lessons For This Repo

- The winning set is not purely VLM-based. Strong methods include diffusion planners, dataset-mixture training, trajectory refinement, and lightweight transformer predictors.
- Front-camera-only systems can be competitive, but this is a leaderboard engineering choice, not proof that side/rear cameras are irrelevant.
- RFS-oriented candidate selection matters. UniPlan and DiffusionLTF both benefit from multiple trajectory proposals.
- Poutine shows the clearest VLM route: auto-labeled language plus trajectory tokens, then small-scale preference optimization.
- Swin-Trajectory is the strongest counterexample to overcomplication: a compact architecture can rank highly if the trajectory interface is engineered well.
- For a minimal-shot proposal, the highest-value synthesis is probably: frozen/cheap semantic critic, maneuver candidates, RFS-aware proposal selection, and a lightweight temporal/trajectory decoder.
