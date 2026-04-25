# WOD-E2E Competitive Analysis

This document separates official prize results from later/live leaderboard results and summarizes what appears to matter competitively on WOD-E2E.

## Official Prize Results Versus Leaderboard

The official 2025 Waymo technical reports and prize results are not identical to the later leaderboard ordering captured in `docs/leaderboard.md`.

Official prize/report results:

- **1st Place:** UniPlan, Lan Feng and Alexandre Alahi, EPFL.
- **2nd Place:** DiffusionLTF, reported under the broader **Open X-AV** work by Long Nguyen, Micha Fauth, Bernhard Jaeger, Daniel Dauner, Maximilian Igl, Andreas Geiger, and Kashyap Chitta from University of Tubingen / Tubingen AI Center / NVIDIA Research.
- **3rd Place:** Swin-Trajectory, Hanyang University and RideFlux.
- **Special Mention:** Poutine, Mila / Universite de Montreal / Polytechnique Montreal / CIFAR AI Chair.

Important nuance:

- Poutine and Poutine-Base rank extremely high in the supplied leaderboard snapshot.
- Poutine was not prize-eligible because of a Quebec-residency exclusion clause in the competition rules, so Waymo lists it as a special mention rather than an official prize placement.
- Later leaderboard entries such as RAP, TTVLM, and NTR may outrank the official prize winners, but they should not be confused with the official 2025 prize results.

## Top Method Families

| Method family | Core pattern | Main advantage | Notes |
|---|---|---|---|
| Poutine | 3B Qwen2.5-VL driving VLM with VLT pretraining and GRPO | Direct RFS preference alignment | Uses CoVLA + WOD-E2E, 72B VLM auto-labels, and less than 500 preference-labeled validation frames for GRPO. |
| DiffusionLTF / Open X-AV | Small diffusion/Latent TransFuser-style planner with data curriculum | Dataset diversity | Uses CARLA, NAVSIM/OpenScene, WOD-P, and WOD-E2E. Key result: strong performance from data strategy, not model scale. |
| UniPlan | DiffusionDrive-style anchored diffusion planner | Candidate generation and data mixture | Uses WOD-E2E + nuPlan, front-camera concatenation, WOD-specific anchors, multi-seed candidate selection. |
| RAP | Rasterization Augmented Planning | Counterfactual data augmentation | Extends UniPlan/RAP lineage with lightweight 3D rasterization, recovery perturbations, cross-agent views, and raster-to-real feature alignment. |
| Swin-Trajectory | Lightweight Swin Transformer waypoint predictor | Simplicity and efficiency | Uses structured ego state and waypoint queries; reports 14 ms on RTX 4090. |
| HMVLM / AutoVLA / dVLM-AD | VLM/VLA reasoning and trajectory generation | Semantic long-tail reasoning | CoT, action tokenization, diffusion language modeling, or structured prompting. |
| NTR | Not publicly reported | Unknown, likely motion-prediction transfer | Authors Jiahui Li and Jiawei Sun are associated with Prof. Marcelo H. Ang Jr.'s NUS group and prior Waymo Motion Prediction work. |

## DiffusionLTF / Open X-AV Finding

The most important lesson from the 2nd-place report is that DiffusionLTF did not win by using a massive foundation model.

Reported pattern:

- Backbone: small ResNet-style visual backbone, specifically reported as ResNet34 in the project synthesis.
- Training budget: approximately one day on A100-scale hardware, per project synthesis.
- Method: small diffusion/Latent TransFuser-style planner.
- Data strategy: pretrain broadly, then post-train on WOD-E2E.

Open X-AV data mixture:

- **CARLA:** synthetic, perception-rich driving scenes.
- **NAVSIM/OpenScene:** real challenging filtered driving data.
- **WOD-P:** Waymo Perception data, valuable because camera calibration/domain matches WOD-E2E.
- **WOD-E2E:** long-tail planning target domain.

Implication:

> The competitive edge was mostly curriculum and domain coverage, not architecture scale.

For this repo, that matters because the strongest minimal-shot route may be a compact model with an aggressively designed data/augmentation curriculum rather than a giant VLM-only system.

## NTR Hypothesis

NTR is not currently backed by a public method paper in the sources found so far.

What is known from supplied metadata and public affiliation search:

- Authors include Jiahui Li and Jiawei Sun.
- They appear connected to Prof. Marcelo H. Ang Jr.'s group at the National University of Singapore.
- The group has prior Waymo Motion Prediction challenge success, including RMP-YOLO.

Working hypothesis:

> NTR likely transfers motion-prediction expertise, such as MTR-style scene tokenization or trajectory refinement, into WOD-E2E trajectory prediction.

This should remain marked as a hypothesis until a public report, paper, repo, or author page confirms the architecture.

## Pattern Across Top Methods

| Factor | Poutine | DiffusionLTF | UniPlan / RAP | NTR |
|---|---|---|---|---|
| Architecture novelty | Low-medium | Low | Low-medium | Unknown |
| Data diversity | High | Very high | High | Unknown |
| Direct RFS alignment | Yes, GRPO/RFS | No direct public evidence | No direct public evidence | Unknown |
| Candidate generation | Text/trajectory tokens | Diffusion proposals | Diffusion/raster-augmented proposals | Unknown |
| Key differentiator | Preference tuning | Data curriculum | Candidate/data augmentation | Unknown |

Overall read:

- The leaderboard ceiling around `~8.05` RFS is being approached by multiple families.
- There is no evidence yet that one fundamentally new backbone has broken the task.
- Gains are coming from data coverage, candidate generation, augmentation, and RFS-aligned selection.
- RAP is the clearest genuine augmentation idea: counterfactual/recovery data without photorealistic rendering.

## Spotlight Gap

The supplied leaderboard shows Spotlight remains the hardest cluster:

- Top Spotlight scores are only around `7.1-7.2`.
- Overall top methods reach `~8.05`.
- Construction and Single-lane can exceed `8.6`.

Interpretation:

- Spotlight contains manually selected hard cases where ordinary imitation, standard diffusion proposals, and prompt-based VLM reasoning still fail.
- This is the highest-value target for a novel submission.
- A project that improves Spotlight without sacrificing the other clusters would be more interesting than another small overall leaderboard bump.

## Implications For This Repo

The proposal should be sharpened around **Spotlight Reflex**, described in `docs/spotlight-reflex.md`:

- Do not pitch novelty as "Mamba backbone" alone.
- Pitch it as **Spotlight-targeted long-tail memory and candidate retrieval**.
- Treat the SSM as a compact temporal compression layer over 12 seconds of camera/ego history.
- Treat the latent maneuver library as a candidate generator/reranker, not a direct action head.
- Add exact RFS trust-region selection at 3s and 5s, using the known lateral/longitudinal thresholds and speed scaling.
- Treat a validation-trained RFS verifier as an optional preference-calibrated variant, not as the strictest zero-shot claim.
- Consider data strategy as first-class: WOD-P domain matching, synthetic counterfactuals, and non-AV public driving data may matter more than architecture size.

Expected outcome framing:

- `7.8-8.1` RFS would be leaderboard-contending if the reasoning pipeline and verifier both work.
- `7.4-7.8` RFS is still a strong zero-shot generalization result.
- `6.5-7.2` RFS indicates the frozen model or candidate library is not handling Waymo geometry and cluster-specific control well enough.

The defensible thesis:

> Top WOD-E2E methods are converging through data diversity, candidate generation, and preference alignment. The remaining unsolved cluster is Spotlight. A credible novel submission should target Spotlight with temporal memory, semantic uncertainty, counterfactual candidate generation, and RFS-aware trajectory selection.
