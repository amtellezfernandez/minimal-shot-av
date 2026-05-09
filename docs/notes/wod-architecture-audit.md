# WOD Architecture Audit

This audit covers the active WOD-E2E model path as of the local-best
`r175` speed-routed fallback run.

## Current Position

The active path is a fast structured baseline, not a finished scene-understanding
architecture. It uses ego history, route intent, ridge trajectory regression,
residual candidates, and an RFS-calibrated structured selector. This is useful
as a measurement harness and transparent baseline, but it should not be presented
as strict zero-shot autonomy or as a demonstrated Cosmos-backed world model.

Current local-best evidence:

- Local CV report: `benchmarks/current/wod_contextual_r175_speed_router_local_cv.json`
- Local-best bias audit: `benchmarks/current/wod_model_bias_audit_local_best.json`
- 25% target feasibility audit:
  `artifacts/wod_contextual_r175_speed_router_25pct_target_audit.json`
- Cosmos Predict2.5 diagnostics:
  `artifacts/wod_cosmos_predict25_contextual_r100_frame_delta_kinfallback_diagnostics.json`
- Gated Cosmos scene-understanding loop:
  `artifacts/wod_cosmos25_alllatent_contextual_r200_world_mem3_gate2_cv_local.json`
- Best supervised Cosmos-scene adapter loop:
  `artifacts/wod_cosmos25_scene_adapter_ridge100_pairwise_residuals_intent_localfallback_cv_local.json`
- Best non-linear selector control:
  `artifacts/wod_cosmos25_scene_adapter_ridge100_pairwise_residuals_intent_rff64_ridge50_cv_local.json`
- Best baseline plus scene-auxiliary control:
  `artifacts/wod_base_temporal_sceneaux_r175_originalfallback_cv_local.json`
- Best separated-scene-source control:
  `artifacts/wod_base_temporal_sceneaux_sepscene_r175_originalfallback_cv_local.json`
- Best listwise selector control:
  `artifacts/wod_contextual_r175_speed_listwise_temp1_nofallback_cv_local.json`
- Mean selected RFS: `7.695139906515457`
- Regret to oracle: `1.3731281054166127`
- Worst slice: `intent:3`, regret `1.7239743433405623`

## Good Architecture Choices

- Simulator and WOD/RFS model code are separated by package and test boundaries.
- Segment-grouped cross-validation is used for model-side validation.
- Submission, runtime, benchmark, and audit artifacts are explicit and reproducible.
- The candidate-source design makes failure attribution possible: kinematic,
  learned ridge, temporal-summary, anchor, and world candidates can be compared.
- The current claim boundary is honest: validation-calibrated and not a
  leaderboard/test score.

## Architecture Risks

- The active model does not consume camera semantics. It cannot react to visual
  hazards such as pedestrians, debris, signals, occlusions, or side/rear context.
- Cosmos and scene-token paths exist as feature/cache infrastructure, but they
  are not yet proven to improve selected official-RFS validation CV.
- Cosmos Predict2.5 tokenizer embeddings show weak local oracle signal but no
  selector uptake in the audited run: selected world rate `0.0`, oracle world
  rate `0.020876826722338204`, selected-RFS gain only `+0.0010438413361164578`,
  and worse worst-slice regret than the local baseline.
- The explicit scene-latent world-model path (`--world-latent-source scene`) and
  gated Cosmos-memory candidates still did not beat the non-scene local best.
  The best gated Cosmos-Predict2.5 run from this loop reached mean local RFS
  `7.674024745163919`, below `7.695139906515457`.
- The supervised Cosmos-scene trajectory adapter now produces a stronger
  candidate universe after enabling pairwise and context residual expansions for
  external-embedding models. The best expanded run reached combined oracle RFS
  `9.21447061155328`, above the old oracle `9.06826801193207`, but the best
  selected score from this loop was only `7.580711299550799`, still below the
  local best `7.695139906515457`.
- The selector is the main bottleneck. Candidate oracle remains much higher than
  selected RFS, so candidate generation has signal that the current ranker cannot
  reliably exploit.
- A random-Fourier ridge selector improved the expanded scene branch from
  `7.580711299550799` to `7.636412788384292`, but it did not beat the current
  non-scene selected baseline.
- Adding the scene-conditioned adapter as an auxiliary learned candidate family
  on top of the current base-plus-temporal candidate universe raised oracle to
  `9.26984729439014`; the best selected score from that control was
  `7.6621797795960385`, still below `7.695139906515457`.
- Splitting scene-conditioned candidates into their own `scene` selector source
  improved the best scene-auxiliary selected score to `7.676080466297595`.
  The scene source is oracle on `0.07515657620041753` of frames in that run, but
  the conservative selector chooses it only `0.006263048016701462` of the time;
  when fallback is allowed to choose scene more often, it over-selects scene and
  drops selected RFS.
- The first same-frame listwise softmax selector did not help. The best listwise
  control reached `7.570242150563104`, so the current implementation is useful
  as an experimental path but should not replace the speed-routed ridge selector.
- A 25% selected-RFS improvement over the current local baseline is not feasible
  with the current selected model family: it requires `9.5823247888145` RFS.
  The expanded scene-adapter oracle is high enough to justify a stronger router,
  but the current linear/fallback selector does not convert that oracle into
  selected RFS.
- Speed/intent fallback improves local metrics but is still validation-calibrated
  policy logic, not generalizable scene reasoning.
- Worst-slice regret remains too high for a stronger generalization claim.

## Recommended Next Changes

1. Promote official-RFS confirmation for the current local-best speed-router
   candidate before using it as headline evidence.
2. Keep the scene-conditioned adapter path because it now changes proposed
   trajectories and increases oracle; do not promote it until selected RFS beats
   the baseline.
3. Replace the ridge selector with a stronger listwise selector that is trained
   to choose among whole same-frame candidate sets. Random Fourier ridge helps
   but is not sufficient.
4. Extend world-candidate diagnostics down to per-frame nearest-neighbor
   distance and selected/oracle candidate family so the next selector can learn
   where Cosmos candidates are trustworthy.
5. Keep simulator evidence out of model selection and keep validation preference
   usage declared.

## Acceptance Bar

Do not call the architecture a meaningful solution until it beats the current
official selected-RFS baseline by at least `+0.1` RFS on the same 479-frame
segment-grouped validation CV contract and reduces worst-slice regret.

A 25% improvement target requires a new scene-conditioned candidate generator,
not another selector or fallback sweep over the current candidate set.
