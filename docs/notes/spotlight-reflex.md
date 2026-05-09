# Spotlight Reflex

## Thesis

**Spotlight Reflex** is a simulation-side long-tail driving policy based on counterfactual maneuver retrieval and simulator-native trajectory selection.

The core observation is that the WOD-E2E Spotlight cluster remains unsolved even by the best fine-tuned methods. In the supplied leaderboard snapshot, top overall methods reach about `8.05` RFS, while Spotlight tops out around `7.2`. This suggests Spotlight cases are not just ordinary imitation problems. They require recognizing unusual hazards, ambiguity, and counterfactual risk.

> Spotlight Reflex targets long-tail simulator scenes directly: scene uncertainty, counterfactual hypotheses, maneuver candidates, and bounded trajectory selection.

## Submission Claim

Spotlight Reflex is designed for the SoTA zero-shot AV challenge:

- No AV-dataset fine-tuning.
- No route memorization.
- No leaderboard-specific supervised trajectory training.
- Uses frozen foundation models and structured search.
- Targets WOD-E2E Spotlight, FOD, construction, and other long-tail cases.

Important competition caveat:

- Training a small selector/verifier on validation `preference_trajectories` is not the same as fine-tuning a driving policy on AV demonstrations, but it must be declared explicitly.
- If the strictest interpretation of "zero-shot" forbids any learning from WOD-E2E validation labels, keep the verifier frozen/hand-coded and report the learned verifier as an ablation.
- The honest target is not guaranteed leaderboard victory. A strong zero-shot result around `7.8+` RFS would already be a serious generalization result.

Short pitch:

> Scene evidence identifies hazards and uncertainty; counterfactual hypotheses generate maneuver candidates; a simulator-native trajectory selector chooses a physically plausible trajectory.

## Architecture

### 1. Frozen Scene Critic

Input:

- 8-camera WOD-E2E montage or per-view image packet.
- Past ego trajectory over `(-4s, 0]`.
- Velocity and acceleration history.
- High-level route intent.

Output:

- visible hazards
- occlusions and blind spots
- rare objects or unusual agents
- route conflicts
- scene uncertainty
- suggested meta-behavior: proceed, slow, stop, yield, nudge, evade, lane-change, fallback

The current repo does not use a language critic in the active model path. Candidate scoring is numeric and audited.

### 2. Counterfactual Scene Hypotheses

The system generates a small set of hypotheses for ambiguous long-tail scenes:

- object is harmless debris
- object is lane-blocking debris
- pedestrian may emerge from occlusion
- vehicle may cut in
- construction cones imply lane shift
- route command conflicts with visible free space
- special vehicle requires extra clearance
- animal/debris may be unpredictable

Each hypothesis produces a different maneuver prior.

### 3. Latent Maneuver Library

The maneuver library emits physically plausible 5-second candidate trajectories:

- constant stop
- crawl forward
- maintain speed
- slow/yield
- nudge left
- nudge right
- return to lane center
- lane change left/right
- evasive offset around obstacle
- cut-in response
- emergency fallback

Every candidate must be exactly 20 `(x, y)` points at 4 Hz over `(0, 5s]`.

### 4. Simulator-Native Selector

The selector is simulator-owned. It ranks deterministic maneuver candidates using bounded 3s/5s trajectory regions and simulator-derived references, without importing WOD/RFS metric code or model outputs.

On validation frames with rater labels, use the official local utility:

```python
from waymo_open_dataset.metrics.python import rater_feedback_utils

rater_feedback_utils.get_rater_feedback_score(
    inference_trajectories,       # [B, I, 20, 2]
    inference_probs,              # [B, I]
    rater_specified_trajectories, # List[List[np.ndarray]]
    rater_feedback_labels,        # List[np.ndarray]
    init_speed,                   # [B]
    frequency=4,
    length_seconds=5,
)
```

At simulator inference time, use references produced by the maneuver library and scene state. The selector ranks candidates by whether they land inside acceptable maneuver regions at 3s and 5s.

Metric-critical indices:

- 3s index: `3 * 4 - 1 = 11`
- 5s index: `5 * 4 - 1 = 19`

Base thresholds:

| Time | Lateral | Longitudinal |
|---|---:|---:|
| 3s | `1.0m` | `4.0m` |
| 5s | `1.8m` | `7.2m` |

Speed scaling:

```python
def speed_scale(speed_mps: float) -> float:
    return min(1.0, max(0.5, 0.5 + 0.5 * (speed_mps - 1.4) / (11.0 - 1.4)))
```

Selection-region score sketch:

```python
def selection_region_score(candidate, reference, reference_score, speed_mps, index):
    if index == 11:
        base_lat, base_lng = 1.0, 4.0
    elif index == 19:
        base_lat, base_lng = 1.8, 7.2
    else:
        raise ValueError("selection regions are defined at 3s and 5s")

    scale = speed_scale(speed_mps)
    lat_thresh = scale * base_lat
    lng_thresh = scale * base_lng

    delta_lng = abs(candidate[index, 0] - reference[index, 0])
    delta_lat = abs(candidate[index, 1] - reference[index, 1])
    normalized = max(delta_lat / lat_thresh, delta_lng / lng_thresh)

    if normalized <= 1.0:
        return reference_score

    overshoot = normalized - 1.0
    return max(reference_score * (0.1 ** overshoot), 4.0)
```

Candidate selection rule:

- Generate candidate trajectories.
- Generate pseudo-rater references from high-confidence maneuver hypotheses.
- Score candidates against all references at 3s and 5s.
- Penalize invalid trajectories, route-command violations, and excessive jerk.
- Pick the candidate with highest combined selector score.

This is the inference-time analogue of Poutine's RFS-aligned GRPO, but without training.

### 5. Optional RFS Verifier

The selector can be upgraded with a small verifier trained only on validation rater preferences:

- Input: candidate trajectory, pseudo-rater references, initial speed, route intent, critic uncertainty, and simple trajectory features.
- Target: validation rater score or pairwise preference induced by `preference_trajectories`.
- Output: calibrated preference score used to rerank candidates after model-side WOD/RFS evaluation.

This verifier is the highest-leverage metric-alignment component. It should be kept small and declared clearly:

- no image backbone fine-tuning
- no AV demonstration imitation
- no training on test data
- trained only on validation preference labels, if allowed by the submission rules

If used, report two variants:

- **Strict zero-shot:** frozen critic + maneuver library + simulator-native selector, no learned verifier.
- **Preference-calibrated:** same system plus validation-trained RFS verifier.

## Why This Is Different From The Winners

DiffusionLTF / Open X-AV:

- Strong because of data curriculum.
- Spotlight Reflex instead uses counterfactual inference-time hypotheses when training data is unavailable.

Poutine:

- Strong because it aligns to RFS with GRPO.
- Spotlight Reflex uses simulator-native selection at inference time rather than learning the preference signal.

UniPlan / RAP:

- Strong because of candidate generation and augmentation.
- Spotlight Reflex uses maneuver-library candidates plus semantic counterfactuals.

Swin-Trajectory:

- Strong because it is compact and trajectory-focused.
- Spotlight Reflex keeps the deployed trajectory generator structured and small.

NTR / later leaderboard leaders:

- Strong leaderboard scores likely reflect iteration, precise trajectory execution, and possibly motion-prediction transfer.
- Spotlight Reflex does not try to beat them by imitation. It tries to exploit the remaining Spotlight semantic gap.

## Expected Outcomes

Realistic score bands:

- **Best case:** `7.8-8.1` RFS if scene reasoning, counterfactual candidates, and verifier selection all work.
- **Partial success:** `7.4-7.8` RFS if candidate generation helps but precise execution lags fine-tuned systems.
- **Weak generalization:** `6.5-7.2` RFS if the frozen model struggles with WOD-E2E camera geometry or coordinate conventions.

Cluster expectations:

- Spotlight: highest upside from reasoning and counterfactuals.
- FOD: high upside from rare-object semantics.
- Construction: limited by precise trajectory execution unless templates are strong.
- Single-lane: limited by progress/caution calibration and rater preference.
- Cut-ins: depends on temporal history and side/front view interpretation.

The central bet:

> A zero-shot system can compete hardest where semantic novelty matters more than demonstration-calibrated control.

## Evaluation Plan

Baselines:

- constant stop
- constant velocity
- curvature extrapolation
- route-intent template
- maneuver library without VLM
- VLM critic without counterfactuals
- full system without learned verifier
- full system with learned RFS verifier, if allowed
- full Spotlight Reflex

Primary analysis:

- validation RFS on frames with `preference_trajectories`
- ADE at 3s and 5s
- cluster-level RFS
- Spotlight-specific failure analysis
- invalid trajectory rate
- route-command violation rate

Decision criteria:

- If a validation-only strict zero-shot ablation reaches `7.4+`, it is not a
  hidden-test claim, but it is worth presenting as a generalization result.
- If preference-calibrated reaches `7.8+`, it is a serious leaderboard-contending result.
- If Spotlight improves materially while Construction/Single-lane lag, the story is still strong: the method attacks the unsolved semantic cluster.

Qualitative evidence:

- one Spotlight success
- one Spotlight failure
- one FOD/debris success
- one construction or cut-in case

## Final Pitch

> Spotlight Reflex is a zero-shot policy for WOD-E2E long-tail driving. It uses frozen multimodal scene understanding to generate counterfactual hazard hypotheses, retrieves physically plausible maneuver candidates, and selects trajectories with the simulator-native trajectory selector criterion. It does not attempt to out-scale fine-tuned planners. It targets the unsolved Spotlight gap directly.
