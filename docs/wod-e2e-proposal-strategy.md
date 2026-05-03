# WOD-E2E Proposal Strategy

The current repo has a useful baseline: a ridge trajectory model, residual
candidates, and a structured RFS selector. That is not how the strongest
WOD-E2E systems get most of their advantage.

## What The Leading Methods Actually Do

### UniPlan

UniPlan uses a DiffusionDrive-style anchored diffusion planner. The important
part is not a separate post-hoc selector over arbitrary candidates. It trains a
planner to generate plausible trajectory proposals and a confidence head to
score them. At inference, four seeded models each generate 20 trajectories; the
system selects the highest-confidence trajectory among 80 candidates.

Implication for this repo:

- Candidate generation must be learned.
- Candidate confidence must be trained with the generator, not only bolted on
  afterward.
- WOD-E2E-specific anchors matter.

### DiffusionLTF

DiffusionLTF also follows the diffusion-proposal pattern, but its main
differentiator is data curriculum: CARLA, NAVSIM/OpenScene, WOD-Perception, and
WOD-E2E. It treats proposal diversity and domain coverage as first-class.

Implication for this repo:

- A compact model can be competitive if the data curriculum is strong.
- WOD-Perception-style pretraining is probably more valuable than a larger
  language model for a fast WOD-E2E submitter.

### Swin-Trajectory

Swin-Trajectory is the cleanest fast architecture: visual backbone, structured
ego encoder, waypoint queries, and a trajectory decoder. It does not rely on a
large candidate pool. The trajectory interface is learned end to end.

Implication for this repo:

- A fast visual trajectory decoder is a credible route.
- We should add waypoint queries and cross-attention over camera features before
  adding more ranker tricks.

### Poutine

Poutine wins on preference alignment: vision-language-trajectory pretraining,
then GRPO using RFS from the validation preference labels. The key transfer is
not text itself; it is metric-aligned policy optimization.

Implication for this repo:

- Use RFS labels as a training signal, not only as an evaluator.
- For our no-text vehicle path, the equivalent is a numeric trajectory policy
  optimized with a differentiable or sampled RFS surrogate.

## What We Should Change

The next architecture should be:

```text
8/3 camera encoder + ego/status encoder
    -> waypoint-query trajectory decoder
    -> K learned proposal heads or diffusion proposal decoder
    -> confidence/RFS head trained on validation preference labels
    -> submit highest expected-RFS trajectory
```

This is different from the current baseline:

```text
ridge mean + residual candidates
    -> external structured ranker
    -> best scored candidate
```

## Near-Term Implementation Plan

1. Keep the current ridge stack as the baseline and submission fallback.
2. Add WOD-E2E anchor extraction with K-means over train future trajectories.
3. Train an anchor-residual proposal model:
   - input: past trajectory, speed, acceleration, intent, optional camera tokens
   - output: anchor logits, residual offsets, confidence score
4. Train confidence with RFS-derived targets on validation folds.
5. Compare:
   - ridge mean
   - ridge residual oracle
   - structured selector
   - learned anchor confidence
   - learned anchor oracle

## Current Anchor Experiment

Implemented:

- `src/minimal_shot_av/model/anchor_trajectory_model.py`
- `scripts/train_wod_anchor_trajectory_model.py`
- `scripts/generate_wod_anchor_candidates.py`
- `scripts/evaluate_wod_anchor_model_cv.py`

Initial local-RFS smoke on 100 validation preference frames:

```text
anchors=16
top_k=8
folds=5
anchor_oracle_mean_rfs=8.3362
distance_confidence_selected_mean_rfs=5.5153
rfs_head_selected_mean_rfs=6.2038
```

Interpretation:

- The learned RFS head is better than raw nearest-anchor confidence.
- The anchor proposal oracle has useful headroom.
- The current anchor generator/selector is not yet competitive with the ridge
  CV baseline (`7.6028` official RFS on all 479 validation preference frames).
- Next improvement should target the generator, not just the selector: richer
  status features, camera tokens, WOD-specific anchors from train, and possibly
  diffusion-style residual refinement.

Follow-up local-RFS experiments on the same 100-frame smoke:

```text
anchor router, anchors=8, top_k=8:
  rfs_head_selected_mean_rfs=6.3031

anchor router + per-anchor PCA residuals, anchors=8, top_k=8, pc=2:
  anchor_oracle_mean_rfs=8.6682
  rfs_head_selected_mean_rfs=6.4071

ridge+kinematic baseline on same slice:
  combined_oracle_mean_rfs=8.6790
  combined_ranker_mean_rfs=6.9496

ridge+kinematic+anchor candidates on same slice:
  combined_oracle_mean_rfs=8.9789
  combined_ranker_mean_rfs=6.7418
```

Conclusion:

- Anchor candidates add oracle headroom.
- The current selector cannot reliably exploit that headroom.
- Anchor candidates should remain experimental and disabled by default until a
  stronger proposal confidence head or visual/temporal generator improves
  selected RFS, not only oracle RFS.

Selector diagnostic follow-up:

```text
ridge+kinematic+anchor source diagnostics:
  selected_anchor_rate=0.35
  oracle_anchor_rate=0.20
```

The selector was over-selecting anchor proposals. Grouping anchor candidate
families and adding source indicator features slightly improved the no-anchor
baseline but did not make anchors usable:

```text
no-anchor full validation CV, official RFS:
  before source indicators: 7.4445864970
  after source indicators:  7.4462954054
```

Temporal motion-summary ablation:

```text
100-frame official RFS smoke:
  base selected:              6.9623
  temporal-only selected:     7.1977

full 479-frame official RFS CV:
  base selected:              7.4463
  temporal-only selected:     7.4275
  base + temporal candidates: 7.4738
  base + temporal oracle:     9.0192
  contextual selector:        7.5438
  trajectory ridge=30:        7.6028
```

Interpretation: temporal summaries are useful as an additional proposal source
but not as a replacement feature set. The active benchmark therefore keeps the
base ridge model, adds temporal-summary candidates behind the selector, uses
source-by-speed/source-by-intent contextual selector features, and trains the
trajectory ridge with `ridge=30` because that improved matched full official-RFS
CV from `7.5438` to `7.6028`.

Slice-level bias diagnostic:

```text
worst regret slice: speed:slow
selected RFS:       6.943203386463849
oracle RFS:         8.858640448126808
mean regret:        1.9154370616629595
```

The slow-speed slice remains the next measured target for improvement. The
contextual selector reduced global regret and shifted selected source mix toward
learned candidates, but slow-speed regret remains high and should only be
changed through a new ablation that improves held-out segment-grouped CV.

Anchors remain off by default. Source and contextual interaction indicators stay
in the selector because they improved the validated default path and provide
useful diagnostics.

Rejected contextual default experiment:

```text
contextual selector with default absolute target/ridge=1:
  selected RFS: 7.3890
```

The contextual features were promoted only after the matched `frame_delta`,
`ridge=100` official-RFS run improved over the previous current benchmark.

Submission path reset:

```text
restore train/test shards and official frame list -> readiness audit -> pre-registered blind matrix -> upload all tarballs -> record hidden-test score by SHA-256
```

The active packaging entrypoint is `run_wod_leaderboard_attack.py`, which writes
the WOD-E2E readiness report first and then calls
`prepare_wod_e2e_submission_matrix.py` only after the data gates pass. The
matrix includes strict minimal-shot rows and separately marked
preference-calibrated rows; only the strict rows are eligible for minimal-shot
claim language.

Negative selector-feature experiment:

```text
dynamic-continuity features, no-anchor full validation CV, local RFS:
  selector_ridge=100:  7.5304
  selector_ridge=300:  7.5466
  selector_ridge=1000: 7.5476

source-indicator baseline, no-anchor full validation CV, local RFS:
  selector_ridge=100:  7.5664
```

The continuity features improved a 100-frame smoke but regressed the full
validation split, even with stronger selector regularization. They were removed
from the active feature set.

Negative camera-payload experiment:

```text
100-frame local-RFS smoke, contextual selector:
  no camera payload features:      7.4206
  camera payload interactions:     7.4110
```

Camera payload presence/size features are implemented as an experimental
`camera_contextual` selector mode, but they are not promoted. They do not solve
visual blindness and slightly regress the matched smoke. A real visual sidecar
should use decoded image or learned visual tokens, then prove improvement under
the same segment-grouped CV gate.

Negative decoded-image experiment:

```text
30-frame local-RFS smoke, contextual selector:
  no image features:       6.9532
  decoded image features:  6.6467
```

The experimental `image_contextual` mode decodes grayscale JPEG statistics
inside the selector path. It regressed immediately and is not promoted. The next
visual attempt should cache visual features per frame and use a stronger visual
representation, not per-candidate decoded image statistics.

Negative residual-pool experiments:

```text
full 479-frame local-RFS CV, contextual selector:
  current residual_modes=3:        7.5936
  pairwise residual candidates:    7.6142
  residual_modes=4:                7.6077
  residual_modes=4, ridge=300:     7.6229

full 479-frame official-RFS CV, contextual selector:
  previous ridge=10 benchmark:     7.5438
  pairwise residual candidates:    7.5167
  residual_modes=4:                7.5090
  residual_modes=4, ridge=300:     7.5272
```

The residual-pool changes improved the local backend but regressed official RFS.
They are rejected and must not replace the current promoted configuration.

Negative selector-target experiments:

```text
full 479-frame local-RFS CV, contextual selector:
  frame_delta, ridge=100:      7.5936
  frame_delta, ridge=30:       7.5995
  frame_zscore, ridge=100:     7.6247
  oracle_binary, ridge=100:    7.4982

full 479-frame official-RFS CV, contextual selector:
  previous ridge=10 benchmark: 7.5438
  frame_zscore, ridge=100:     7.5162
```

Frame z-score improved the local backend but regressed official RFS, matching the
residual-pool pattern. The active selector target remains `frame_delta`.

Negative selector-regularization and pairwise-ranking experiments:

```text
full 479-frame official-RFS CV, contextual selector, frame_delta:
  ridge=30:     7.5013
  ridge=100:    7.5438
  ridge=300:    7.5300
  ridge=1000:   7.5253

full 479-frame local-RFS CV, contextual selector, pairwise-delta prototype:
  ridge=100:    7.5769
  ridge=300:    7.5857
  ridge=1000:   7.5813
```

The current `ridge=100` selector remains the best official-RFS configuration.
The pairwise-delta prototype regressed locally and was removed instead of kept
as unused experimental code.

Promoted trajectory-ridge experiment:

```text
full 479-frame local-RFS CV, contextual selector:
  trajectory ridge=1:     7.5318
  trajectory ridge=10:    7.5936
  trajectory ridge=30:    7.6659
  trajectory ridge=100:   7.6175

full 479-frame official-RFS CV, contextual selector:
  trajectory ridge=10:    7.5438
  trajectory ridge=30:    7.6028
```

`ridge=30` is promoted because it improved the full official-RFS CV selected
score and reduced selector regret from `1.4753` to `1.4130`. The promoted
submission/training defaults use this value.

Negative intent-shape selector experiment:

```text
full 479-frame local-RFS CV, trajectory ridge=30:
  contextual selector, ridge=100:            7.6659
  contextual selector, ridge=300:            7.6657
  squared selector, ridge=100:               7.5910
  linear selector, ridge=100:                7.5896
  compact intent-shape selector, ridge=300:  7.6613
```

A broader intent-shape interaction prototype improved local selected RFS to
`7.6800`, but the official run produced no artifact in this environment. The
compact version regressed full local CV, so the experimental selector mode was
removed and the active default remains `contextual`.

Negative anchor-proposal experiments:

```text
full 479-frame local-RFS CV, contextual selector:
  previous ridge=10 candidate pool: selected 7.5936, oracle 9.0714
  anchor_count=8, top_k=2:          selected 7.6256, oracle 9.1842
  anchor_count=16, top_k=2:         selected 7.6042, oracle 9.1707
  anchor_count=16, top_k=4:         selected 7.5748, oracle 9.2256

full 479-frame official-RFS CV, contextual selector:
  previous ridge=10 benchmark:      selected 7.5438, oracle 9.0192
  anchor_count=8, top_k=2:          selected 7.5231, oracle 9.1430
```

Anchors increase official oracle headroom, which is directionally consistent
with winning WOD-E2E approaches, but the current selector cannot reliably choose
the anchor wins. Anchors stay experimental until the selector improves under
official RFS.

## Guardrails

- Do not use simulator scores or simulator scenarios for model selection.
- Do not use text prompts in the active vehicle path.
- Do not tune fixed constants unless backed by validation experiments.
- Keep validation preference labels declared as calibration/RFS training data,
  not as test labels.
