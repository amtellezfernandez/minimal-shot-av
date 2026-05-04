# Grand Commission Submission Candidate

## Submission Claim

The current repository is not yet a completed Grand Commission solution. It is a
submission candidate built around one primary minimal-shot architecture asset
and one auxiliary benchmark asset:

- a lightweight simulator/Spotlight Reflex prototype for reproducible long-tail
  scenario evidence;
- a WOD-E2E model-side benchmark and packaging harness for analysis,
  submission formatting, and failure auditing.

The Grand claim should lead with the minimal-shot architecture behavior: a small
closed-loop policy that uses scene structure, maneuver hypotheses, and safety
selection in unfamiliar long-tail scenarios. The WOD model-side claim is a
transparent non-text benchmark harness, not frontier scene understanding or
strict zero-shot autonomy. The latest confirmed structured selector artifact
improves official validation-CV selected RFS from `7.0223571581211495` constant
velocity to `7.65941846208851`, with a combined candidate oracle at
`9.098014272661512`; because that selector is preference-calibrated on retained
validation labels under segment-grouped CV, it is useful development analysis
but not the centerpiece of the minimal-shot claim.

This submission targets the **Grand Commission**: overall best autonomy architecture.

## What To Submit

- GitHub repo: this codebase, with `README.md`, `models/DECLARATION.md`, and `docs/spotlight-reflex.md`.
- Video or slide deck: show the Spotlight Reflex policy navigating a generated long-tail scene and explain the simulator-native selector.
- Short write-up: use `docs/two-page-writeup.md` as the architecture draft.
- Analysis notebook: `notebooks/wod_e2e_analysis.ipynb`.
- Judging evidence map: `docs/judging-criteria-evidence.md` and
  `artifacts/sota_judging_criteria_audit.json`.

## Demo Commands

Run the main architecture demo:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/grand_spotlight_demo
```

Run an intersection stress case:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy spotlight-reflex \
  --scenario-cluster intersection \
  --seed 3 \
  --artifacts-dir artifacts/grand_intersection_stress_seed3
```

Compare against the baseline policy:

```bash
uv run --no-sync python scripts/run_demo.py \
  --policy baseline \
  --scenario-cluster spotlight \
  --seed 3 \
  --artifacts-dir artifacts/grand_baseline_spotlight_demo
```

Each command writes:

- `latest_rollout.json`
- `latest_rollout.svg`

## Evidence To Highlight

- The policy outputs 20-point, 5-second maneuver trajectories internally.
- Candidate selection uses simulator-native trajectory selector geometry at 3s and 5s.
- Each Spotlight Reflex rollout step now carries a decision explanation:
  matched 3s/5s references, whether the selected trajectory stayed inside each
  trust region, action and horizon clearance, progress/speed bonuses, penalties,
  and the top candidate summaries.
- Each step also carries label-free world geometry signals: obstacle pressure,
  route blockage, corridor blockage, side clearances, and preferred escape side.
  The regression tests verify these signals are unchanged when object names,
  labels, clusters, and tags are relabeled while geometry is held fixed.
- The policy is deterministic and reproducible by seed.
- The primary simulator submission uses no learned AV-specific model and no
  WOD-E2E fine-tuning.
- Optional model-side WOD analysis now includes a neural anchor-residual
  proposal ensemble. Its strongest official held-out subset result is
  `7.737680847131364` RFS on `159` frames with `9.178972912996967` oracle RFS.
  This is stronger than the non-neural champion on that subset, but the
  full-479-frame promoted WOD evidence remains `7.65941846208851`; it is not a
  strict zero-shot claim and it does not beat the `8.0461` leaderboard snapshot
  target.
- WOD validation-CV evidence is declared as auxiliary development analysis, not
  proof of strict zero-shot WOD-E2E generalisation.
- The built-in tests cover selector geometry, maneuver generation, selection behavior, CLI artifacts, and deterministic simulator seeds.
- The judging-criteria audit maps the submission to technical excellence,
  novelty, feasibility, and adherence to the brief.
- The current model-side world-model experiment should be presented as a
  negative/early result: it adds oracle headroom, but the selector cannot yet
  exploit it reliably.

## Boundaries

- This is a runnable architecture prototype, not a completed WOD-E2E leaderboard submission.
- The WOD-E2E validation split is retained locally and can be used for analysis,
  parser smoke tests, and preference-label experiments. The full train/test
  download needs to be restored before making strict train-set or hidden-test
  claims.
- Any WOD selector trained or calibrated from retained validation preference
  labels must be described as minimal-shot development evidence, not strict
  zero-shot behavior.
- The current scene critic is procedural/context-derived, not a deployed VLM
  camera stack.
- The strongest honest claim is infrastructure plus reproducible closed-loop
  prototype behavior. A stronger solution claim requires the acceptance bar in
  `docs/solution-reset.md`.

## Grand Submission Checklist

- [ ] Record or export a 1-5 minute video/slide deck for Spotlight Reflex.
- [ ] Include one understood stress case; if using a failure case, choose a run that still fails after the latest generator repair.
- [ ] Complete `models/DECLARATION.md` with any base model actually used in the final demo.
- [ ] Keep the no-AV-finetuning claim explicit.
- [ ] Include test command output: `uv run --no-sync python -m unittest discover -s tests`.
