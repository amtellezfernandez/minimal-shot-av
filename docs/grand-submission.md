# Grand Commission Submission Candidate

## Submission Claim

The current repository is not yet a completed Grand Commission solution. It is a
submission candidate built around two honest assets:

- a WOD-E2E model-side benchmark and packaging harness;
- a lightweight simulator/Spotlight Reflex prototype for reproducible long-tail
  scenario evidence.

The model-side claim is currently a transparent non-text WOD-E2E baseline, not
frontier scene understanding. The latest confirmed structured selector run
improves official validation-CV selected RFS from `7.0223571581211495` constant
velocity to `7.657089971818379`, with a combined candidate oracle at
`9.098014272661512`. That is useful evidence for candidate diversity and
fallback/gating, but it is not enough to call the architecture solved.

This submission targets the **Grand Commission**: overall best autonomy architecture.

## What To Submit

- GitHub repo: this codebase, with `README.md`, `models/DECLARATION.md`, and `docs/spotlight-reflex.md`.
- Video or slide deck: show the Spotlight Reflex policy navigating a generated long-tail scene and explain the simulator-native selector.
- Short write-up: use `docs/two-page-writeup.md` as the architecture draft.
- Analysis notebook: `notebooks/wod_e2e_analysis.ipynb`.

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
- The policy is deterministic and reproducible by seed.
- The current implementation uses no learned AV-specific model and no WOD-E2E fine-tuning.
- The built-in tests cover selector geometry, maneuver generation, selection behavior, CLI artifacts, and deterministic simulator seeds.
- The current model-side world-model experiment should be presented as a
  negative/early result: it adds oracle headroom, but the selector cannot yet
  exploit it reliably.

## Boundaries

- This is a runnable architecture prototype, not a completed WOD-E2E leaderboard submission.
- The WOD-E2E validation split is retained locally and can be used for analysis,
  parser smoke tests, and preference-label experiments. The full train/test
  download needs to be restored before making strict train-set or hidden-test
  claims.
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
