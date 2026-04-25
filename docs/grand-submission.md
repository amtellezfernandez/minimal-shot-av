# Grand Commission Submission: Spotlight Reflex

## Submission Claim

Spotlight Reflex is a minimal-shot autonomy architecture for rare long-tail driving scenes. It uses structured scene reasoning, deterministic maneuver generation, and exact RFS trust-region selection rather than AV-dataset fine-tuning or route memorization.

This submission targets the **Grand Commission**: overall best autonomy architecture.

## What To Submit

- GitHub repo: this codebase, with `README.md`, `models/DECLARATION.md`, and `docs/spotlight-reflex.md`.
- Video or slide deck: show the Spotlight Reflex policy navigating a generated long-tail scene and explain the RFS-style selector.
- Short write-up: use `docs/two-page-writeup.md` as the architecture draft.
- Analysis material: use `notebooks/README.md` as the planned WOD-E2E exploration notebook outline.

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
- Candidate selection uses exact RFS trust-region geometry at 3s and 5s.
- The policy is deterministic and reproducible by seed.
- The current implementation uses no learned AV-specific model and no WOD-E2E fine-tuning.
- The built-in tests cover RFS geometry, maneuver generation, selection behavior, CLI artifacts, and deterministic simulator seeds.

## Boundaries

- This is a runnable architecture prototype, not a completed WOD-E2E leaderboard submission.
- The WOD-E2E validation split is present locally and can be used for analysis,
  parser smoke tests, and preference-label experiments. Train/test TFRecords
  are not present in this workspace.
- The current scene critic is procedural/context-derived, not a deployed VLM
  camera stack.
- The strongest honest claim is minimal-shot architecture and reproducible closed-loop prototype behavior.

## Grand Submission Checklist

- [ ] Record or export a 1-5 minute video/slide deck for Spotlight Reflex.
- [ ] Include one understood stress case; if using a failure case, choose a run that still fails after the latest generator repair.
- [ ] Complete `models/DECLARATION.md` with any base model actually used in the final demo.
- [ ] Keep the no-AV-finetuning claim explicit.
- [ ] Include test command output: `uv run --no-sync python -m unittest discover -s tests`.
