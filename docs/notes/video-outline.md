# Video / Slide Deck Outline

## Slide 1: Thesis

Project title: **Predictive Reflex for Minimal-Shot AV**.

One sentence:

> A fast structured trajectory policy predicts WOD-E2E long-tail trajectories with non-text candidate generators and an RFS-calibrated selector.

## Slide 2: Why WOD-E2E

Explain the benchmark:

- rare long-tail scenes instead of ordinary driving
- 8-camera 360-degree input
- 12 seconds of context
- 5 seconds of future ego waypoints
- Rater Feedback Score as the primary metric

## Slide 3: Constraint

State the hard rule:

- no AV-dataset fine-tuning for the submitted policy
- all base models declared
- validation used for analysis and calibration reporting, not hidden training claims

## Slide 4: Architecture

Show the pipeline:

- WOD-E2E ego history and route intent
- kinematic and learned residual candidate generators
- anchor/residual proposal experiments
- source-aware numeric ranker
- RFS trust-region verifier
- `(20, 2)` waypoint output

## Slide 5: Validation Demo

Show one WOD-E2E-like case:

- past ego trajectory and route intent
- candidate trajectory set
- selector scores
- generated future path
- why the choice is plausible under RFS

## Slide 6: Failure Case

Show one failure:

- what the system predicted
- what the rater/logged alternatives suggest
- which component failed
- what the next iteration would change

## Slide 7: Results Table

Include the smallest honest table available:

- constant-velocity baseline
- route-following baseline
- kinematic candidate baseline
- ridge residual candidate baseline
- full Predictive Reflex, if implemented

Report:

- validation RFS
- ADE at 3 and 5 seconds
- invalid trajectory rate
- fallback rate
- runtime per target frame

## Slide 8: Funding Milestone

State what the prize enables:

- complete WOD-E2E submission writer and proto validator
- reproducible notebook
- stronger rater-aware selector
- edge-sized trajectory decoder
- deeper failure analysis across scenario clusters
