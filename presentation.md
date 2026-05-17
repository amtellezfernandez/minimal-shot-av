---
title: Spotlight Reflex Presentation
description: Canonical presentation entry point for the CoRL-2027 branch
---

# Spotlight Reflex Presentation

The canonical deck for this branch is the Beamer presentation:

- [`presentation.pdf`](presentation.pdf)
- [`presentation-sota.pdf`](presentation-sota.pdf)
- [`docs/presentation.pdf`](docs/presentation.pdf)
- [`docs/presentation.tex`](docs/presentation.tex)

This root Markdown file is intentionally only an index. The slide deck follows
the main-branch presentation format and keeps the story compact:

1. who built it, what was built, and what this is not
2. motivation and failure case
3. grounded reasoning context
4. why the custom simulator exists and why AlpaSim was added after it
5. core idea: reason from geometry, not memory
6. six geometry scalars, nine ManeuverTokens, and the decision-flow diagram
7. simulation and AlpaSim transfer results
8. WOD-E2E selector results
9. oracle-gap and next-step analysis

Supporting artifacts:

- [`README.md`](README.md)
- [`docs/simulation.md`](docs/simulation.md)
- [`docs/wod-e2e-system-walkthrough.md`](docs/wod-e2e-system-walkthrough.md)
- [`docs/corl2027/paper.pdf`](docs/corl2027/paper.pdf)

For a more technical view of the experiment logs, AlpaSim transfer harness, and
ongoing paper work, use the
[`CoRL-2027` branch](https://github.com/amtellezfernandez/minimal-shot-av/tree/CoRL-2027).
