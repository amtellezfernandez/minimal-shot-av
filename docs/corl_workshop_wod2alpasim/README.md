# WOD2ALPAsim Workshop Paper

This directory contains a second, workshop-scoped paper draft:

```text
WOD2ALPAsim:
Extending AlpaSim into a Closed-Loop Harness for WOD-Style Drivers
```

The draft is intentionally artifact-oriented. It frames the contribution as the
full-stack AlpaSim integration work needed to run WOD-style and minimal-shot
drivers as native AlpaSim external-driver plugins:

- patched AlpaSim driver/deployment surfaces
- project-owned AlpaSim model plugins
- policy-facing signal reconstruction
- reproducible setup, readiness, and launch tooling
- an adapter pattern for future AlpaSim bridges

Build locally from this directory:

```bash
make
```

The current draft uses the existing CoRL style files in `../corl2027/`.

Reviewer-facing artifact checks referenced by the paper live in:

```bash
pytest tests/test_alpasim_integration.py tests/test_alpasim_setup_scripts.py
```
