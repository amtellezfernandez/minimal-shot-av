# WOD2ALPAsim Workshop Paper

This directory contains a second, workshop-scoped paper draft:

```text
WOD2ALPAsim:
Extending AlpaSim's External-Driver Contract for WOD-Style Policies
```

The draft frames the contribution as an AlpaSim external-driver contract extension for
WOD-style and minimal-shot policies:

- patched AlpaSim driver/deployment code
- project-owned AlpaSim model plugins
- policy-facing signal reconstruction
- reproducible setup, readiness, and launch tooling
- a reference implementation for future AlpaSim external-driver adapters

Build locally from this directory:

```bash
make
```

The current draft uses the existing CoRL style files in `../corl2027/`.

Checks referenced by the paper live in:

```bash
pytest tests/test_alpasim_integration.py tests/test_alpasim_setup_scripts.py
```
