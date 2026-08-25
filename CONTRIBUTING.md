# Contributing

This is a research repository backing a CoRL 2027 submission. Issues and pull
requests are welcome, with the caveat that result-bearing artifacts under
`artifacts/corl2027/` are frozen evidence and are only regenerated via the
documented audit scripts.

## Setup

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md).

## Before opening a PR

```bash
./.venv/bin/python scripts/check_code_quality.py   # line length + compat gates
./.venv/bin/python -m pytest tests -q              # unit + smoke suite
```

Repo conventions (enforced by `tests/test_architecture_boundaries.py`):

- All real logic lives in `src/minimal_shot_av/`; `scripts/*.py` are generated
  thin wrappers, one per CLI command. New commands need a wrapper with the
  standard template and a registration in the ownership sets in
  `tests/test_architecture_boundaries.py`.
- `model/` code must not import `simulator/` code and vice versa; `neutral/`
  imports neither.
- Target Python 3.10 compatibility (no `StrEnum`, `tomllib`, `typing.Self`,
  etc. — see `[tool.minimal_shot_av.quality]` in `pyproject.toml`).

## Licensing

First-party contributions are accepted under the repo's Apache-2.0 license.
Do not add code to the vendored `LASTVLA/` or `third_party/` trees unless it is
clearly marked as first-party in the corresponding README.
