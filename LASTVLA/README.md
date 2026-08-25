# LASTVLA — Vendored External VLA Projects

Everything under `LASTVLA/` is **vendored external-origin material** from published
Xiaomi EV / Xiaomi Research projects, plus clearly-marked first-party experiment
additions. Nothing here is claimed as first-party work except where listed below.

The repo root `LICENSE` (Apache-2.0) covers first-party code only; the vendored
material below remains under its upstream terms.

## `onevl/` — copy of Xiaomi Research OneVL

- Upstream: <https://github.com/xiaomi-research/onevl> (default branch `main`;
  vendored copy corresponds to the upstream state as of late May 2026, upstream
  head at the time of this audit: `f4e6f05572b2f66611259bcd5563529833f30469`).
- Model weights: <https://huggingface.co/collections/xiaomi-research/onevl-models>
  (built on Qwen3-VL-4B-Instruct; visual tokenizer from BAAI Emu3.5-VisionTokenizer —
  see their respective licenses).
- **License status:** the upstream README declares Apache-2.0 and links to a
  `LICENSE` file, but upstream ships no license text (verified against the GitHub
  API on 2026-07-05, which reports no detectable license for the repository).
  This copy therefore contains no `LICENSE` file either. Treat the code as
  Apache-2.0 per the upstream README declaration, but note the upstream gap.

### First-party additions inside `onevl/` (not upstream code)

- `scripts/` — candidate-bank generation, NAVSIM v1/v2 scoring conversion, and
  summary pipelines written for this repo's CoRL diagnostics.
- `docs/navsim_zero_perception_probe.md` — first-party methodology note.
- `output/` — experiment outputs produced by this repo (NAVSIM v2 mapped smoke
  artifacts and preserved run logs).

## `LaST-VLA/` — upstream announcement README + assets

- Upstream paper repo for "LaST-VLA: Thinking in Latent Spatio-Temporal Space"
  (arXiv:2603.01928; Tsinghua / Xiaomi EV / University of Macau).
- Contains only the upstream `README.md`, `LICENSE` (Apache-2.0), and `assets/`.
  No source code is vendored; upstream had released no code at copy time.
