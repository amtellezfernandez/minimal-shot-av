# Handoff

## Current state

This repo currently has two active lines of work that were tied together:

1. `docs/corl2027/paper.tex`: the paper draft now includes a bounded OneVL/NAVSIM v2 diagnostic section.
2. `LASTVLA/onevl`: preserved OneVL candidate-bank artifacts, scoring outputs, and summary scripts for mapped NAVSIM v2 smoke subsets.

Worktree status was clean at handoff time.

Recent commits:

- `beb5e8b` Add reproducible NAVSIM v2 mapped scaling summary
- `862611f` Preserve interrupted NAVSIM v2 g10 intermediates
- `3b9d994` Update paper with NAVSIM v2 OneVL diagnostics
- `c9b0ba2` Preserve CoRL artifact outputs

## Paper status

Primary draft:

- [docs/corl2027/paper.tex](/home/amdev/sota/minimal-shot-av/docs/corl2027/paper.tex)
- [docs/corl2027/paper.bib](/home/amdev/sota/minimal-shot-av/docs/corl2027/paper.bib)

What changed:

- Added a bounded `Released VLA Check on NAVSIM v2` subsection.
- Added OneVL citation (`lu2026onevl`).
- Updated framing so the OneVL result is used as an interface-level diagnostic, not a claim that reranking is solved.
- Abstract and limitations now explicitly state that the NAVSIM v2 evidence is smoke-scale and mapped-subset only.

Where to look in the paper:

- `docs/corl2027/paper.tex:372` subsection start for the NAVSIM v2 result
- `docs/corl2027/paper.tex:393` table caption for mapped smoke audit
- `docs/corl2027/paper.tex:696` and `:718` for limitation/claim-boundary language

## OneVL / NAVSIM v2 status

Root path:

- [LASTVLA/onevl](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl)

The repo contains:

- candidate generation for OneVL
- NAVSIM scoring/eval conversion scripts
- mapped NAVSIM v2 smoke artifacts under `output/navsim_v2/smoke`
- zero-perception ridge-probe outputs for some groups

Important scripts:

- [LASTVLA/onevl/scripts/summarize_navsim_v2_mapped_scaling.py](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/scripts/summarize_navsim_v2_mapped_scaling.py)
- [LASTVLA/onevl/scripts/summarize_navsim_k_scaling.py](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/scripts/summarize_navsim_k_scaling.py)
- [LASTVLA/onevl/scripts/summarize_candidate_bank_variance.py](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/scripts/summarize_candidate_bank_variance.py)
- [LASTVLA/onevl/scripts/train_navsim_v2_ridge_probe.py](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/scripts/train_navsim_v2_ridge_probe.py)

README note for rebuilding the mapped summary:

- [LASTVLA/onevl/README.md](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/README.md)

## Preserved NAVSIM v2 smoke artifacts

Directory:

- [LASTVLA/onevl/output/navsim_v2/smoke](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke)

Most useful summary artifacts:

- [navhard_smoke_g1_k_scaling_summary.md](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_k_scaling_summary.md)
- [navhard_smoke_g1_k4_k8_seed_variance_summary.md](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_k4_k8_seed_variance_summary.md)
- [navhard_smoke_g2_k4_k8_seed0_comparison.md](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g2_k4_k8_seed0_comparison.md)
- [navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.md](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.md)
- [navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.svg](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.svg)

Interrupted larger run preserved for later restart:

- [navhard_smoke_g10_interrupted_summary.md](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g10_interrupted_summary.md)
- [navhard_smoke_g10_interrupted_summary.json](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g10_interrupted_summary.json)
- [navhard_smoke_g10_k1_k2_k4_k8_seed0_interrupted.sh](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g10_k1_k2_k4_k8_seed0_interrupted.sh)
- [navhard_smoke_g10_k1_k2_k4_k8_seed0_interrupted.log](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g10_k1_k2_k4_k8_seed0_interrupted.log)
- [navhard_smoke_g10_sample7_seed0_t0.8_p0.95_partial51.json](/home/amdev/sota/minimal-shot-av/LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g10_sample7_seed0_t0.8_p0.95_partial51.json)

## Key empirical readouts already in repo

### g1 K-scaling

From `navhard_smoke_g1_k_scaling_summary.md`:

- proxy oracle rises from `0.6033` at `K=1` to `0.6958` at `K=8`
- proxy oracle reaches `0.7079` at `K=32`
- interpretation: headroom appears by `K=8` and then starts saturating

### g1 seed variance

From `navhard_smoke_g1_k4_k8_seed_variance_summary.md`:

- `K=8` improves mean proxy oracle over `K=4`
- proxy variance across seeds decreases at `K=8`
- fixed branch identity remains unstable, which supports a density/selection story rather than a fixed-branch story

### cross-group mapped scaling (g1-g4, seed 0)

From `navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.md`:

- `K=8` consistently raises the proxy oracle over `K=4`
- official best-fixed EPDMS gains vary by group
- on `g4`, official best-fixed is nearly tied while proxy headroom still increases
- zero-perception probe underperforms the proxy top-1 baseline on `g2-g4`

Representative numbers:

| Group | K | Top-1 EPDMS | Best fixed EPDMS | Proxy oracle | Zero-perception proxy |
| --- | ---: | ---: | ---: | ---: | ---: |
| g1 | 4 | 0.000000 | 0.362018 | 0.661465 | - |
| g1 | 8 | 0.000000 | 0.800662 | 0.695776 | 0.608252 |
| g2 | 8 | 0.045811 | 0.418909 | 0.614206 | 0.515941 |
| g3 | 8 | 0.064128 | 0.436662 | 0.601387 | 0.461974 |
| g4 | 8 | 0.157245 | 0.379370 | 0.696682 | 0.540848 |

Current conclusion supported by the saved evidence:

- candidate-bank headroom is real
- increasing `K` helps expose that headroom
- the current blind low-capacity selector does not recover it reliably on NAVSIM v2 mapped smoke groups

## What is safe to claim

Safe:

- OneVL can be adapted to produce candidate banks and scored on NAVSIM v2 mapped smoke subsets.
- Candidate-bank headroom exists and grows with `K` in the saved smoke runs.
- Official fixed-candidate gains do not scale monotonically with proxy headroom.
- The current zero-perception ridge probe does not solve selector quality on these v2 mapped subsets.

Not safe:

- full NAVSIM v2 benchmark claim
- full-validation ranking claim
- claim that reranking beats OneVL in deployed selection
- claim that the probe generalizes broadly beyond mapped smoke subsets

## Recommended next steps

Priority order:

1. Keep the current paper framing. Do not rewrite around a solved reranker story.
2. If GPU time becomes available again, resume from the preserved `g10` script rather than reconstructing the run from memory.
3. If more experiments are needed, prefer:
   - larger mapped subsets first
   - official score preservation
   - train/test split discipline for any learned selector
4. Keep the OneVL section framed as a diagnostic audit unless a stronger held-out selector result is obtained.

## Commands

Rebuild the mapped cross-group summary:

```bash
python3 LASTVLA/onevl/scripts/summarize_navsim_v2_mapped_scaling.py \
  --smoke-dir LASTVLA/onevl/output/navsim_v2/smoke \
  --groups g1,g2,g3,g4 \
  --ks 4,8 \
  --seed 0 \
  --out-json LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.json \
  --out-md LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.md \
  --out-svg LASTVLA/onevl/output/navsim_v2/smoke/navhard_smoke_g1_g2_g3_g4_k4_k8_seed0_scaling.svg
```

Rebuild the paper:

```bash
cd docs/corl2027
pdflatex paper.tex
bibtex paper
pdflatex paper.tex
pdflatex paper.tex
```

## Short resume brief

If resuming later, the practical summary is:

- the paper already contains the bounded OneVL/NAVSIM v2 diagnostic
- the expensive v2 smoke artifacts are preserved and pushed
- the interrupted larger `g10` run was stopped intentionally and saved
- the current evidence supports a candidate-bank headroom story, not a selector-success story
