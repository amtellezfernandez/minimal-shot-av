# NAVSIM Methodology Surface

NAVSIM is the public reproducibility surface for the paper's per-axis
non-co-monotonicity claim. It should not be used to validate the AlpaSim
controller-proxy mismatch claim.

Use it for:
- public intervention matrices over ManeuverToken variants;
- per-axis tradeoff checks across collision, drivable-area/offroad, progress,
  TTC, and PDMS/EPDMS;
- reviewer-reproducible evidence that clamping, hard veto, and source decay can
  move different axes in opposite directions.

Do not use it for:
- reactive closed-loop controller-proxy attribution;
- realized-clearance erosion under an AlpaSim controller;
- claims about controller execution mismatch.

NAVSIM evaluates predicted trajectories with a public non-reactive or
pseudo-closed-loop metric surface. That is exactly why it is useful as an
independent benchmark for non-co-monotonicity, and exactly why AlpaSim remains
necessary for the controller-proxy rung.

Agent entry point for NAVSIM devkit runs:

```text
minimal_shot_av.model.navsim_maneuver_token_agent:ManeuverTokenNavsimAgent
```

Run four variants with the same scene split:
- `variant=raw`
- `variant=clamped`
- `variant=hard_veto`
- `variant=source_decay`

Local in-memory runner used for the first public smoke matrix:

```bash
PYTHONPATH=$PWD/src \
python3 scripts/run_navsim_in_memory_matrix.py \
  --navsim-devkit-root /tmp/navsim-devkit \
  --openscene-data-root /tmp/navsim_workspace/dataset \
  --nuplan-maps-root /tmp/navsim_workspace/dataset/maps \
  --max-scenes 20 \
  --output-dir artifacts/corl2027/navsim_matrix20
```

This bypasses NAVSIM metric-cache serialization but still uses NAVSIM's official
`MetricCacheProcessor.compute_metric_cache` and `pdm_score` in memory.

Analysis command after NAVSIM evaluation exports per-scene CSV/JSON metrics:

```bash
python3 scripts/analyze_navsim_intervention_matrix.py \
  --baseline raw \
  --run raw=artifacts/corl2027/navsim_matrix20/raw.csv \
  --run clamped=artifacts/corl2027/navsim_matrix20/clamped.csv \
  --run hard_veto=artifacts/corl2027/navsim_matrix20/hard_veto.csv \
  --run source_decay=artifacts/corl2027/navsim_matrix20/source_decay.csv \
  --output-json artifacts/corl2027/navsim_matrix20_intervention_matrix.json \
  --output-markdown artifacts/corl2027/navsim_matrix20_intervention_matrix.md
```

The output report is written to:
- `artifacts/corl2027/navsim_matrix20_intervention_matrix.json`
- `artifacts/corl2027/navsim_matrix20_intervention_matrix.md`

Current 20-scene mini result:
- `clamped`: offroad and score improve, progress worsens.
- `hard_veto`: offroad and score improve, progress worsens.
- `source_decay`: offroad and score improve, progress worsens.
- `Non-co-monotone interventions`: 3/3.
