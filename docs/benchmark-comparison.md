# Benchmark Comparison

This repo now treats benchmark comparison as a first-class artifact, not prose.

The neutral comparison CLI is:

```bash
PYTHONPATH=src python3 -m minimal_shot_av.neutral.benchmark_compare SUBJECT.json BASELINE.json
```

Reports must use the same `suite` and shared metric names. The comparator refuses
mixed-suite comparisons, so a WOD-E2E RFS number cannot be compared to an
AlpaSim closed-loop score.

WOD-E2E reports also carry explicit fairness metadata. The comparator rejects
comparisons unless `evaluation_contract`, `split`, `frame_count`,
`score_backend`, and `selection_mode` all match and are known. This prevents
validation-set, leaderboard-test, oracle-selection, deployable-ranker, and
smoke-check numbers from being treated as the same result.

Metric reports can be generated from raw artifacts with:

```bash
PYTHONPATH=src python3 -m minimal_shot_av.neutral.benchmark_reports wod-eval artifacts/model_diagnostic_logged_future_official.json --system spotlight_reflex_logged_future_smoke --suite wod_e2e_smoke_frame --output benchmarks/current/spotlight_reflex_wod_smoke.json
PYTHONPATH=src python3 -m minimal_shot_av.neutral.benchmark_reports scenario-eval artifacts/agnostic_sim_eval_wod_1_10/scenario_eval.json --system spotlight_reflex_procedural_wod --suite procedural_wod_simulator --output benchmarks/current/spotlight_reflex_procedural_wod.json
```

For a real Alpamayo 1.5 comparison, ingest the aggregate output from an
AlpaSim PhysicalAI-AV-NuRec run and require the published comparison metric.
The report must declare the same evaluation and sensor contract as the baseline;
otherwise the comparator rejects the comparison.

```bash
PYTHONPATH=src python3 -m minimal_shot_av.neutral.benchmark_reports alpasim-metrics runs/YOUR_NUREC_RUN/aggregate/metrics_results.txt --system spotlight_reflex_alpasim_nurec --suite physicalai_nurec_alpasim --require-metric alpasim_score --metadata-json '{"evaluation_contract":"physicalai_nurec_alpasim_closed_loop","scenario_set":"physicalai_av_nurec_910","score_backend":"alpasim","sensor_contract":"published_alpamayo_1_5_model_card","camera_ids":"published_multi_camera_rgb","context_length":"published_alpamayo_1_5","ego_history_hz":10,"output_horizon":"6.4s_64_waypoints_10hz","route_command_source":"navigation_guidance","alpasim_version":"published_model_card_unspecified"}' --output benchmarks/current/spotlight_reflex_alpasim_nurec.json
PYTHONPATH=src python3 -m minimal_shot_av.neutral.benchmark_compare benchmarks/current/spotlight_reflex_alpasim_nurec.json benchmarks/baselines/alpamayo_1_5_alpasim.json
```

If the AlpaSim output only contains component metrics such as
`collision_at_fault`, `offroad`, `dist_to_gt_trajectory`, or
`duration_frac_20s`, the report builder can still record them, but it will not
claim a win over Alpamayo 1.5 without `alpasim_score`.

The current checked-in Spotlight Reflex AlpaSim driver requests only
`camera_front_wide_120fov`. That should be treated as a separate front-camera
ablation unless Alpamayo 1.5 is run with the same `camera_ids`, context length,
route-command source, output horizon, AlpaSim version, scenario set, and score
backend.

After the three required AlpaSim runs exist, generate both comparison tracks
with:

```bash
PYTHONPATH=src python3 scripts/produce_alpasim_comparable_reports.py \
  --published-ours-run runs/spotlight_reflex_published_contract \
  --front-ours-run runs/spotlight_reflex_front_camera \
  --front-alpamayo-run runs/alpamayo_1_5_front_camera
```

This writes:

- `benchmarks/current/spotlight_reflex_alpasim_published_contract.json`
- `benchmarks/current/spotlight_reflex_alpasim_front_camera.json`
- `benchmarks/current/alpamayo_1_5_alpasim_front_camera.json`
- `benchmarks/current/comparisons/spotlight_reflex_vs_alpamayo_1_5_published_contract.json`
- `benchmarks/current/comparisons/spotlight_reflex_vs_alpamayo_1_5_front_camera.json`

Default AlpaSim aggregates are commonly written as
`aggregate/metrics_results.txt`; JSON and CSV are also accepted if produced by
your run pipeline. Unknown metric names are rejected until their direction is
declared in code, so an error metric cannot accidentally be treated as
higher-is-better.

## Alpamayo 1.5

The checked-in Alpamayo 1.5 baseline is:

- `benchmarks/baselines/alpamayo_1_5_alpasim.json`
- suite: `physicalai_nurec_alpasim`
- metric: `alpasim_score = 0.81`
- uncertainty: `0.01`
- source: `https://huggingface.co/nvidia/Alpamayo-1.5-10B`

The model card reports `0.81 +/- 0.01` closed-loop AlpaSim score on 910
PhysicalAI-AV-NuRec scenarios. I did not find a public Alpamayo 1.5 WOD-E2E RFS
leaderboard number in the repo metadata or public model card, so WOD-E2E
comparisons should use WOD baselines from `docs/leaderboard.md` until we run or
obtain a real Alpamayo WOD-E2E result.

The comparator uses declared uncertainty as the required margin. A measured
`0.82` against Alpamayo's `0.81 +/- 0.01` is treated as not clearly better; the
subject must exceed the baseline by more than `0.01`.

## Current Real Numbers

- WOD smoke sanity: `benchmarks/current/spotlight_reflex_wod_smoke.json`
  records official Waymo RFS `4.793513563812969` on one matched diagnostic
  frame. This is not a validation-set score.
- WOD one-frame preference-reference smoke checks:
  `benchmarks/current/wod_preference_oracle_candidate_smoke.json` records RFS
  `6.5`, and `benchmarks/current/wod_best_preference_candidate_smoke.json`
  records RFS `10.0`. These confirm official-RFS plumbing and candidate
  selection semantics on one validation frame; they are not model performance
  claims.
- Model-stack experiment ledger:
  `benchmarks/experiments/model_stack_experiments_20260426.json` records the
  non-text candidate, ranker-selection, and output-validator experiments with exact test
  commands, pass/fail results, and runtime measurements.
- Non-text kinematic model baseline:
  `benchmarks/current/wod_kinematic_non_text_smoke_first.json` records official
  RFS `10.0` on the same one-frame smoke contract using only WOD ego-history
  kinematics. This is not validation performance; it is a warning that VLM
  outputs must beat cheap non-language baselines on the full validation set.
- Bounded non-text WOD validation baseline:
  `benchmarks/current/wod_kinematic_non_text_records2000_first.json` records
  official RFS `8.0` mean / `9.0` median on 3 preference frames found in the
  first 2,000 raw validation records. This is still not the full 479-frame
  validation score, but it is a stronger sanity check than the one-frame smoke.
- Full non-text WOD validation baseline:
  `benchmarks/current/wod_kinematic_non_text_val479_first.json` records official
  RFS `7.02235769749286` mean and `7.441072996771592` median on all 479
  validation preference frames. The same four-candidate set has validation-only
  oracle headroom of `7.852351864082588` mean RFS, so selection/ranking is an
  immediate model-side bottleneck. Its report metadata declares
  `selection_mode = first`; it is not comparable to oracle or leaderboard-test
  reports.
- WOD model-side evaluation now preserves multiple candidate trajectories per
  frame. The default selection mode is `first`; `--candidate-selection best_validation`
  is validation-only and reports candidate-set headroom through
  `mean_best_candidate_rfs` and `mean_selection_regret`.
- Deployable candidate selection uses a trained WOD preference ranker:
  `--candidate-selection ranker --ranker artifacts/wod_preference_ranker.full.json`.
  This does not use simulator feedback or validation labels at inference time.
- Current kinematic-candidate ranker check:
  `benchmarks/current/wod_kinematic_non_text_val479_ranker.json` records the
  same official RFS mean as first-candidate selection, `7.02235769749286`. The
  diagnostic artifact reports that the ranker selected candidate index `0` on
  all 479 frames, while the oracle best candidate was index `0` on only 305
  frames. The saved ranker did not know three candidate names in this candidate
  set: `constant_acceleration`, `constant_heading_change`, and `hold_position`.
- Candidate-distribution-matched selector:
  `benchmarks/current/wod_kinematic_ranker_val479_cv.json` records segment-grouped
  5-fold cross-validation after scoring the exact four-candidate kinematic set
  with official RFS labels. Mean selected RFS improves from the candidate-index-0
  baseline `7.02235769749286` to `7.096202559469913`, with oracle headroom
  `7.852351864082588`. This is validation selector evidence, not a
  leaderboard-test result.
- Previous WOD trajectory baseline:
  `benchmarks/current/wod_ridge_trajectory_cv_official.json` records
  segment-grouped 5-fold official-RFS validation CV for the non-text ridge
  trajectory model with auxiliary temporal-summary candidates. Mean selected RFS
  is `7.602814916652251`, with combined oracle headroom `9.015857402777051`.
- Current promoted structured-selector report:
  `artifacts/wod_fastkin_gate_ridge175_scene016_cv_official.json`
  records segment-grouped 5-fold official-RFS validation CV after adding the
  fallback-enabled fast-kinematic/scene-gate selector path. Mean selected RFS is
  `7.659197835744162`, with combined oracle headroom
  `9.098014272661512`. This is a small conservative improvement over the
  previous `7.657089971818379` report:
  `artifacts/wod_fastkin_gate_ridge175_scene016_breakthrough_audit.json`
  records a passed validation-CV improvement audit.
- Current lightweight world-model ablation:
  `artifacts/world_sweeps/wod_cv_479_world_ego_l8_m0_official.json` records
  segment-grouped 5-fold official-RFS validation CV after adding an ego-temporal
  world-model imagined-future candidate. Mean selected RFS is
  `7.606198495114426`, with combined oracle headroom `9.046125057713054`.
  This is a tiny positive result, not a strong solution. Memory-neighbor
  candidates improved oracle headroom in local sweeps but regressed selected RFS.
  Scene-token mode needs cached image features before it is practical to sweep.
- Current model bias audit:
  `benchmarks/current/wod_model_bias_audit.json` records status `warn`. It flags
  validation-preference bias, ego-history-only visual blindness, and a `1.4130`
  RFS selector gap to the candidate oracle. Slice diagnostics show the worst
  current regret on `intent:2`: selected RFS `6.7820637675739555` versus oracle
  `8.535338832632307`. The score should therefore be described as
  validation-calibrated model evidence, not a strict zero-shot or
  leaderboard-test claim.
- Current online runtime:
  `benchmarks/current/wod_online_runtime.json` records the loaded numeric
  trajectory loop over 5,000 synthetic online frames. It measures base
  candidate generation, temporal auxiliary candidates, feature extraction, and
  contextual ranker selection only. It excludes TFRecord parsing, JPEG decode,
  official RFS scoring, and offline training. Current p95 total runtime is
  `1.40214185 ms` for 14 candidates on the recorded CPU environment,
  below the strict `14 ms` target. The comparator-ready report is
  `benchmarks/current/wod_online_runtime_metric_report.json`, and
  `benchmarks/current/wod_online_runtime_vs_14ms_budget.json` records a pass
  against `benchmarks/baselines/realtime_14ms_budget.json`.
- Procedural WOD simulator sanity:
  `benchmarks/current/spotlight_reflex_procedural_wod.json` records 110
  independent simulator runs with `success_rate = 1.0`, `collision_rate = 0.0`,
  and `benchmark_pass_rate = 1.0`. This is simulator evidence only.
- WOD public baseline: `benchmarks/baselines/wod_e2e_public_baseline.json`
  records `rfs_overall = 6.2365` and `spotlight_rfs = 5.4946` from
  `docs/leaderboard.md`.
- Alpamayo 1.5 public AlpaSim baseline:
  `benchmarks/baselines/alpamayo_1_5_alpasim.json` records
  `alpasim_score = 0.81`.

## Next Measurement Required

To compare directly against Alpamayo 1.5, run our model stack on the same
PhysicalAI-AV-NuRec AlpaSim 910-scenario suite and emit a report with:

```json
{
  "system": "spotlight_reflex_or_zero_shot_model",
  "suite": "physicalai_nurec_alpasim",
  "source": "path/to/alpasim/results",
  "metadata": {
    "evaluation_contract": "physicalai_nurec_alpasim_closed_loop",
    "scenario_set": "physicalai_av_nurec_910",
    "score_backend": "alpasim",
    "sensor_contract": "published_alpamayo_1_5_model_card",
    "camera_ids": "published_multi_camera_rgb",
    "context_length": "published_alpamayo_1_5",
    "ego_history_hz": 10,
    "output_horizon": "6.4s_64_waypoints_10hz",
    "route_command_source": "navigation_guidance",
    "alpasim_version": "published_model_card_unspecified"
  },
  "metrics": {
    "alpasim_score": {
      "value": 0.0,
      "higher_is_better": true,
      "unit": "score"
    }
  }
}
```

Until that exists, any claim that we beat Alpamayo 1.5 would be unsupported.
