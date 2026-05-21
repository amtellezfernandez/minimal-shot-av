from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.cli.commands import analyze_alpasim_transfer_matrix as _cmd

BINARY_METRICS = _cmd.BINARY_METRICS
CONTINUOUS_METRICS = _cmd.CONTINUOUS_METRICS
METRIC_VALIDITY_FLAGS = _cmd.METRIC_VALIDITY_FLAGS
MODEL_LABELS = _cmd.MODEL_LABELS
_axis_conflicts = _cmd._axis_conflicts
_bootstrap_mean_interval = _cmd._bootstrap_mean_interval
_discover_batches = _cmd._discover_batches
_load_batch_rows = _cmd._load_batch_rows
_markdown_report = _cmd._markdown_report
_mcnemar_exact_p = _cmd._mcnemar_exact_p
_metric_warning_flags = _cmd._metric_warning_flags
_model_summary = _cmd._model_summary
_numeric_summary = _cmd._numeric_summary
_paired_comparison = _cmd._paired_comparison
_parse_args = _cmd._parse_args
_safe_rate = _cmd._safe_rate
_sign_test_two_sided_p = _cmd._sign_test_two_sided_p
_truthy = _cmd._truthy
_wilson_interval = _cmd._wilson_interval
_percentile = _cmd._percentile
_fmt_rate = _cmd._fmt_rate
_fmt_rate_with_validity = _cmd._fmt_rate_with_validity
_fmt_float = _cmd._fmt_float
_fmt_signed = _cmd._fmt_signed
_fmt_p = _cmd._fmt_p
summarize_run = _cmd.summarize_run


def analyze_matrix(
    matrix_dir: Path,
    *,
    baseline_model: str,
    scene_preset: str | None,
    max_dist_to_gt: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    matrix_dir = matrix_dir.resolve()
    batches = _discover_batches(matrix_dir, scene_preset=scene_preset)
    if baseline_model not in batches:
        raise SystemExit(f"Baseline model not found in matrix dir: {baseline_model}")

    by_model = {
        model: _load_batch_rows(batch_dir, max_dist_to_gt=max_dist_to_gt)
        for model, batch_dir in sorted(batches.items())
    }
    baseline_rows = by_model[baseline_model]
    baseline_clips = set(baseline_rows)
    common_clips_all = sorted(set.intersection(*(set(rows) for rows in by_model.values()))) if by_model else []

    per_model = {}
    comparisons = {}
    for model, rows in by_model.items():
        per_model[model] = _model_summary(rows)
        if model == baseline_model:
            continue
        common = sorted(baseline_clips & set(rows))
        comparisons[model] = _paired_comparison(
            baseline_rows,
            rows,
            common,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )

    return {
        "schema": "alpasim_transfer_matrix_analysis_v1",
        "matrix_dir": str(matrix_dir),
        "baseline_model": baseline_model,
        "baseline_label": MODEL_LABELS.get(baseline_model, baseline_model),
        "scene_preset": scene_preset,
        "models": list(sorted(by_model)),
        "started_model_count": len(by_model),
        "all_models_common_scene_count": len(common_clips_all),
        "all_models_common_clip_ids": common_clips_all,
        "per_model": per_model,
        "per_model_on_common_scenes": {
            model: _model_summary({clip_id: rows[clip_id] for clip_id in common_clips_all})
            for model, rows in by_model.items()
        },
        "comparisons_vs_baseline": comparisons,
    }


def _load_batch_rows(batch_dir: Path, *, max_dist_to_gt: float) -> dict[str, dict[str, Any]]:
    rows_by_clip: dict[str, dict[str, Any]] = {}
    for run_dir in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        metrics_path = run_dir / "aggregate" / "metrics_unprocessed.parquet"
        if not metrics_path.is_file():
            continue
        rows = summarize_run(run_dir, max_dist_to_gt=max_dist_to_gt)
        if len(rows) != 1:
            raise ValueError(f"Expected exactly one rollout row for {run_dir}, found {len(rows)}")
        row = rows[0]
        row.update(_metric_warning_flags(run_dir))
        clip_id = str(row["clipgt_id"])
        if clip_id in rows_by_clip:
            raise ValueError(f"Duplicate clipgt_id {clip_id} found in {batch_dir}")
        rows_by_clip[clip_id] = row
    return rows_by_clip


def main() -> None:
    args = _parse_args()
    report = analyze_matrix(
        args.matrix_dir,
        baseline_model=args.baseline_model,
        scene_preset=args.scene_preset,
        max_dist_to_gt=args.max_dist_to_gt,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(_cmd.json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown_report(report)
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
