#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summarize_alpasim_episodes import summarize_run


BINARY_METRICS = (
    "raw_collision_any",
    "raw_offroad",
    "raw_wrong_lane",
)

CONTINUOUS_METRICS = (
    "raw_progress",
    "raw_dist_traveled_m",
    "raw_dist_to_gt_trajectory",
)

METRIC_VALIDITY_FLAGS = {
    "raw_offroad": "offroad_metric_valid",
    "raw_wrong_lane": "wrong_lane_metric_valid",
}

MODEL_LABELS = {
    "token_dagger_iter2": "raw_iter2",
    "token_dagger_iter2_clamped": "clamped_iter2",
    "token_dagger_iter2_axis_constrained_clamped": "axis_constrained_clamped",
    "token_dagger_iter2_axis_constrained_oracle_actor_clamped": "oracle_actor_axis_constrained_clamped",
    "token_dagger_iter2_axis_lexicographic_oracle_actor_clamped": "oracle_actor_axis_lexicographic_clamped",
    "token_dagger_iter2_actor_axis_oracle_actor_clamped": "actor_axis_oracle_actor_clamped",
    "token_dagger_iter2_hybrid_clamped": "hybrid_clamped",
    "token_dagger_srcdecay": "srcdecay",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run paired, scene-aligned transfer analysis on a completed AlpaSim matrix. "
            "Each variant is aligned by clipgt_id and compared against the baseline model."
        )
    )
    parser.add_argument("matrix_dir", type=Path, help="Completed AlpaSim transfer matrix directory.")
    parser.add_argument("--baseline-model", default="token_dagger_iter2")
    parser.add_argument(
        "--scene-preset",
        default=None,
        help="Optional preset suffix to analyze when the matrix dir contains multiple presets.",
    )
    parser.add_argument("--max-dist-to-gt", type=float, default=4.0)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-markdown", type=Path, default=None)
    return parser.parse_args()


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
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = _markdown_report(report)
    if args.output_markdown is not None:
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


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


def _discover_batches(matrix_dir: Path, *, scene_preset: str | None) -> dict[str, Path]:
    by_model: dict[str, list[tuple[str, Path]]] = {}
    for child in sorted(matrix_dir.iterdir()):
        if not child.is_dir():
            continue
        if "__" not in child.name:
            continue
        model, preset = child.name.split("__", 1)
        if model not in MODEL_LABELS and not (child / "batch-status.json").exists():
            continue
        by_model.setdefault(model, []).append((preset, child))
    batches: dict[str, Path] = {}
    for model, entries in by_model.items():
        if scene_preset is not None:
            matched = [path for preset, path in entries if preset == scene_preset]
            if matched:
                batches[model] = matched[0]
            continue
        if len(entries) > 1:
            presets = ", ".join(sorted(preset for preset, _ in entries))
            raise SystemExit(
                f"Multiple presets found for model {model}: {presets}. "
                "Pass --scene-preset to choose one."
            )
        batches[model] = entries[0][1]
    if not batches:
        raise SystemExit(f"No batch directories found in {matrix_dir}")
    return batches


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


def _model_summary(rows_by_clip: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows_by_clip.values())
    summary: dict[str, Any] = {
        "scene_count": len(rows),
        "metric_warning_counts": {
            "offroad_metric_invalid": sum(1 for row in rows if not row.get("offroad_metric_valid", True)),
            "wrong_lane_metric_invalid": sum(1 for row in rows if not row.get("wrong_lane_metric_valid", True)),
        },
    }
    for metric in BINARY_METRICS:
        valid_flag = METRIC_VALIDITY_FLAGS.get(metric)
        valid_rows = [row for row in rows if valid_flag is None or row.get(valid_flag, True)]
        hits = sum(1 for row in valid_rows if _truthy(row.get(metric)))
        summary[metric] = {
            "count": hits,
            "valid_count": len(valid_rows),
            "invalid_count": len(rows) - len(valid_rows),
            "rate": _safe_rate(hits, len(valid_rows)),
            "wilson95": _wilson_interval(hits, len(valid_rows)) if valid_rows else None,
        }
    for metric in CONTINUOUS_METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary[metric] = _numeric_summary(values)
    return summary


def _paired_comparison(
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
    common_clip_ids: list[str],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    report: dict[str, Any] = {"scene_count": len(common_clip_ids), "clip_ids": common_clip_ids}
    binary_report: dict[str, Any] = {}
    for metric in BINARY_METRICS:
        valid_flag = METRIC_VALIDITY_FLAGS.get(metric)
        metric_clip_ids = [
            clip_id
            for clip_id in common_clip_ids
            if valid_flag is None
            or (
                baseline_rows[clip_id].get(valid_flag, True)
                and candidate_rows[clip_id].get(valid_flag, True)
            )
        ]
        baseline_values = [_truthy(baseline_rows[clip_id].get(metric)) for clip_id in metric_clip_ids]
        candidate_values = [_truthy(candidate_rows[clip_id].get(metric)) for clip_id in metric_clip_ids]
        b01 = sum(1 for b, c in zip(baseline_values, candidate_values) if (not b) and c)
        b10 = sum(1 for b, c in zip(baseline_values, candidate_values) if b and (not c))
        baseline_rate = _safe_rate(sum(baseline_values), len(baseline_values))
        candidate_rate = _safe_rate(sum(candidate_values), len(candidate_values))
        binary_report[metric] = {
            "scene_count": len(metric_clip_ids),
            "invalid_scene_count": len(common_clip_ids) - len(metric_clip_ids),
            "baseline_rate": baseline_rate,
            "candidate_rate": candidate_rate,
            "delta_rate": None if baseline_rate is None or candidate_rate is None else candidate_rate - baseline_rate,
            "discordant_candidate_only": b01,
            "discordant_baseline_only": b10,
            "mcnemar_exact_p": _mcnemar_exact_p(b01, b10),
            "improved_scenes": b10,
            "worsened_scenes": b01,
        }
    continuous_report: dict[str, Any] = {}
    for metric in CONTINUOUS_METRICS:
        deltas = [
            float(candidate_rows[clip_id][metric]) - float(baseline_rows[clip_id][metric])
            for clip_id in common_clip_ids
            if baseline_rows[clip_id].get(metric) is not None and candidate_rows[clip_id].get(metric) is not None
        ]
        continuous_report[metric] = {
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "median_delta": statistics.median(deltas) if deltas else None,
            "bootstrap_mean95": _bootstrap_mean_interval(deltas, bootstrap_samples=bootstrap_samples, seed=seed),
            "positive_count": sum(1 for value in deltas if value > 0.0),
            "negative_count": sum(1 for value in deltas if value < 0.0),
            "sign_test_two_sided_p": _sign_test_two_sided_p(
                sum(1 for value in deltas if value > 0.0),
                sum(1 for value in deltas if value < 0.0),
            ),
        }
    report["binary_metrics"] = binary_report
    report["continuous_metrics"] = continuous_report
    report["axis_conflicts"] = _axis_conflicts(common_clip_ids, baseline_rows, candidate_rows)
    return report


def _axis_conflicts(
    clip_ids: list[str],
    baseline_rows: dict[str, dict[str, Any]],
    candidate_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    progress_up_collision_up = 0
    progress_up_offroad_up = 0
    wrong_lane_down_collision_flat_or_up = 0
    dist_to_gt_down_collision_up = 0
    progress_up_offroad_up_valid = 0
    wrong_lane_down_collision_flat_or_up_valid = 0
    for clip_id in clip_ids:
        base = baseline_rows[clip_id]
        cand = candidate_rows[clip_id]
        base_progress = float(base["raw_progress"])
        cand_progress = float(cand["raw_progress"])
        base_collision = _truthy(base["raw_collision_any"])
        cand_collision = _truthy(cand["raw_collision_any"])
        base_offroad = _truthy(base["raw_offroad"])
        cand_offroad = _truthy(cand["raw_offroad"])
        base_wrong_lane = _truthy(base["raw_wrong_lane"])
        cand_wrong_lane = _truthy(cand["raw_wrong_lane"])
        base_dist_gt = float(base["raw_dist_to_gt_trajectory"])
        cand_dist_gt = float(cand["raw_dist_to_gt_trajectory"])
        if cand_progress > base_progress and cand_collision > base_collision:
            progress_up_collision_up += 1
        offroad_valid = base.get("offroad_metric_valid", True) and cand.get("offroad_metric_valid", True)
        if offroad_valid:
            progress_up_offroad_up_valid += 1
            if cand_progress > base_progress and cand_offroad > base_offroad:
                progress_up_offroad_up += 1
        wrong_lane_valid = base.get("wrong_lane_metric_valid", True) and cand.get("wrong_lane_metric_valid", True)
        if wrong_lane_valid:
            wrong_lane_down_collision_flat_or_up_valid += 1
            if cand_wrong_lane < base_wrong_lane and cand_collision >= base_collision:
                wrong_lane_down_collision_flat_or_up += 1
        if cand_dist_gt < base_dist_gt and cand_collision > base_collision:
            dist_to_gt_down_collision_up += 1
    return {
        "progress_up_collision_up": progress_up_collision_up,
        "progress_up_offroad_up": progress_up_offroad_up,
        "progress_up_offroad_up_valid_scenes": progress_up_offroad_up_valid,
        "wrong_lane_down_collision_flat_or_up": wrong_lane_down_collision_flat_or_up,
        "wrong_lane_down_collision_flat_or_up_valid_scenes": wrong_lane_down_collision_flat_or_up_valid,
        "dist_to_gt_down_collision_up": dist_to_gt_down_collision_up,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    lines = []
    lines.append(
        f"Paired summary table uses the {report['all_models_common_scene_count']} scenes shared by the "
        f"{report['started_model_count']} started models."
    )
    lines.append("")
    lines.append(
        f"| Model | N | Collision | Offroad | Wrong lane | Progress | Dist. (m) | Dist.-GT |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in report["models"]:
        summary = report["per_model_on_common_scenes"][model]
        lines.append(
            "| "
            + " | ".join(
                [
                    MODEL_LABELS.get(model, model),
                    str(summary["scene_count"]),
                    _fmt_rate(summary["raw_collision_any"]["rate"]),
                    _fmt_rate_with_validity(summary["raw_offroad"]),
                    _fmt_rate_with_validity(summary["raw_wrong_lane"]),
                    _fmt_float(summary["raw_progress"]["mean"]),
                    _fmt_float(summary["raw_dist_traveled_m"]["mean"]),
                    _fmt_float(summary["raw_dist_to_gt_trajectory"]["mean"]),
                ]
            )
            + " |"
        )
    for model, comparison in report["comparisons_vs_baseline"].items():
        lines.append("")
        lines.append(f"### {MODEL_LABELS.get(model, model)} vs {report['baseline_label']}")
        lines.append("| Axis | Delta | Paired test |")
        lines.append("| --- | ---: | --- |")
        for metric in BINARY_METRICS:
            row = comparison["binary_metrics"][metric]
            lines.append(
                f"| {metric} | {_fmt_signed(row['delta_rate'])} | "
                f"n={row['scene_count']}"
                + (f", invalid={row['invalid_scene_count']}" if row["invalid_scene_count"] else "")
                + "; "
                f"McNemar p={_fmt_p(row['mcnemar_exact_p'])}; "
                f"better={row['improved_scenes']}, worse={row['worsened_scenes']} |"
            )
        for metric in CONTINUOUS_METRICS:
            row = comparison["continuous_metrics"][metric]
            ci = row["bootstrap_mean95"]
            ci_text = "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(
                f"| {metric} | {_fmt_signed(row['mean_delta'])} | "
                f"bootstrap95={ci_text}; sign p={_fmt_p(row['sign_test_two_sided_p'])} |"
            )
    return "\n".join(lines)


def _metric_warning_flags(run_dir: Path) -> dict[str, Any]:
    text_path = run_dir / "aggregate" / "metrics_results.txt"
    text = text_path.read_text(encoding="utf-8") if text_path.is_file() else ""
    offroad_missing = "No offroad column found in the metrics dataframe." in text
    return {
        "offroad_metric_valid": not offroad_missing,
        "wrong_lane_metric_valid": not offroad_missing,
    }


def _bootstrap_mean_interval(values: list[float], *, bootstrap_samples: int, seed: int) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    for _ in range(max(1, bootstrap_samples)):
        sample = [values[rng.randrange(len(values))] for _ in range(len(values))]
        means.append(statistics.fmean(sample))
    means.sort()
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _mcnemar_exact_p(b01: int, b10: int) -> float | None:
    n = b01 + b10
    if n <= 0:
        return None
    k = min(b01, b10)
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * cumulative)


def _sign_test_two_sided_p(positive: int, negative: int) -> float | None:
    n = positive + negative
    if n <= 0:
        return None
    k = min(positive, negative)
    cumulative = sum(math.comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2.0 * cumulative)


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None}
    ordered = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(ordered),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2.0 * total)
    spread = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total)
    return [(centre - spread) / denom, (centre + spread) / denom]


def _percentile(ordered: list[float], q: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    return float(value) > 0.0


def _safe_rate(successes: int, total: int) -> float | None:
    if total <= 0:
        return None
    return successes / total


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_rate_with_validity(summary: dict[str, Any]) -> str:
    rate = _fmt_rate(summary.get("rate"))
    invalid = int(summary.get("invalid_count", 0) or 0)
    valid = int(summary.get("valid_count", 0) or 0)
    if invalid <= 0:
        return rate
    return f"{rate} ({valid} valid, {invalid} invalid)"


def _fmt_float(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _fmt_signed(value: float | None) -> str:
    return "n/a" if value is None else f"{value:+.3f}"


def _fmt_p(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


if __name__ == "__main__":
    main()
