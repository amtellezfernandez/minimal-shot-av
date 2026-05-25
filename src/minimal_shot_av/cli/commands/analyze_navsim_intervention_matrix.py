#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "navsim_intervention_matrix.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "navsim_intervention_matrix.md"

BINARY_METRIC_ALIASES = {
    "collision": (
        "collision",
        "collision_any",
        "has_collision",
        "at_fault_collision",
        "no_at_fault_collisions",
        "nc",
    ),
    "offroad": (
        "offroad",
        "offroad_any",
        "drivable_area_violation",
        "drivable_area_compliance",
        "dac",
    ),
}

CONTINUOUS_METRIC_ALIASES = {
    "progress": ("progress", "ego_progress", "ep"),
    "ttc": ("ttc", "time_to_collision", "time_to_collision_within_bound"),
    "score": ("score", "pdm_score", "pdms", "epdms"),
}

LOWER_IS_BETTER = {"collision", "offroad"}
HIGHER_IS_BETTER = {"progress", "ttc", "score"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a NAVSIM public-benchmark intervention matrix for per-axis non-co-monotonicity. "
            "Inputs are per-scene NAVSIM metric CSV/JSON files for baseline and intervention variants."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Named NAVSIM metric file. Pass at least two, e.g. raw=raw.csv clamped=clamped.csv.",
    )
    parser.add_argument("--baseline", required=True, help="Baseline run name from --run.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runs = _parse_run_args(args.run)
    report = analyze_matrix(runs, baseline=args.baseline)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def analyze_matrix(runs: dict[str, Path], *, baseline: str) -> dict[str, Any]:
    if baseline not in runs:
        raise ValueError(f"baseline {baseline!r} not found in runs: {sorted(runs)}")
    per_run = {name: _load_metric_rows(path) for name, path in sorted(runs.items())}
    baseline_rows = per_run[baseline]
    common_scene_ids = sorted(set.intersection(*(set(rows) for rows in per_run.values())))
    per_model = {
        name: _model_summary(rows, scene_ids=common_scene_ids)
        for name, rows in sorted(per_run.items())
    }
    comparisons = {}
    for name, rows in sorted(per_run.items()):
        if name == baseline:
            continue
        comparisons[name] = _paired_comparison(baseline_rows, rows, common_scene_ids)
    return {
        "schema": "navsim_intervention_matrix_v1",
        "benchmark_scope": (
            "NAVSIM public non-reactive/pseudo-closed-loop metric surface. This supports "
            "public verification of per-axis non-co-monotonicity, not controller-proxy mismatch."
        ),
        "baseline": baseline,
        "run_paths": {name: str(path) for name, path in sorted(runs.items())},
        "common_scene_count": len(common_scene_ids),
        "per_model": per_model,
        "comparisons_vs_baseline": comparisons,
        "non_co_monotone_intervention_count": sum(
            1 for comparison in comparisons.values() if comparison["axis_delta"]["class"] == "tradeoff"
        ),
        "limitations": [
            "NAVSIM is not a fully reactive closed-loop controller-proxy surface.",
            (
                "Use this report for public per-axis intervention validation; "
                "keep AlpaSim for controller-proxy localization."
            ),
        ],
    }


def _paired_comparison(
    baseline_rows: dict[str, dict[str, float]],
    candidate_rows: dict[str, dict[str, float]],
    scene_ids: list[str],
) -> dict[str, Any]:
    axis_delta = _axis_delta(
        _aggregate_metrics(baseline_rows, scene_ids=scene_ids),
        _aggregate_metrics(candidate_rows, scene_ids=scene_ids),
    )
    conflict_counts = {
        "progress_up_collision_up": 0,
        "progress_up_offroad_up": 0,
        "offroad_down_collision_up": 0,
        "score_up_collision_up": 0,
    }
    paired_rows = []
    for scene_id in scene_ids:
        base = baseline_rows[scene_id]
        cand = candidate_rows[scene_id]
        row = {
            "scene_id": scene_id,
            "delta_collision": _metric_delta(base, cand, "collision"),
            "delta_offroad": _metric_delta(base, cand, "offroad"),
            "delta_progress": _metric_delta(base, cand, "progress"),
            "delta_score": _metric_delta(base, cand, "score"),
        }
        if _positive(row["delta_progress"]) and _positive(row["delta_collision"]):
            conflict_counts["progress_up_collision_up"] += 1
        if _positive(row["delta_progress"]) and _positive(row["delta_offroad"]):
            conflict_counts["progress_up_offroad_up"] += 1
        if _negative(row["delta_offroad"]) and _positive(row["delta_collision"]):
            conflict_counts["offroad_down_collision_up"] += 1
        if _positive(row["delta_score"]) and _positive(row["delta_collision"]):
            conflict_counts["score_up_collision_up"] += 1
        paired_rows.append(row)
    return {
        "scene_count": len(scene_ids),
        "axis_delta": axis_delta,
        "axis_conflict_counts": conflict_counts,
        "paired_delta_summary": {
            "collision": _distribution([row["delta_collision"] for row in paired_rows]),
            "offroad": _distribution([row["delta_offroad"] for row in paired_rows]),
            "progress": _distribution([row["delta_progress"] for row in paired_rows]),
            "score": _distribution([row["delta_score"] for row in paired_rows]),
        },
    }


def _axis_delta(baseline: dict[str, float | None], candidate: dict[str, float | None]) -> dict[str, Any]:
    improved = []
    worsened = []
    unchanged = []
    deltas = {}
    for metric in ("collision", "offroad", "progress", "ttc", "score"):
        base = baseline.get(metric)
        cand = candidate.get(metric)
        if base is None or cand is None:
            continue
        signed = cand - base
        deltas[metric] = signed
        if metric in LOWER_IS_BETTER:
            signed = -signed
        if signed > 1e-9:
            improved.append(metric)
        elif signed < -1e-9:
            worsened.append(metric)
        else:
            unchanged.append(metric)
    if improved and worsened:
        klass = "tradeoff"
    elif improved:
        klass = "dominates"
    elif worsened:
        klass = "regresses"
    else:
        klass = "unchanged"
    return {
        "class": klass,
        "raw_candidate_minus_baseline": deltas,
        "improved_axes": improved,
        "worsened_axes": worsened,
        "unchanged_axes": unchanged,
    }


def _model_summary(rows: dict[str, dict[str, float]], *, scene_ids: list[str]) -> dict[str, Any]:
    aggregates = _aggregate_metrics(rows, scene_ids=scene_ids)
    return {"scene_count": len(scene_ids), "metrics": aggregates}


def _aggregate_metrics(
    rows: dict[str, dict[str, float]],
    *,
    scene_ids: list[str],
) -> dict[str, float | None]:
    output = {}
    for metric in ("collision", "offroad", "progress", "ttc", "score"):
        values = [rows[scene_id].get(metric) for scene_id in scene_ids]
        values = [value for value in values if value is not None and math.isfinite(value)]
        output[metric] = None if not values else statistics.fmean(values)
    return output


def _load_metric_rows(path: Path) -> dict[str, dict[str, float]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = _rows_from_json_payload(payload)
    else:
        rows = _rows_from_csv(path)
    normalized = {}
    for index, row in enumerate(rows):
        scene_id = _scene_id(row, default=f"row_{index:06d}")
        normalized[scene_id] = _normalize_metrics(row)
    return normalized


def _rows_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _rows_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "results", "scenes", "metrics"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        if all(isinstance(value, dict) for value in payload.values()):
            rows = []
            for key, value in payload.items():
                row = dict(value)
                row.setdefault("scene_id", key)
                rows.append(row)
            return rows
    raise ValueError("unsupported NAVSIM metric JSON format")


def _normalize_metrics(row: dict[str, Any]) -> dict[str, float]:
    normalized = {}
    lower = {str(key).lower(): value for key, value in row.items()}
    for metric, aliases in BINARY_METRIC_ALIASES.items():
        value = _first_value(lower, aliases)
        if value is None:
            continue
        parsed = _as_float(value)
        if parsed is None:
            continue
        if metric == "collision" and _matched_alias(lower, aliases) in {"no_at_fault_collisions", "nc"}:
            parsed = 1.0 - parsed
        if metric == "offroad" and _matched_alias(lower, aliases) in {"drivable_area_compliance", "dac"}:
            parsed = 1.0 - parsed
        normalized[metric] = parsed
    for metric, aliases in CONTINUOUS_METRIC_ALIASES.items():
        value = _first_value(lower, aliases)
        parsed = _as_float(value)
        if parsed is not None:
            normalized[metric] = parsed
    return normalized


def _scene_id(row: dict[str, Any], *, default: str) -> str:
    for key in ("scene_id", "scene_token", "token", "initial_token", "id", "log_name"):
        if key in row and row[key]:
            return str(row[key])
    for key, value in row.items():
        if str(key).lower() in {"scene_id", "scene_token", "token", "initial_token"} and value:
            return str(value)
    return default


def _metric_delta(base: dict[str, float], cand: dict[str, float], metric: str) -> float | None:
    if metric not in base or metric not in cand:
        return None
    return cand[metric] - base[metric]


def _distribution(values: list[float | None]) -> dict[str, Any]:
    parsed = [value for value in values if value is not None and math.isfinite(value)]
    if not parsed:
        return {"count": 0}
    return {
        "count": len(parsed),
        "mean": statistics.fmean(parsed),
        "median": statistics.median(parsed),
        "min": min(parsed),
        "max": max(parsed),
        "positive_count": sum(value > 1e-9 for value in parsed),
        "negative_count": sum(value < -1e-9 for value in parsed),
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# NAVSIM intervention matrix",
        "",
        report["benchmark_scope"],
        "",
        f"Common scenes: `{report['common_scene_count']}`",
        f"Non-co-monotone interventions: `{report['non_co_monotone_intervention_count']}`",
        "",
        "## Per-Model Metrics",
        "",
        "| Model | Scenes | Collision | Offroad | Progress | TTC | Score |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["per_model"].items():
        metrics = row["metrics"]
        lines.append(
            f"| {name} | {row['scene_count']} | {_fmt(metrics.get('collision'))} | "
            f"{_fmt(metrics.get('offroad'))} | {_fmt(metrics.get('progress'))} | "
            f"{_fmt(metrics.get('ttc'))} | {_fmt(metrics.get('score'))} |"
        )
    lines.extend(["", "## Comparisons Vs Baseline", ""])
    for name, comparison in report["comparisons_vs_baseline"].items():
        axis = comparison["axis_delta"]
        lines.append(
            f"- `{name}`: `{axis['class']}`; improved={axis['improved_axes']}; "
            f"worsened={axis['worsened_axes']}; conflicts={comparison['axis_conflict_counts']}"
        )
    lines.extend(["", "## Limitation", ""])
    for limitation in report["limitations"]:
        lines.append(f"- {limitation}")
    return "\n".join(lines)


def _parse_run_args(items: list[str]) -> dict[str, Path]:
    runs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--run must be NAME=PATH, got {item!r}")
        name, raw_path = item.split("=", 1)
        if not name:
            raise ValueError(f"empty run name in {item!r}")
        runs[name] = Path(raw_path)
    if len(runs) < 2:
        raise ValueError("at least two --run entries are required")
    return runs


def _first_value(row: dict[str, Any], aliases: tuple[str, ...]) -> Any | None:
    alias = _matched_alias(row, aliases)
    return None if alias is None else row.get(alias)


def _matched_alias(row: dict[str, Any], aliases: tuple[str, ...]) -> str | None:
    for alias in aliases:
        if alias in row:
            return alias
    return None


def _as_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive(value: float | None) -> bool:
    return value is not None and value > 1e-9


def _negative(value: float | None) -> bool:
    return value is not None and value < -1e-9


def _fmt(value: Any) -> str:
    parsed = _as_float(value)
    return "NA" if parsed is None else f"{parsed:.4f}"


if __name__ == "__main__":
    main()
