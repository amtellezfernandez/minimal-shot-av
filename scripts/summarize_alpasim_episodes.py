from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


RAW_EVENT_METRICS = (
    "collision_any",
    "collision_front",
    "collision_lateral",
    "collision_rear",
    "offroad",
    "wrong_lane",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize per-episode AlpaSim outcomes from rollout metrics. "
            "This intentionally reports raw incident timing and route-deviation "
            "cutoffs, because aggregate AlpaSim metrics may truncate later events."
        )
    )
    parser.add_argument("run_dir", nargs="+", type=Path, help="AlpaSim run directory/directories.")
    parser.add_argument(
        "--max-dist-to-gt",
        type=float,
        default=4.0,
        help="Route-deviation cutoff used by AlpaSim aggregation modifiers.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv", "json"),
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def _metric_rows(metrics: pd.DataFrame, name: str) -> pd.DataFrame:
    return metrics[metrics["name"] == name].sort_values("timestamps_us")


def _first_time_s(metrics: pd.DataFrame, name: str, *, threshold: float = 0.0) -> float | None:
    rows = _metric_rows(metrics, name)
    if rows.empty:
        return None
    values = pd.to_numeric(rows["values"], errors="coerce")
    hits = rows[values > threshold]
    if hits.empty:
        return None
    return float((hits["timestamps_us"].iloc[0] - metrics["timestamps_us"].min()) / 1e6)


def _first_time_ge_s(metrics: pd.DataFrame, name: str, *, threshold: float) -> float | None:
    rows = _metric_rows(metrics, name)
    if rows.empty:
        return None
    values = pd.to_numeric(rows["values"], errors="coerce")
    hits = rows[values >= threshold]
    if hits.empty:
        return None
    return float((hits["timestamps_us"].iloc[0] - metrics["timestamps_us"].min()) / 1e6)


def _aggregate_metric(metrics: pd.DataFrame, name: str) -> float | None:
    rows = _metric_rows(metrics, name)
    if rows.empty:
        return None
    values = pd.to_numeric(rows["values"], errors="coerce")
    aggregation = str(rows["time_aggregation"].iloc[0])
    if aggregation == "max":
        return float(values.max())
    if aggregation == "mean":
        return float(values.mean())
    return float(values.iloc[-1])


def _event_label(candidates: list[tuple[str, float | None]]) -> str:
    present = [(name, t) for name, t in candidates if t is not None and math.isfinite(t)]
    if not present:
        return "none"
    name, seconds = min(present, key=lambda item: item[1])
    return f"{name}@{seconds:.1f}s"


def _controller_summary(run_dir: Path, rollout_id: str) -> dict[str, float | None]:
    matches = list((run_dir / "controller").glob(f"*{rollout_id}*.csv"))
    if not matches:
        return {
            "final_speed_mps": None,
            "min_speed_mps": None,
            "mean_speed_mps": None,
            "final_x": None,
            "final_y": None,
        }
    controller = pd.read_csv(matches[0])
    if "vx" not in controller or "vy" not in controller:
        return {
            "final_speed_mps": None,
            "min_speed_mps": None,
            "mean_speed_mps": None,
            "final_x": _last_float(controller, "x"),
            "final_y": _last_float(controller, "y"),
        }
    speed = (
        pd.to_numeric(controller["vx"], errors="coerce") ** 2
        + pd.to_numeric(controller["vy"], errors="coerce") ** 2
    ) ** 0.5
    return {
        "final_speed_mps": float(speed.iloc[-1]),
        "min_speed_mps": float(speed.min()),
        "mean_speed_mps": float(speed.mean()),
        "final_x": _last_float(controller, "x"),
        "final_y": _last_float(controller, "y"),
    }


def _last_float(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame or frame.empty:
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").iloc[-1])


def summarize_run(run_dir: Path, *, max_dist_to_gt: float) -> list[dict[str, Any]]:
    metrics_path = run_dir / "aggregate" / "metrics_unprocessed.parquet"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_parquet(metrics_path)
    rows: list[dict[str, Any]] = []
    for (clipgt_id, rollout_id), group in metrics.groupby(["clipgt_id", "rollout_id"], sort=True):
        collision_first_s = _first_time_s(group, "collision_any")
        offroad_first_s = _first_time_s(group, "offroad")
        dist_cutoff_s = _first_time_ge_s(group, "dist_to_gt_trajectory", threshold=max_dist_to_gt)
        first_raw_incident = _event_label(
            [
                ("collision", collision_first_s),
                ("offroad", offroad_first_s),
            ]
        )
        first_score_cutoff = _event_label(
            [
                (f"dist_to_gt>={max_dist_to_gt:g}m", dist_cutoff_s),
                ("collision", collision_first_s),
                ("offroad", offroad_first_s),
            ]
        )
        row: dict[str, Any] = {
            "run": run_dir.name,
            "clipgt_id": clipgt_id,
            "rollout_id": rollout_id,
            "metric_span_s": float((group["timestamps_us"].max() - group["timestamps_us"].min()) / 1e6),
            "first_score_cutoff": first_score_cutoff,
            "first_raw_incident": first_raw_incident,
            "dist_gt4_first_s": dist_cutoff_s,
            "collision_first_s": collision_first_s,
            "offroad_first_s": offroad_first_s,
            "raw_collision_any": _aggregate_metric(group, "collision_any"),
            "raw_offroad": _aggregate_metric(group, "offroad"),
            "raw_wrong_lane": _aggregate_metric(group, "wrong_lane"),
            "raw_dist_traveled_m": _aggregate_metric(group, "dist_traveled_m"),
            "raw_progress": _aggregate_metric(group, "progress"),
            "raw_progress_rel": _aggregate_metric(group, "progress_rel"),
            "raw_dist_to_gt_trajectory": _aggregate_metric(group, "dist_to_gt_trajectory"),
            "raw_plan_deviation": _aggregate_metric(group, "plan_deviation"),
        }
        for event_metric in RAW_EVENT_METRICS:
            row[f"{event_metric}_first_s"] = _first_time_s(group, event_metric)
        row.update(_controller_summary(run_dir, rollout_id))
        rows.append(row)
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    columns = [
        "run",
        "clipgt_id",
        "first_score_cutoff",
        "first_raw_incident",
        "raw_collision_any",
        "raw_offroad",
        "raw_wrong_lane",
        "raw_progress",
        "raw_dist_traveled_m",
        "raw_dist_to_gt_trajectory",
        "final_speed_mps",
        "mean_speed_mps",
    ]
    print("| " + " | ".join(columns) + " |")
    print("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        print("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |")


def main() -> None:
    args = _parse_args()
    rows: list[dict[str, Any]] = []
    for run_dir in args.run_dir:
        rows.extend(summarize_run(run_dir, max_dist_to_gt=args.max_dist_to_gt))
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        writer = csv.DictWriter(
            sys.stdout,
            fieldnames=sorted({key for row in rows for key in row}),
        )
        writer.writeheader()
        writer.writerows(rows)
        sys.stdout.flush()
    else:
        _print_markdown(rows)


if __name__ == "__main__":
    main()
