#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.analyze_candidate_recoverable_regret import read_json_or_jsonl
from minimal_shot_av.cli.commands.validate_candidate_generator_dump import validate_candidate_generator_dump


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "candidate_sets_with_metrics.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "candidate_metric_join_report.json"
DEFAULT_REPORT_MD = ROOT / "artifacts" / "corl2027" / "candidate_metric_join_report.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach evaluator metrics to candidate_set_v1 dumps. Metrics can come from NAVSIM, replay, "
            "or any evaluator that emits scene_id/candidate_id rows."
        )
    )
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidate_payload = read_json_or_jsonl(args.candidate_json)
    metrics_payload = read_json_or_jsonl(args.metrics_json)
    enriched, report = attach_candidate_metrics(
        candidate_payload,
        metrics_payload,
        candidate_json=str(args.candidate_json),
        metrics_json=str(args.metrics_json),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_report(report) + "\n", encoding="utf-8")
    print(markdown_report(report))
    return 0


def attach_candidate_metrics(
    candidate_payload: Any,
    metrics_payload: Any,
    *,
    candidate_json: str,
    metrics_json: str,
    near_miss_threshold_m: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_sets = extract_candidate_sets(candidate_payload)
    metric_rows = extract_metric_rows(metrics_payload, near_miss_threshold_m=near_miss_threshold_m)
    metric_by_key = {(row["scene_id"], row["candidate_id"]): row for row in metric_rows}
    matched = 0
    missing = 0
    duplicate_counter = duplicate_metric_counter(metric_rows)
    enriched_sets = []
    for candidate_set in candidate_sets:
        copied_set = json.loads(json.dumps(candidate_set))
        scene_id = str(copied_set.get("scene_id", ""))
        for candidate in copied_set.get("candidates", []):
            candidate_id = str(candidate.get("candidate_id", ""))
            row = metric_by_key.get((scene_id, candidate_id))
            if row is None:
                missing += 1
                continue
            candidate_metrics = dict(candidate.get("metrics", {}))
            candidate_metrics.update(row["metrics"])
            candidate["metrics"] = candidate_metrics
            matched += 1
        enriched_sets.append(copied_set)
    enriched = {
        "schema": "candidate_set_v1",
        "source_format": source_format(candidate_payload),
        "candidate_sets": enriched_sets,
    }
    validation = validate_candidate_generator_dump(enriched_sets, input_json=candidate_json)
    report = {
        "schema": "candidate_metric_join_report_v1",
        "candidate_json": candidate_json,
        "metrics_json": metrics_json,
        "scene_count": len(enriched_sets),
        "candidate_count": sum(len(row.get("candidates", [])) for row in enriched_sets),
        "metric_row_count": len(metric_rows),
        "matched_candidate_count": matched,
        "missing_metric_candidate_count": missing,
        "duplicate_metric_key_count": sum(1 for count in duplicate_counter.values() if count > 1),
        "metric_name_histogram": metric_name_histogram(metric_rows),
        "validation": validation,
    }
    return enriched, report


def extract_candidate_sets(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and "candidate_sets" in payload:
        return [dict(row) for row in payload["candidate_sets"]]
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    raise ValueError("candidate input must be candidate_set_v1 JSON or JSONL/list candidate sets")


def source_format(payload: Any) -> str:
    if isinstance(payload, Mapping):
        return str(payload.get("source_format") or payload.get("schema") or "candidate_set_v1")
    return "candidate_set_jsonl"


def extract_metric_rows(payload: Any, *, near_miss_threshold_m: float) -> list[dict[str, Any]]:
    raw_rows: list[Any]
    if isinstance(payload, Mapping) and "metrics" in payload:
        raw_rows = list(payload["metrics"])
    elif isinstance(payload, Mapping) and "rows" in payload:
        raw_rows = list(payload["rows"])
    elif isinstance(payload, list):
        raw_rows = payload
    else:
        raise ValueError("metrics input must be JSONL/list or contain `metrics`/`rows`")
    rows = [normalize_metric_row(row, near_miss_threshold_m=near_miss_threshold_m) for row in raw_rows]
    return [row for row in rows if row is not None]


def normalize_metric_row(row: Mapping[str, Any], *, near_miss_threshold_m: float) -> dict[str, Any] | None:
    scene_id = row.get("scene_id")
    candidate_id = row.get("candidate_id", row.get("id", row.get("token")))
    if scene_id is None or candidate_id is None:
        return None
    metrics = dict(row.get("metrics", {}))
    for source_key, target_key in (
        ("progress_m", "progress_m"),
        ("final_progress_m", "progress_m"),
        ("route_progress_m", "progress_m"),
        ("ade_3s_m", "ade_3s_m"),
        ("ade_m", "ade_3s_m"),
        ("fde_3s_m", "fde_3s_m"),
        ("fde_m", "fde_3s_m"),
        ("clearance_m", "clearance_m"),
        ("realized_min_clearance_m", "clearance_m"),
        ("min_clearance_m", "clearance_m"),
        ("pdms", "pdms"),
        ("epdms", "epdms"),
        ("score", "raw_score"),
        ("metric_score", "raw_score"),
    ):
        if source_key in row and row[source_key] is not None:
            metrics[target_key] = row[source_key]
    replay_fail = row.get("replay_fail", row.get("replay_infeasible", row.get("failed")))
    if replay_fail is None and metrics.get("clearance_m") is not None:
        replay_fail = float(metrics["clearance_m"]) < float(near_miss_threshold_m)
    if replay_fail is not None:
        metrics["replay_fail"] = bool(replay_fail)
    return {
        "scene_id": str(scene_id),
        "candidate_id": str(candidate_id),
        "metrics": normalize_metric_values(metrics),
    }


def normalize_metric_values(metrics: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {}
    for key, value in metrics.items():
        if isinstance(value, bool) or value is None:
            normalized[str(key)] = value
            continue
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            normalized[str(key)] = value
    return normalized


def duplicate_metric_counter(rows: list[Mapping[str, Any]]) -> Counter[tuple[str, str]]:
    counter: Counter[tuple[str, str]] = Counter()
    for row in rows:
        counter.update([(str(row["scene_id"]), str(row["candidate_id"]))])
    return counter


def metric_name_histogram(rows: list[Mapping[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(str(key) for key in row.get("metrics", {}))
    return dict(sorted(counter.items()))


def markdown_report(report: Mapping[str, Any]) -> str:
    validation = report["validation"]
    lines = [
        "# Candidate Metric Join",
        "",
        f"- Candidate JSON: `{report['candidate_json']}`",
        f"- Metrics JSON: `{report['metrics_json']}`",
        f"- Readiness after join: `{validation['readiness']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| scenes | {report['scene_count']} |",
        f"| candidates | {report['candidate_count']} |",
        f"| metric rows | {report['metric_row_count']} |",
        f"| matched candidates | {report['matched_candidate_count']} |",
        f"| missing metrics | {report['missing_metric_candidate_count']} |",
        f"| duplicate metric keys | {report['duplicate_metric_key_count']} |",
        "",
        f"- Metric names: `{json.dumps(report['metric_name_histogram'], sort_keys=True)}`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
