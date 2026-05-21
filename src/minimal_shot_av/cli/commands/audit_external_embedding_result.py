#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BASELINE = ROOT / "benchmarks" / "current" / "wod_ridge_trajectory_cv_official.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate an external-embedding WOD CV result against the baseline.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--min-rfs-gain", type=float, default=0.1)
    parser.add_argument(
        "--slice-regret-tolerance",
        type=float,
        default=0.0,
        help="Allowed increase in worst-slice regret versus baseline.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_external_embedding_result(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        min_rfs_gain=args.min_rfs_gain,
        slice_regret_tolerance=args.slice_regret_tolerance,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


def audit_external_embedding_result(
    *,
    candidate_path: Path,
    baseline_path: Path,
    min_rfs_gain: float = 0.1,
    slice_regret_tolerance: float = 0.0,
) -> dict[str, Any]:
    candidate = _load_report_with_slices(candidate_path)
    baseline = _load_report_with_slices(baseline_path)
    candidate_rfs = _metric(candidate, "combined_with_world_ranker_mean_rfs", "combined_ranker_mean_rfs")
    baseline_rfs = _metric(baseline, "combined_ranker_mean_rfs")
    gain = candidate_rfs - baseline_rfs
    candidate_worst = _worst_slice_regret(candidate)
    baseline_worst = _worst_slice_regret(baseline)
    world_diagnostics = _world_candidate_diagnostics(candidate)
    failures: list[dict[str, str]] = []
    checks: list[dict[str, object]] = []

    if str(candidate.get("world_model", _metadata(candidate).get("world_model", ""))) != "external_embeddings":
        failures.append(
            {"id": "not_external_embeddings", "detail": "Candidate report is not external_embeddings mode."}
        )
    if "embedding_source" not in candidate and "embedding_source" not in _metadata(candidate):
        failures.append(
            {"id": "missing_embedding_source", "detail": "Candidate report does not identify embedding_source."}
        )
    if gain < min_rfs_gain:
        failures.append(
            {
                "id": "insufficient_rfs_gain",
                "detail": f"RFS gain {gain:.4f} is below required +{min_rfs_gain:.4f}.",
            }
        )
    if candidate_worst is None or baseline_worst is None:
        failures.append({"id": "missing_slice_regret", "detail": "Candidate or baseline is missing slice diagnostics."})
    elif candidate_worst["mean_regret"] > baseline_worst["mean_regret"] + slice_regret_tolerance:
        failures.append(
            {
                "id": "worst_slice_regret_worse",
                "detail": (
                    f"Worst-slice regret {candidate_worst['mean_regret']:.4f} exceeds baseline "
                    f"{baseline_worst['mean_regret']:.4f}."
                ),
            }
        )

    checks.append(
        {
            "id": "selected_rfs_gain",
            "candidate": candidate_rfs,
            "baseline": baseline_rfs,
            "gain": gain,
            "required_gain": min_rfs_gain,
        }
    )
    checks.append(
        {
            "id": "worst_slice_regret",
            "candidate": candidate_worst,
            "baseline": baseline_worst,
            "tolerance": slice_regret_tolerance,
        }
    )
    checks.append({"id": "world_candidate_diagnostics", **world_diagnostics})
    return {
        "audit_type": "external_embedding_result_gate",
        "status": "pass" if not failures else "fail",
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
        "checks": checks,
        "failures": failures,
    }


def _load_report_with_slices(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(report.get("slices"), dict):
        return report
    source = report.get("source")
    if isinstance(source, str):
        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if source_path.exists():
            source_report = json.loads(source_path.read_text(encoding="utf-8"))
            if isinstance(source_report.get("slices"), dict):
                merged = dict(report)
                merged["slices"] = source_report["slices"]
                return merged
    return report


def _metric(report: dict[str, Any], *names: str) -> float:
    metrics = report.get("metrics", {})
    for name in names:
        if name in report and isinstance(report[name], int | float):
            return float(report[name])
        if isinstance(metrics, dict) and name in metrics:
            value = metrics[name]
            if isinstance(value, dict):
                return float(value["value"])
            return float(value)
    raise KeyError(f"report is missing metric {names[0]!r}")


def _metadata(report: dict[str, Any]) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    return metadata if isinstance(metadata, dict) else {}


def _worst_slice_regret(report: dict[str, Any]) -> dict[str, Any] | None:
    slices = report.get("slices", {})
    if not isinstance(slices, dict) or not slices:
        return None
    rows = [
        {
            "slice": str(name),
            "frames": int(data["frames"]),
            "mean_regret": float(data["mean_regret"]),
        }
        for name, data in slices.items()
        if isinstance(data, dict) and int(data.get("frames", 0)) > 0
    ]
    if not rows:
        return None
    return max(rows, key=lambda row: (row["mean_regret"], row["slice"]))


def _world_candidate_diagnostics(report: dict[str, Any]) -> dict[str, Any]:
    slices = report.get("slices", {})
    if not isinstance(slices, dict):
        return {
            "selected_world_rate": _optional_float(report.get("selected_world_rate")),
            "oracle_world_rate": _optional_float(report.get("oracle_world_rate")),
            "slice_count": 0,
            "top_world_oracle_slices": [],
            "top_world_underselection_slices": [],
        }
    rows: list[dict[str, Any]] = []
    for name, raw_slice in slices.items():
        if not isinstance(raw_slice, dict) or int(raw_slice.get("frames", 0)) <= 0:
            continue
        selected_world = _source_rate(raw_slice, "selected_source_rates", "world")
        oracle_world = _source_rate(raw_slice, "oracle_source_rates", "world")
        rows.append(
            {
                "slice": str(name),
                "frames": int(raw_slice["frames"]),
                "selected_world_rate": selected_world,
                "oracle_world_rate": oracle_world,
                "world_underselection_rate": max(0.0, oracle_world - selected_world),
                "mean_regret": float(raw_slice.get("mean_regret", 0.0)),
            }
        )
    return {
        "selected_world_rate": _optional_float(report.get("selected_world_rate")),
        "oracle_world_rate": _optional_float(report.get("oracle_world_rate")),
        "slice_count": len(rows),
        "top_world_oracle_slices": sorted(
            rows,
            key=lambda row: (row["oracle_world_rate"], row["mean_regret"], row["slice"]),
            reverse=True,
        )[:5],
        "top_world_underselection_slices": sorted(
            rows,
            key=lambda row: (row["world_underselection_rate"], row["mean_regret"], row["slice"]),
            reverse=True,
        )[:5],
    }


def _source_rate(slice_report: dict[str, Any], field: str, source: str) -> float:
    rates = slice_report.get(field, {})
    if not isinstance(rates, dict):
        return 0.0
    return _optional_float(rates.get(source)) or 0.0


def _optional_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
