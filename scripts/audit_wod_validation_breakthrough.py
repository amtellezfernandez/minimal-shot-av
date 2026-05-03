#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "artifacts" / "wod_bestbase_crossfit_kingate_fastonly_rate020_rerun_ridge175_cv_local.json"
DEFAULT_CANDIDATE = ROOT / "artifacts" / "wod_breakthrough" / "promoted_report.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether a WOD validation-CV run clears the breakthrough bar.")
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--min-rfs-gain", type=float, default=0.1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_breakthrough(args.candidate, args.baseline, min_rfs_gain=args.min_rfs_gain)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


def audit_breakthrough(candidate_path: Path, baseline_path: Path, *, min_rfs_gain: float = 0.1) -> dict[str, Any]:
    if not candidate_path.is_file():
        baseline = _load_report(baseline_path)
        baseline_rfs = _metric(baseline, "combined_ranker_mean_rfs")
        baseline_normalized = _metric(baseline, "combined_ranker_mean_normalized_rfs")
        baseline_worst = _worst_slice(baseline)
        return {
            "schema": "wod_validation_breakthrough_audit_v1",
            "candidate": str(candidate_path),
            "baseline": str(baseline_path),
            "passed": False,
            "failures": [{"id": "candidate_report_missing", "path": str(candidate_path)}],
            "metrics": {
                "candidate_selected_rfs": 0.0,
                "baseline_selected_rfs": baseline_rfs,
                "selected_rfs_gain": -baseline_rfs,
                "candidate_normalized_rfs": 0.0,
                "baseline_normalized_rfs": baseline_normalized,
                "normalized_rfs_gain": -baseline_normalized,
                "candidate_worst_slice": None,
                "baseline_worst_slice": baseline_worst,
            },
            "claim_boundary": "validation_cv_only_not_hidden_test_not_production",
        }
    candidate_wrapper = _load_json(candidate_path)
    candidate = _load_report(candidate_path, payload=candidate_wrapper)
    baseline = _load_report(baseline_path)
    candidate_rfs = _metric(candidate, "combined_ranker_mean_rfs")
    baseline_rfs = _metric(baseline, "combined_ranker_mean_rfs")
    candidate_normalized = _metric(candidate, "combined_ranker_mean_normalized_rfs")
    baseline_normalized = _metric(baseline, "combined_ranker_mean_normalized_rfs")
    candidate_worst = _worst_slice(candidate)
    baseline_worst = _worst_slice(baseline)
    gain = candidate_rfs - baseline_rfs
    normalized_gain = candidate_normalized - baseline_normalized
    failures: list[dict[str, Any]] = []
    failures.extend(_wrapper_failures(candidate_wrapper))
    if candidate.get("benchmark_type") != "segment_grouped_cross_validation":
        failures.append({"id": "not_segment_grouped_cv", "detail": "candidate is not segment-grouped CV"})
    comparable_failures = _comparability_failures(candidate, baseline)
    failures.extend(comparable_failures)
    if bool(candidate.get("hidden_test_confirmed", False)):
        failures.append({"id": "hidden_test_claim", "detail": "candidate report contains a hidden-test claim"})
    if gain < float(min_rfs_gain):
        failures.append(
            {
                "id": "selected_rfs_gain_too_small",
                "candidate": candidate_rfs,
                "baseline": baseline_rfs,
                "gain": gain,
                "target": float(min_rfs_gain),
            }
        )
    if normalized_gain <= 0.0:
        failures.append(
            {
                "id": "normalized_rfs_not_improved",
                "candidate": candidate_normalized,
                "baseline": baseline_normalized,
            }
        )
    if candidate_worst is None or baseline_worst is None:
        failures.append({"id": "missing_worst_slice", "detail": "candidate or baseline has no slice diagnostics"})
    elif candidate_worst["mean_regret"] > baseline_worst["mean_regret"] + 1e-9:
        failures.append(
            {
                "id": "worst_slice_regret_worse",
                "candidate": candidate_worst,
                "baseline": baseline_worst,
            }
        )
    return {
        "schema": "wod_validation_breakthrough_audit_v1",
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
        "passed": not failures,
        "failures": failures,
        "metrics": {
            "candidate_selected_rfs": candidate_rfs,
            "baseline_selected_rfs": baseline_rfs,
            "selected_rfs_gain": gain,
            "candidate_normalized_rfs": candidate_normalized,
            "baseline_normalized_rfs": baseline_normalized,
            "normalized_rfs_gain": normalized_gain,
            "candidate_worst_slice": candidate_worst,
            "baseline_worst_slice": baseline_worst,
        },
        "claim_boundary": "validation_cv_only_not_hidden_test_not_production",
    }


def _load_report(path: Path, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = _load_json(path) if payload is None else payload
    nested_path = payload.get("best_report_path") or payload.get("report_path")
    if isinstance(nested_path, str) and nested_path:
        nested = Path(nested_path)
        if not nested.is_absolute():
            nested = path.parent / nested
        return _load_report(nested)
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _metric(report: dict[str, Any], key: str) -> float:
    value = report.get(key)
    return float(value) if isinstance(value, (int, float)) else 0.0


def _worst_slice(report: dict[str, Any]) -> dict[str, Any] | None:
    slices = report.get("slices", {})
    if not isinstance(slices, dict) or not slices:
        return None
    candidates: list[dict[str, Any]] = []
    for name, payload in slices.items():
        if not isinstance(payload, dict) or "mean_regret" not in payload:
            continue
        candidates.append(
            {
                "slice": str(name),
                "frames": int(payload.get("frames", 0)),
                "mean_regret": float(payload["mean_regret"]),
            }
        )
    if not candidates:
        return None
    return max(candidates, key=lambda row: (row["mean_regret"], row["slice"]))


def _wrapper_failures(wrapper: dict[str, Any]) -> list[dict[str, Any]]:
    nested_path = wrapper.get("best_report_path") or wrapper.get("report_path")
    if not isinstance(nested_path, str) or not nested_path:
        return []
    schema = wrapper.get("schema")
    role = wrapper.get("report_role")
    promoted = wrapper.get("promoted")
    if schema != "wod_validation_breakthrough_candidate_report_v1" or role != "validation_cv_candidate":
        return [
            {
                "id": "candidate_wrapper_not_validation_cv_candidate",
                "schema": schema,
                "report_role": role,
                "detail": "candidate wrapper is not a full validation-CV breakthrough candidate report",
            }
        ]
    if promoted is not True:
        return [
            {
                "id": "candidate_not_promoted",
                "detail": "candidate wrapper did not clear the validation-CV promotion gate",
            }
        ]
    return []


def _comparability_failures(candidate: dict[str, Any], baseline: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for key in ("frames", "fold_count", "score_backend"):
        candidate_value = candidate.get(key)
        baseline_value = baseline.get(key)
        if candidate_value is None or baseline_value is None:
            continue
        if candidate_value != baseline_value:
            failures.append(
                {
                    "id": "not_comparable_cv_protocol",
                    "field": key,
                    "candidate": candidate_value,
                    "baseline": baseline_value,
                }
            )
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
