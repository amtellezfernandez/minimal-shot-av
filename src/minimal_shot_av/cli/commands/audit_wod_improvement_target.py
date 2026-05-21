#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit whether a WOD CV report can meet a target RFS improvement.")
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--target-relative-gain", type=float, default=0.25)
    parser.add_argument("--max-worst-slice-regret-increase", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_improvement_target(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        target_relative_gain=args.target_relative_gain,
        max_worst_slice_regret_increase=args.max_worst_slice_regret_increase,
    )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 1


def audit_improvement_target(
    *,
    candidate_path: Path,
    baseline_path: Path,
    target_relative_gain: float,
    max_worst_slice_regret_increase: float = 0.0,
) -> dict[str, Any]:
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_score = _metric(baseline, "combined_ranker_mean_rfs")
    candidate_score = _metric(candidate, "combined_ranker_mean_rfs")
    oracle_score = _metric(candidate, "combined_oracle_mean_rfs")
    failures: list[dict[str, str]] = []
    if baseline_score <= 0.0:
        failures.append(
            {
                "id": "invalid_baseline_score",
                "detail": f"Baseline score must be positive for a relative gain target, got {baseline_score:.6f}.",
            }
        )
        required_score = float("inf")
    else:
        required_score = baseline_score * (1.0 + target_relative_gain)
    target_gap = required_score - candidate_score
    oracle_gap = required_score - oracle_score
    candidate_worst = _worst_slice(candidate)
    baseline_worst = _worst_slice(baseline)

    if candidate_score < required_score:
        failures.append(
            {
                "id": "target_not_met",
                "detail": f"Candidate score {candidate_score:.6f} is below required {required_score:.6f}.",
            }
        )
    if oracle_score < required_score:
        failures.append(
            {
                "id": "target_above_candidate_oracle",
                "detail": f"Current candidate oracle {oracle_score:.6f} is below required {required_score:.6f}.",
            }
        )
    if candidate_worst is None or baseline_worst is None:
        failures.append(
            {
                "id": "missing_slice_diagnostics",
                "detail": "Both candidate and baseline must include slice diagnostics for non-biased target audit.",
            }
        )
    else:
        allowed = baseline_worst["mean_regret"] + max_worst_slice_regret_increase
        if candidate_worst["mean_regret"] > allowed:
            failures.append(
                {
                    "id": "worst_slice_regret_not_non_biased",
                    "detail": (
                        f"Candidate worst-slice regret {candidate_worst['mean_regret']:.6f} "
                        f"exceeds allowed {allowed:.6f}."
                    ),
                }
            )

    return {
        "audit_type": "wod_improvement_target",
        "status": "pass" if not failures else "fail",
        "candidate": str(candidate_path),
        "baseline": str(baseline_path),
        "target_relative_gain": target_relative_gain,
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "candidate_relative_gain": _relative_gain(candidate_score, baseline_score),
        "candidate_oracle_score": oracle_score,
        "candidate_oracle_relative_gain": _relative_gain(oracle_score, baseline_score),
        "required_score": required_score,
        "target_gap": target_gap,
        "oracle_gap": oracle_gap,
        "baseline_worst_slice": baseline_worst,
        "candidate_worst_slice": candidate_worst,
        "failures": failures,
    }


def _metric(report: dict[str, Any], name: str) -> float:
    if isinstance(report.get(name), (int, float)) and not isinstance(report.get(name), bool):
        return float(report[name])
    metrics = report.get("metrics", {})
    if isinstance(metrics, dict) and name in metrics:
        raw = metrics[name]
        if isinstance(raw, dict):
            return float(raw["value"])
        return float(raw)
    raise KeyError(f"missing metric {name!r}")


def _relative_gain(value: float, baseline: float) -> float | None:
    if baseline <= 0.0:
        return None
    return (value - baseline) / baseline


def _worst_slice(report: dict[str, Any]) -> dict[str, Any] | None:
    slices = report.get("slices", {})
    if not isinstance(slices, dict):
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
    return max(rows, key=lambda row: (row["mean_regret"], row["slice"])) if rows else None


if __name__ == "__main__":
    raise SystemExit(main())
