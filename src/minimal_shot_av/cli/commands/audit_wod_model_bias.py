#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_BENCHMARK = ROOT / "benchmarks" / "current" / "wod_ridge_trajectory_cv_official.json"
DEFAULT_DECLARATION = ROOT / "models" / "DECLARATION.md"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit bias and leakage risks in the WOD-E2E model path.")
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--declaration", type=Path, default=DEFAULT_DECLARATION)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = audit_bias(args.benchmark, args.declaration)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["status"] != "fail" else 1


def audit_bias(benchmark_path: Path, declaration_path: Path) -> dict[str, Any]:
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    slice_source = _load_slice_source(benchmark, benchmark_path)
    declaration = declaration_path.read_text(encoding="utf-8")
    metadata = _benchmark_metadata(benchmark)
    metrics = _benchmark_metrics(benchmark)
    selected_rates = _object(metadata.get("selected_source_rates"))
    risks: list[dict[str, str]] = []
    mitigations: list[dict[str, str]] = []

    _check_validation_calibration(metadata, declaration, risks, mitigations)
    _check_visual_blindness(declaration, risks)
    _check_cross_validation(metadata, risks, mitigations)
    _check_oracle_gap(metrics, risks)
    _check_source_concentration(selected_rates, risks, mitigations)
    slice_summary = _slice_summary(_object(slice_source.get("slices")))
    _check_slice_bias(slice_summary, risks)
    _check_simulator_independence(benchmark, declaration, risks, mitigations)

    status = "warn" if risks else "pass"
    return {
        "audit_type": "wod_model_bias",
        "status": status,
        "benchmark": str(benchmark_path),
        "declaration": str(declaration_path),
        "risks": risks,
        "mitigations": mitigations,
        "summary": {
            "risk_count": len(risks),
            "mitigation_count": len(mitigations),
            "score_is_leaderboard_claim": False,
            "strict_zero_shot_claim_allowed": False,
        },
        "slice_summary": slice_summary,
    }


def _check_validation_calibration(
    metadata: dict[str, Any],
    declaration: str,
    risks: list[dict[str, str]],
    mitigations: list[dict[str, str]],
) -> None:
    if "validation" in str(metadata.get("split", "")) and "preference_trajectories" in declaration:
        risks.append(
            {
                "id": "validation_preference_bias",
                "severity": "high",
                "detail": (
                    "Selector development uses WOD-E2E validation preference labels, "
                    "so the score is calibration evidence, not a strict zero-shot claim."
                ),
            }
        )
    if "not a leaderboard/test score" in declaration:
        mitigations.append(
            {
                "id": "leaderboard_claim_blocked",
                "detail": "Declaration explicitly marks validation RFS as internal development evidence.",
            }
        )


def _check_visual_blindness(declaration: str, risks: list[dict[str, str]]) -> None:
    if "Input cameras: not used in current active path" in declaration:
        risks.append(
            {
                "id": "visual_blindness_bias",
                "severity": "high",
                "detail": (
                    "The active model is ego-history-only, so it is biased toward motion priors "
                    "and cannot react to visual hazards such as debris, pedestrians, signals, or cross-traffic."
                ),
            }
        )


def _check_cross_validation(
    metadata: dict[str, Any],
    risks: list[dict[str, str]],
    mitigations: list[dict[str, str]],
) -> None:
    split = str(metadata.get("split", ""))
    fold_count = int(metadata.get("fold_count", 0)) if isinstance(metadata.get("fold_count"), int) else 0
    if "segment_grouped_cv" in split and fold_count >= 2:
        mitigations.append(
            {
                "id": "segment_grouped_cv",
                "detail": f"Benchmark uses segment-grouped cross-validation with {fold_count} folds.",
            }
        )
    else:
        risks.append(
            {
                "id": "ungrouped_or_missing_cv",
                "severity": "high",
                "detail": "Benchmark is missing segment-grouped cross-validation metadata.",
            }
        )


def _check_oracle_gap(metrics: dict[str, Any], risks: list[dict[str, str]]) -> None:
    selected = _metric_value(metrics, "combined_ranker_mean_rfs")
    oracle = _metric_value(metrics, "combined_oracle_mean_rfs")
    if selected is None or oracle is None:
        return
    gap = oracle - selected
    if gap > 1.0:
        risks.append(
            {
                "id": "selector_oracle_gap",
                "severity": "medium",
                "detail": (
                    f"Candidate oracle exceeds selected RFS by {gap:.4f}; the selector is still "
                    "biased toward the wrong candidate family on many frames."
                ),
            }
        )


def _check_source_concentration(
    selected_rates: dict[str, Any],
    risks: list[dict[str, str]],
    mitigations: list[dict[str, str]],
) -> None:
    if not selected_rates:
        risks.append(
            {
                "id": "missing_source_diagnostics",
                "severity": "medium",
                "detail": "Benchmark does not report selected candidate source rates.",
            }
        )
        return
    max_source, max_rate = max(
        ((str(source), float(rate)) for source, rate in selected_rates.items()),
        key=lambda item: item[1],
    )
    if max_rate > 0.9:
        risks.append(
            {
                "id": "source_concentration",
                "severity": "medium",
                "detail": (
                    f"Selector chooses {max_source} candidates on {max_rate:.1%} of frames, "
                    "which suggests source-family bias."
                ),
            }
        )
    else:
        mitigations.append(
            {
                "id": "source_diversity",
                "detail": f"Largest selected source share is {max_source} at {max_rate:.1%}.",
            }
        )


def _check_simulator_independence(
    benchmark: dict[str, Any],
    declaration: str,
    risks: list[dict[str, str]],
    mitigations: list[dict[str, str]],
) -> None:
    haystack = json.dumps(benchmark).lower() + "\n" + declaration.lower()
    if "simulator metrics were not used" in haystack and "simulator metrics are not used" in haystack:
        mitigations.append(
            {
                "id": "simulator_independence_declared",
                "detail": "Benchmark and declaration both state simulator metrics were not used for model selection.",
            }
        )
    else:
        risks.append(
            {
                "id": "simulator_independence_missing",
                "severity": "high",
                "detail": "Could not confirm simulator metrics were excluded from model selection.",
            }
        )


def _check_slice_bias(slice_summary: dict[str, Any], risks: list[dict[str, str]]) -> None:
    worst = slice_summary.get("worst_regret_slice")
    if not isinstance(worst, dict):
        return
    regret = float(worst.get("mean_regret", 0.0))
    if regret > 1.5:
        risks.append(
            {
                "id": "slice_regret_bias",
                "severity": "medium",
                "detail": (
                    f"Worst slice is {worst.get('slice')} with mean regret {regret:.4f}; "
                    "global RFS hides uneven selector quality."
                ),
            }
        )


def _slice_summary(slices: dict[str, Any]) -> dict[str, Any]:
    if not slices:
        return {"available": False}
    ordered = sorted(
        (
            {
                "slice": str(name),
                "frames": int(data["frames"]),
                "selected_mean_rfs": float(data["selected_mean_rfs"]),
                "oracle_mean_rfs": float(data["oracle_mean_rfs"]),
                "mean_regret": float(data["mean_regret"]),
                "selected_source_rates": dict(data.get("selected_source_rates", {})),
                "oracle_source_rates": dict(data.get("oracle_source_rates", {})),
            }
            for name, data in slices.items()
            if isinstance(data, dict) and int(data.get("frames", 0)) > 0
        ),
        key=lambda item: item["mean_regret"],
        reverse=True,
    )
    return {
        "available": True,
        "worst_regret_slice": ordered[0] if ordered else None,
        "top_regret_slices": ordered[:5],
    }


def _load_slice_source(benchmark: dict[str, Any], benchmark_path: Path) -> dict[str, Any]:
    if isinstance(benchmark.get("slices"), dict):
        return benchmark
    source = benchmark.get("source")
    if not isinstance(source, str):
        return {}
    candidates = [Path(source)]
    if not Path(source).is_absolute():
        candidates.append(ROOT / source)
        candidates.append(benchmark_path.parent / source)
    for candidate in candidates:
        if candidate.is_file():
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                return loaded if isinstance(loaded, dict) else {}
            except json.JSONDecodeError:
                return {}
    return {}


def _metric_value(metrics: dict[str, Any], name: str) -> float | None:
    raw = metrics.get(name)
    if isinstance(raw, dict) and isinstance(raw.get("value"), (int, float)) and not isinstance(raw.get("value"), bool):
        return float(raw["value"])
    return None


def _benchmark_metadata(benchmark: dict[str, Any]) -> dict[str, Any]:
    metadata = _object(benchmark.get("metadata"))
    if metadata:
        return metadata
    if benchmark.get("benchmark_type") != "segment_grouped_cross_validation":
        return {}
    selected_rates = {
        source: benchmark.get(f"selected_{source}_rate")
        for source in ("learned", "temporal", "kinematic", "anchor", "world")
        if isinstance(benchmark.get(f"selected_{source}_rate"), (int, float))
    }
    return {
        "split": f"validation_{benchmark.get('frames', 0)}_preference_frames_segment_grouped_cv",
        "fold_count": benchmark.get("fold_count"),
        "frame_count": benchmark.get("frames"),
        "score_backend": benchmark.get("score_backend"),
        "selected_source_rates": selected_rates,
    }


def _benchmark_metrics(benchmark: dict[str, Any]) -> dict[str, Any]:
    metrics = _object(benchmark.get("metrics"))
    if metrics:
        return metrics
    metric_names = (
        "combined_ranker_mean_rfs",
        "combined_oracle_mean_rfs",
        "combined_ranker_regret_to_oracle",
        "combined_ranker_top1_oracle_match_rate",
    )
    return {
        name: {"value": benchmark[name]}
        for name in metric_names
        if isinstance(benchmark.get(name), (int, float)) and not isinstance(benchmark.get(name), bool)
    }


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())
