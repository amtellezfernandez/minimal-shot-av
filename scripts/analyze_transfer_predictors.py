#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERNAL_PROXY = ROOT / "artifacts" / "internal_proxy_transfer_axis_smoke.json"
DEFAULT_ALPASIM = ROOT / "artifacts" / "alpasim_matrix10_analysis.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "transfer_predictor_analysis.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "transfer_predictor_analysis.md"

VARIANT_TO_ALPASIM_MODEL = {
    "clamped_iter2": "token_dagger_iter2_clamped",
    "hybrid_veto_iter2": "token_dagger_iter2_hybrid_clamped",
    "axis_constrained_iter2": "token_dagger_iter2_axis_constrained_clamped",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare internal proxy-perturbation signatures against external AlpaSim "
            "outcomes. This is a prediction audit, not a new rollout."
        )
    )
    parser.add_argument("--internal-proxy", type=Path, default=DEFAULT_INTERNAL_PROXY)
    parser.add_argument("--alpasim-analysis", type=Path, default=DEFAULT_ALPASIM)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_predictor_report(
        internal_proxy=_load_json(args.internal_proxy),
        alpasim_analysis=_load_json(args.alpasim_analysis),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def build_predictor_report(
    *,
    internal_proxy: dict[str, Any],
    alpasim_analysis: dict[str, Any],
) -> dict[str, Any]:
    internal = _internal_signatures(internal_proxy)
    external = _external_outcomes(alpasim_analysis)
    rows = []
    for variant, model in VARIANT_TO_ALPASIM_MODEL.items():
        signature = internal.get(variant)
        outcome = external.get(model)
        prediction = _predict_external_class(signature)
        rows.append(
            {
                "internal_variant": variant,
                "alpasim_model": model,
                "internal_signature": signature,
                "prediction": prediction,
                "external_outcome": outcome,
                "status": _prediction_status(prediction, outcome),
            }
        )
    confirmed = [row for row in rows if row["status"] == "confirmed"]
    missing = [row for row in rows if row["status"] == "needs_external_run"]
    contradicted = [row for row in rows if row["status"] == "contradicted"]
    return {
        "schema": "transfer_predictor_analysis_v1",
        "rows": rows,
        "summary": {
            "mapped_interventions": len(rows),
            "confirmed_predictions": len(confirmed),
            "missing_external_runs": len(missing),
            "contradicted_predictions": len(contradicted),
            "ready_for_claim": bool(rows and not missing and not contradicted),
        },
    }


def _internal_signatures(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    by_perturbation = payload.get("by_perturbation", {})
    signatures: dict[str, dict[str, Any]] = {}
    for variant in VARIANT_TO_ALPASIM_MODEL:
        perturbation_rows = []
        for perturbation, block in sorted(by_perturbation.items()):
            summary = block.get("summary", {})
            raw = summary.get("raw_iter2")
            candidate = summary.get(variant)
            if not raw or not candidate:
                continue
            outcome = _axis_outcome(
                raw,
                candidate,
                axes={
                    "pass": "higher",
                    "collision": "lower",
                    "lane_violation_any": "lower",
                    "offroad_proxy_any": "lower",
                    "total_progress_m": "higher",
                },
            )
            perturbation_rows.append({"perturbation": perturbation, **outcome})
        signatures[variant] = _summarize_internal_variant(perturbation_rows)
    return signatures


def _summarize_internal_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    class_counts = {"dominates": 0, "tradeoff": 0, "regresses": 0, "unchanged": 0}
    worsened_axes: dict[str, int] = {}
    improved_axes: dict[str, int] = {}
    for row in rows:
        class_counts[row["class"]] += 1
        for axis in row["worsened_axes"]:
            worsened_axes[axis] = worsened_axes.get(axis, 0) + 1
        for axis in row["improved_axes"]:
            improved_axes[axis] = improved_axes.get(axis, 0) + 1
    total = len(rows)
    return {
        "perturbation_count": total,
        "class_counts": class_counts,
        "tradeoff_rate": class_counts["tradeoff"] / total if total else None,
        "dominance_rate": class_counts["dominates"] / total if total else None,
        "worsened_axes": worsened_axes,
        "improved_axes": improved_axes,
        "per_perturbation": rows,
    }


def _external_outcomes(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_model = payload.get("per_model_on_common_scenes") or payload.get("per_model", {})
    raw = per_model.get("token_dagger_iter2")
    if not raw:
        return {}
    baseline = _flatten_alpasim(raw)
    outcomes = {}
    for model, metrics in sorted(per_model.items()):
        if model == "token_dagger_iter2":
            continue
        outcomes[model] = _axis_outcome(
            baseline,
            _flatten_alpasim(metrics),
            axes={
                "collision": "lower",
                "offroad": "lower",
                "wrong_lane": "lower",
                "progress": "higher",
                "dist_to_gt": "lower",
            },
        )
    return outcomes


def _predict_external_class(signature: dict[str, Any] | None) -> dict[str, Any]:
    if not signature or not signature.get("perturbation_count"):
        return {
            "predicted_class": "unknown",
            "reason": "no internal proxy signature available",
        }
    if signature["class_counts"]["regresses"] > 0:
        return {
            "predicted_class": "not_uniform_dominance",
            "reason": "internal proxy sweep already contains regressions",
        }
    if signature["class_counts"]["tradeoff"] > 0:
        return {
            "predicted_class": "not_uniform_dominance",
            "reason": "internal proxy sweep contains non-co-monotone tradeoffs",
        }
    if signature["class_counts"]["dominates"] == signature["perturbation_count"]:
        return {
            "predicted_class": "possible_dominance",
            "reason": "internal proxy sweep shows dominance on every perturbation",
        }
    return {
        "predicted_class": "unknown",
        "reason": "internal proxy signature is mixed without a tradeoff",
    }


def _prediction_status(
    prediction: dict[str, Any],
    outcome: dict[str, Any] | None,
) -> str:
    if outcome is None:
        return "needs_external_run"
    predicted = prediction["predicted_class"]
    if predicted == "not_uniform_dominance":
        return "confirmed" if outcome["class"] != "dominates" else "contradicted"
    if predicted == "possible_dominance":
        return "confirmed" if outcome["class"] == "dominates" else "contradicted"
    return "unscored"


def _axis_outcome(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    axes: dict[str, str],
    tol: float = 1e-9,
) -> dict[str, Any]:
    improved = []
    worsened = []
    unchanged = []
    signed_improvements = {}
    for axis, direction in axes.items():
        base = _as_float(baseline.get(axis))
        cand = _as_float(candidate.get(axis))
        if base is None or cand is None:
            continue
        signed = cand - base if direction == "higher" else base - cand
        signed_improvements[axis] = signed
        if signed > tol:
            improved.append(axis)
        elif signed < -tol:
            worsened.append(axis)
        else:
            unchanged.append(axis)
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
        "improved_axes": improved,
        "worsened_axes": worsened,
        "unchanged_axes": unchanged,
        "signed_improvements": signed_improvements,
    }


def _flatten_alpasim(metrics: dict[str, Any]) -> dict[str, float | None]:
    return {
        "collision": _metric_rate(metrics, "raw_collision_any"),
        "offroad": _metric_rate(metrics, "raw_offroad"),
        "wrong_lane": _metric_rate(metrics, "raw_wrong_lane"),
        "progress": _metric_mean(metrics, "raw_progress"),
        "dist_to_gt": _metric_mean(metrics, "raw_dist_to_gt_trajectory"),
    }


def _metric_rate(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        return _as_float(value.get("rate"))
    return _as_float(value)


def _metric_mean(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        return _as_float(value.get("mean"))
    return _as_float(value)


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Transfer Predictor Analysis",
        "",
        "| Internal variant | External model | Prediction | External class | Status |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        external = row["external_outcome"]
        external_class = "missing" if external is None else external["class"]
        lines.append(
            f"| `{row['internal_variant']}` | `{row['alpasim_model']}` | "
            f"`{row['prediction']['predicted_class']}` | `{external_class}` | "
            f"`{row['status']}` |"
        )
    summary = report["summary"]
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"Confirmed predictions: {summary['confirmed_predictions']} / {summary['mapped_interventions']}.",
            f"Missing external runs: {summary['missing_external_runs']}.",
            f"Contradicted predictions: {summary['contradicted_predictions']}.",
            f"Ready for claim: {str(summary['ready_for_claim']).lower()}.",
        ]
    )
    return "\n".join(lines)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
