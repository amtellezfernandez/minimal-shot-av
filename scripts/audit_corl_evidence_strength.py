#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INTERNAL_PROXY = ROOT / "artifacts" / "internal_proxy_transfer_medium.json"
DEFAULT_ALPASIM = ROOT / "artifacts" / "alpasim_matrix10_analysis.json"
DEFAULT_WOD = ROOT / "artifacts" / "wod_grounding_ablation_table.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl_evidence_strength_audit.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl_evidence_strength_audit.md"

INTERNAL_BASELINE = "raw_iter2"
ALPASIM_BASELINE = "token_dagger_iter2"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the current CoRL evidence supports a diagnostic paper, a "
            "strong CoRL paper, or a Best-Paper-level method claim."
        )
    )
    parser.add_argument("--internal-proxy", type=Path, default=DEFAULT_INTERNAL_PROXY)
    parser.add_argument("--alpasim-analysis", type=Path, default=DEFAULT_ALPASIM)
    parser.add_argument("--wod-grounding", type=Path, default=DEFAULT_WOD)
    parser.add_argument("--min-best-paper-alpasim-scenes", type=int, default=30)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_audit(
        internal_proxy=_load_json(args.internal_proxy),
        alpasim_analysis=_load_json(args.alpasim_analysis),
        wod_grounding=_load_json(args.wod_grounding),
        min_best_paper_alpasim_scenes=args.min_best_paper_alpasim_scenes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def build_audit(
    *,
    internal_proxy: dict[str, Any],
    alpasim_analysis: dict[str, Any],
    wod_grounding: dict[str, Any],
    min_best_paper_alpasim_scenes: int = 30,
) -> dict[str, Any]:
    internal = _internal_audit(internal_proxy)
    alpasim = _alpasim_audit(alpasim_analysis)
    wod = _wod_audit(wod_grounding)
    gates = _evidence_gates(
        internal=internal,
        alpasim=alpasim,
        wod=wod,
        min_best_paper_alpasim_scenes=min_best_paper_alpasim_scenes,
    )
    conclusion = _conclusion(gates)
    return {
        "schema": "corl_evidence_strength_audit_v1",
        "conclusion": conclusion,
        "gates": gates,
        "internal_proxy": internal,
        "alpasim": alpasim,
        "wod_grounding": wod,
    }


def _internal_audit(payload: dict[str, Any]) -> dict[str, Any]:
    by_perturbation = payload.get("by_perturbation", {})
    config = payload.get("config", {})
    variant_outcomes: dict[str, dict[str, int]] = {}
    perturbation_outcomes: dict[str, dict[str, Any]] = {}
    for perturbation, block in sorted(by_perturbation.items()):
        summary = block.get("summary", {})
        baseline = summary.get(INTERNAL_BASELINE)
        if not baseline:
            continue
        per_variant = {}
        for variant, metrics in sorted(summary.items()):
            if variant == INTERNAL_BASELINE:
                continue
            outcome = _classify_axis_delta(
                baseline,
                metrics,
                axes={
                    "pass": "higher",
                    "collision": "lower",
                    "lane_violation_any": "lower",
                    "offroad_proxy_any": "lower",
                },
            )
            per_variant[variant] = outcome
            _bump_outcome_counts(variant_outcomes.setdefault(variant, _empty_counts()), outcome["class"])
        perturbation_outcomes[perturbation] = per_variant
    return {
        "task_count": int(config.get("tasks", 0) or 0),
        "perturbation_count": len(by_perturbation),
        "variant_outcomes_vs_raw": variant_outcomes,
        "perturbation_outcomes_vs_raw": perturbation_outcomes,
        "tradeoff_count": sum(counts["tradeoff"] for counts in variant_outcomes.values()),
        "dominance_count": sum(counts["dominates"] for counts in variant_outcomes.values()),
    }


def _alpasim_audit(payload: dict[str, Any]) -> dict[str, Any]:
    per_model = payload.get("per_model_on_common_scenes") or payload.get("per_model", {})
    baseline = per_model.get(ALPASIM_BASELINE)
    model_outcomes = {}
    if baseline:
        for model, metrics in sorted(per_model.items()):
            if model == ALPASIM_BASELINE:
                continue
            model_outcomes[model] = _classify_axis_delta(
                _flatten_alpasim_metrics(baseline),
                _flatten_alpasim_metrics(metrics),
                axes={
                    "collision": "lower",
                    "offroad": "lower",
                    "wrong_lane": "lower",
                    "progress": "higher",
                    "dist_to_gt": "lower",
                },
            )
    scene_count = int(payload.get("all_models_common_scene_count", 0) or 0)
    return {
        "common_scene_count": scene_count,
        "model_outcomes_vs_raw": model_outcomes,
        "tradeoff_count": sum(1 for item in model_outcomes.values() if item["class"] == "tradeoff"),
        "dominance_count": sum(1 for item in model_outcomes.values() if item["class"] == "dominates"),
    }


def _wod_audit(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows", []))
    if not rows:
        return {
            "row_count": 0,
            "best_rfs": None,
            "scalar_rfs": None,
            "best_gain_over_scalar": None,
            "best_name": None,
            "grounding_head_sensitive": False,
        }
    scalar = next((row for row in rows if "scalar" in str(row.get("name", "")).lower()), rows[0])
    best = max(rows, key=lambda row: float(row.get("combined_ranker_mean_rfs", float("-inf"))))
    internvla = next((row for row in rows if "internvla only" in str(row.get("name", "")).lower()), None)
    scalar_rfs = float(scalar["combined_ranker_mean_rfs"])
    best_rfs = float(best["combined_ranker_mean_rfs"])
    internvla_rfs = None if internvla is None else float(internvla["combined_ranker_mean_rfs"])
    return {
        "row_count": len(rows),
        "best_name": str(best.get("name", "")),
        "best_rfs": best_rfs,
        "scalar_rfs": scalar_rfs,
        "best_gain_over_scalar": best_rfs - scalar_rfs,
        "internvla_only_rfs": internvla_rfs,
        "grounding_head_sensitive": bool(
            internvla_rfs is not None
            and internvla_rfs < scalar_rfs
            and best_rfs > scalar_rfs
        ),
        "heldout_test_claim": bool(payload.get("heldout_test_claim", False)),
    }


def _evidence_gates(
    *,
    internal: dict[str, Any],
    alpasim: dict[str, Any],
    wod: dict[str, Any],
    min_best_paper_alpasim_scenes: int,
) -> dict[str, Any]:
    controlled_tradeoffs = internal["tradeoff_count"] > 0
    external_tradeoffs = alpasim["tradeoff_count"] > 0
    positive_external_method = alpasim["dominance_count"] > 0
    gates = {
        "diagnostic_internal_proxy_supported": {
            "pass": internal["task_count"] >= 288 and controlled_tradeoffs,
            "current": internal["task_count"],
            "target": ">=288 cases with at least one controlled tradeoff",
        },
        "external_transfer_supported": {
            "pass": alpasim["common_scene_count"] >= 10 and external_tradeoffs,
            "current": alpasim["common_scene_count"],
            "target": ">=10 matched scenes with at least one external tradeoff",
        },
        "best_paper_external_scale": {
            "pass": alpasim["common_scene_count"] >= min_best_paper_alpasim_scenes,
            "current": alpasim["common_scene_count"],
            "target": f">={min_best_paper_alpasim_scenes} matched scenes",
        },
        "positive_method_uniform_dominance": {
            "pass": positive_external_method,
            "current": alpasim["dominance_count"],
            "target": ">=1 external intervention that improves at least one axis without worsening another",
        },
        "grounding_prior_signal": {
            "pass": bool(wod["best_gain_over_scalar"] is not None and wod["best_gain_over_scalar"] > 0.0),
            "current": wod["best_gain_over_scalar"],
            "target": ">0 RFS over scalar-only selector",
        },
        "grounding_hidden_test_claim": {
            "pass": bool(wod.get("heldout_test_claim")),
            "current": wod.get("heldout_test_claim"),
            "target": "true held-out or test-set grounding result",
        },
    }
    return gates


def _conclusion(gates: dict[str, Any]) -> dict[str, Any]:
    passed = {name for name, gate in gates.items() if gate["pass"]}
    blockers = [name for name, gate in gates.items() if not gate["pass"]]
    if {
        "diagnostic_internal_proxy_supported",
        "external_transfer_supported",
        "grounding_prior_signal",
    }.issubset(passed):
        level = "strong_diagnostic"
    else:
        level = "incomplete"
    if not blockers:
        level = "best_paper_ready"
    return {
        "level": level,
        "blockers": blockers,
        "summary": _level_summary(level, blockers),
    }


def _level_summary(level: str, blockers: list[str]) -> str:
    if level == "best_paper_ready":
        return "All configured gates pass; the evidence can support a Best-Paper-level methods claim."
    if level == "strong_diagnostic":
        return (
            "The evidence supports a strong diagnostic CoRL paper, but not a Best-Paper-level "
            f"method claim. Blocking gates: {', '.join(blockers)}."
        )
    return f"The evidence is incomplete. Blocking gates: {', '.join(blockers)}."


def _classify_axis_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    axes: dict[str, str],
    tol: float = 1e-9,
) -> dict[str, Any]:
    deltas = {}
    improved = []
    worsened = []
    unchanged = []
    for axis, direction in axes.items():
        base = _as_float(baseline.get(axis))
        cand = _as_float(candidate.get(axis))
        if base is None or cand is None:
            continue
        signed = cand - base if direction == "higher" else base - cand
        deltas[axis] = signed
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
        "signed_improvements": deltas,
    }


def _flatten_alpasim_metrics(metrics: dict[str, Any]) -> dict[str, float | None]:
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


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _empty_counts() -> dict[str, int]:
    return {"dominates": 0, "tradeoff": 0, "regresses": 0, "unchanged": 0}


def _bump_outcome_counts(counts: dict[str, int], outcome: str) -> None:
    counts[outcome] = int(counts.get(outcome, 0)) + 1


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# CoRL Evidence Strength Audit",
        "",
        f"Conclusion: `{report['conclusion']['level']}`.",
        "",
        report["conclusion"]["summary"],
        "",
        "## Gates",
        "",
        "| Gate | Pass | Current | Target |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, gate in report["gates"].items():
        lines.append(
            f"| `{name}` | {str(bool(gate['pass'])).lower()} | "
            f"{_fmt(gate['current'])} | {gate['target']} |"
        )
    lines.extend(
        [
            "",
            "## Internal Proxy Perturbation",
            "",
            f"Tasks: {report['internal_proxy']['task_count']}; "
            f"perturbations: {report['internal_proxy']['perturbation_count']}; "
            f"tradeoff outcomes: {report['internal_proxy']['tradeoff_count']}; "
            f"dominance outcomes: {report['internal_proxy']['dominance_count']}.",
            "",
            "| Variant | Dominates | Tradeoff | Regresses | Unchanged |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for variant, counts in sorted(report["internal_proxy"]["variant_outcomes_vs_raw"].items()):
        lines.append(
            f"| `{variant}` | {counts['dominates']} | {counts['tradeoff']} | "
            f"{counts['regresses']} | {counts['unchanged']} |"
        )
    lines.extend(
        [
            "",
            "## AlpaSim External Matrix",
            "",
            f"Common scenes: {report['alpasim']['common_scene_count']}; "
            f"tradeoff outcomes: {report['alpasim']['tradeoff_count']}; "
            f"dominance outcomes: {report['alpasim']['dominance_count']}.",
            "",
            "| Model | Class | Improved axes | Worsened axes |",
            "| --- | --- | --- | --- |",
        ]
    )
    for model, outcome in sorted(report["alpasim"]["model_outcomes_vs_raw"].items()):
        lines.append(
            f"| `{model}` | `{outcome['class']}` | "
            f"{', '.join(outcome['improved_axes']) or '-'} | "
            f"{', '.join(outcome['worsened_axes']) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## WOD Grounding",
            "",
            f"Best: `{report['wod_grounding']['best_name']}`; "
            f"RFS gain over scalar-only: {_fmt(report['wod_grounding']['best_gain_over_scalar'])}; "
            f"head-sensitive grounding: {str(report['wod_grounding']['grounding_head_sensitive']).lower()}.",
        ]
    )
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
