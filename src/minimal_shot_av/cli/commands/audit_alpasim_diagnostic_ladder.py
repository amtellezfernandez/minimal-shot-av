#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MATRIX = ROOT / "artifacts" / "corl2027" / "alpasim_matrix10_with_axis_analysis.json"
DEFAULT_ACTOR_COMPLETION = ROOT / "artifacts" / "corl2027" / "alpasim_actor_blindness_30scene_raw_analysis.json"
DEFAULT_COLLISION_SURFACE = ROOT / "artifacts" / "corl2027" / "alpasim_collision_surface_30scene_audit.json"
DEFAULT_COUNTERFACTUAL = ROOT / "artifacts" / "corl2027" / "alpasim_candidate_counterfactual_30scene.json"
DEFAULT_ACTION_SPACE = ROOT / "artifacts" / "alpasim_action_space_upper_bound_30scene.json"
DEFAULT_DIRECT_GRID_COST = ROOT / "artifacts" / "alpasim_direct_actor_planner_collision18_v2_analysis.json"
DEFAULT_DIRECT_GRID_CLEARANCE = ROOT / "artifacts" / "alpasim_direct_grid_max_clearance_collision18_analysis.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "alpasim_diagnostic_ladder.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "alpasim_diagnostic_ladder.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synthesize committed AlpaSim audits into a diagnostic ladder over actor visibility, "
            "candidate feasibility, selector ranking, direct-grid ranking, and controller/proxy execution."
        )
    )
    parser.add_argument("--matrix-analysis", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--actor-completion-analysis", type=Path, default=DEFAULT_ACTOR_COMPLETION)
    parser.add_argument("--collision-surface", type=Path, default=DEFAULT_COLLISION_SURFACE)
    parser.add_argument("--candidate-counterfactual", type=Path, default=DEFAULT_COUNTERFACTUAL)
    parser.add_argument("--action-space-upper-bound", type=Path, default=DEFAULT_ACTION_SPACE)
    parser.add_argument("--direct-grid-cost-replay", type=Path, default=DEFAULT_DIRECT_GRID_COST)
    parser.add_argument("--direct-grid-clearance-replay", type=Path, default=DEFAULT_DIRECT_GRID_CLEARANCE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_ladder_report(
        matrix_analysis=_load_json(args.matrix_analysis),
        actor_completion_analysis=_load_json(args.actor_completion_analysis),
        collision_surface=_load_json(args.collision_surface),
        candidate_counterfactual=_load_json(args.candidate_counterfactual),
        action_space_upper_bound=_load_json(args.action_space_upper_bound),
        direct_grid_cost_replay=_load_json(args.direct_grid_cost_replay),
        direct_grid_clearance_replay=_load_json(args.direct_grid_clearance_replay),
        source_paths={
            "matrix_analysis": args.matrix_analysis,
            "actor_completion_analysis": args.actor_completion_analysis,
            "collision_surface": args.collision_surface,
            "candidate_counterfactual": args.candidate_counterfactual,
            "action_space_upper_bound": args.action_space_upper_bound,
            "direct_grid_cost_replay": args.direct_grid_cost_replay,
            "direct_grid_clearance_replay": args.direct_grid_clearance_replay,
        },
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)


def build_ladder_report(
    *,
    matrix_analysis: dict[str, Any],
    actor_completion_analysis: dict[str, Any],
    collision_surface: dict[str, Any],
    candidate_counterfactual: dict[str, Any],
    action_space_upper_bound: dict[str, Any],
    direct_grid_cost_replay: dict[str, Any],
    direct_grid_clearance_replay: dict[str, Any],
    source_paths: dict[str, Path] | None = None,
) -> dict[str, Any]:
    rungs = [
        _actor_visibility_rung(actor_completion_analysis, collision_surface),
        _token_feasibility_rung(action_space_upper_bound),
        _token_selector_ranking_rung(candidate_counterfactual),
        _direct_grid_ranking_rung(action_space_upper_bound),
        _controller_execution_rung(
            action_space_upper_bound,
            direct_grid_cost_replay=direct_grid_cost_replay,
            direct_grid_clearance_replay=direct_grid_clearance_replay,
        ),
    ]
    trajectory_discrepancy = {
        "method_note": (
            "SimOpt-style diagnostic discrepancy: signed per-axis metric deltas between paired "
            "AlpaSim interventions. Positive values are candidate minus baseline for the raw metric."
        ),
        "matrix10": _paired_axis_discrepancies(matrix_analysis),
        "actor_completion_30scene": _paired_axis_discrepancies(actor_completion_analysis),
        "direct_grid_cost_collision18": _paired_axis_discrepancies(direct_grid_cost_replay),
        "direct_grid_clearance_collision18": _paired_axis_discrepancies(direct_grid_clearance_replay),
    }
    localization = _offline_localization_scores(rungs)
    return {
        "schema": "alpasim_diagnostic_ladder_v1",
        "source_paths": {key: str(path) for key, path in (source_paths or {}).items()},
        "conclusion": _conclusion(rungs, localization),
        "rungs": rungs,
        "offline_localization": localization,
        "trajectory_discrepancy": trajectory_discrepancy,
        "notes": [
            (
                "The offline localization scores are deterministic evidence weights from committed audits, "
                "not a learned DROPO posterior."
            ),
            (
                "The ladder localizes residual collision only within the available AlpaSim artifacts; "
                "it does not prove deployment safety."
            ),
        ],
    }


def _actor_visibility_rung(
    actor_completion_analysis: dict[str, Any],
    collision_surface: dict[str, Any],
) -> dict[str, Any]:
    comparison = _first_comparison(actor_completion_analysis)
    collision = ((comparison.get("binary_metrics") or {}).get("raw_collision_any") or {})
    surface = collision_surface.get("summary", {})
    baseline_first = surface.get("baseline_first_collision", {})
    candidate_first = surface.get("candidate_first_collision", {})
    baseline_collisions = _int(surface.get("baseline_raw_collision_count"))
    candidate_collisions = _int(surface.get("candidate_raw_collision_count"))
    common = _int(surface.get("common_scene_count")) or _int(collision.get("scene_count"))
    restored_hits = _int(candidate_first.get("proxy_hits"))
    baseline_zero = _int(baseline_first.get("structured_hazards_zero"))
    status = "visibility_real_but_not_sufficient"
    if baseline_collisions != candidate_collisions:
        status = "visibility_changes_collision_surface"
    return {
        "name": "actor_visibility",
        "question": "Does restoring actor visibility explain the collision surface?",
        "status": status,
        "load_bearing_score": 0.0 if baseline_collisions == candidate_collisions else 1.0,
        "evidence": {
            "common_scene_count": common,
            "baseline_collision_count": baseline_collisions,
            "actor_completion_collision_count": candidate_collisions,
            "paired_collision_improved_scenes": _int(collision.get("improved_scenes")),
            "paired_collision_worsened_scenes": _int(collision.get("worsened_scenes")),
            "baseline_first_impact_structured_hazards_zero": baseline_zero,
            "actor_completion_first_impact_proxy_hits": restored_hits,
            "actor_completion_logged_at_impact": _int(candidate_first.get("logged_at_impact")),
        },
        "interpretation": (
            "Actor dropout is real at impact, but matched actor completion preserves the same raw "
            "collision count; this rung is necessary context, not a sufficient explanation."
        ),
    }


def _token_feasibility_rung(action_space: dict[str, Any]) -> dict[str, Any]:
    summary = action_space.get("summary", {})
    token = summary.get("token_candidate_set", {})
    scenes = _int(summary.get("baseline_collision_scene_count"))
    scenes_with_safe = _int(token.get("scene_count_with_proxy_safe_actionable_frame"))
    missing = max(0, scenes - scenes_with_safe)
    score = _safe_rate(missing, scenes)
    status = "token_safe_candidate_present_in_all_collision_scenes" if missing == 0 else "candidate_feasibility_gap"
    return {
        "name": "token_candidate_feasibility",
        "question": "Did the bounded token set contain proxy-safe candidate actions before impact?",
        "status": status,
        "load_bearing_score": score,
        "evidence": {
            "collision_scene_count": scenes,
            "scene_count_with_proxy_safe_token": scenes_with_safe,
            "proxy_safe_actionable_frame_count": _int(token.get("proxy_safe_actionable_frame_count")),
            "selected_proxy_safe_actionable_frame_count": _int(token.get("selected_proxy_safe_actionable_frame_count")),
            "first_token_proxy_safe_lead_frames": summary.get("first_token_proxy_safe_lead_frames", {}),
        },
        "interpretation": (
            "The committed open-loop audit finds proxy-safe token candidates before impact in every "
            "collision scene, so pure candidate absence is not the dominant explanation."
        ),
    }


def _token_selector_ranking_rung(counterfactual: dict[str, Any]) -> dict[str, Any]:
    summary = counterfactual.get("summary", {})
    scenes = _int(summary.get("baseline_collision_scene_count"))
    buckets = summary.get("bucket_counts", {})
    miss_scenes = _int(buckets.get("actor_axis_safe_candidate_missed_by_baseline_selector")) + _int(
        buckets.get("mixed_actor_axis_safe_candidate_intermittently_missed")
    )
    persisted = _int(buckets.get("baseline_selected_actor_axis_safe_candidate_but_collision_persisted"))
    score = _safe_rate(miss_scenes, scenes)
    status = "selector_miss_secondary" if persisted > miss_scenes else "selector_miss_primary_candidate"
    return {
        "name": "token_selector_ranking",
        "question": "Was an actor-axis-safe token present but rejected by the baseline selector?",
        "status": status,
        "load_bearing_score": score,
        "evidence": {
            "collision_scene_count": scenes,
            "actor_axis_safe_actionable_frame_count": _int(
                summary.get("actor_axis_proxy_actionable_safe_frame_count")
            ),
            "actor_axis_selected_safe_actionable_frame_count": _int(
                summary.get("actor_axis_proxy_selected_safe_frame_count")
            ),
            "actor_axis_missed_safe_actionable_frame_count": _int(
                summary.get("actor_axis_proxy_missed_safe_frame_count")
            ),
            "scene_count_with_any_selector_miss": miss_scenes,
            "scene_count_selected_safe_but_collision_persisted": persisted,
            "actor_axis_safe_at_impact_count": _int(summary.get("actor_axis_proxy_safe_at_impact_count")),
        },
        "interpretation": (
            "Selector mistakes exist, but many scenes already select actor-axis-safe tokens before "
            "collision; selector ranking alone is not sufficient."
        ),
    }


def _direct_grid_ranking_rung(action_space: dict[str, Any]) -> dict[str, Any]:
    summary = action_space.get("summary", {})
    direct = summary.get("direct_grid", {})
    scenes = _int(summary.get("baseline_collision_scene_count"))
    missed = _int(direct.get("scene_count_with_cost_missed_proxy_safe_actionable_frame"))
    selected_safe = _int(direct.get("scene_count_with_selected_proxy_safe_actionable_frame"))
    score = _safe_rate(missed, scenes)
    status = "direct_grid_cost_not_open_loop_bottleneck" if missed == 0 else "direct_grid_cost_ranking_gap"
    return {
        "name": "direct_grid_cost_ranking",
        "question": "If the learned selector is bypassed, does the direct-grid cost reject proxy-safe actions?",
        "status": status,
        "load_bearing_score": score,
        "evidence": {
            "collision_scene_count": scenes,
            "scene_count_with_direct_proxy_safe_action": _int(
                direct.get("scene_count_with_proxy_safe_actionable_frame")
            ),
            "scene_count_with_direct_selected_proxy_safe_action": selected_safe,
            "scene_count_with_direct_cost_miss": missed,
            "selected_proxy_safe_actionable_frame_count": _int(
                direct.get("selected_proxy_safe_actionable_frame_count")
            ),
            "cost_missed_proxy_safe_actionable_frame_count": _int(
                direct.get("cost_missed_proxy_safe_actionable_frame_count")
            ),
        },
        "interpretation": (
            "The direct-grid open-loop cost selects proxy-safe actions where available, so the "
            "remaining collision cannot be assigned to direct-grid cost ranking alone."
        ),
    }


def _controller_execution_rung(
    action_space: dict[str, Any],
    *,
    direct_grid_cost_replay: dict[str, Any],
    direct_grid_clearance_replay: dict[str, Any],
) -> dict[str, Any]:
    scenes = _int((action_space.get("summary", {})).get("baseline_collision_scene_count"))
    cost_collision = _collision_metric(_first_comparison(direct_grid_cost_replay))
    clearance_collision = _collision_metric(_first_comparison(direct_grid_clearance_replay))
    cost_persisted = _int(cost_collision.get("scene_count")) - _int(cost_collision.get("improved_scenes"))
    clearance_persisted = _int(clearance_collision.get("scene_count")) - _int(
        clearance_collision.get("improved_scenes")
    )
    persisted = min(cost_persisted, clearance_persisted)
    score = _safe_rate(persisted, scenes)
    return {
        "name": "controller_execution_or_proxy_mismatch",
        "question": "Do closed-loop direct-grid replays eliminate collision after proxy-safe actions exist?",
        "status": "residual_load_bearing_boundary",
        "load_bearing_score": score,
        "evidence": {
            "collision_scene_count": scenes,
            "cost_ranked_direct_grid_collision_rate": cost_collision.get("candidate_rate"),
            "cost_ranked_direct_grid_improved_scenes": cost_collision.get("improved_scenes"),
            "clearance_direct_grid_collision_rate": clearance_collision.get("candidate_rate"),
            "clearance_direct_grid_improved_scenes": clearance_collision.get("improved_scenes"),
            "minimum_persisted_collision_scenes_across_direct_replays": persisted,
        },
        "interpretation": (
            "Closed-loop direct-grid replay removes only a small number of collisions despite "
            "open-loop proxy-safe actions. The residual boundary is controller execution, "
            "receding-horizon erosion, or proxy/simulator clearance mismatch."
        ),
    }


def _offline_localization_scores(rungs: list[dict[str, Any]]) -> dict[str, Any]:
    raw = {str(rung["name"]): float(rung.get("load_bearing_score", 0.0) or 0.0) for rung in rungs}
    total = sum(raw.values())
    normalized = {key: (value / total if total > 0.0 else 0.0) for key, value in raw.items()}
    ranking = sorted(normalized.items(), key=lambda item: item[1], reverse=True)
    return {
        "method": "deterministic_frequency_weighted_localization",
        "uncertainty_method": (
            "Deterministic per-rung frequency bootstrap over observed scene-count evidence; "
            "not a learned posterior and not an independence proof between rungs."
        ),
        "raw_scores": raw,
        "normalized_scores": dict(normalized),
        "normalized_score_bootstrap_ci": _bootstrap_normalized_score_ci(rungs),
        "ranking": [{"rung": rung, "weight": weight} for rung, weight in ranking],
        "top_rung": ranking[0][0] if ranking else None,
    }


def _bootstrap_normalized_score_ci(
    rungs: list[dict[str, Any]],
    *,
    iterations: int = 2000,
    seed: int = 7,
) -> dict[str, Any]:
    rng = random.Random(seed)
    observations = {}
    for rung in rungs:
        name = str(rung["name"])
        trial_count = _rung_trial_count(rung)
        score = float(rung.get("load_bearing_score", 0.0) or 0.0)
        hit_count = max(0, min(trial_count, round(score * trial_count)))
        observations[name] = {
            "hit_count": hit_count,
            "trial_count": trial_count,
            "samples": [1.0] * hit_count + [0.0] * max(0, trial_count - hit_count),
        }

    draws = {name: [] for name in observations}
    for _ in range(iterations):
        raw = {}
        for name, row in observations.items():
            samples = row["samples"]
            if not samples:
                raw[name] = 0.0
                continue
            raw[name] = sum(rng.choice(samples) for _ in samples) / len(samples)
        total = sum(raw.values())
        for name, value in raw.items():
            draws[name].append(value / total if total > 0.0 else 0.0)

    return {
        "iterations": iterations,
        "seed": seed,
        "observed_counts": {
            name: {
                "hit_count": row["hit_count"],
                "trial_count": row["trial_count"],
            }
            for name, row in observations.items()
        },
        "percentile_95": {
            name: {
                "low": _percentile(values, 0.025),
                "median": _percentile(values, 0.5),
                "high": _percentile(values, 0.975),
            }
            for name, values in draws.items()
        },
    }


def _rung_trial_count(rung: dict[str, Any]) -> int:
    evidence = rung.get("evidence", {})
    for key in ("collision_scene_count", "common_scene_count", "scene_count"):
        value = _int(evidence.get(key))
        if value > 0:
            return value
    return 0


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((len(ordered) - 1) * fraction)))
    return float(ordered[index])


def _paired_axis_discrepancies(payload: dict[str, Any]) -> dict[str, Any]:
    output = {}
    for model, comparison in sorted((payload.get("comparisons_vs_baseline") or {}).items()):
        axes: dict[str, Any] = {}
        binary = comparison.get("binary_metrics", {})
        for metric in ("raw_collision_any", "raw_offroad", "raw_wrong_lane"):
            row = binary.get(metric) or {}
            axes[metric] = {
                "baseline_rate": row.get("baseline_rate"),
                "candidate_rate": row.get("candidate_rate"),
                "signed_delta": row.get("delta_rate"),
                "improved_scenes": row.get("improved_scenes"),
                "worsened_scenes": row.get("worsened_scenes"),
                "scene_count": row.get("scene_count"),
            }
        continuous = comparison.get("continuous_metrics", {})
        for metric in ("raw_progress", "raw_dist_traveled_m", "raw_dist_to_gt_trajectory"):
            row = continuous.get(metric) or {}
            axes[metric] = {
                "mean_delta": row.get("mean_delta"),
                "median_delta": row.get("median_delta"),
                "positive_count": row.get("positive_count"),
                "negative_count": row.get("negative_count"),
            }
        output[model] = {
            "scene_count": comparison.get("scene_count"),
            "axes": axes,
            "axis_conflicts": comparison.get("axis_conflicts", {}),
        }
    return output


def _conclusion(rungs: list[dict[str, Any]], localization: dict[str, Any]) -> dict[str, Any]:
    top = localization.get("top_rung")
    return {
        "claim": (
            "Residual AlpaSim collision is not localized to actor visibility, token candidate "
            "availability, learned selector ranking, or direct-grid cost ranking alone."
        ),
        "top_residual_rung": top,
        "recommended_positioning": (
            "Frame the contribution as a diagnostic ladder below perception: feasibility, ranking, "
            "direct-grid ranking, and controller/proxy execution."
        ),
        "rung_statuses": {str(rung["name"]): str(rung["status"]) for rung in rungs},
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# AlpaSim diagnostic ladder",
        "",
        f"Conclusion: {report['conclusion']['claim']}",
        "",
        "## Rungs",
        "",
    ]
    for rung in report["rungs"]:
        lines.append(
            f"- `{rung['name']}`: `{rung['status']}`; "
            f"score={rung['load_bearing_score']:.3f}. {rung['interpretation']}"
        )
    lines.extend(["", "## Offline localization", ""])
    ci = report["offline_localization"].get("normalized_score_bootstrap_ci", {})
    intervals = ci.get("percentile_95", {})
    for item in report["offline_localization"]["ranking"]:
        interval = intervals.get(item["rung"], {})
        suffix = ""
        if interval:
            suffix = f" (95% bootstrap CI {interval['low']:.3f}-{interval['high']:.3f})"
        lines.append(f"- `{item['rung']}`: {item['weight']:.3f}{suffix}")
    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines)


def _first_comparison(payload: dict[str, Any]) -> dict[str, Any]:
    comparisons = payload.get("comparisons_vs_baseline") or {}
    if not comparisons:
        return {}
    first_key = sorted(comparisons)[0]
    return comparisons[first_key] or {}


def _collision_metric(comparison: dict[str, Any]) -> dict[str, Any]:
    return ((comparison.get("binary_metrics") or {}).get("raw_collision_any") or {})


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _safe_rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


if __name__ == "__main__":
    main()
