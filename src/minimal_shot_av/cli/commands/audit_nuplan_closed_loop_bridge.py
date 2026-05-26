#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping

from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    DEFAULT_NEAR_MISS_THRESHOLD_M,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    REPLAY_INFEASIBLE_RUNG,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    _actor_state_at_time,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    _build_actor_tracks,
)
from minimal_shot_av.cli.commands.audit_nuplan_selected_token_realized_clearance import (
    _nuplan_replay_imports,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _constrained_candidate_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _features
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_boundary_intervention_selector import (
    load_boundary_intervention_policy,
)
from minimal_shot_av.cli.commands.train_nuplan_boundary_intervention_selector import (
    predict_boundary_delta,
)
from minimal_shot_av.cli.commands.train_nuplan_boundary_intervention_selector import (
    select_with_boundary_intervention,
)
from minimal_shot_av.cli.commands.train_nuplan_selective_regret_selector import load_selective_regret_policy
from minimal_shot_av.cli.commands.train_nuplan_selective_regret_selector import select_with_selective_regret
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import select_with_constrained_replay_risk
from minimal_shot_av.cli.commands.train_nuplan_failure_gate_selector import build_safe_token_fn
from minimal_shot_av.cli.commands.train_nuplan_failure_gate_selector import fallback_meets_progress_retention
from minimal_shot_av.cli.commands.train_nuplan_failure_gate_selector import failure_gate_feature_row
from minimal_shot_av.cli.commands.train_nuplan_failure_gate_selector import load_failure_gate_policy
from minimal_shot_av.cli.commands.train_nuplan_failure_gate_selector import predict_failure_probability
from minimal_shot_av.cli.commands.train_nuplan_failure_gate_selector import select_with_failure_gate_policy
from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import predict_meta_lambda_value
from minimal_shot_av.cli.commands.run_nuplan_replay_calibration_experiments import select_with_meta_lambda_value_policy
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector
ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_mini_rollout50_realized_clearance.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_mini_rollout50_closed_loop_bridge.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_mini_rollout50_closed_loop_bridge.md"
DEFAULT_CALIBRATION_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_mini_closed_loop_bridge_calibration.json"
DEFAULT_CALIBRATION_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_mini_closed_loop_bridge_calibration.md"
DEFAULT_MAPS_ROOT = ROOT / "workspace" / "nuplan" / "maps" / "maps"
DEFAULT_MAP_VERSION = "nuplan-maps-v1.0"
BUFFER_VARIANTS_METERS = (0.0, 0.5, 1.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the selected ManeuverToken through nuPlan's closed-loop simulation stack on the decisive replay "
            "subset and compare replay classes against executed outcomes."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--maps-root", type=Path, default=DEFAULT_MAPS_ROOT)
    parser.add_argument("--map-version", default=DEFAULT_MAP_VERSION)
    parser.add_argument("--sensor-root", type=Path)
    parser.add_argument("--replay-infeasible-limit", type=int)
    parser.add_argument("--safe-match-limit", type=int)
    parser.add_argument("--near-miss-threshold-m", type=float, default=DEFAULT_NEAR_MISS_THRESHOLD_M)
    parser.add_argument("--calibrate-clearance", action="store_true")
    parser.add_argument("--calibration-json", type=Path, default=DEFAULT_CALIBRATION_JSON)
    parser.add_argument("--calibration-markdown", type=Path, default=DEFAULT_CALIBRATION_MARKDOWN)
    parser.add_argument(
        "--compare-selectors",
        action="store_true",
        help="Run proxy, replay-calibrated, and replay-oracle selectors on the same bridge scenes.",
    )
    parser.add_argument(
        "--comparison-selector-model",
        type=Path,
        help="Replay-calibrated selector artifact used when --compare-selectors is enabled.",
    )
    parser.add_argument(
        "--comparison-failure-gate-model",
        type=Path,
        help="Optional embedded failure-gate policy artifact used instead of scalar replay calibration.",
    )
    parser.add_argument(
        "--comparison-holdout-only",
        action="store_true",
        help="If the selector artifact records a DB split, evaluate only its holdout DBs.",
    )
    parser.add_argument(
        "--comparison-constrained-risk-threshold",
        type=float,
        help="If set, compare constrained replay-risk selection instead of scalar risk-penalty selection.",
    )
    parser.add_argument(
        "--comparison-meta-value-policy",
        type=Path,
        help="Optional meta-lambda value policy JSON used instead of scalar risk-penalty selection.",
    )
    parser.add_argument(
        "--comparison-boundary-intervention-model",
        type=Path,
        help="Optional boundary-intervention selector artifact used instead of scalar/meta replay calibration.",
    )
    parser.add_argument(
        "--comparison-selective-regret-model",
        type=Path,
        help="Optional selective-regret selector artifact used instead of scalar/meta replay calibration.",
    )
    parser.add_argument(
        "--comparison-border-only",
        action="store_true",
        help="Select bridge scenes from selector-border cases instead of replay-class matched cases.",
    )
    parser.add_argument(
        "--comparison-border-limit",
        type=int,
        help="Maximum number of selector-border scenes to evaluate when --comparison-border-only is enabled.",
    )
    parser.add_argument(
        "--comparison-border-selector-model",
        type=Path,
        help="Optional reference selector model used to define selector-border scenes.",
    )
    parser.add_argument(
        "--comparison-border-meta-value-policy",
        type=Path,
        help="Optional meta policy paired with --comparison-border-selector-model for selector-border sampling.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = json.loads(args.input_json.read_text(encoding="utf-8"))
    executor = lambda scene: execute_scene_selected_token_closed_loop(
        scene,
        maps_root=args.maps_root,
        map_version=str(args.map_version),
        sensor_root=args.sensor_root,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    if args.compare_selectors:
        if args.comparison_selector_model is None and args.comparison_failure_gate_model is None:
            raise ValueError("--compare-selectors requires a comparison selector or failure-gate model")
        split_model_path = args.comparison_selector_model or args.comparison_failure_gate_model
        report = _filter_to_selector_holdout_if_requested(
            report,
            selector_model_path=split_model_path,
            enabled=bool(args.comparison_holdout_only),
        )
        enriched = enrich_report_with_closed_loop_selector_comparison(
            report,
            replay_calibrated_selector=(
                None if args.comparison_selector_model is None else load_selector(args.comparison_selector_model)
            ),
            failure_gate_policy=(
                None
                if args.comparison_failure_gate_model is None
                else load_failure_gate_policy(args.comparison_failure_gate_model)
            ),
            boundary_intervention_policy=(
                None
                if args.comparison_boundary_intervention_model is None
                else load_boundary_intervention_policy(args.comparison_boundary_intervention_model)
            ),
            selective_regret_policy=(
                None
                if args.comparison_selective_regret_model is None
                else load_selective_regret_policy(args.comparison_selective_regret_model)
            ),
            meta_value_policy=(
                None
                if args.comparison_meta_value_policy is None
                else json.loads(args.comparison_meta_value_policy.read_text(encoding="utf-8"))
            ),
            executor=executor,
            replay_infeasible_limit=args.replay_infeasible_limit,
            safe_match_limit=args.safe_match_limit,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            constrained_risk_threshold=args.comparison_constrained_risk_threshold,
            comparison_border_only=bool(args.comparison_border_only),
            comparison_border_limit=args.comparison_border_limit,
            comparison_border_selector=(
                None
                if args.comparison_border_selector_model is None
                else load_selector(args.comparison_border_selector_model)
            ),
            comparison_border_meta_value_policy=(
                None
                if args.comparison_border_meta_value_policy is None
                else json.loads(args.comparison_border_meta_value_policy.read_text(encoding="utf-8"))
            ),
        )
    else:
        enriched = enrich_report_with_closed_loop_execution(
            report,
            executor=executor,
            replay_infeasible_limit=args.replay_infeasible_limit,
            safe_match_limit=args.safe_match_limit,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(enriched, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(enriched)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    if args.calibrate_clearance:
        calibration = build_bridge_calibration_report(
            enriched,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        )
        args.calibration_json.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_json.write_text(json.dumps(calibration, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        args.calibration_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.calibration_markdown.write_text(markdown_calibration_report(calibration) + "\n", encoding="utf-8")
    print(markdown)
    return 0


def enrich_report_with_closed_loop_execution(
    report: Mapping[str, Any],
    *,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    replay_infeasible_limit: int | None = None,
    safe_match_limit: int | None = None,
    near_miss_threshold_m: float = DEFAULT_NEAR_MISS_THRESHOLD_M,
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(report))
    selected = select_bridge_scenes(
        enriched.get("scenes", []),
        replay_infeasible_limit=replay_infeasible_limit,
        safe_match_limit=safe_match_limit,
    )
    executed_scenes: list[dict[str, Any]] = []
    for scene in selected:
        executed = dict(scene)
        executed.update(executor(executed))
        executed["executed_failure_rung"] = classify_executed_failure_rung(
            replay_failure_rung=str(scene.get("failure_rung", "")),
            executed_min_clearance_m=executed.get("selected_token_executed_min_clearance_m"),
            near_miss_threshold_m=float(near_miss_threshold_m),
        )
        executed_scenes.append(executed)
    enriched["schema"] = "nuplan_selected_token_closed_loop_bridge_v1"
    enriched["bridge_scene_count"] = len(executed_scenes)
    enriched["bridge_scenes"] = executed_scenes
    enriched["closed_loop_bridge_table"] = _closed_loop_bridge_table(executed_scenes)
    enriched["closed_loop_bridge_summary"] = _closed_loop_bridge_summary(executed_scenes)
    return enriched


def enrich_report_with_closed_loop_selector_comparison(
    report: Mapping[str, Any],
    *,
    replay_calibrated_selector: Any | None,
    failure_gate_policy: Mapping[str, Any] | None = None,
    boundary_intervention_policy: Mapping[str, Any] | None = None,
    selective_regret_policy: Mapping[str, Any] | None = None,
    meta_value_policy: Mapping[str, Any] | None = None,
    executor: Callable[[dict[str, Any]], dict[str, Any]],
    replay_infeasible_limit: int | None,
    safe_match_limit: int | None,
    near_miss_threshold_m: float = DEFAULT_NEAR_MISS_THRESHOLD_M,
    constrained_risk_threshold: float | None = None,
    comparison_border_only: bool = False,
    comparison_border_limit: int | None = None,
    comparison_border_selector: Any | None = None,
    comparison_border_meta_value_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    enriched = json.loads(json.dumps(report))
    calibrated_name = "replay_calibrated"
    if replay_calibrated_selector is None and failure_gate_policy is None:
        raise ValueError("selector comparison requires a replay selector or failure-gate policy")
    calibrated_fn = lambda scene: _select_with_score_model(scene, replay_calibrated_selector)
    calibrated_margin_fn = lambda scene: _score_model_decision(scene, replay_calibrated_selector)
    if failure_gate_policy is not None:
        calibrated_name = "failure_gate"
        calibrated_fn = lambda scene: select_with_failure_gate_policy(scene, failure_gate_policy)
        calibrated_margin_fn = lambda scene: _failure_gate_decision(scene, failure_gate_policy)
    if boundary_intervention_policy is not None:
        if replay_calibrated_selector is None:
            raise ValueError("boundary intervention comparison requires --comparison-selector-model")
        calibrated_name = "boundary_intervention"
        calibrated_fn = lambda scene: select_with_boundary_intervention(
            scene,
            baseline_selector=replay_calibrated_selector,
            policy=boundary_intervention_policy,
        )
        calibrated_margin_fn = lambda scene: _boundary_intervention_decision(
            scene,
            baseline_selector=replay_calibrated_selector,
            boundary_intervention_policy=boundary_intervention_policy,
        )
    if selective_regret_policy is not None:
        if replay_calibrated_selector is None:
            raise ValueError("selective regret comparison requires --comparison-selector-model")
        calibrated_name = "selective_regret"
        calibrated_fn = lambda scene: select_with_selective_regret(
            scene,
            baseline_selector=replay_calibrated_selector,
            regret_selector=selective_regret_policy["regret_model"],
            top_k=int(selective_regret_policy["config"].get("top_k", 2)),
            margin_threshold=float(selective_regret_policy["config"].get("margin_threshold", 1.5)),
            intervention_threshold=float(selective_regret_policy["config"].get("intervention_threshold", 0.0)),
        )
        calibrated_margin_fn = lambda scene: _selective_regret_decision(
            scene,
            baseline_selector=replay_calibrated_selector,
            selective_regret_policy=selective_regret_policy,
        )
    if constrained_risk_threshold is not None:
        if replay_calibrated_selector is None:
            raise ValueError("constrained-risk comparison requires --comparison-selector-model")
        calibrated_name = f"replay_constrained_risk_{float(constrained_risk_threshold):g}"
        calibrated_fn = lambda scene: select_with_constrained_replay_risk(
            scene,
            replay_calibrated_selector,
            risk_threshold=float(constrained_risk_threshold),
        )
        calibrated_margin_fn = lambda scene: _constrained_risk_decision(
            scene,
            replay_calibrated_selector,
            risk_threshold=float(constrained_risk_threshold),
        )
    if meta_value_policy is not None:
        if replay_calibrated_selector is None:
            raise ValueError("meta-value comparison requires --comparison-selector-model")
        calibrated_name = "meta_lambda_value"
        calibrated_fn = lambda scene: select_with_meta_lambda_value_policy(
            scene,
            replay_calibrated_selector,
            meta_value_policy,
            proxy_score_weight=float(getattr(replay_calibrated_selector, "proxy_score_weight", 1.0)),
            progress_weight=float(getattr(replay_calibrated_selector, "progress_weight", 0.0)),
            unsafe_penalty=float(getattr(replay_calibrated_selector, "unsafe_penalty", 2.0)),
        )
        calibrated_margin_fn = lambda scene: _meta_value_decision(
            scene,
            replay_calibrated_selector,
            meta_value_policy,
            proxy_score_weight=float(getattr(replay_calibrated_selector, "proxy_score_weight", 1.0)),
            progress_weight=float(getattr(replay_calibrated_selector, "progress_weight", 0.0)),
            unsafe_penalty=float(getattr(replay_calibrated_selector, "unsafe_penalty", 2.0)),
        )
    if comparison_border_only:
        reference_selector = (
            replay_calibrated_selector
            if comparison_border_selector is None
            else comparison_border_selector
        )
        if comparison_border_meta_value_policy is not None:
            reference_name = "border_reference_meta_lambda_value"
            border_reference_fn = lambda scene: _meta_value_decision(
                scene,
                reference_selector,
                comparison_border_meta_value_policy,
                proxy_score_weight=float(getattr(reference_selector, "proxy_score_weight", 1.0)),
                progress_weight=float(getattr(reference_selector, "progress_weight", 0.0)),
                unsafe_penalty=float(getattr(reference_selector, "unsafe_penalty", 2.0)),
            )
        else:
            reference_name = "border_reference"
            border_reference_fn = lambda scene: _score_model_decision(scene, reference_selector)
        bridge_base_scenes, border_summary = select_selector_border_scenes(
            enriched.get("scenes", []),
            selector_a_name=calibrated_name,
            selector_a_decision=calibrated_margin_fn,
            selector_b_name=reference_name,
            selector_b_decision=border_reference_fn,
            limit=comparison_border_limit,
        )
        enriched["selector_border_sampling"] = border_summary
    else:
        bridge_base_scenes = select_bridge_scenes(
            enriched.get("scenes", []),
            replay_infeasible_limit=replay_infeasible_limit,
            safe_match_limit=safe_match_limit,
        )
    selector_fns: list[tuple[str, Callable[[Mapping[str, Any]], str]]] = [
        ("proxy_selector", lambda scene: str(scene.get("selected_token", ""))),
        (calibrated_name, calibrated_fn),
        ("replay_oracle", _select_replay_oracle_token),
    ]
    if comparison_border_selector is not None:
        if comparison_border_meta_value_policy is not None:
            selector_fns.insert(
                2,
                (
                    "border_reference_meta_lambda_value",
                    lambda scene: select_with_meta_lambda_value_policy(
                        scene,
                        comparison_border_selector,
                        comparison_border_meta_value_policy,
                        proxy_score_weight=float(getattr(comparison_border_selector, "proxy_score_weight", 1.0)),
                        progress_weight=float(getattr(comparison_border_selector, "progress_weight", 0.0)),
                        unsafe_penalty=float(getattr(comparison_border_selector, "unsafe_penalty", 2.0)),
                    ),
                ),
            )
        else:
            selector_fns.insert(
                2,
                ("border_reference", lambda scene: _select_with_score_model(scene, comparison_border_selector)),
            )
    bridge_scenes: list[dict[str, Any]] = []
    for base_scene in bridge_base_scenes:
        for selector_name, selector_fn in selector_fns:
            token = selector_fn(base_scene)
            scene = scene_with_selected_candidate(base_scene, token=token, selector_name=selector_name)
            scene.update(executor(scene))
            scene["executed_failure_rung"] = classify_executed_failure_rung(
                replay_failure_rung=str(scene.get("failure_rung", "")),
                executed_min_clearance_m=scene.get("selected_token_executed_min_clearance_m"),
                near_miss_threshold_m=float(near_miss_threshold_m),
            )
            bridge_scenes.append(scene)
    enriched["schema"] = "nuplan_selector_closed_loop_bridge_comparison_v1"
    enriched["bridge_base_scene_count"] = len(bridge_base_scenes)
    enriched["bridge_scene_count"] = len(bridge_scenes)
    enriched["bridge_scenes"] = bridge_scenes
    enriched["closed_loop_bridge_table"] = _closed_loop_bridge_table(bridge_scenes)
    enriched["closed_loop_bridge_summary"] = _closed_loop_bridge_summary(bridge_scenes)
    enriched["closed_loop_selector_comparison"] = _selector_comparison_table(bridge_scenes)
    return enriched


def select_bridge_scenes(
    scenes: list[dict[str, Any]],
    *,
    replay_infeasible_limit: int | None = None,
    safe_match_limit: int | None,
) -> list[dict[str, Any]]:
    replay_infeasible = [
        scene
        for scene in scenes
        if bool(scene.get("proxy_safe_selected")) and str(scene.get("failure_rung")) == REPLAY_INFEASIBLE_RUNG
    ]
    if replay_infeasible_limit is not None:
        replay_infeasible = replay_infeasible[: max(0, int(replay_infeasible_limit))]
    replay_safe = [
        scene
        for scene in scenes
        if bool(scene.get("proxy_safe_selected")) and not bool(scene.get("selected_token_realized_near_miss"))
    ]
    if safe_match_limit is None:
        safe_match_limit = min(len(replay_infeasible), len(replay_safe))
    matched_safe = _match_replay_safe_scenes(replay_infeasible, replay_safe, safe_match_limit)
    selected: list[dict[str, Any]] = []
    for scene in replay_infeasible:
        row = dict(scene)
        row["bridge_replay_class"] = "replay_infeasible"
        selected.append(row)
    for scene in matched_safe:
        row = dict(scene)
        row["bridge_replay_class"] = "replay_safe"
        selected.append(row)
    return selected


def select_selector_border_scenes(
    scenes: list[dict[str, Any]],
    *,
    selector_a_name: str,
    selector_a_decision: Callable[[Mapping[str, Any]], dict[str, Any]],
    selector_b_name: str,
    selector_b_decision: Callable[[Mapping[str, Any]], dict[str, Any]],
    limit: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ranked: list[tuple[tuple[int, float], dict[str, Any]]] = []
    disagreement_count = 0
    for scene in scenes:
        a_decision = selector_a_decision(scene)
        b_decision = selector_b_decision(scene)
        disagree = str(a_decision["token"]) != str(b_decision["token"])
        disagreement_count += 1 if disagree else 0
        min_margin = min(float(a_decision["margin"]), float(b_decision["margin"]))
        row = dict(scene)
        row["bridge_replay_class"] = (
            "replay_infeasible"
            if bool(scene.get("proxy_safe_selected")) and str(scene.get("failure_rung")) == REPLAY_INFEASIBLE_RUNG
            else "replay_safe"
        )
        row["selector_border"] = {
            selector_a_name: a_decision,
            selector_b_name: b_decision,
            "disagree": disagree,
            "min_margin": _optional_round(min_margin),
        }
        ranked.append((((0 if disagree else 1), min_margin), row))
    ranked.sort(key=lambda item: item[0])
    selected = [row for _, row in ranked]
    if limit is not None:
        selected = selected[: max(0, int(limit))]
    summary = {
        "mode": "selector_border",
        "selector_a": selector_a_name,
        "selector_b": selector_b_name,
        "candidate_scene_count": len(scenes),
        "selected_scene_count": len(selected),
        "total_disagreement_count": disagreement_count,
        "selected_disagreement_count": sum(
            1 for row in selected if bool(row.get("selector_border", {}).get("disagree"))
        ),
    }
    return selected, summary


def scene_with_selected_candidate(
    scene: Mapping[str, Any],
    *,
    token: str,
    selector_name: str,
) -> dict[str, Any]:
    candidate = _candidate_by_token(scene).get(str(token))
    replay = _replay_by_token(scene).get(str(token))
    updated = json.loads(json.dumps(scene))
    updated["bridge_selector"] = selector_name
    updated["bridge_base_selected_token"] = scene.get("selected_token")
    if candidate is None:
        updated["bridge_selected_token_available"] = False
        return updated
    updated["bridge_selected_token_available"] = True
    updated["selected_token"] = str(token)
    updated["selected_token_rollout"] = dict(candidate)
    updated["selected_token_min_proxy_clearance_m"] = candidate.get("min_proxy_clearance_m")
    updated["selected_token_proxy_safe"] = bool(candidate.get("proxy_safe"))
    updated["proxy_safe_selected"] = bool(candidate.get("proxy_safe"))
    updated["selected_token_score"] = candidate.get("score")
    updated["selection_source"] = selector_name
    if replay is not None:
        updated["selected_token_realized_clearance_trace"] = replay.get("realized_clearance_trace", [])
        updated["selected_token_realized_collision"] = replay.get("realized_collision")
        updated["selected_token_realized_min_clearance_m"] = replay.get("realized_min_clearance_m")
        updated["selected_token_realized_min_clearance_t"] = replay.get("realized_min_clearance_t")
        updated["selected_token_realized_near_miss"] = replay.get("realized_near_miss")
    updated["failure_rung"] = _selector_variant_replay_rung(updated)
    return updated


def _score_model_decision(scene: Mapping[str, Any], selector: Any) -> dict[str, Any]:
    scored = _candidate_scores(
        scene,
        score_fn=lambda candidate: float(
            selector.predict_score(candidate["features"] if "features" in candidate else _features(scene, candidate))
        ),
    )
    return _decision_from_scored_candidates(scored)


def _failure_gate_decision(scene: Mapping[str, Any], policy: Mapping[str, Any]) -> dict[str, Any]:
    safe_token_fn = build_safe_token_fn(
        policy["safe_selector"],
        fallback_mode=str(policy.get("fallback_mode", "score")),
        fallback_risk_threshold=float(policy.get("fallback_risk_threshold", 0.5)),
    )
    fallback_token = str(safe_token_fn(scene))
    probability = predict_failure_probability(
        policy["gate_model"],
        failure_gate_feature_row(scene, fallback_token),
    )
    threshold = float(policy.get("gate_threshold", 0.5))
    progress_allowed = fallback_meets_progress_retention(
        scene,
        fallback_token=fallback_token,
        min_fallback_progress_retention=float(policy.get("min_fallback_progress_retention", 0.0)),
    )
    token = fallback_token if probability >= threshold and progress_allowed else str(scene.get("selected_token", ""))
    return {
        "token": token,
        "score": float(probability),
        "margin": abs(float(probability) - threshold),
        "fallback_token": fallback_token,
        "switch_probability": float(probability),
        "gate_threshold": threshold,
        "progress_allowed": progress_allowed,
    }


def _constrained_risk_decision(scene: Mapping[str, Any], selector: Any, *, risk_threshold: float) -> dict[str, Any]:
    scored_rows = []
    for candidate in list(scene.get("candidates", [])):
        features = candidate["features"] if "features" in candidate else _features(scene, candidate)
        risk = float(selector.predict_risk(features))
        utility = _constrained_candidate_utility(
            features,
            selector,
            proxy_score_weight=None,
            progress_weight=None,
            unsafe_penalty=None,
        )
        score = utility if risk <= float(risk_threshold) else -1.0e6 - risk
        scored_rows.append(_candidate_score_row(candidate, score))
    return _decision_from_scored_candidates(scored_rows)


def _meta_value_decision(
    scene: Mapping[str, Any],
    selector: Any,
    policy: Mapping[str, Any],
    *,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> dict[str, Any]:
    selected_lambda = float(predict_meta_lambda_value(scene, selector, policy))
    scored = _candidate_scores(
        scene,
        score_fn=lambda candidate: _lambda_risk_candidate_score(
            candidate["features"] if "features" in candidate else _features(scene, candidate),
            selector,
            risk_weight=selected_lambda,
            proxy_score_weight=proxy_score_weight,
            progress_weight=progress_weight,
            unsafe_penalty=unsafe_penalty,
        ),
    )
    decision = _decision_from_scored_candidates(scored)
    decision["lambda"] = selected_lambda
    return decision


def _selective_regret_decision(
    scene: Mapping[str, Any],
    *,
    baseline_selector: Any,
    selective_regret_policy: Mapping[str, Any],
) -> dict[str, Any]:
    config = dict(selective_regret_policy["config"])
    token = select_with_selective_regret(
        scene,
        baseline_selector=baseline_selector,
        regret_selector=selective_regret_policy["regret_model"],
        top_k=int(config.get("top_k", 2)),
        margin_threshold=float(config.get("margin_threshold", 1.5)),
        intervention_threshold=float(config.get("intervention_threshold", 0.0)),
    )
    ranked = _candidate_scores(
        scene,
        score_fn=lambda candidate: float(
            baseline_selector.predict_score(
                candidate["features"] if "features" in candidate else _features(scene, candidate)
            )
        ),
    )
    decision = _decision_from_scored_candidates(ranked)
    decision["token"] = token
    return decision


def _boundary_intervention_decision(
    scene: Mapping[str, Any],
    *,
    baseline_selector: Any,
    boundary_intervention_policy: Mapping[str, Any],
) -> dict[str, Any]:
    token = select_with_boundary_intervention(
        scene,
        baseline_selector=baseline_selector,
        policy=boundary_intervention_policy,
    )
    ranked = _candidate_scores(
        scene,
        score_fn=lambda candidate: float(
            baseline_selector.predict_score(
                candidate["features"] if "features" in candidate else _features(scene, candidate)
            )
        ),
    )
    decision = _decision_from_scored_candidates(ranked)
    decision["token"] = token
    by_token = {
        str(candidate.get("token", "")): candidate
        for candidate in list(scene.get("candidates", []))
    }
    baseline_candidate = by_token.get(str(ranked[0]["token"])) if ranked else None
    chosen_candidate = by_token.get(token)
    if baseline_candidate is not None and chosen_candidate is not None and len(ranked) >= 2:
        baseline_features = (
            baseline_candidate["features"] if "features" in baseline_candidate else _features(scene, baseline_candidate)
        )
        chosen_features = (
            chosen_candidate["features"] if "features" in chosen_candidate else _features(scene, chosen_candidate)
        )
        decision["predicted_delta"] = predict_boundary_delta(
            {
                "scene_margin": float(ranked[0]["score"] - ranked[1]["score"]),
                "score_delta": float(
                    baseline_selector.predict_score(chosen_features)
                    - baseline_selector.predict_score(baseline_features)
                ),
                "progress_delta": float(
                    chosen_features["candidate_final_progress_m"] - baseline_features["candidate_final_progress_m"]
                ),
                "proxy_safe_delta": float(
                    chosen_features["candidate_proxy_safe"] - baseline_features["candidate_proxy_safe"]
                ),
                "clearance_delta": float(
                    chosen_features["candidate_min_proxy_clearance_m"]
                    - baseline_features["candidate_min_proxy_clearance_m"]
                ),
                "speed_scale_delta": float(
                    chosen_features["candidate_speed_scale"] - baseline_features["candidate_speed_scale"]
                ),
                "lateral_offset_delta": float(
                    chosen_features["candidate_lateral_offset_m"] - baseline_features["candidate_lateral_offset_m"]
                ),
                "obstacle_pressure": float(baseline_features["obstacle_pressure"]),
                "route_blockage": float(baseline_features["route_blockage"]),
                "nearest_actor_distance_m": float(baseline_features["nearest_actor_distance_m"]),
                "rear_closing_actor_count": float(baseline_features["rear_closing_actor_count"]),
                "crossing_actor_count": float(baseline_features["crossing_actor_count"]),
                "baseline_progress_m": float(baseline_features["candidate_final_progress_m"]),
                "baseline_proxy_clearance_m": float(baseline_features["candidate_min_proxy_clearance_m"]),
            },
            boundary_intervention_policy,
        )
    return decision


def _candidate_scores(
    scene: Mapping[str, Any],
    *,
    score_fn: Callable[[Mapping[str, Any]], float],
) -> list[dict[str, Any]]:
    return [
        _candidate_score_row(candidate, score_fn(candidate))
        for candidate in list(scene.get("candidates", []))
    ]


def _candidate_score_row(candidate: Mapping[str, Any], score: float) -> dict[str, Any]:
    return {
        "token": str(candidate.get("token", "")),
        "score": float(score),
        "clearance": float(candidate.get("min_proxy_clearance_m", 0.0)),
        "progress": float(candidate.get("final_progress_m", 0.0)),
    }


def _decision_from_scored_candidates(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {"token": "", "margin": math.inf}
    ordered = sorted(
        scored,
        key=lambda row: (float(row["score"]), float(row["clearance"]), float(row["progress"])),
        reverse=True,
    )
    margin = math.inf if len(ordered) < 2 else float(ordered[0]["score"]) - float(ordered[1]["score"])
    return {
        "token": str(ordered[0]["token"]),
        "margin": margin,
        "score": float(ordered[0]["score"]),
    }


def _lambda_risk_candidate_score(
    features: Mapping[str, float],
    selector: Any,
    *,
    risk_weight: float,
    proxy_score_weight: float,
    progress_weight: float,
    unsafe_penalty: float,
) -> float:
    risk = float(selector.predict_risk(features))
    proxy_score = float(features["candidate_score_heuristic"])
    progress = float(features["candidate_final_progress_m"])
    proxy_safe = bool(float(features["candidate_proxy_safe"]) >= 0.5)
    penalty = 0.0 if proxy_safe else float(unsafe_penalty)
    return (
        float(proxy_score_weight) * proxy_score
        + float(progress_weight) * progress
        - float(risk_weight) * risk
        - penalty
    )


def _selector_variant_replay_rung(scene: Mapping[str, Any]) -> str:
    if not bool(scene.get("safe_token_existed", True)):
        return "no safe token existed"
    if not bool(scene.get("selected_token_proxy_safe")):
        return "safe token existed but selector missed it"
    replay_min = scene.get("selected_token_realized_min_clearance_m")
    if replay_min is None:
        return "metric/spec ambiguity"
    if float(replay_min) < DEFAULT_NEAR_MISS_THRESHOLD_M:
        return REPLAY_INFEASIBLE_RUNG
    return "resolved safe"


def _filter_to_selector_holdout_if_requested(
    report: Mapping[str, Any],
    *,
    selector_model_path: Path,
    enabled: bool,
) -> dict[str, Any]:
    copied = json.loads(json.dumps(report))
    if not enabled:
        return copied
    payload = json.loads(selector_model_path.read_text(encoding="utf-8"))
    holdout_dbs = set(payload.get("training", {}).get("split", {}).get("holdout_dbs", []))
    if not holdout_dbs:
        return copied
    holdout_names = {Path(path).name for path in holdout_dbs}
    scenes = [
        scene
        for scene in copied.get("scenes", [])
        if str(scene.get("source_db_file")) in holdout_dbs
        or Path(str(scene.get("source_db_file", ""))).name in holdout_names
    ]
    copied["scenes"] = scenes
    copied["scene_count"] = len(scenes)
    copied["comparison_filter"] = {
        "filter": "selector_model_holdout_dbs",
        "selector_model": str(selector_model_path),
        "holdout_db_count": len(holdout_dbs),
        "scene_count": len(scenes),
    }
    return copied


def _match_replay_safe_scenes(
    replay_infeasible: list[dict[str, Any]],
    replay_safe: list[dict[str, Any]],
    safe_match_limit: int,
) -> list[dict[str, Any]]:
    remaining = list(replay_safe)
    matched: list[dict[str, Any]] = []
    for scene in replay_infeasible:
        if len(matched) >= safe_match_limit or not remaining:
            break
        selected = _pop_best_replay_safe_match(scene, remaining)
        matched.append(selected)
    return matched


def _pop_best_replay_safe_match(scene: dict[str, Any], remaining: list[dict[str, Any]]) -> dict[str, Any]:
    def score(candidate: dict[str, Any]) -> tuple[int, int, float]:
        return (
            int(str(candidate.get("selected_token")) == str(scene.get("selected_token"))),
            int(str(candidate.get("scenario_type")) == str(scene.get("scenario_type"))),
            -abs(
                float(candidate.get("selected_token_min_proxy_clearance_m", 0.0))
                - float(scene.get("selected_token_min_proxy_clearance_m", 0.0))
            ),
        )

    best_index = max(range(len(remaining)), key=lambda index: score(remaining[index]))
    return remaining.pop(best_index)


def classify_executed_failure_rung(
    *,
    replay_failure_rung: str,
    executed_min_clearance_m: float | None,
    near_miss_threshold_m: float,
) -> str:
    if replay_failure_rung == REPLAY_INFEASIBLE_RUNG:
        if executed_min_clearance_m is not None and float(executed_min_clearance_m) < near_miss_threshold_m:
            return "proxy_safe_but_realized_failed"
        return "replay_infeasible_but_executed_safe"
    if executed_min_clearance_m is not None and float(executed_min_clearance_m) < near_miss_threshold_m:
        return "replay_safe_but_executed_failed"
    return "executed_safe"


def execute_scene_selected_token_closed_loop(
    scene: dict[str, Any],
    *,
    maps_root: Path,
    map_version: str,
    sensor_root: Path | None,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    imports = _nuplan_closed_loop_imports()
    replay_query = _nuplan_replay_imports()
    if not maps_root.exists():
        raise FileNotFoundError(f"maps root not found: {maps_root}")
    dt_s = float(scene.get("selected_token_rollout_dt_s", 0.5))
    horizon_s = float(scene.get("selected_token_rollout_horizon_s", 4.0))
    scenario = imports["NuPlanScenario"](
        data_root=str(Path(str(scene["source_db_file"])).parent),
        log_file_load_path=str(scene["source_db_file"]),
        initial_lidar_token=str(scene["scenario_token"]),
        initial_lidar_timestamp=int(scene["sensor_timestamp_us"]),
        scenario_type=str(scene.get("scenario_type") or "unknown"),
        map_root=str(maps_root),
        map_version=str(map_version),
        map_name=str(scene["map_name"]),
        scenario_extraction_info=imports["ScenarioExtractionInfo"](
            scenario_name=str(scene.get("scenario_type") or "unknown"),
            scenario_duration=horizon_s,
            extraction_offset=0.0,
            subsample_ratio=1.0,
        ),
        ego_vehicle_parameters=imports["get_pacifica_parameters"](),
        sensor_root=None if sensor_root is None else str(sensor_root),
    )
    tracked_objects = list(
        replay_query["get_tracked_objects_for_lidarpc_token_from_db"](
            str(scene["source_db_file"]),
            str(scene["scenario_token"]),
        )
    )
    actor_tracks = _build_actor_tracks(
        query=replay_query,
        source_db_file=str(scene["source_db_file"]),
        tracked_objects=tracked_objects,
        start_timestamp_us=int(scene["sensor_timestamp_us"]),
        end_timestamp_us=int(scene["sensor_timestamp_us"] + round(horizon_s * 1_000_000)),
    )
    planner = _SelectedTokenPlanner(
        imports=imports,
        scene=scene,
        fixed_states=_ego_states_from_rollout_dict(
            imports=imports,
            ego_state=scenario.initial_ego_state,
            rollout=dict(scene["selected_token_rollout"]),
            dt_s=dt_s,
        ),
    )
    simulation = imports["Simulation"](
        simulation_setup=imports["SimulationSetup"](
            time_controller=imports["StepSimulationTimeController"](scenario),
            observations=imports["TracksObservation"](scenario),
            ego_controller=imports["PerfectTrackingController"](scenario),
            scenario=scenario,
        ),
        callback=None,
        simulation_history_buffer_duration=2.0,
    )
    runner = imports["SimulationRunner"](simulation, planner)
    runner.run()
    return _closed_loop_metrics(
        scene=scene,
        history=simulation.history,
        near_miss_threshold_m=near_miss_threshold_m,
        actor_tracks=actor_tracks,
        rollout_dt_s=dt_s,
        rollout_horizon_s=horizon_s,
        fixed_states=planner._fixed_states,
        initial_ego_state=scenario.initial_ego_state,
    )


class _SelectedTokenPlanner:
    requires_scenario = False

    def __init__(self, *, imports: dict[str, Any], scene: dict[str, Any], fixed_states: list[Any]) -> None:
        self._imports = imports
        self._scene = scene
        self._fixed_states = fixed_states
        self._full_trajectory = imports["InterpolatedTrajectory"](fixed_states)

    def name(self) -> str:
        return "selected_maneuvertoken_closed_loop_bridge"

    def initialize(self, initialization: Any) -> None:
        del initialization

    def observation_type(self) -> type[Any]:
        return self._imports["DetectionsTracks"]

    def compute_planner_trajectory(self, current_input: Any) -> Any:
        current_time = current_input.iteration.time_point
        current_state = self._full_trajectory.get_state_at_time(current_time)
        future_states = [state for state in self._fixed_states if state.time_point.time_us > current_time.time_us]
        states = [current_state] + future_states
        if len(states) == 1:
            states.append(self._fixed_states[-1])
        return self._imports["InterpolatedTrajectory"](states)

    def compute_trajectory(self, current_input: Any) -> Any:
        return self.compute_planner_trajectory(current_input)

    def generate_planner_report(self, clear_stats: bool = True) -> Any:
        return {"name": self.name(), "clear_stats": bool(clear_stats)}


def _ego_states_from_rollout_dict(
    *,
    imports: dict[str, Any],
    ego_state: Any,
    rollout: dict[str, Any],
    dt_s: float,
) -> list[Any]:
    states = [ego_state]
    base_x = float(ego_state.rear_axle.x)
    base_y = float(ego_state.rear_axle.y)
    base_heading = float(ego_state.rear_axle.heading)
    vehicle_parameters = ego_state.car_footprint.vehicle_parameters
    previous_global = (base_x, base_y)
    previous_velocity = (
        float(ego_state.dynamic_car_state.rear_axle_velocity_2d.x),
        float(ego_state.dynamic_car_state.rear_axle_velocity_2d.y),
    )
    poses = list(rollout.get("poses", []))
    for index, pose in enumerate(poses):
        x_global, y_global = _local_to_global(
            origin_x=base_x,
            origin_y=base_y,
            origin_heading=base_heading,
            local_x=float(pose[0]),
            local_y=float(pose[1]),
        )
        heading_global = _normalize_angle(base_heading + float(pose[2]))
        velocity = (
            (x_global - previous_global[0]) / dt_s,
            (y_global - previous_global[1]) / dt_s,
        )
        acceleration = (
            (velocity[0] - previous_velocity[0]) / dt_s,
            (velocity[1] - previous_velocity[1]) / dt_s,
        )
        states.append(
            imports["EgoState"].build_from_rear_axle(
                rear_axle_pose=imports["StateSE2"](x_global, y_global, heading_global),
                rear_axle_velocity_2d=imports["StateVector2D"](velocity[0], velocity[1]),
                rear_axle_acceleration_2d=imports["StateVector2D"](acceleration[0], acceleration[1]),
                tire_steering_angle=float(getattr(ego_state, "tire_steering_angle", 0.0)),
                time_point=imports["TimePoint"](int(ego_state.time_us + round((index + 1) * dt_s * 1_000_000))),
                vehicle_parameters=vehicle_parameters,
            )
        )
        previous_global = (x_global, y_global)
        previous_velocity = velocity
    return states


def _closed_loop_metrics(
    *,
    scene: dict[str, Any],
    history: Any,
    near_miss_threshold_m: float,
    actor_tracks: list[dict[str, Any]],
    rollout_dt_s: float,
    rollout_horizon_s: float,
    fixed_states: list[Any],
    initial_ego_state: Any,
) -> dict[str, Any]:
    trace = []
    min_clearance_m = math.inf
    min_time_s = None
    impact_time_s = None
    first_erosion_time_s = None
    initial_time_s = None
    proxy_trace = list(scene.get("selected_token_realized_clearance_trace") or [])
    proxy_clearance_variants = _proxy_metric_variants(
        scene=scene,
        actor_tracks=actor_tracks,
        thresholds_m=BUFFER_VARIANTS_METERS,
        rollout_dt_s=rollout_dt_s,
        initial_ego_state=initial_ego_state,
    )
    aligned_executed_variants = _aligned_executed_metric_variants(
        history=history,
        actor_tracks=actor_tracks,
        thresholds_m=BUFFER_VARIANTS_METERS,
        rollout_dt_s=rollout_dt_s,
        rollout_horizon_s=rollout_horizon_s,
    )
    executed_clearance_variants = {buffer_m: math.inf for buffer_m in BUFFER_VARIANTS_METERS}
    replay_nearest = _nearest_replay_actor(
        scene=scene,
        actor_tracks=actor_tracks,
        initial_ego_state=initial_ego_state,
    )
    trajectory_trace = []
    for sample in history.data:
        time_s = float(sample.iteration.time_point.time_s)
        if initial_time_s is None:
            initial_time_s = time_s
        rel_time_s = round(time_s - initial_time_s, 6)
        if rel_time_s <= 0.0:
            continue
        metrics = _observation_clearance_metrics(sample.ego_state, sample.observation)
        executed_clearance = float(metrics["box_clearance_m"])
        proxy_clearance = _trace_value_at_time(proxy_trace, rel_time_s)
        if executed_clearance < min_clearance_m:
            min_clearance_m = executed_clearance
            min_time_s = rel_time_s
        for buffer_m in BUFFER_VARIANTS_METERS:
            value = float(metrics["inflated_clearance_by_buffer_m"][buffer_m])
            executed_clearance_variants[buffer_m] = min(executed_clearance_variants[buffer_m], value)
        if (
            first_erosion_time_s is None
            and proxy_clearance is not None
            and executed_clearance + 1.0e-6 < proxy_clearance
        ):
            first_erosion_time_s = rel_time_s
        if impact_time_s is None and executed_clearance < near_miss_threshold_m:
            impact_time_s = rel_time_s
        trace.append(
            {
                "time_s": rel_time_s,
                "proxy_clearance_m": _optional_round(proxy_clearance),
                "executed_center_distance_m": round(float(metrics["center_distance_m"]), 6),
                "executed_box_clearance_m": round(executed_clearance, 6),
                "executed_inflated_clearance_0_5m": round(
                    float(metrics["inflated_clearance_by_buffer_m"][0.5]), 6
                ),
                "executed_inflated_clearance_1_0m": round(
                    float(metrics["inflated_clearance_by_buffer_m"][1.0]), 6
                ),
                "executed_clearance_m": round(executed_clearance, 6),
                "nearest_actor_id": metrics["nearest_actor_id"],
                "nearest_actor_type": metrics["nearest_actor_type"],
                "nearest_actor_is_dynamic": metrics["nearest_actor_is_dynamic"],
            }
        )
        trajectory_trace.append(
            {
                "time_s": rel_time_s,
                "executed_x_m": round(float(sample.ego_state.rear_axle.x), 6),
                "executed_y_m": round(float(sample.ego_state.rear_axle.y), 6),
            }
        )
    if not math.isfinite(min_clearance_m):
        min_clearance_m = math.nan
    trajectory_agreement = _trajectory_agreement(
        trajectory_trace=trajectory_trace,
        fixed_states=fixed_states,
    )
    failure_reason = _bridge_failure_reason(
        scene=scene,
        replay_nearest_actor_id=replay_nearest["actor_id"],
        executed_trace=trace,
        trajectory_agreement=trajectory_agreement,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    return {
        "selected_token_executed_min_clearance_m": _optional_round(min_clearance_m),
        "selected_token_executed_min_clearance_t": _optional_round(min_time_s),
        "selected_token_executed_collision": (
            None if not math.isfinite(min_clearance_m) else bool(min_clearance_m < 0.0)
        ),
        "selected_token_executed_near_miss": (
            None if not math.isfinite(min_clearance_m) else bool(min_clearance_m < near_miss_threshold_m)
        ),
        "first_erosion_time_s": _optional_round(first_erosion_time_s),
        "time_from_first_erosion_to_impact_s": (
            None
            if first_erosion_time_s is None or impact_time_s is None or impact_time_s < first_erosion_time_s
            else round(float(impact_time_s - first_erosion_time_s), 6)
        ),
        "executed_clearance_source": "nuplan_closed_loop",
        "selected_token_executed_clearance_trace": trace,
        "replay_nearest_actor_id": replay_nearest["actor_id"],
        "replay_nearest_actor_type": replay_nearest["actor_type"],
        "executed_nearest_actor_id": _nearest_executed_actor_id(trace, min_time_s),
        "executed_nearest_actor_type": _nearest_executed_actor_type(trace, min_time_s),
        "ego_pose_source": "selected_token_rollout_rear_axle",
        "actor_geometry_source": "nuplan_tracked_object_oriented_box",
        "ego_footprint_geometry": "pacifica_oriented_box",
        "actor_footprint_geometry": "oriented_box",
        "threshold_m": float(near_miss_threshold_m),
        "failure_reason": failure_reason,
        "replay_horizon_s": rollout_horizon_s,
        "executed_horizon_s": _executed_horizon_s(trace),
        "num_replay_steps": len(proxy_trace),
        "num_executed_steps": len(trace),
        "dt_replay_s": rollout_dt_s,
        "dt_executed_s": _executed_dt_s(trace),
        "proxy_metric_variants": proxy_clearance_variants,
        "executed_metric_variants": {
            "center_distance_m": _minimum_trace_value(trace, "executed_center_distance_m"),
            "box_clearance_m": _optional_round(min_clearance_m),
            "inflated_box_clearance_0_0m": _optional_round(executed_clearance_variants[0.0]),
            "inflated_box_clearance_0_5m": _optional_round(executed_clearance_variants[0.5]),
            "inflated_box_clearance_1_0m": _optional_round(executed_clearance_variants[1.0]),
        },
        "aligned_executed_metric_variants": aligned_executed_variants,
        "trajectory_agreement": trajectory_agreement,
    }


def _observation_clearance_metrics(ego_state: Any, observation: Any) -> dict[str, Any]:
    ego_center = ego_state.center
    ego_radius = 0.5 * math.hypot(
        float(ego_state.car_footprint.vehicle_parameters.width),
        float(ego_state.car_footprint.vehicle_parameters.length),
    )
    min_center_distance_m = math.inf
    min_clearance_m = math.inf
    min_inflated = {buffer_m: math.inf for buffer_m in BUFFER_VARIANTS_METERS}
    nearest_actor_id = None
    nearest_actor_type = None
    nearest_actor_is_dynamic = None
    for obj in observation.tracked_objects:
        box = getattr(obj, "box", None) or getattr(obj, "oriented_box", None)
        if box is None:
            continue
        radius = 0.5 * math.hypot(float(box.width), float(box.length))
        center_distance = math.hypot(
            float(ego_center.x) - float(obj.center.x),
            float(ego_center.y) - float(obj.center.y),
        )
        clearance = center_distance - ego_radius - radius
        if clearance < min_clearance_m:
            nearest_actor_id = str(getattr(obj.metadata, "track_token", None) or getattr(obj.metadata, "token", None))
            nearest_actor_type = str(getattr(getattr(obj, "tracked_object_type", None), "name", "unknown")).lower()
            nearest_actor_is_dynamic = bool(hasattr(obj, "velocity"))
        min_center_distance_m = min(min_center_distance_m, center_distance)
        min_clearance_m = min(min_clearance_m, clearance)
        for buffer_m in BUFFER_VARIANTS_METERS:
            min_inflated[buffer_m] = min(min_inflated[buffer_m], clearance - float(buffer_m))
    return {
        "center_distance_m": min_center_distance_m,
        "box_clearance_m": min_clearance_m,
        "inflated_clearance_by_buffer_m": min_inflated,
        "nearest_actor_id": nearest_actor_id,
        "nearest_actor_type": nearest_actor_type,
        "nearest_actor_is_dynamic": nearest_actor_is_dynamic,
    }


def _trace_value_at_time(trace: list[dict[str, Any]], time_s: float) -> float | None:
    if not trace:
        return None
    nearest = min(trace, key=lambda row: abs(float(row["time_s"]) - time_s))
    value = nearest.get("realized_clearance_m")
    return None if value is None else float(value)


def _minimum_trace_value(trace: list[dict[str, Any]], key: str) -> float | None:
    values = [float(row[key]) for row in trace if row.get(key) is not None]
    return _optional_round(min(values)) if values else None


def _nearest_replay_actor(
    *,
    scene: dict[str, Any],
    actor_tracks: list[dict[str, Any]],
    initial_ego_state: Any,
) -> dict[str, str | None]:
    min_time_s = scene.get("selected_token_realized_min_clearance_t")
    if min_time_s is None or not actor_tracks:
        return {"actor_id": None, "actor_type": None}
    poses = list(scene.get("selected_token_rollout", {}).get("poses", []))
    if not poses:
        return {"actor_id": None, "actor_type": None}
    index = max(
        0,
        min(
            len(poses) - 1,
            int(round(float(min_time_s) / float(scene.get("selected_token_rollout_dt_s", 0.5)))) - 1,
        ),
    )
    pose = poses[index]
    global_x, global_y = _local_to_global(
        origin_x=float(initial_ego_state.rear_axle.x),
        origin_y=float(initial_ego_state.rear_axle.y),
        origin_heading=float(initial_ego_state.rear_axle.heading),
        local_x=float(pose[0]),
        local_y=float(pose[1]),
    )
    nearest = None
    nearest_distance = math.inf
    for actor in actor_tracks:
        actor_state = _actor_state_at_time(actor, t_s=float(min_time_s))
        distance = math.hypot(global_x - float(actor_state["x_m"]), global_y - float(actor_state["y_m"]))
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = actor
    return {
        "actor_id": None if nearest is None else str(nearest.get("token")),
        "actor_type": None,
    }


def _nearest_executed_actor_id(trace: list[dict[str, Any]], min_time_s: float | None) -> str | None:
    if min_time_s is None:
        return None
    row = min(trace, key=lambda item: abs(float(item["time_s"]) - float(min_time_s)))
    value = row.get("nearest_actor_id")
    return None if value is None else str(value)


def _nearest_executed_actor_type(trace: list[dict[str, Any]], min_time_s: float | None) -> str | None:
    if min_time_s is None:
        return None
    row = min(trace, key=lambda item: abs(float(item["time_s"]) - float(min_time_s)))
    value = row.get("nearest_actor_type")
    return None if value is None else str(value)


def _executed_horizon_s(trace: list[dict[str, Any]]) -> float:
    return 0.0 if not trace else round(max(float(row["time_s"]) for row in trace), 6)


def _executed_dt_s(trace: list[dict[str, Any]]) -> float | None:
    if len(trace) < 2:
        return None
    return round(float(trace[1]["time_s"]) - float(trace[0]["time_s"]), 6)


def _proxy_metric_variants(
    *,
    scene: dict[str, Any],
    actor_tracks: list[dict[str, Any]],
    thresholds_m: tuple[float, ...],
    rollout_dt_s: float,
    initial_ego_state: Any,
) -> dict[str, float | None]:
    poses = list(scene.get("selected_token_rollout", {}).get("poses", []))
    if not poses or not actor_tracks:
        return {}
    variants = {
        "center_distance_m": math.inf,
        "box_clearance_m": math.inf,
    }
    for buffer_m in thresholds_m:
        variants[f"inflated_box_clearance_{str(buffer_m).replace('.', '_')}m"] = math.inf
    for index, pose in enumerate(poses):
        t_s = (index + 1) * rollout_dt_s
        global_x, global_y = _local_to_global(
            origin_x=float(initial_ego_state.rear_axle.x),
            origin_y=float(initial_ego_state.rear_axle.y),
            origin_heading=float(initial_ego_state.rear_axle.heading),
            local_x=float(pose[0]),
            local_y=float(pose[1]),
        )
        actor = min(
            actor_tracks,
            key=lambda row: math.hypot(
                global_x - float(_actor_state_at_time(row, t_s=t_s)["x_m"]),
                global_y - float(_actor_state_at_time(row, t_s=t_s)["y_m"]),
            ),
        )
        actor_state = _actor_state_at_time(actor, t_s=t_s)
        center_distance = math.hypot(
            global_x - float(actor_state["x_m"]),
            global_y - float(actor_state["y_m"]),
        )
        actor_radius = float(actor_state["radius_m"])
        ego_radius = 1.0
        clearance = center_distance - ego_radius - actor_radius
        variants["center_distance_m"] = min(float(variants["center_distance_m"]), center_distance)
        variants["box_clearance_m"] = min(float(variants["box_clearance_m"]), clearance)
        for buffer_m in thresholds_m:
            key = f"inflated_box_clearance_{str(buffer_m).replace('.', '_')}m"
            variants[key] = min(float(variants[key]), clearance - float(buffer_m))
    return {key: _optional_round(value) for key, value in variants.items()}


def _aligned_executed_metric_variants(
    *,
    history: Any,
    actor_tracks: list[dict[str, Any]],
    thresholds_m: tuple[float, ...],
    rollout_dt_s: float,
    rollout_horizon_s: float,
) -> dict[str, float | None]:
    samples = _sample_history_at_replay_dt(
        history=history,
        rollout_dt_s=rollout_dt_s,
        rollout_horizon_s=rollout_horizon_s,
    )
    if not samples or not actor_tracks:
        return {}
    variants = {
        "center_distance_m": math.inf,
        "box_clearance_m": math.inf,
    }
    for buffer_m in thresholds_m:
        variants[f"inflated_box_clearance_{str(buffer_m).replace('.', '_')}m"] = math.inf
    for t_s, ego_state in samples:
        # Match the log-replay clearance definition exactly: rear-axle point with a 1m ego proxy radius.
        ego_x = float(ego_state.rear_axle.x)
        ego_y = float(ego_state.rear_axle.y)
        ego_radius = 1.0
        nearest_center_distance = math.inf
        nearest_clearance = math.inf
        for actor in actor_tracks:
            actor_state = _actor_state_at_time(actor, t_s=t_s)
            actor_radius = float(actor_state["radius_m"])
            center_distance = math.hypot(
                ego_x - float(actor_state["x_m"]),
                ego_y - float(actor_state["y_m"]),
            )
            clearance = center_distance - ego_radius - actor_radius
            nearest_center_distance = min(nearest_center_distance, center_distance)
            nearest_clearance = min(nearest_clearance, clearance)
        variants["center_distance_m"] = min(float(variants["center_distance_m"]), nearest_center_distance)
        variants["box_clearance_m"] = min(float(variants["box_clearance_m"]), nearest_clearance)
        for buffer_m in thresholds_m:
            key = f"inflated_box_clearance_{str(buffer_m).replace('.', '_')}m"
            variants[key] = min(float(variants[key]), nearest_clearance - float(buffer_m))
    return {key: _optional_round(value) for key, value in variants.items()}


def _sample_history_at_replay_dt(
    *,
    history: Any,
    rollout_dt_s: float,
    rollout_horizon_s: float,
) -> list[tuple[float, Any]]:
    if not history.data:
        return []
    initial_time_s = float(history.data[0].iteration.time_point.time_s)
    samples: list[tuple[float, Any]] = []
    target_times = []
    step_count = max(1, int(round(rollout_horizon_s / max(rollout_dt_s, 1.0e-6))))
    for index in range(step_count):
        target_times.append(round((index + 1) * rollout_dt_s, 6))
    for target_time in target_times:
        nearest = min(
            history.data,
            key=lambda sample: abs(
                (float(sample.iteration.time_point.time_s) - initial_time_s) - target_time
            ),
        )
        rel_time_s = round(float(nearest.iteration.time_point.time_s) - initial_time_s, 6)
        if rel_time_s <= 0.0:
            continue
        samples.append((target_time, nearest.ego_state))
    return samples


def _trajectory_agreement(
    *,
    trajectory_trace: list[dict[str, Any]],
    fixed_states: list[Any],
) -> dict[str, float | None]:
    deviations = []
    for row in trajectory_trace:
        if not fixed_states:
            break
        target_index = max(0, min(len(fixed_states) - 1, int(round(float(row["time_s"]) / 0.5))))
        target_state = fixed_states[target_index]
        deviation = math.hypot(
            float(row["executed_x_m"]) - float(target_state.rear_axle.x),
            float(row["executed_y_m"]) - float(target_state.rear_axle.y),
        )
        deviations.append(deviation)
    if not deviations:
        return {"mean_trajectory_deviation_m": None, "max_trajectory_deviation_m": None, "final_pose_deviation_m": None}
    return {
        "mean_trajectory_deviation_m": _optional_round(sum(deviations) / len(deviations)),
        "max_trajectory_deviation_m": _optional_round(max(deviations)),
        "final_pose_deviation_m": _optional_round(deviations[-1]),
    }


def _bridge_failure_reason(
    *,
    scene: dict[str, Any],
    replay_nearest_actor_id: str | None,
    executed_trace: list[dict[str, Any]],
    trajectory_agreement: dict[str, float | None],
    near_miss_threshold_m: float,
) -> str:
    if not executed_trace:
        return "missing_execution_trace"
    if (
        trajectory_agreement.get("max_trajectory_deviation_m") is not None
        and float(trajectory_agreement["max_trajectory_deviation_m"]) > 2.0
    ):
        return "trajectory_execution_mismatch"
    executed_actor_id = _nearest_executed_actor_id(
        executed_trace,
        scene.get("selected_token_executed_min_clearance_t"),
    )
    if (
        replay_nearest_actor_id is not None
        and executed_actor_id is not None
        and replay_nearest_actor_id != executed_actor_id
    ):
        return "actor_set_mismatch"
    center_distance = _minimum_trace_value(executed_trace, "executed_center_distance_m")
    box_clearance = _minimum_trace_value(executed_trace, "executed_box_clearance_m")
    if (
        center_distance is not None
        and box_clearance is not None
        and float(center_distance) >= near_miss_threshold_m
        and float(box_clearance) < near_miss_threshold_m
    ):
        return "footprint_inflation_mismatch"
    return "threshold_or_geometry_mismatch"


def build_bridge_calibration_report(
    report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    scenes = list(report.get("bridge_scenes", []))
    calibration_rows = [
        _bridge_calibration_row(scene, near_miss_threshold_m=near_miss_threshold_m)
        for scene in scenes
    ]
    return {
        "schema": "nuplan_selected_token_closed_loop_bridge_calibration_v1",
        "scene_count": len(calibration_rows),
        "threshold_m": float(near_miss_threshold_m),
        "scene_rows": calibration_rows,
        "class_summary_table": _calibration_class_summary_table(calibration_rows),
        "metric_variant_table": _metric_variant_table(calibration_rows, near_miss_threshold_m=near_miss_threshold_m),
        "nearest_actor_agreement_table": _nearest_actor_agreement_table(calibration_rows),
        "horizon_alignment_table": _horizon_alignment_table(calibration_rows),
        "trajectory_agreement_table": _trajectory_agreement_table(calibration_rows),
    }


def _bridge_calibration_row(scene: dict[str, Any], *, near_miss_threshold_m: float) -> dict[str, Any]:
    return {
        "scene_id": scene["scene_id"],
        "replay_class": scene["bridge_replay_class"],
        "selected_token": scene["selected_token"],
        "replay_min_clearance_m": scene.get("selected_token_realized_min_clearance_m"),
        "executed_min_clearance_m": scene.get("selected_token_executed_min_clearance_m"),
        "replay_min_clearance_t": scene.get("selected_token_realized_min_clearance_t"),
        "executed_min_clearance_t": scene.get("selected_token_executed_min_clearance_t"),
        "replay_nearest_actor_id": scene.get("replay_nearest_actor_id"),
        "executed_nearest_actor_id": scene.get("executed_nearest_actor_id"),
        "replay_nearest_actor_type": scene.get("replay_nearest_actor_type"),
        "executed_nearest_actor_type": scene.get("executed_nearest_actor_type"),
        "ego_pose_source": scene.get("ego_pose_source"),
        "actor_geometry_source": scene.get("actor_geometry_source"),
        "ego_footprint_geometry": scene.get("ego_footprint_geometry"),
        "actor_footprint_geometry": scene.get("actor_footprint_geometry"),
        "threshold_m": float(scene.get("threshold_m", near_miss_threshold_m)),
        "failure_reason": scene.get("failure_reason"),
        "replay_horizon_s": scene.get("replay_horizon_s"),
        "executed_horizon_s": scene.get("executed_horizon_s"),
        "num_replay_steps": scene.get("num_replay_steps"),
        "num_executed_steps": scene.get("num_executed_steps"),
        "dt_replay_s": scene.get("dt_replay_s"),
        "dt_executed_s": scene.get("dt_executed_s"),
        "proxy_metric_variants": dict(scene.get("proxy_metric_variants", {})),
        "executed_metric_variants": dict(scene.get("executed_metric_variants", {})),
        "aligned_executed_metric_variants": dict(scene.get("aligned_executed_metric_variants", {})),
        "trajectory_agreement": dict(scene.get("trajectory_agreement", {})),
        "executed_failed": bool(
            str(scene.get("executed_failure_rung"))
            in {"proxy_safe_but_realized_failed", "replay_safe_but_executed_failed"}
        ),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = len(ordered) // 2
    if len(ordered) % 2:
        return _optional_round(ordered[index])
    return _optional_round((ordered[index - 1] + ordered[index]) / 2.0)


def _calibration_class_summary_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for replay_class in ("replay_safe", "replay_infeasible"):
        subset = [row for row in rows if row["replay_class"] == replay_class]
        table.append(
            {
                "replay_class": replay_class,
                "count": len(subset),
                "executed_fail": sum(1 for row in subset if row["executed_failed"]),
                "median_replay_clearance_m": _median(
                    [
                        float(row["replay_min_clearance_m"])
                        for row in subset
                        if row["replay_min_clearance_m"] is not None
                    ]
                ),
                "median_executed_clearance_m": _median(
                    [
                        float(row["executed_min_clearance_m"])
                        for row in subset
                        if row["executed_min_clearance_m"] is not None
                    ]
                ),
            }
        )
    return table


def _metric_variant_table(rows: list[dict[str, Any]], *, near_miss_threshold_m: float) -> list[dict[str, Any]]:
    variants = [
        ("center distance, no footprint", "center_distance_m"),
        ("box clearance, no inflation", "box_clearance_m"),
        ("box clearance + 0.0m buffer", "inflated_box_clearance_0_0m"),
        ("box clearance + 0.5m buffer", "inflated_box_clearance_0_5m"),
        ("box clearance + 1.0m buffer", "inflated_box_clearance_1_0m"),
    ]
    table = []
    for label, key in variants:
        table.append(
            {
                "metric_variant": label,
                "surface": "executed_observation",
                "replay_safe_controls_failed": _metric_failed_count(
                    rows,
                    replay_class="replay_safe",
                    metric_key=key,
                    source_key="executed_metric_variants",
                    near_miss_threshold_m=near_miss_threshold_m,
                ),
                "replay_infeasible_failed": _metric_failed_count(
                    rows,
                    replay_class="replay_infeasible",
                    metric_key=key,
                    source_key="executed_metric_variants",
                    near_miss_threshold_m=near_miss_threshold_m,
                ),
            }
        )
    for label, key in variants:
        table.append(
            {
                "metric_variant": label,
                "surface": "aligned_logged_actors",
                "replay_safe_controls_failed": _metric_failed_count(
                    rows,
                    replay_class="replay_safe",
                    metric_key=key,
                    source_key="aligned_executed_metric_variants",
                    near_miss_threshold_m=near_miss_threshold_m,
                ),
                "replay_infeasible_failed": _metric_failed_count(
                    rows,
                    replay_class="replay_infeasible",
                    metric_key=key,
                    source_key="aligned_executed_metric_variants",
                    near_miss_threshold_m=near_miss_threshold_m,
                ),
            }
        )
    return table


def _metric_failed_count(
    rows: list[dict[str, Any]],
    *,
    replay_class: str,
    metric_key: str,
    source_key: str,
    near_miss_threshold_m: float,
) -> int:
    count = 0
    for row in rows:
        if row["replay_class"] != replay_class:
            continue
        value = row.get(source_key, {}).get(metric_key)
        if value is not None and float(value) < near_miss_threshold_m:
            count += 1
    return count


def _nearest_actor_agreement_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    same = 0
    different = 0
    missing = 0
    for row in rows:
        replay_id = row.get("replay_nearest_actor_id")
        executed_id = row.get("executed_nearest_actor_id")
        if replay_id is None or executed_id is None:
            missing += 1
        elif replay_id == executed_id:
            same += 1
        else:
            different += 1
    return [
        {"agreement": "same_actor", "count": same},
        {"agreement": "different_actor", "count": different},
        {"agreement": "missing_actor_id", "count": missing},
    ]


def _horizon_alignment_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "median_replay_horizon_s",
            "value": _median(
                [float(row["replay_horizon_s"]) for row in rows if row["replay_horizon_s"] is not None]
            ),
        },
        {
            "metric": "median_executed_horizon_s",
            "value": _median(
                [float(row["executed_horizon_s"]) for row in rows if row["executed_horizon_s"] is not None]
            ),
        },
        {
            "metric": "median_dt_replay_s",
            "value": _median([float(row["dt_replay_s"]) for row in rows if row["dt_replay_s"] is not None]),
        },
        {
            "metric": "median_dt_executed_s",
            "value": _median([float(row["dt_executed_s"]) for row in rows if row["dt_executed_s"] is not None]),
        },
    ]


def _trajectory_agreement_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "metric": "median_mean_trajectory_deviation_m",
            "value": _median(
                [
                    float(row["trajectory_agreement"]["mean_trajectory_deviation_m"])
                    for row in rows
                    if row["trajectory_agreement"].get("mean_trajectory_deviation_m") is not None
                ]
            ),
        },
        {
            "metric": "median_max_trajectory_deviation_m",
            "value": _median(
                [
                    float(row["trajectory_agreement"]["max_trajectory_deviation_m"])
                    for row in rows
                    if row["trajectory_agreement"].get("max_trajectory_deviation_m") is not None
                ]
            ),
        },
        {
            "metric": "median_final_pose_deviation_m",
            "value": _median(
                [
                    float(row["trajectory_agreement"]["final_pose_deviation_m"])
                    for row in rows
                    if row["trajectory_agreement"].get("final_pose_deviation_m") is not None
                ]
            ),
        },
    ]


def _closed_loop_bridge_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for scene in scenes:
        key = (str(scene["bridge_replay_class"]), str(scene["executed_failure_rung"]))
        counts[key] = counts.get(key, 0) + 1
    rows = []
    for replay_class in ("replay_infeasible", "replay_safe"):
        for executed_class in (
            "proxy_safe_but_realized_failed",
            "replay_infeasible_but_executed_safe",
            "replay_safe_but_executed_failed",
            "executed_safe",
        ):
            rows.append(
                {
                    "replay_class": replay_class,
                    "executed_class": executed_class,
                    "count": counts.get((replay_class, executed_class), 0),
                }
            )
    return rows


def _closed_loop_bridge_summary(scenes: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "replay_infeasible_scene_count": sum(
            1 for scene in scenes if scene["bridge_replay_class"] == "replay_infeasible"
        ),
        "replay_safe_scene_count": sum(1 for scene in scenes if scene["bridge_replay_class"] == "replay_safe"),
        "executed_failure_count": sum(
            1
            for scene in scenes
            if str(scene["executed_failure_rung"])
            in {"proxy_safe_but_realized_failed", "replay_safe_but_executed_failed"}
        ),
        "executed_safe_count": sum(
            1
            for scene in scenes
            if str(scene["executed_failure_rung"]) in {"executed_safe", "replay_infeasible_but_executed_safe"}
        ),
    }


def _selector_comparison_table(scenes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selectors = sorted({str(scene.get("bridge_selector", "selected_token")) for scene in scenes})
    rows = []
    for selector in selectors:
        selected = [scene for scene in scenes if str(scene.get("bridge_selector", "selected_token")) == selector]
        failure_count = sum(
            1
            for scene in selected
            if str(scene["executed_failure_rung"])
            in {"proxy_safe_but_realized_failed", "replay_safe_but_executed_failed"}
        )
        clearances = [
            float(scene["selected_token_executed_min_clearance_m"])
            for scene in selected
            if scene.get("selected_token_executed_min_clearance_m") is not None
        ]
        aligned_clearances = [
            float(scene["aligned_executed_metric_variants"]["box_clearance_m"])
            for scene in selected
            if scene.get("aligned_executed_metric_variants", {}).get("box_clearance_m") is not None
        ]
        aligned_failure_count = sum(1 for clearance in aligned_clearances if clearance < DEFAULT_NEAR_MISS_THRESHOLD_M)
        progress_values = [
            float(scene["selected_token_rollout"]["final_progress_m"])
            for scene in selected
            if scene.get("selected_token_rollout", {}).get("final_progress_m") is not None
        ]
        utilities = [
            candidate_logged_ego_utility(scene["selected_token_rollout"], scene)
            for scene in selected
            if isinstance(scene.get("selected_token_rollout"), Mapping)
        ]
        utilities = [utility for utility in utilities if utility is not None]
        rows.append(
            {
                "selector": selector,
                "scene_count": len(selected),
                "executed_failure_count": failure_count,
                "executed_failure_rate": round(failure_count / len(selected), 6) if selected else 0.0,
                "mean_executed_min_clearance_m": _optional_round(sum(clearances) / len(clearances))
                if clearances
                else None,
                "aligned_logged_actor_failure_count": aligned_failure_count,
                "aligned_logged_actor_failure_rate": round(aligned_failure_count / len(aligned_clearances), 6)
                if aligned_clearances
                else None,
                "mean_aligned_logged_actor_clearance_m": _optional_round(
                    sum(aligned_clearances) / len(aligned_clearances)
                )
                if aligned_clearances
                else None,
                "mean_selected_progress_m": _optional_round(sum(progress_values) / len(progress_values))
                if progress_values
                else None,
                "mean_ade_3s_m": _mean_utility(utilities, "ade_3s_m"),
                "mean_fde_3s_m": _mean_utility(utilities, "fde_3s_m"),
            }
        )
    return rows


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Closed-Loop Bridge Audit",
        "",
        f"- Bridge scene count: `{report['bridge_scene_count']}`",
        f"- Bridge base scene count: `{report['bridge_base_scene_count']}`"
        if "bridge_base_scene_count" in report
        else "",
        f"- Replay-infeasible scenes: `{report['closed_loop_bridge_summary']['replay_infeasible_scene_count']}`",
        f"- Replay-safe scenes: `{report['closed_loop_bridge_summary']['replay_safe_scene_count']}`",
        f"- Executed failures: `{report['closed_loop_bridge_summary']['executed_failure_count']}`",
        f"- Executed safe: `{report['closed_loop_bridge_summary']['executed_safe_count']}`",
        "",
        "## Replay Vs Execution",
        "",
        "| Replay class | Executed class | Count |",
        "|---|---|---:|",
    ]
    for row in report["closed_loop_bridge_table"]:
        lines.append(f"| {row['replay_class']} | {row['executed_class']} | {row['count']} |")
    if report.get("closed_loop_selector_comparison"):
        lines.extend(
            [
                "",
                "## Selector Comparison",
                "",
                "Absolute box-clearance failures use nuPlan executed geometry. Aligned replay failures use the "
                "logged-actor replay-equivalent clearance already used by the replay study, so this table should be "
                "read as directional transfer evidence rather than absolute closed-loop safety.",
                "",
                "| Selector | Scenes | Executed box failures | Box fail rate | Aligned replay failures | "
                "Aligned fail rate | Mean aligned clearance | Progress | ADE3 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in report["closed_loop_selector_comparison"]:
            aligned_rate = row["aligned_logged_actor_failure_rate"]
            aligned_clearance = row["mean_aligned_logged_actor_clearance_m"]
            progress = row.get("mean_selected_progress_m")
            ade = row.get("mean_ade_3s_m")
            lines.append(
                f"| {row['selector']} | {row['scene_count']} | {row['executed_failure_count']} | "
                f"{row['executed_failure_rate']:.3f} | "
                f"{row['aligned_logged_actor_failure_count']} | "
                f"{'n/a' if aligned_rate is None else f'{aligned_rate:.3f}'} | "
                f"{'n/a' if aligned_clearance is None else f'{aligned_clearance:.3f}'} | "
                f"{'n/a' if progress is None else f'{progress:.3f}'} | "
                f"{'n/a' if ade is None else f'{ade:.3f}'} |"
            )
    return "\n".join(lines)


def markdown_calibration_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Closed-Loop Bridge Calibration",
        "",
        f"- Scene count: `{report['scene_count']}`",
        f"- Threshold: `{report['threshold_m']:.3f} m`",
        "",
        "## Replay Class Summary",
        "",
        "| Replay class | Count | Executed fail | Median replay clearance | Median executed clearance |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["class_summary_table"]:
        lines.append(
            f"| {row['replay_class']} | {row['count']} | {row['executed_fail']} | "
            f"{_format_optional(row['median_replay_clearance_m'])} | "
            f"{_format_optional(row['median_executed_clearance_m'])} |"
        )
    lines.extend(
        [
            "",
            "## Metric Variants",
            "",
            "| Surface | Metric variant | Replay-safe controls failed | Replay-infeasible failed |",
            "|---|---|---:|---:|",
        ]
    )
    for row in report["metric_variant_table"]:
        lines.append(
            f"| {row['surface']} | {row['metric_variant']} | "
            f"{row['replay_safe_controls_failed']} | {row['replay_infeasible_failed']} |"
        )
    lines.extend(
        [
            "",
            "## Nearest Actor Agreement",
            "",
            "| Agreement | Count |",
            "|---|---:|",
        ]
    )
    for row in report["nearest_actor_agreement_table"]:
        lines.append(f"| {row['agreement']} | {row['count']} |")
    lines.extend(
        [
            "",
            "## Horizon Alignment",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for row in report["horizon_alignment_table"]:
        lines.append(f"| {row['metric']} | {_format_optional(row['value'])} |")
    lines.extend(
        [
            "",
            "## Trajectory Agreement",
            "",
            "| Metric | Value |",
            "|---|---:|",
        ]
    )
    for row in report["trajectory_agreement_table"]:
        lines.append(f"| {row['metric']} | {_format_optional(row['value'])} |")
    return "\n".join(lines)


def _format_optional(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _local_to_global(
    *,
    origin_x: float,
    origin_y: float,
    origin_heading: float,
    local_x: float,
    local_y: float,
) -> tuple[float, float]:
    cos_h = math.cos(origin_heading)
    sin_h = math.sin(origin_heading)
    return (
        origin_x + cos_h * local_x - sin_h * local_y,
        origin_y + sin_h * local_x + cos_h * local_y,
    )


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _optional_round(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return None if not math.isfinite(value) else round(value, 6)


def _mean_utility(rows: list[Mapping[str, float]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return None
    return _optional_round(sum(values) / len(values))


def _nuplan_closed_loop_imports() -> dict[str, Any]:
    try:
        from nuplan.common.actor_state.ego_state import EgoState
        from nuplan.common.actor_state.state_representation import StateSE2
        from nuplan.common.actor_state.state_representation import StateVector2D
        from nuplan.common.actor_state.state_representation import TimePoint
        from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
        from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
        from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
        from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
        from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
        from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
        from nuplan.planning.simulation.simulation import Simulation
        from nuplan.planning.simulation.simulation_setup import SimulationSetup
        from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import (
            StepSimulationTimeController,
        )
        from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError(
            "nuPlan closed-loop dependencies are unavailable. Run `./scripts/bootstrap_nuplan_env.sh` first."
        ) from exc
    return {
        "DetectionsTracks": DetectionsTracks,
        "EgoState": EgoState,
        "InterpolatedTrajectory": InterpolatedTrajectory,
        "NuPlanScenario": NuPlanScenario,
        "PerfectTrackingController": PerfectTrackingController,
        "ScenarioExtractionInfo": ScenarioExtractionInfo,
        "Simulation": Simulation,
        "SimulationRunner": SimulationRunner,
        "SimulationSetup": SimulationSetup,
        "StateSE2": StateSE2,
        "StateVector2D": StateVector2D,
        "StepSimulationTimeController": StepSimulationTimeController,
        "TimePoint": TimePoint,
        "TracksObservation": TracksObservation,
        "get_pacifica_parameters": get_pacifica_parameters,
    }


if __name__ == "__main__":
    raise SystemExit(main())
