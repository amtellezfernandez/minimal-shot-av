#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _features
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.cli.commands.train_nuplan_replay_value_selector import build_replay_value_examples
from minimal_shot_av.model.nuplan_maneuver_token_adapter import TOKEN_ORDER
from minimal_shot_av.model.nuplan_maneuver_token_selector import SELECTOR_FEATURE_NAMES
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_mlp_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expert" / "replay.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_regime_value_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_regime_value_selector_eval.json"
DEFAULT_REPORT_MD = ROOT / "artifacts" / "corl2027" / "nuplan_regime_value_selector_eval.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a candidate-level selector with a two-regime replay-recovery objective."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--marginal-progress-retention", type=float, default=0.5)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--ade-weight", type=float, default=2.0)
    parser.add_argument("--marginal-bonus", type=float, default=60.0)
    parser.add_argument("--intervention-bonus", type=float, default=40.0)
    parser.add_argument("--low-retention-penalty", type=float, default=80.0)
    parser.add_argument("--safe-proxy-keep-bonus", type=float, default=60.0)
    parser.add_argument("--safe-proxy-switch-penalty", type=float, default=25.0)
    parser.add_argument("--baseline-fail-penalty", type=float, default=250.0)
    parser.add_argument("--feature-set", choices=("base", "regime_augmented"), default="regime_augmented")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    objective_config = {
        "marginal_progress_retention": float(args.marginal_progress_retention),
        "fail_penalty": float(args.fail_penalty),
        "progress_weight": float(args.progress_weight),
        "ade_weight": float(args.ade_weight),
        "marginal_bonus": float(args.marginal_bonus),
        "intervention_bonus": float(args.intervention_bonus),
        "low_retention_penalty": float(args.low_retention_penalty),
        "safe_proxy_keep_bonus": float(args.safe_proxy_keep_bonus),
        "safe_proxy_switch_penalty": float(args.safe_proxy_switch_penalty),
    }
    train_examples = build_regime_value_examples(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        feature_set=str(args.feature_set),
        **objective_config,
    )
    holdout_examples = build_regime_value_examples(
        holdout_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        feature_set=str(args.feature_set),
        **objective_config,
    )
    feature_names = regime_value_feature_names() if str(args.feature_set) == "regime_augmented" else None
    selector, train_metrics = fit_replay_value_mlp_selector(
        train_examples,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        feature_names=feature_names,
    )
    baseline_examples = build_replay_value_examples(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.baseline_fail_penalty),
        objective_progress_weight=float(args.progress_weight),
        objective_ade_weight=float(args.ade_weight),
    )
    baseline_selector, baseline_metrics = fit_replay_value_mlp_selector(
        baseline_examples,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed) + 991,
    )
    report = {
        "schema": "nuplan_regime_value_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "objective": objective_config,
        "feature_set": str(args.feature_set),
        "baseline_objective": {
            "fail_penalty": float(args.baseline_fail_penalty),
            "progress_weight": float(args.progress_weight),
            "ade_weight": float(args.ade_weight),
        },
        "train_fit_metrics": train_metrics,
        "baseline_fit_metrics": baseline_metrics,
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
        "train": evaluate_regime_value_family(
            train_report,
            regime_value_selector=selector,
            baseline_selector=baseline_selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            objective_config=objective_config,
            feature_set=str(args.feature_set),
        ),
        "holdout": evaluate_regime_value_family(
            holdout_report,
            regime_value_selector=selector,
            baseline_selector=baseline_selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            objective_config=objective_config,
            feature_set=str(args.feature_set),
        ),
        "train_regime_counts": regime_type_counts(
            train_report,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            marginal_progress_retention=float(args.marginal_progress_retention),
        ),
        "holdout_regime_counts": regime_type_counts(
            holdout_report,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
            marginal_progress_retention=float(args.marginal_progress_retention),
        ),
    }
    payload = {
        **selector.to_payload(),
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "split": split,
            "objective": objective_config,
            "feature_set": str(args.feature_set),
            "train_fit_metrics": train_metrics,
            "train_example_count": len(train_examples),
            "holdout_example_count": len(holdout_examples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_regime_value_report(report) + "\n", encoding="utf-8")
    print(markdown_regime_value_report(report))
    return 0


def build_regime_value_examples(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    feature_set: str = "base",
    marginal_progress_retention: float,
    fail_penalty: float,
    progress_weight: float,
    ade_weight: float,
    marginal_bonus: float,
    intervention_bonus: float,
    low_retention_penalty: float,
    safe_proxy_keep_bonus: float,
    safe_proxy_switch_penalty: float,
) -> list[dict[str, Any]]:
    examples = []
    for scene in replay_report.get("scenes", []):
        candidate_by_token = _candidate_by_token(scene)
        replay_by_token = _replay_by_token(scene)
        proxy_token = str(scene.get("selected_token", ""))
        safe_proxy = _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m)
        marginal_safe_exists = any(
            _token_is_replay_safe(scene, token, near_miss_threshold_m)
            and token_progress_retention(scene, token) >= float(marginal_progress_retention)
            for token in candidate_by_token
        )
        safe_exists = any(_token_is_replay_safe(scene, token, near_miss_threshold_m) for token in candidate_by_token)
        for token, candidate in candidate_by_token.items():
            replay = replay_by_token.get(str(token))
            if replay is None or replay.get("realized_min_clearance_m") is None:
                continue
            utility = candidate_logged_ego_utility(candidate, scene)
            if utility is None:
                continue
            replay_fail = float(replay["realized_min_clearance_m"]) < float(near_miss_threshold_m)
            retention = token_progress_retention(scene, str(token))
            objective_value = (
                float(progress_weight) * float(candidate.get("final_progress_m", 0.0))
                - float(ade_weight) * float(utility["ade_3s_m"])
            )
            if replay_fail:
                objective_value -= float(fail_penalty)
            elif safe_proxy:
                objective_value += float(safe_proxy_keep_bonus) if str(token) == proxy_token else 0.0
                objective_value -= float(safe_proxy_switch_penalty) if str(token) != proxy_token else 0.0
            elif marginal_safe_exists:
                if retention >= float(marginal_progress_retention):
                    objective_value += float(marginal_bonus)
                else:
                    objective_value -= float(low_retention_penalty)
            elif safe_exists:
                objective_value += float(intervention_bonus)
            examples.append(
                {
                    "scene_id": str(scene["scene_id"]),
                    "source_db_file": str(scene.get("source_db_file", "")),
                    "token": str(token),
                    "objective_value": float(objective_value),
                    "regime_type": scene_regime_type(
                        scene,
                        near_miss_threshold_m=near_miss_threshold_m,
                        marginal_progress_retention=marginal_progress_retention,
                    ),
                    "features": regime_value_candidate_features(
                        scene,
                        candidate,
                        feature_set=feature_set,
                    ),
                }
            )
    if not examples:
        raise ValueError("cannot train regime-value selector on an empty replay report")
    return examples


def evaluate_regime_value_family(
    replay_report: Mapping[str, Any],
    *,
    regime_value_selector: Any,
    baseline_selector: Any,
    near_miss_threshold_m: float,
    objective_config: Mapping[str, float],
    feature_set: str,
) -> dict[str, Any]:
    return {
        "proxy_selector": evaluate_selector_policy(
            replay_report,
            selector_name="proxy_selector",
            selected_token_fn=lambda scene: str(scene.get("selected_token", "")),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "safety_value_baseline": evaluate_selector_policy(
            replay_report,
            selector_name="safety_value_baseline",
            selected_token_fn=lambda scene: _select_with_score_model(scene, baseline_selector),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "regime_value": evaluate_selector_policy(
            replay_report,
            selector_name="regime_value",
            selected_token_fn=lambda scene: select_with_regime_value_model(
                scene,
                regime_value_selector,
                feature_set=feature_set,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "regime_objective_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="regime_objective_oracle",
            selected_token_fn=lambda scene: select_regime_objective_oracle(
                scene,
                near_miss_threshold_m=near_miss_threshold_m,
                **objective_config,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }


def select_with_regime_value_model(scene: Mapping[str, Any], selector: Any, *, feature_set: str) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    selected = max(
        candidates,
        key=lambda candidate: (
            selector.predict_score(
                regime_value_candidate_features(
                    scene,
                    candidate,
                    feature_set=feature_set,
                )
            ),
            float(candidate.get("min_proxy_clearance_m", 0.0)),
            float(candidate.get("final_progress_m", 0.0)),
        ),
    )
    return str(selected.get("token", scene.get("selected_token", "")))


def select_regime_objective_oracle(
    scene: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    marginal_progress_retention: float,
    fail_penalty: float,
    progress_weight: float,
    ade_weight: float,
    marginal_bonus: float,
    intervention_bonus: float,
    low_retention_penalty: float,
    safe_proxy_keep_bonus: float,
    safe_proxy_switch_penalty: float,
) -> str:
    report = {"scenes": [scene]}
    examples = build_regime_value_examples(
        report,
        near_miss_threshold_m=near_miss_threshold_m,
        feature_set="base",
        marginal_progress_retention=marginal_progress_retention,
        fail_penalty=fail_penalty,
        progress_weight=progress_weight,
        ade_weight=ade_weight,
        marginal_bonus=marginal_bonus,
        intervention_bonus=intervention_bonus,
        low_retention_penalty=low_retention_penalty,
        safe_proxy_keep_bonus=safe_proxy_keep_bonus,
        safe_proxy_switch_penalty=safe_proxy_switch_penalty,
    )
    if not examples:
        return str(scene.get("selected_token", ""))
    return str(max(examples, key=lambda example: float(example["objective_value"]))["token"])


def regime_value_candidate_features(
    scene: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    feature_set: str,
) -> dict[str, float]:
    if feature_set == "base":
        return candidate.get("features") if "features" in candidate else _features(scene, candidate)
    row = selector_feature_row(scene, candidate)
    candidate_by_token = _candidate_by_token(scene)
    proxy_token = str(scene.get("selected_token", ""))
    token = str(candidate.get("token", ""))
    proxy_candidate = candidate_by_token.get(proxy_token)
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0)) if proxy_candidate is not None else 0.0
    proxy_clearance = (
        float(proxy_candidate.get("min_proxy_clearance_m", 0.0)) if proxy_candidate is not None else 0.0
    )
    proxy_score = float(proxy_candidate.get("score", 0.0)) if proxy_candidate is not None else 0.0
    progress = float(candidate.get("final_progress_m", 0.0))
    clearance = float(candidate.get("min_proxy_clearance_m", 0.0))
    score = float(candidate.get("score", 0.0))
    progress_values = [float(row.get("final_progress_m", 0.0)) for row in candidate_by_token.values()]
    clearance_values = [float(row.get("min_proxy_clearance_m", 0.0)) for row in candidate_by_token.values()]
    score_values = [float(row.get("score", 0.0)) for row in candidate_by_token.values()]
    retention_values = [token_progress_retention(scene, name) for name in candidate_by_token]
    proxy_safe_count = sum(1 for row in candidate_by_token.values() if bool(row.get("proxy_safe", False)))
    row.update(
        {
            "candidate_is_selected_proxy": 1.0 if token == proxy_token else 0.0,
            "candidate_progress_retention": token_progress_retention(scene, token),
            "candidate_minus_proxy_progress_m": progress - proxy_progress,
            "candidate_minus_proxy_clearance_m": clearance - proxy_clearance,
            "candidate_minus_proxy_score": score - proxy_score,
            "candidate_progress_rank": float(descending_rank(progress, progress_values)),
            "candidate_clearance_rank": float(descending_rank(clearance, clearance_values)),
            "candidate_score_rank": float(descending_rank(score, score_values)),
            "candidate_progress_fraction_of_max": ratio(progress, max(progress_values) if progress_values else 0.0),
            "candidate_clearance_fraction_of_max": ratio(clearance, max(clearance_values) if clearance_values else 0.0),
            "candidate_score_fraction_of_max": ratio(score, max(score_values) if score_values else 0.0),
            "candidate_low_retention_lt025": 1.0 if token_progress_retention(scene, token) < 0.25 else 0.0,
            "candidate_marginal_retention_ge050": 1.0 if token_progress_retention(scene, token) >= 0.5 else 0.0,
            "candidate_high_retention_ge075": 1.0 if token_progress_retention(scene, token) >= 0.75 else 0.0,
            "candidate_low_motion": 1.0 if float(candidate.get("speed_scale", 0.0)) <= 0.25 else 0.0,
            "candidate_stop_or_crawl": 1.0 if token in {"stop", "crawl"} else 0.0,
            "proxy_final_progress_m": proxy_progress,
            "proxy_min_proxy_clearance_m": proxy_clearance,
            "proxy_score": proxy_score,
            "candidate_count": float(len(candidate_by_token)),
            "candidate_proxy_safe_count": float(proxy_safe_count),
            "candidate_proxy_safe_rate": ratio(proxy_safe_count, len(candidate_by_token)),
            "candidate_retention_ge025_count": float(sum(1 for value in retention_values if value >= 0.25)),
            "candidate_retention_ge050_count": float(sum(1 for value in retention_values if value >= 0.5)),
            "candidate_retention_ge075_count": float(sum(1 for value in retention_values if value >= 0.75)),
            "max_candidate_progress_m": max(progress_values) if progress_values else 0.0,
            "mean_candidate_progress_m": mean(progress_values),
            "max_candidate_clearance_m": max(clearance_values) if clearance_values else 0.0,
            "mean_candidate_clearance_m": mean(clearance_values),
            "max_candidate_score": max(score_values) if score_values else 0.0,
            "mean_candidate_score": mean(score_values),
        }
    )
    for name in TOKEN_ORDER:
        row[f"token_is_{name}"] = 1.0 if token == name else 0.0
    return row


def regime_value_feature_names() -> tuple[str, ...]:
    names = list(SELECTOR_FEATURE_NAMES)
    names.extend(
        [
            "candidate_is_selected_proxy",
            "candidate_progress_retention",
            "candidate_minus_proxy_progress_m",
            "candidate_minus_proxy_clearance_m",
            "candidate_minus_proxy_score",
            "candidate_progress_rank",
            "candidate_clearance_rank",
            "candidate_score_rank",
            "candidate_progress_fraction_of_max",
            "candidate_clearance_fraction_of_max",
            "candidate_score_fraction_of_max",
            "candidate_low_retention_lt025",
            "candidate_marginal_retention_ge050",
            "candidate_high_retention_ge075",
            "candidate_low_motion",
            "candidate_stop_or_crawl",
            "proxy_final_progress_m",
            "proxy_min_proxy_clearance_m",
            "proxy_score",
            "candidate_count",
            "candidate_proxy_safe_count",
            "candidate_proxy_safe_rate",
            "candidate_retention_ge025_count",
            "candidate_retention_ge050_count",
            "candidate_retention_ge075_count",
            "max_candidate_progress_m",
            "mean_candidate_progress_m",
            "max_candidate_clearance_m",
            "mean_candidate_clearance_m",
            "max_candidate_score",
            "mean_candidate_score",
        ]
    )
    names.extend(f"token_is_{name}" for name in TOKEN_ORDER)
    return tuple(names)


def scene_regime_type(
    scene: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    marginal_progress_retention: float,
) -> str:
    proxy_token = str(scene.get("selected_token", ""))
    if _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m):
        return "safe_proxy_keep"
    candidate_by_token = _candidate_by_token(scene)
    safe_tokens = [
        token for token in candidate_by_token if _token_is_replay_safe(scene, token, near_miss_threshold_m)
    ]
    if not safe_tokens:
        return "unrecoverable_no_safe_token"
    if any(token_progress_retention(scene, token) >= float(marginal_progress_retention) for token in safe_tokens):
        return "marginal_recovery_available"
    return "intervention_recovery_required"


def regime_type_counts(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    marginal_progress_retention: float,
) -> dict[str, int]:
    counts = Counter(
        scene_regime_type(
            scene,
            near_miss_threshold_m=near_miss_threshold_m,
            marginal_progress_retention=marginal_progress_retention,
        )
        for scene in replay_report.get("scenes", [])
    )
    return dict(sorted(counts.items()))


def token_progress_retention(scene: Mapping[str, Any], token: str) -> float:
    candidate_by_token = _candidate_by_token(scene)
    proxy_candidate = candidate_by_token.get(str(scene.get("selected_token", "")))
    candidate = candidate_by_token.get(str(token))
    if candidate is None:
        return 0.0
    progress = float(candidate.get("final_progress_m", 0.0))
    if proxy_candidate is None:
        return 1.0 if progress >= 0.0 else 0.0
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0))
    if proxy_progress <= 1.0e-6:
        return 1.0 if progress >= proxy_progress else 0.0
    return float(progress / proxy_progress)


def descending_rank(value: float, values: list[float]) -> int:
    return 1 + sum(1 for other in values if other > value)


def ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= 1.0e-8:
        return 0.0
    return float(numerator) / float(denominator)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def markdown_regime_value_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Regime-Value Selector",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Marginal retention floor: `{report['objective']['marginal_progress_retention']:.3f}`",
        "",
        "## Regime Counts",
        "",
        f"- Train: `{json.dumps(report['train_regime_counts'], sort_keys=True)}`",
        f"- Holdout: `{json.dumps(report['holdout_regime_counts'], sort_keys=True)}`",
        "",
    ]
    for split_name in ("train", "holdout"):
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for selector_name, metrics in report[split_name].items():
            lines.append(
                f"| {selector_name} | {metrics['selected_replay_infeasible_count']} | "
                f"{metrics['selected_replay_infeasible_rate']:.3f} | "
                f"{metrics['recoverable_proxy_optimism_fixed_count']}/"
                f"{metrics['recoverable_proxy_optimism_count']} | "
                f"{metrics['mean_selected_progress_m']:.3f} | "
                f"{_optional_metric(metrics['mean_ade_3s_m'])} | "
                f"{_optional_metric(metrics['mean_fde_3s_m'])} | "
                f"{metrics['selected_token_entropy']:.3f} |"
            )
        lines.extend(["", "### Token Histograms", ""])
        for selector_name, metrics in report[split_name].items():
            lines.append(f"- `{selector_name}`: `{json.dumps(metrics['selected_token_histogram'], sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def _optional_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
