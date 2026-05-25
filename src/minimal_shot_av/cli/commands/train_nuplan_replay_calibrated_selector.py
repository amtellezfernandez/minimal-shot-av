#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
from typing import Any
from typing import Mapping

from minimal_shot_av.model.nuplan_maneuver_token_selector import build_replay_calibration_examples
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_calibrated_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import load_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction250" / "replay.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibrated_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibrated_selector_eval.json"
DEFAULT_REPORT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_replay_calibrated_selector_eval.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a replay-calibrated ManeuverToken selector from all-token log-replay labels."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MARKDOWN)
    parser.add_argument("--baseline-selector-model", type=Path)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--risk-weight", type=float, default=4.0)
    parser.add_argument("--proxy-score-weight", type=float, default=1.0)
    parser.add_argument("--progress-weight", type=float, default=0.02)
    parser.add_argument("--unsafe-penalty", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    train_examples = build_replay_calibration_examples(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    holdout_examples = build_replay_calibration_examples(
        holdout_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    selector, risk_metrics = fit_replay_calibrated_selector(
        train_examples,
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        seed=int(args.seed),
        risk_weight=float(args.risk_weight),
        proxy_score_weight=float(args.proxy_score_weight),
        progress_weight=float(args.progress_weight),
        unsafe_penalty=float(args.unsafe_penalty),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    baseline_selector = None if args.baseline_selector_model is None else load_selector(args.baseline_selector_model)
    evaluation = {
        "schema": "nuplan_replay_calibrated_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "risk_head_train_metrics": risk_metrics,
        "train": evaluate_selector_family(
            train_report,
            replay_calibrated_selector=selector,
            baseline_selector=baseline_selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "holdout": evaluate_selector_family(
            holdout_report,
            replay_calibrated_selector=selector,
            baseline_selector=baseline_selector,
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "train_example_count": len(train_examples),
        "holdout_example_count": len(holdout_examples),
    }
    payload = {
        **selector.to_payload(),
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "split": split,
            "risk_head_train_metrics": risk_metrics,
            "train_example_count": len(train_examples),
            "holdout_example_count": len(holdout_examples),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(evaluation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_evaluation(evaluation)
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def split_replay_report_by_db(
    replay_report: Mapping[str, Any],
    *,
    holdout_fraction: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    scenes = list(replay_report.get("scenes", []))
    dbs = sorted({str(scene.get("source_db_file", "")) for scene in scenes})
    rng = random.Random(seed)
    shuffled = list(dbs)
    rng.shuffle(shuffled)
    if len(shuffled) <= 1:
        holdout_dbs = set(shuffled)
    else:
        holdout_count = max(1, min(len(shuffled) - 1, round(len(shuffled) * holdout_fraction)))
        holdout_dbs = set(shuffled[:holdout_count])
    train_dbs = [db for db in dbs if db not in holdout_dbs]
    train_scenes = [scene for scene in scenes if str(scene.get("source_db_file", "")) in train_dbs]
    holdout_scenes = [scene for scene in scenes if str(scene.get("source_db_file", "")) in holdout_dbs]
    return (
        _report_with_scenes(replay_report, train_scenes),
        _report_with_scenes(replay_report, holdout_scenes),
        {
            "split_unit": "source_db_file",
            "seed": int(seed),
            "holdout_fraction": float(holdout_fraction),
            "train_db_count": len(train_dbs),
            "holdout_db_count": len(holdout_dbs),
            "train_scene_count": len(train_scenes),
            "holdout_scene_count": len(holdout_scenes),
            "train_dbs": train_dbs,
            "holdout_dbs": sorted(holdout_dbs),
        },
    )


def evaluate_selector_family(
    replay_report: Mapping[str, Any],
    *,
    replay_calibrated_selector: Any,
    baseline_selector: Any | None,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    rows = {
        "proxy_selector": evaluate_selector_policy(
            replay_report,
            selector_name="proxy_selector",
            selected_token_fn=lambda scene: str(scene["selected_token"]),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_calibrated": evaluate_selector_policy(
            replay_report,
            selector_name="replay_calibrated",
            selected_token_fn=lambda scene: _select_with_score_model(scene, replay_calibrated_selector),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }
    if baseline_selector is not None:
        rows["expert_mlp"] = evaluate_selector_policy(
            replay_report,
            selector_name="expert_mlp",
            selected_token_fn=lambda scene: _select_with_score_model(scene, baseline_selector),
            near_miss_threshold_m=near_miss_threshold_m,
        )
    return rows


def evaluate_selector_policy(
    replay_report: Mapping[str, Any],
    *,
    selector_name: str,
    selected_token_fn,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    scenes = list(replay_report.get("scenes", []))
    histogram: Counter[str] = Counter()
    replay_infeasible = 0
    replay_safe = 0
    replay_missing = 0
    proxy_safe_selected = 0
    recoverable = 0
    recovered = 0
    no_replay_safe_token = 0
    total_progress = 0.0
    total_proxy_score = 0.0
    ade_values = []
    fde_values = []
    heading_error_values = []
    for scene in scenes:
        selected_token = str(selected_token_fn(scene))
        candidate = _candidate_by_token(scene).get(selected_token)
        replay = _replay_by_token(scene).get(selected_token)
        histogram[selected_token] += 1
        if candidate is not None:
            proxy_safe_selected += 1 if bool(candidate.get("proxy_safe")) else 0
            total_progress += float(candidate.get("final_progress_m", 0.0))
            total_proxy_score += float(candidate.get("score", 0.0))
            utility = candidate_logged_ego_utility(candidate, scene)
            if utility is not None:
                ade_values.append(float(utility["ade_3s_m"]))
                fde_values.append(float(utility["fde_3s_m"]))
                heading_error_values.append(float(utility["heading_error_3s_rad"]))
        if replay is None or replay.get("realized_min_clearance_m") is None:
            replay_missing += 1
        elif float(replay["realized_min_clearance_m"]) < near_miss_threshold_m:
            replay_infeasible += 1
        else:
            replay_safe += 1
        baseline_fails = _selected_token_fails(scene, str(scene["selected_token"]), near_miss_threshold_m)
        oracle_token = scene.get("oracle_log_replay_safe_token")
        if baseline_fails and oracle_token is not None and str(oracle_token) != str(scene["selected_token"]):
            recoverable += 1
            if _token_is_replay_safe(scene, selected_token, near_miss_threshold_m):
                recovered += 1
        if oracle_token is None:
            no_replay_safe_token += 1
    scene_count = len(scenes)
    return {
        "selector_name": selector_name,
        "scene_count": scene_count,
        "selected_replay_infeasible_count": replay_infeasible,
        "selected_replay_infeasible_rate": round(replay_infeasible / scene_count, 6) if scene_count else 0.0,
        "selected_replay_safe_count": replay_safe,
        "selected_replay_missing_count": replay_missing,
        "proxy_safe_selected_count": proxy_safe_selected,
        "proxy_safe_selected_rate": round(proxy_safe_selected / scene_count, 6) if scene_count else 0.0,
        "recoverable_proxy_optimism_count": recoverable,
        "recoverable_proxy_optimism_fixed_count": recovered,
        "recoverable_proxy_optimism_recovery_rate": round(recovered / recoverable, 6) if recoverable else 0.0,
        "no_replay_safe_token_count": no_replay_safe_token,
        "mean_selected_progress_m": round(total_progress / scene_count, 6) if scene_count else 0.0,
        "mean_selected_proxy_score": round(total_proxy_score / scene_count, 6) if scene_count else 0.0,
        "mean_ade_3s_m": _mean_or_none(ade_values),
        "mean_fde_3s_m": _mean_or_none(fde_values),
        "mean_heading_error_3s_rad": _mean_or_none(heading_error_values),
        "expert_distance_available_count": len(ade_values),
        "selected_token_entropy": _token_entropy(histogram),
        "selected_token_histogram": dict(sorted(histogram.items())),
    }


def candidate_logged_ego_utility(
    candidate: Mapping[str, Any],
    scene: Mapping[str, Any],
    *,
    horizon_s: float = 3.0,
    dt_s: float = 0.5,
) -> dict[str, float] | None:
    expert = list(scene.get("expert_trajectory", []))
    poses = list(candidate.get("poses", []))
    count = min(len(expert), len(poses), max(1, int(round(horizon_s / max(dt_s, 1.0e-6)))))
    if count <= 0:
        return None
    distances = []
    for index in range(count):
        pose = poses[index]
        row = expert[index]
        px = float(pose[0])
        py = float(pose[1])
        ex = float(row["x_m"]) if isinstance(row, Mapping) else float(row[0])
        ey = float(row["y_m"]) if isinstance(row, Mapping) else float(row[1])
        distances.append(math.hypot(px - ex, py - ey))
    last_pose = poses[count - 1]
    last_expert = expert[count - 1]
    expert_heading = (
        float(last_expert.get("heading_rad", 0.0))
        if isinstance(last_expert, Mapping)
        else (float(last_expert[2]) if len(last_expert) > 2 else 0.0)
    )
    pose_heading = float(last_pose[2]) if len(last_pose) > 2 else 0.0
    return {
        "ade_3s_m": round(sum(distances) / count, 6),
        "fde_3s_m": round(distances[-1], 6),
        "heading_error_3s_rad": round(abs(_normalize_angle(pose_heading - expert_heading)), 6),
    }


def markdown_evaluation(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Replay-Calibrated Selector Evaluation",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Near-miss threshold: `{report['near_miss_threshold_m']:.2f} m`",
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


def _report_with_scenes(report: Mapping[str, Any], scenes: list[Mapping[str, Any]]) -> dict[str, Any]:
    copied = json.loads(json.dumps(report))
    copied["scenes"] = scenes
    copied["scene_count"] = len(scenes)
    return copied


def _select_with_score_model(scene: Mapping[str, Any], selector: Any) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    selected = max(
        candidates,
        key=lambda candidate: (
            selector.predict_score(candidate["features"] if "features" in candidate else _features(scene, candidate)),
            float(candidate.get("min_proxy_clearance_m", 0.0)),
            float(candidate.get("final_progress_m", 0.0)),
        ),
    )
    return str(selected["token"])


def select_with_constrained_replay_risk(
    scene: Mapping[str, Any],
    selector: Any,
    *,
    risk_threshold: float,
    proxy_score_weight: float | None = None,
    progress_weight: float | None = None,
    unsafe_penalty: float | None = None,
) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    scored = []
    for candidate in candidates:
        features = candidate["features"] if "features" in candidate else _features(scene, candidate)
        risk = float(selector.predict_risk(features))
        utility = _constrained_candidate_utility(
            features,
            selector,
            proxy_score_weight=proxy_score_weight,
            progress_weight=progress_weight,
            unsafe_penalty=unsafe_penalty,
        )
        row = {
            "candidate": candidate,
            "risk": risk,
            "utility": utility,
            "clearance": float(candidate.get("min_proxy_clearance_m", 0.0)),
            "progress": float(candidate.get("final_progress_m", 0.0)),
        }
        scored.append(row)
    feasible = [row for row in scored if row["risk"] <= float(risk_threshold)]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (row["utility"], row["clearance"], row["progress"], -row["risk"]),
        )
    else:
        selected = min(
            scored,
            key=lambda row: (row["risk"], -row["utility"], -row["clearance"], -row["progress"]),
        )
    return str(selected["candidate"]["token"])


def _constrained_candidate_utility(
    features: Mapping[str, float],
    selector: Any,
    *,
    proxy_score_weight: float | None,
    progress_weight: float | None,
    unsafe_penalty: float | None,
) -> float:
    proxy_weight = float(
        getattr(selector, "proxy_score_weight", 1.0) if proxy_score_weight is None else proxy_score_weight
    )
    progress_w = float(getattr(selector, "progress_weight", 0.0) if progress_weight is None else progress_weight)
    unsafe = float(getattr(selector, "unsafe_penalty", 2.0) if unsafe_penalty is None else unsafe_penalty)
    proxy_safe = bool(float(features["candidate_proxy_safe"]) >= 0.5)
    return (
        proxy_weight * float(features["candidate_score_heuristic"])
        + progress_w * float(features["candidate_final_progress_m"])
        - (0.0 if proxy_safe else unsafe)
    )


def _features(scene: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    return selector_feature_row(scene, candidate)


def _select_replay_oracle_token(scene: Mapping[str, Any]) -> str:
    oracle = scene.get("oracle_log_replay_safe_token")
    if oracle is not None:
        return str(oracle)
    candidates = [
        row
        for row in scene.get("candidate_replay_evaluations", [])
        if row.get("realized_min_clearance_m") is not None
    ]
    if not candidates:
        return str(scene.get("selected_token", ""))
    return str(max(candidates, key=lambda row: float(row["realized_min_clearance_m"]))["token"])


def _candidate_by_token(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(candidate["token"]): candidate for candidate in scene.get("candidates", [])}


def _replay_by_token(scene: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(candidate["token"]): candidate for candidate in scene.get("candidate_replay_evaluations", [])}


def _selected_token_fails(scene: Mapping[str, Any], token: str, near_miss_threshold_m: float) -> bool:
    replay = _replay_by_token(scene).get(token)
    if replay is None or replay.get("realized_min_clearance_m") is None:
        return False
    return float(replay["realized_min_clearance_m"]) < near_miss_threshold_m


def _token_is_replay_safe(scene: Mapping[str, Any], token: str, near_miss_threshold_m: float) -> bool:
    replay = _replay_by_token(scene).get(token)
    if replay is None or replay.get("realized_min_clearance_m") is None:
        return False
    return float(replay["realized_min_clearance_m"]) >= near_miss_threshold_m


def _token_entropy(histogram: Mapping[str, int]) -> float:
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in histogram.values():
        probability = float(count) / float(total)
        entropy -= probability * math.log(probability, 2)
    return round(entropy, 6)


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _optional_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


if __name__ == "__main__":
    raise SystemExit(main())
