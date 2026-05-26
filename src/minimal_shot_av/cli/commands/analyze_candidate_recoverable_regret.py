#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expert" / "replay.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "candidate_recoverable_regret.json"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "candidate_recoverable_regret.md"
DEFAULT_TARGETS_JSONL = ROOT / "artifacts" / "corl2027" / "candidate_recoverable_regret_targets.jsonl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze recoverable regret for arbitrary planner candidate sets. "
            "Accepts a generic candidate-set JSON/JSONL or a nuPlan replay report."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--targets-jsonl", type=Path, default=DEFAULT_TARGETS_JSONL)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--failure-penalty", type=float, default=250.0)
    parser.add_argument("--progress-weight", type=float, default=1.0)
    parser.add_argument("--ade-weight", type=float, default=2.0)
    parser.add_argument("--fde-weight", type=float, default=0.0)
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    parser.add_argument(
        "--utility-regret-threshold",
        type=float,
        default=5.0,
        help="Minimum positive value gap counted as recoverable utility regret when top-1 is already safe.",
    )
    parser.add_argument("--target-limit", type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = read_json_or_jsonl(args.input_json)
    candidate_sets = normalize_candidate_sets(
        payload,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
    )
    score_config = {
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "failure_penalty": float(args.failure_penalty),
        "progress_weight": float(args.progress_weight),
        "ade_weight": float(args.ade_weight),
        "fde_weight": float(args.fde_weight),
        "clearance_weight": float(args.clearance_weight),
    }
    report = analyze_candidate_recoverable_regret(
        candidate_sets,
        input_json=str(args.input_json),
        score_config=score_config,
        utility_regret_threshold=float(args.utility_regret_threshold),
        target_limit=args.target_limit,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown_report(report) + "\n", encoding="utf-8")
    args.targets_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.targets_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in report["bootstrap_targets"]),
        encoding="utf-8",
    )
    print(markdown_report(report))
    return 0


def read_json_or_jsonl(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if not stripped:
        raise ValueError(f"empty input: {path}")
    if stripped[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [json.loads(line) for line in text.splitlines() if line.strip()]
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def normalize_candidate_sets(payload: Any, *, near_miss_threshold_m: float) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping) and "candidate_sets" in payload:
        return [
            normalize_generic_candidate_set(row, near_miss_threshold_m=near_miss_threshold_m)
            for row in payload["candidate_sets"]
        ]
    if isinstance(payload, list):
        return [
            normalize_generic_candidate_set(row, near_miss_threshold_m=near_miss_threshold_m)
            for row in payload
        ]
    if isinstance(payload, Mapping) and "scenes" in payload:
        return [
            normalize_nuplan_scene(scene, near_miss_threshold_m=near_miss_threshold_m)
            for scene in payload["scenes"]
        ]
    raise ValueError(
        "input must contain candidate_sets, be JSONL/list candidate sets, "
        "or be a nuPlan replay report with scenes"
    )


def normalize_generic_candidate_set(row: Mapping[str, Any], *, near_miss_threshold_m: float) -> dict[str, Any]:
    candidates = [
        normalize_generic_candidate(candidate, near_miss_threshold_m=near_miss_threshold_m)
        for candidate in row.get("candidates", [])
    ]
    top1_id = str(row.get("top1_candidate_id") or row.get("selected_candidate_id") or "")
    if not top1_id:
        for candidate in candidates:
            if bool(candidate.get("is_top1", False)):
                top1_id = str(candidate["candidate_id"])
                break
    if not top1_id and candidates:
        top1_id = str(candidates[0]["candidate_id"])
    return {
        "scene_id": str(row.get("scene_id", "")),
        "source": str(row.get("source") or row.get("source_db_file") or ""),
        "scenario_type": str(row.get("scenario_type", "")),
        "generator": str(row.get("generator") or row.get("planner") or ""),
        "generator_iteration": row.get("generator_iteration", row.get("iteration")),
        "top1_candidate_id": top1_id,
        "candidates": candidates,
    }


def normalize_generic_candidate(candidate: Mapping[str, Any], *, near_miss_threshold_m: float) -> dict[str, Any]:
    metrics = dict(candidate.get("metrics", {}))
    clearance = first_number(
        metrics,
        "clearance_m",
        "realized_min_clearance_m",
        "min_clearance_m",
        "replay_min_clearance_m",
    )
    replay_fail = metrics.get("replay_fail", metrics.get("replay_infeasible", metrics.get("failed")))
    if replay_fail is None and clearance is not None:
        replay_fail = float(clearance) < float(near_miss_threshold_m)
    return {
        "candidate_id": str(candidate.get("candidate_id") or candidate.get("id") or candidate.get("token") or ""),
        "token": None if candidate.get("token") is None else str(candidate.get("token")),
        "is_top1": bool(candidate.get("is_top1", False)),
        "trajectory": candidate.get("trajectory", candidate.get("poses")),
        "metrics": {
            "replay_fail": bool(replay_fail) if replay_fail is not None else None,
            "progress_m": first_number(metrics, "progress_m", "final_progress_m", "route_progress_m"),
            "ade_3s_m": first_number(metrics, "ade_3s_m", "ade_m"),
            "fde_3s_m": first_number(metrics, "fde_3s_m", "fde_m"),
            "clearance_m": clearance,
            "raw_score": first_number(metrics, "score", "value", "metric_score"),
        },
    }


def normalize_nuplan_scene(scene: Mapping[str, Any], *, near_miss_threshold_m: float) -> dict[str, Any]:
    replay_by_token = _replay_by_token(scene)
    candidates = []
    for token, candidate in _candidate_by_token(scene).items():
        replay = replay_by_token.get(token, {})
        clearance = replay.get("realized_min_clearance_m")
        utility = candidate_logged_ego_utility(candidate, scene) or {}
        candidates.append(
            {
                "candidate_id": str(token),
                "token": str(token),
                "is_top1": str(token) == str(scene.get("selected_token", "")),
                "trajectory": candidate.get("poses"),
                "metrics": {
                    "replay_fail": (
                        None
                        if clearance is None
                        else float(clearance) < float(near_miss_threshold_m)
                    ),
                    "progress_m": float(candidate.get("final_progress_m", 0.0)),
                    "ade_3s_m": optional_float(utility.get("ade_3s_m")),
                    "fde_3s_m": optional_float(utility.get("fde_3s_m")),
                    "clearance_m": optional_float(clearance),
                    "raw_score": optional_float(candidate.get("score")),
                },
            }
        )
    return {
        "scene_id": str(scene.get("scene_id", "")),
        "source": str(scene.get("source_db_file", "")),
        "scenario_type": str(scene.get("scenario_type", "")),
        "generator": "nuplan_maneuver_token",
        "generator_iteration": scene.get("generator_iteration"),
        "top1_candidate_id": str(scene.get("selected_token", "")),
        "candidates": candidates,
    }


def analyze_candidate_recoverable_regret(
    candidate_sets: list[Mapping[str, Any]],
    *,
    input_json: str,
    score_config: Mapping[str, float],
    utility_regret_threshold: float,
    target_limit: int | None,
) -> dict[str, Any]:
    records = [
        scene_regret_record(
            candidate_set,
            score_config=score_config,
            utility_regret_threshold=utility_regret_threshold,
        )
        for candidate_set in candidate_sets
    ]
    valid_records = [record for record in records if bool(record.get("valid"))]
    targets = [
        bootstrap_target(record)
        for record in valid_records
        if record["regret_type"] in {"recoverable_safety_regret", "recoverable_utility_regret"}
    ]
    if target_limit is not None:
        targets = targets[: int(target_limit)]
    return {
        "schema": "candidate_recoverable_regret_v1",
        "input_json": input_json,
        "score_config": dict(score_config),
        "utility_regret_threshold": float(utility_regret_threshold),
        "claim_boundary": (
            "Model-agnostic recoverable-regret audit. It measures top-1, oracle@K, generation gap, "
            "recoverable safety regret, and recoverable utility regret for candidate generators; it does "
            "not assume ManeuverToken actions."
        ),
        "scene_count": len(candidate_sets),
        "valid_scene_count": len(valid_records),
        "candidate_distribution": candidate_distribution(candidate_sets),
        "gap_decomposition": gap_decomposition(valid_records),
        "score_summary": score_summary(valid_records),
        "utility_regret_sensitivity": utility_regret_sensitivity(valid_records),
        "per_generator_iteration": per_iteration_summary(valid_records),
        "regret_records": records,
        "bootstrap_target_count": len(targets),
        "bootstrap_target_regret_type_histogram": dict(sorted(Counter(row["regret_type"] for row in targets).items())),
        "bootstrap_targets": targets,
    }


def scene_regret_record(
    candidate_set: Mapping[str, Any],
    *,
    score_config: Mapping[str, float],
    utility_regret_threshold: float,
) -> dict[str, Any]:
    candidates = [
        {**candidate, "value": candidate_value(candidate, score_config)}
        for candidate in candidate_set.get("candidates", [])
    ]
    scored = [candidate for candidate in candidates if candidate["value"] is not None]
    top1_id = str(candidate_set.get("top1_candidate_id", ""))
    top1 = next((candidate for candidate in scored if str(candidate["candidate_id"]) == top1_id), None)
    if top1 is None and scored:
        top1 = scored[0]
    if top1 is None or not scored:
        return base_record(candidate_set, valid=False, candidates=candidates)
    oracle = max(scored, key=lambda candidate: (float(candidate["value"]), str(candidate["candidate_id"])))
    ranked = sorted(
        scored,
        key=lambda candidate: (float(candidate["value"]), str(candidate["candidate_id"])),
        reverse=True,
    )
    top1_value = float(top1["value"])
    oracle_value = float(oracle["value"])
    regret = max(0.0, oracle_value - top1_value)
    top1_safe = not bool(top1["metrics"].get("replay_fail", False))
    oracle_safe = not bool(oracle["metrics"].get("replay_fail", False))
    if not oracle_safe:
        regret_type = "generation_gap"
    elif not top1_safe and oracle_safe:
        regret_type = "recoverable_safety_regret"
    elif top1_safe and oracle_safe and regret >= float(utility_regret_threshold):
        regret_type = "recoverable_utility_regret"
    elif regret > 1.0e-6:
        regret_type = "minor_utility_regret"
    else:
        regret_type = "no_regret"
    return {
        **base_record(candidate_set, valid=True, candidates=candidates),
        "top1_candidate_id": str(top1["candidate_id"]),
        "oracle_candidate_id": str(oracle["candidate_id"]),
        "top1_value": top1_value,
        "oracle_at_k_value": oracle_value,
        "recoverable_regret": round(regret, 6),
        "top1_replay_fail": not top1_safe,
        "oracle_at_k_replay_fail": not oracle_safe,
        "regret_type": regret_type,
        "top1_metrics": dict(top1["metrics"]),
        "oracle_metrics": dict(oracle["metrics"]),
        "top1_token": top1.get("token"),
        "oracle_token": oracle.get("token"),
        "top1_trajectory": top1.get("trajectory"),
        "oracle_trajectory": oracle.get("trajectory"),
        "ranked_candidate_ids": [str(candidate["candidate_id"]) for candidate in ranked],
    }


def base_record(
    candidate_set: Mapping[str, Any],
    *,
    valid: bool,
    candidates: list[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "scene_id": str(candidate_set.get("scene_id", "")),
        "source": str(candidate_set.get("source", "")),
        "scenario_type": str(candidate_set.get("scenario_type", "")),
        "generator": str(candidate_set.get("generator", "")),
        "generator_iteration": candidate_set.get("generator_iteration"),
        "valid": bool(valid),
        "candidate_count": len(candidates),
        "top1_candidate_id": str(candidate_set.get("top1_candidate_id", "")),
    }


def candidate_value(candidate: Mapping[str, Any], score_config: Mapping[str, float]) -> float | None:
    metrics = dict(candidate.get("metrics", {}))
    replay_fail = metrics.get("replay_fail")
    progress = optional_float(metrics.get("progress_m"))
    ade = optional_float(metrics.get("ade_3s_m")) or 0.0
    fde = optional_float(metrics.get("fde_3s_m")) or 0.0
    clearance = optional_float(metrics.get("clearance_m")) or 0.0
    if replay_fail is None or progress is None:
        return None
    return round(
        -float(score_config["failure_penalty"]) * (1.0 if bool(replay_fail) else 0.0)
        + float(score_config["progress_weight"]) * progress
        - float(score_config["ade_weight"]) * ade
        - float(score_config["fde_weight"]) * fde
        + float(score_config["clearance_weight"]) * clearance,
        6,
    )


def bootstrap_target(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scene_id": record["scene_id"],
        "source": record["source"],
        "scenario_type": record["scenario_type"],
        "generator": record["generator"],
        "generator_iteration": record.get("generator_iteration"),
        "top1_candidate_id": record["top1_candidate_id"],
        "teacher_candidate_id": record["oracle_candidate_id"],
        "recoverable_regret": record["recoverable_regret"],
        "regret_type": record["regret_type"],
        "top1_value": record["top1_value"],
        "teacher_value": record["oracle_at_k_value"],
        "top1_metrics": record["top1_metrics"],
        "teacher_metrics": record["oracle_metrics"],
        "top1_token": record.get("top1_token"),
        "teacher_token": record.get("oracle_token"),
        "top1_trajectory": record.get("top1_trajectory"),
        "teacher_trajectory": record.get("oracle_trajectory"),
    }


def gap_decomposition(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    histogram = Counter(str(record["regret_type"]) for record in records)
    total = len(records)
    return {
        "generation_gap_count": int(histogram.get("generation_gap", 0)),
        "generation_gap_rate": rate(histogram.get("generation_gap", 0), total),
        "recoverable_safety_regret_count": int(histogram.get("recoverable_safety_regret", 0)),
        "recoverable_safety_regret_rate": rate(histogram.get("recoverable_safety_regret", 0), total),
        "recoverable_utility_regret_count": int(histogram.get("recoverable_utility_regret", 0)),
        "recoverable_utility_regret_rate": rate(histogram.get("recoverable_utility_regret", 0), total),
        "minor_utility_regret_count": int(histogram.get("minor_utility_regret", 0)),
        "no_regret_count": int(histogram.get("no_regret", 0)),
        "regret_type_histogram": dict(sorted(histogram.items())),
    }


def score_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    top1_scores = [float(record["top1_value"]) for record in records]
    oracle_scores = [float(record["oracle_at_k_value"]) for record in records]
    regrets = [float(record["recoverable_regret"]) for record in records]
    positive = sum(1 for regret in regrets if regret > 1.0e-6)
    top1_fail = sum(1 for record in records if bool(record["top1_replay_fail"]))
    oracle_fail = sum(1 for record in records if bool(record["oracle_at_k_replay_fail"]))
    return {
        "mean_top1_value": mean(top1_scores),
        "mean_oracle_at_k_value": mean(oracle_scores),
        "mean_recoverable_regret": mean(regrets),
        "median_recoverable_regret": median(regrets),
        "positive_regret_count": positive,
        "positive_regret_rate": rate(positive, len(records)),
        "top1_replay_fail_count": top1_fail,
        "top1_replay_fail_rate": rate(top1_fail, len(records)),
        "oracle_at_k_replay_fail_count": oracle_fail,
        "oracle_at_k_replay_fail_rate": rate(oracle_fail, len(records)),
    }


def utility_regret_sensitivity(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    thresholds = (0.001, 1.0, 5.0, 10.0)
    safe_records = [
        record
        for record in records
        if not bool(record["top1_replay_fail"]) and not bool(record["oracle_at_k_replay_fail"])
    ]
    return {
        f"{threshold:g}": {
            "recoverable_utility_regret_count": sum(
                1 for record in safe_records if float(record["recoverable_regret"]) >= threshold
            ),
            "recoverable_utility_regret_rate": rate(
                sum(1 for record in safe_records if float(record["recoverable_regret"]) >= threshold),
                len(records),
            ),
        }
        for threshold in thresholds
    }


def per_iteration_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        iteration = record.get("generator_iteration")
        key = "none" if iteration is None else str(iteration)
        grouped[key].append(record)
    return {
        key: {
            "scene_count": len(rows),
            "gap_decomposition": gap_decomposition(rows),
            "score_summary": score_summary(rows),
        }
        for key, rows in sorted(grouped.items())
    }


def candidate_distribution(candidate_sets: list[Mapping[str, Any]]) -> dict[str, Any]:
    counts = [len(list(row.get("candidates", []))) for row in candidate_sets]
    top1_histogram = Counter(str(row.get("top1_candidate_id", "")) for row in candidate_sets)
    candidate_ids = Counter()
    generators = Counter()
    for row in candidate_sets:
        generators.update([str(row.get("generator", ""))])
        candidate_ids.update(str(candidate.get("candidate_id", "")) for candidate in row.get("candidates", []))
    return {
        "mean_candidate_count": mean([float(value) for value in counts]),
        "min_candidate_count": min(counts) if counts else 0,
        "max_candidate_count": max(counts) if counts else 0,
        "top1_candidate_entropy": entropy(top1_histogram),
        "top1_candidate_histogram": dict(sorted(top1_histogram.items())),
        "candidate_id_entropy": entropy(candidate_ids),
        "generator_histogram": dict(sorted(generators.items())),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    gap = report["gap_decomposition"]
    score = report["score_summary"]
    lines = [
        "# Candidate Recoverable-Regret Audit",
        "",
        f"- Input: `{report['input_json']}`",
        f"- Scenes: `{report['scene_count']}`",
        f"- Valid scenes: `{report['valid_scene_count']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        "",
        "## Gap Decomposition",
        "",
        "| Regret type | Count | Rate |",
        "|---|---:|---:|",
        f"| generation gap | {gap['generation_gap_count']} | {gap['generation_gap_rate']:.3f} |",
        "| recoverable safety regret | "
        f"{gap['recoverable_safety_regret_count']} | {gap['recoverable_safety_regret_rate']:.3f} |",
        "| recoverable utility regret | "
        f"{gap['recoverable_utility_regret_count']} | {gap['recoverable_utility_regret_rate']:.3f} |",
        f"| minor utility regret | {gap['minor_utility_regret_count']} | n/a |",
        f"| no regret | {gap['no_regret_count']} | n/a |",
        "",
        "## Utility-Regret Sensitivity",
        "",
        "| Utility threshold | Count | Rate |",
        "|---:|---:|---:|",
    ]
    for threshold, metrics in report["utility_regret_sensitivity"].items():
        lines.append(
            f"| {threshold} | {metrics['recoverable_utility_regret_count']} | "
            f"{metrics['recoverable_utility_regret_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
        "## Top-1 vs Oracle@K",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean top-1 value | {score['mean_top1_value']:.3f} |",
        f"| mean oracle@K value | {score['mean_oracle_at_k_value']:.3f} |",
        f"| mean recoverable regret | {score['mean_recoverable_regret']:.3f} |",
        f"| positive regret rate | {score['positive_regret_rate']:.3f} |",
        f"| top-1 replay fail rate | {score['top1_replay_fail_rate']:.3f} |",
        f"| oracle@K replay fail rate | {score['oracle_at_k_replay_fail_rate']:.3f} |",
        "",
        "## Candidate Surface",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| mean candidates | {report['candidate_distribution']['mean_candidate_count']:.3f} |",
        f"| top-1 entropy | {report['candidate_distribution']['top1_candidate_entropy']:.3f} |",
        f"| candidate-id entropy | {report['candidate_distribution']['candidate_id_entropy']:.3f} |",
        "",
        "## Bootstrap Targets",
        "",
        f"- Target count: `{report['bootstrap_target_count']}`",
        f"- Target types: `{json.dumps(report['bootstrap_target_regret_type_histogram'], sort_keys=True)}`",
        "",
        ]
    )
    return "\n".join(lines)


def first_number(mapping: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = optional_float(mapping.get(key))
        if value is not None:
            return value
    return None


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return round(ordered[middle], 6)
    return round(0.5 * (ordered[middle - 1] + ordered[middle]), 6)


def rate(numerator: int | float, denominator: int | float) -> float:
    return round(float(numerator) / float(denominator), 6) if denominator else 0.0


def entropy(histogram: Mapping[str, int]) -> float:
    total = sum(int(count) for count in histogram.values())
    if total <= 0:
        return 0.0
    value = 0.0
    for count in histogram.values():
        probability = float(count) / float(total)
        value -= probability * math_log2(probability)
    return round(value, 6)


def math_log2(value: float) -> float:
    # Keep this file dependency-light for wrappers that import it directly.
    import math

    return math.log(value, 2)


if __name__ == "__main__":
    raise SystemExit(main())
