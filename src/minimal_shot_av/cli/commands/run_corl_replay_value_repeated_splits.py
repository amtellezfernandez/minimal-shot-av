#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path
from statistics import mean
from statistics import pstdev
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.run_nuplan_recoverable_regret_bootstrap import (
    run_recoverable_regret_iterations,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import (
    split_replay_report_by_db,
)
from minimal_shot_av.cli.commands.train_nuplan_replay_value_selector import (
    build_replay_value_examples,
    evaluate_selector_family,
)
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_mlp_selector


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expanded" / "replay.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "replay_value_repeated_splits_summary.json"
DEFAULT_OUTPUT_CSV = ROOT / "artifacts" / "corl2027" / "replay_value_repeated_splits_summary.csv"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "replay_value_repeated_splits_summary.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated DB-split CoRL replay-value comparisons on the expanded 21-token replay surface."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--split-count", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--scene-token-iterations", type=int, default=3)
    parser.add_argument("--scene-token-hidden-dim", type=int, default=48)
    parser.add_argument("--scene-token-epochs", type=int, default=700)
    parser.add_argument("--scene-token-learning-rate", type=float, default=0.03)
    parser.add_argument("--scene-token-regret-sample-weight", type=float, default=4.0)
    parser.add_argument("--scene-token-changed-target-weight", type=float, default=2.0)
    parser.add_argument("--replay-value-hidden-dim", type=int, default=32)
    parser.add_argument("--replay-value-epochs", type=int, default=400)
    parser.add_argument("--replay-value-learning-rate", type=float, default=0.03)
    parser.add_argument("--fail-penalty", type=float, default=250.0)
    parser.add_argument("--safe-progress-weight", type=float, default=1.0)
    parser.add_argument("--balanced-progress-weight", type=float, default=2.0)
    parser.add_argument("--ade-weight", type=float, default=2.0)
    parser.add_argument("--clearance-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    report = run_repeated_splits(
        replay_report,
        input_replay_json=str(args.input_replay_json),
        split_count=int(args.split_count),
        seed_start=int(args.seed_start),
        holdout_fraction=float(args.holdout_fraction),
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        scene_token_iterations=int(args.scene_token_iterations),
        scene_token_hidden_dim=int(args.scene_token_hidden_dim),
        scene_token_epochs=int(args.scene_token_epochs),
        scene_token_learning_rate=float(args.scene_token_learning_rate),
        scene_token_regret_sample_weight=float(args.scene_token_regret_sample_weight),
        scene_token_changed_target_weight=float(args.scene_token_changed_target_weight),
        replay_value_hidden_dim=int(args.replay_value_hidden_dim),
        replay_value_epochs=int(args.replay_value_epochs),
        replay_value_learning_rate=float(args.replay_value_learning_rate),
        fail_penalty=float(args.fail_penalty),
        safe_progress_weight=float(args.safe_progress_weight),
        balanced_progress_weight=float(args.balanced_progress_weight),
        ade_weight=float(args.ade_weight),
        clearance_weight=float(args.clearance_weight),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.write_text(report_csv(report), encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(report_markdown(report) + "\n", encoding="utf-8")
    print(report_markdown(report))
    return 0


def run_repeated_splits(
    replay_report: Mapping[str, Any],
    *,
    input_replay_json: str,
    split_count: int,
    seed_start: int,
    holdout_fraction: float,
    near_miss_threshold_m: float,
    scene_token_iterations: int,
    scene_token_hidden_dim: int,
    scene_token_epochs: int,
    scene_token_learning_rate: float,
    scene_token_regret_sample_weight: float,
    scene_token_changed_target_weight: float,
    replay_value_hidden_dim: int,
    replay_value_epochs: int,
    replay_value_learning_rate: float,
    fail_penalty: float,
    safe_progress_weight: float,
    balanced_progress_weight: float,
    ade_weight: float,
    clearance_weight: float,
) -> dict[str, Any]:
    safe_score_config = {
        "near_miss_threshold_m": near_miss_threshold_m,
        "fail_penalty": fail_penalty,
        "progress_weight": safe_progress_weight,
        "ade_weight": ade_weight,
        "clearance_weight": clearance_weight,
    }
    split_rows: list[dict[str, Any]] = []
    for offset in range(max(1, split_count)):
        seed = seed_start + offset
        train_report, holdout_report, split = split_replay_report_by_db(
            replay_report,
            holdout_fraction=holdout_fraction,
            seed=seed,
        )
        scene_iterations, _scene_model, _scene_fit = run_recoverable_regret_iterations(
            train_report,
            holdout_report,
            iterations=scene_token_iterations,
            score_config=safe_score_config,
            hidden_dim=scene_token_hidden_dim,
            epochs=scene_token_epochs,
            learning_rate=scene_token_learning_rate,
            seed=seed,
            regret_sample_weight=scene_token_regret_sample_weight,
            changed_target_weight=scene_token_changed_target_weight,
            student_kind="scene_token_mlp",
        )
        scene_holdout = scene_iterations[-1]["holdout"]
        proxy_row = _selector_row(scene_holdout, "g0_proxy_top1")
        scene_row = _selector_row(scene_holdout, "g1_student_top1")

        safe_eval = _train_and_eval_replay_value(
            train_report=train_report,
            holdout_report=holdout_report,
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            progress_weight=safe_progress_weight,
            ade_weight=ade_weight,
            hidden_dim=replay_value_hidden_dim,
            epochs=replay_value_epochs,
            learning_rate=replay_value_learning_rate,
            seed=seed,
        )
        balanced_eval = _train_and_eval_replay_value(
            train_report=train_report,
            holdout_report=holdout_report,
            near_miss_threshold_m=near_miss_threshold_m,
            fail_penalty=fail_penalty,
            progress_weight=balanced_progress_weight,
            ade_weight=ade_weight,
            hidden_dim=replay_value_hidden_dim,
            epochs=replay_value_epochs,
            learning_rate=replay_value_learning_rate,
            seed=seed,
        )
        split_rows.append(
            {
                "seed": seed,
                "split": split,
                "selectors": {
                    "proxy_top1": _named_metrics(proxy_row),
                    "scene_token_student": _named_metrics(scene_row),
                    "replay_value_safe": _named_metrics(safe_eval["holdout"]["replay_value"]),
                    "replay_value_balanced": _named_metrics(balanced_eval["holdout"]["replay_value"]),
                    "replay_value_oracle": _named_metrics(balanced_eval["holdout"]["replay_oracle"]),
                },
            }
        )
    summary_rows = [
        _aggregate_selector(split_rows, "proxy_top1", "Proxy top-1"),
        _aggregate_selector(split_rows, "scene_token_student", "Scene-token student"),
        _aggregate_selector(split_rows, "replay_value_safe", "Replay-value, safety-heavy"),
        _aggregate_selector(split_rows, "replay_value_balanced", "Replay-value, balanced"),
        _aggregate_selector(split_rows, "replay_value_oracle", "Replay-value oracle"),
    ]
    return {
        "schema": "corl_replay_value_repeated_splits_v1",
        "input_replay_json": input_replay_json,
        "split_count": max(1, split_count),
        "seed_start": seed_start,
        "holdout_fraction": holdout_fraction,
        "near_miss_threshold_m": near_miss_threshold_m,
        "scene_token_config": {
            "iterations": scene_token_iterations,
            "hidden_dim": scene_token_hidden_dim,
            "epochs": scene_token_epochs,
            "learning_rate": scene_token_learning_rate,
            "regret_sample_weight": scene_token_regret_sample_weight,
            "changed_target_weight": scene_token_changed_target_weight,
        },
        "replay_value_safe_objective": {
            "fail_penalty": fail_penalty,
            "progress_weight": safe_progress_weight,
            "ade_weight": ade_weight,
        },
        "replay_value_balanced_objective": {
            "fail_penalty": fail_penalty,
            "progress_weight": balanced_progress_weight,
            "ade_weight": ade_weight,
        },
        "splits": split_rows,
        "rows": summary_rows,
    }


def _train_and_eval_replay_value(
    *,
    train_report: Mapping[str, Any],
    holdout_report: Mapping[str, Any],
    near_miss_threshold_m: float,
    fail_penalty: float,
    progress_weight: float,
    ade_weight: float,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    train_examples = build_replay_value_examples(
        train_report,
        near_miss_threshold_m=near_miss_threshold_m,
        fail_penalty=fail_penalty,
        objective_progress_weight=progress_weight,
        objective_ade_weight=ade_weight,
    )
    selector, train_fit_metrics = fit_replay_value_mlp_selector(
        train_examples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    return {
        "train_fit_metrics": train_fit_metrics,
        "holdout": evaluate_selector_family(
            holdout_report,
            replay_value_selector=selector,
            baseline_selector=None,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }


def _selector_row(report: Mapping[str, Any], selector_name: str) -> Mapping[str, Any]:
    if selector_name not in report:
        raise KeyError(f"missing selector {selector_name!r}")
    return report[selector_name]


def _named_metrics(row: Mapping[str, Any]) -> dict[str, float]:
    return {
        "replay_fail_rate": float(row["selected_replay_infeasible_rate"]),
        "progress_m": float(row["mean_selected_progress_m"]),
        "ade_3s_m": _float_or_none(row.get("mean_ade_3s_m")),
        "fde_3s_m": _float_or_none(row.get("mean_fde_3s_m")),
        "entropy": float(row["selected_token_entropy"]),
    }


def _aggregate_selector(split_rows: list[Mapping[str, Any]], selector_id: str, label: str) -> dict[str, Any]:
    metrics = [row["selectors"][selector_id] for row in split_rows]
    return {
        "id": selector_id,
        "label": label,
        "replay_fail_rate": _mean_std(metrics, "replay_fail_rate"),
        "progress_m": _mean_std(metrics, "progress_m"),
        "ade_3s_m": _mean_std(metrics, "ade_3s_m"),
        "fde_3s_m": _mean_std(metrics, "fde_3s_m"),
        "entropy": _mean_std(metrics, "entropy"),
    }


def _mean_std(rows: list[Mapping[str, Any]], key: str) -> dict[str, float | None]:
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return {"mean": None, "std": None}
    return {
        "mean": round(float(mean(values)), 6),
        "std": round(float(pstdev(values)), 6),
    }


def report_csv(report: Mapping[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "label",
            "replay_fail_rate_mean",
            "replay_fail_rate_std",
            "progress_m_mean",
            "progress_m_std",
            "ade_3s_m_mean",
            "ade_3s_m_std",
            "fde_3s_m_mean",
            "fde_3s_m_std",
            "entropy_mean",
            "entropy_std",
        ]
    )
    for row in report["rows"]:
        writer.writerow(
            [
                row["id"],
                row["label"],
                _csv_metric(row["replay_fail_rate"]["mean"]),
                _csv_metric(row["replay_fail_rate"]["std"]),
                _csv_metric(row["progress_m"]["mean"]),
                _csv_metric(row["progress_m"]["std"]),
                _csv_metric(row["ade_3s_m"]["mean"]),
                _csv_metric(row["ade_3s_m"]["std"]),
                _csv_metric(row["fde_3s_m"]["mean"]),
                _csv_metric(row["fde_3s_m"]["std"]),
                _csv_metric(row["entropy"]["mean"]),
                _csv_metric(row["entropy"]["std"]),
            ]
        )
    return buffer.getvalue()


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# CoRL Replay-Value Repeated-Split Summary",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Split count: `{report['split_count']}`",
        f"- Seed start: `{report['seed_start']}`",
        f"- Holdout fraction: `{report['holdout_fraction']:.2f}`",
        "",
        "| Selector | Replay fail | Progress (m) | ADE3 (m) | FDE3 (m) | Entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {_metric_mean_std(row['replay_fail_rate'])} | "
            f"{_metric_mean_std(row['progress_m'])} | "
            f"{_metric_mean_std(row['ade_3s_m'])} | "
            f"{_metric_mean_std(row['fde_3s_m'])} | "
            f"{_metric_mean_std(row['entropy'])} |"
        )
    lines.extend(["", "## Per-split Seeds", ""])
    for split in report["splits"]:
        lines.append(
            f"- seed `{split['seed']}`: "
            f"proxy `{split['selectors']['proxy_top1']['replay_fail_rate']:.3f}`, "
            f"scene-token `{split['selectors']['scene_token_student']['replay_fail_rate']:.3f}`, "
            f"balanced `{split['selectors']['replay_value_balanced']['replay_fail_rate']:.3f}`"
        )
    return "\n".join(lines)


def _metric_mean_std(metric: Mapping[str, Any]) -> str:
    if metric["mean"] is None:
        return "n/a"
    return f"{float(metric['mean']):.3f} ± {float(metric['std']):.3f}"


def _csv_metric(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6f}"


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


if __name__ == "__main__":
    raise SystemExit(main())
