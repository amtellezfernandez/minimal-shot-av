#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping

from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _replay_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import candidate_logged_ego_utility


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expert" / "replay.json"
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_replay_utility_barrier.json"
DEFAULT_OUTPUT_MARKDOWN = ROOT / "artifacts" / "corl2027" / "nuplan_replay_utility_barrier.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure whether recoverable replay failures require marginal or intervention-like progress loss."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MARKDOWN)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--retention-thresholds", default="0.25,0.5,0.75,0.9")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    thresholds = [float(value) for value in str(args.retention_thresholds).split(",") if value.strip()]
    report = analyze_utility_barrier(
        replay_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        retention_thresholds=thresholds,
    )
    report["input_replay_json"] = str(args.input_replay_json)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = markdown_report(report)
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    return 0


def analyze_utility_barrier(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    retention_thresholds: list[float],
) -> dict[str, Any]:
    rows = [
        utility_barrier_row(scene, near_miss_threshold_m=near_miss_threshold_m)
        for scene in replay_report.get("scenes", [])
    ]
    recoverable = [row for row in rows if row is not None]
    retention_rows = retention_summary(recoverable, retention_thresholds=retention_thresholds)
    return {
        "schema": "nuplan_replay_utility_barrier_v1",
        "near_miss_threshold_m": float(near_miss_threshold_m),
        "scene_count": len(list(replay_report.get("scenes", []))),
        "recoverable_proxy_failure_count": len(recoverable),
        "summary": summarize_barrier_rows(recoverable),
        "retention_summary": retention_rows,
        "retention_bin_token_histogram": retention_bin_token_histogram(recoverable),
        "oracle_token_histogram": dict(sorted(Counter(row["best_replay_safe_token"] for row in recoverable).items())),
        "proxy_token_histogram": dict(sorted(Counter(row["proxy_token"] for row in recoverable).items())),
        "rows": recoverable,
    }


def utility_barrier_row(
    scene: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
) -> dict[str, Any] | None:
    proxy_token = str(scene.get("selected_token", ""))
    if _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m):
        return None
    candidate_by_token = _candidate_by_token(scene)
    proxy_candidate = candidate_by_token.get(proxy_token)
    if proxy_candidate is None:
        return None
    replay_safe_candidates = [
        candidate
        for candidate in candidate_by_token.values()
        if _token_is_replay_safe(scene, str(candidate.get("token", "")), near_miss_threshold_m)
    ]
    if not replay_safe_candidates:
        return None
    best_safe = max(
        replay_safe_candidates,
        key=lambda candidate: (
            float(candidate.get("final_progress_m", 0.0)),
            replay_clearance(scene, str(candidate.get("token", ""))),
            float(candidate.get("score", 0.0)),
        ),
    )
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0))
    safe_progress = float(best_safe.get("final_progress_m", 0.0))
    progress_drop = proxy_progress - safe_progress
    progress_retention = safe_progress / proxy_progress if proxy_progress > 1.0e-6 else 1.0
    proxy_utility = candidate_logged_ego_utility(proxy_candidate, scene)
    safe_utility = candidate_logged_ego_utility(best_safe, scene)
    return {
        "scene_id": str(scene.get("scene_id", "")),
        "source_db_file": str(scene.get("source_db_file", "")),
        "proxy_token": proxy_token,
        "best_replay_safe_token": str(best_safe.get("token", "")),
        "proxy_progress_m": round(proxy_progress, 6),
        "best_replay_safe_progress_m": round(safe_progress, 6),
        "progress_drop_m": round(progress_drop, 6),
        "progress_retention": round(progress_retention, 6),
        "relative_progress_loss": round(max(0.0, 1.0 - progress_retention), 6),
        "proxy_replay_clearance_m": replay_clearance(scene, proxy_token),
        "best_replay_safe_clearance_m": replay_clearance(scene, str(best_safe.get("token", ""))),
        "proxy_ade_3s_m": None if proxy_utility is None else proxy_utility["ade_3s_m"],
        "best_replay_safe_ade_3s_m": None if safe_utility is None else safe_utility["ade_3s_m"],
        "intervention_like": bool(progress_retention < 0.25),
    }


def replay_clearance(scene: Mapping[str, Any], token: str) -> float | None:
    replay = _replay_by_token(scene).get(str(token))
    if replay is None or replay.get("realized_min_clearance_m") is None:
        return None
    return round(float(replay["realized_min_clearance_m"]), 6)


def summarize_barrier_rows(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    retentions = [float(row["progress_retention"]) for row in rows]
    drops = [float(row["progress_drop_m"]) for row in rows]
    proxy_progress = [float(row["proxy_progress_m"]) for row in rows]
    safe_progress = [float(row["best_replay_safe_progress_m"]) for row in rows]
    return {
        "progress_retention_mean": mean_or_none(retentions),
        "progress_retention_median": percentile_or_none(retentions, 0.5),
        "progress_retention_p25": percentile_or_none(retentions, 0.25),
        "progress_retention_p75": percentile_or_none(retentions, 0.75),
        "progress_drop_m_mean": mean_or_none(drops),
        "progress_drop_m_median": percentile_or_none(drops, 0.5),
        "proxy_progress_m_mean": mean_or_none(proxy_progress),
        "best_replay_safe_progress_m_mean": mean_or_none(safe_progress),
        "intervention_like_count": sum(1 for row in rows if bool(row["intervention_like"])),
        "intervention_like_rate": rate(sum(1 for row in rows if bool(row["intervention_like"])), len(rows)),
    }


def retention_summary(
    rows: list[Mapping[str, Any]],
    *,
    retention_thresholds: list[float],
) -> list[dict[str, Any]]:
    output = []
    for threshold in retention_thresholds:
        retained = [row for row in rows if float(row["progress_retention"]) >= float(threshold)]
        output.append(
            {
                "min_progress_retention": float(threshold),
                "recoverable_count": len(retained),
                "recoverable_rate": rate(len(retained), len(rows)),
                "blocked_count": len(rows) - len(retained),
                "blocked_rate": rate(len(rows) - len(retained), len(rows)),
                "mean_safe_progress_m": mean_or_none(
                    [float(row["best_replay_safe_progress_m"]) for row in retained]
                ),
            }
        )
    return output


def retention_bin_token_histogram(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    bins = [
        ("lt_0.25", lambda value: value < 0.25),
        ("0.25_to_0.50", lambda value: 0.25 <= value < 0.5),
        ("0.50_to_0.75", lambda value: 0.5 <= value < 0.75),
        ("gte_0.75", lambda value: value >= 0.75),
    ]
    output = []
    for name, predicate in bins:
        group = [row for row in rows if predicate(float(row["progress_retention"]))]
        output.append(
            {
                "retention_bin": name,
                "count": len(group),
                "rate": rate(len(group), len(rows)),
                "best_replay_safe_token_histogram": dict(
                    sorted(Counter(str(row["best_replay_safe_token"]) for row in group).items())
                ),
            }
        )
    return output


def markdown_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# nuPlan Replay Utility-Barrier Audit",
        "",
        f"- Input replay JSON: `{report.get('input_replay_json', '')}`",
        f"- Scene count: `{report['scene_count']}`",
        f"- Recoverable proxy failures: `{report['recoverable_proxy_failure_count']}`",
        f"- Mean progress retention of best replay-safe token: `{summary['progress_retention_mean']:.3f}`",
        f"- Median progress retention: `{summary['progress_retention_median']:.3f}`",
        f"- Intervention-like recovery rate `<0.25 retention`: `{summary['intervention_like_rate']:.3f}`",
        "",
        "## Retention Thresholds",
        "",
        "| Min retention | Recoverable | Rate | Blocked | Blocked rate | Mean safe progress |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["retention_summary"]:
        lines.append(
            f"| {row['min_progress_retention']:.2f} | {row['recoverable_count']} | "
            f"{row['recoverable_rate']:.3f} | {row['blocked_count']} | {row['blocked_rate']:.3f} | "
            f"{optional_metric(row['mean_safe_progress_m'])} |"
        )
    lines.extend(
        [
            "",
            "## Retention Bins",
            "",
            "| Retention bin | Count | Rate | Best replay-safe tokens |",
            "|---|---:|---:|---|",
        ]
    )
    for row in report["retention_bin_token_histogram"]:
        lines.append(
            f"| {row['retention_bin']} | {row['count']} | {row['rate']:.3f} | "
            f"`{json.dumps(row['best_replay_safe_token_histogram'], sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Token Histograms",
            "",
            f"- Proxy tokens: `{json.dumps(report['proxy_token_histogram'], sort_keys=True)}`",
            f"- Best replay-safe tokens: `{json.dumps(report['oracle_token_histogram'], sort_keys=True)}`",
            "",
            "## Interpretation",
            "",
            "- This audit measures progress loss from the failing proxy token to the best replay-safe token.",
            "- Low retention means recovery is intervention-like, not a marginal progress-preserving correction.",
        ]
    )
    return "\n".join(lines)


def mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def percentile_or_none(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(float(quantile) * (len(ordered) - 1)))))
    return round(ordered[index], 6)


def rate(count: int, total: int) -> float:
    return round(float(count) / float(total), 6) if total else 0.0


def optional_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
