#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Mapping


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SCENE_TOKEN_JSON = (
    ROOT / "artifacts" / "corl2027" / "nuplan_recoverable_regret_bootstrap_interaction1000_expanded_iter3.json"
)
DEFAULT_REPLAY_VALUE_SAFE_JSON = (
    ROOT / "artifacts" / "corl2027" / "nuplan_replay_value_selector_interaction1000_expanded_mlp_eval.json"
)
DEFAULT_REPLAY_VALUE_BALANCED_JSON = (
    ROOT / "artifacts" / "corl2027" / "nuplan_replay_value_selector_interaction1000_expanded_balanced_eval.json"
)
DEFAULT_OUTPUT_JSON = ROOT / "artifacts" / "corl2027" / "replay_value_paper_summary.json"
DEFAULT_OUTPUT_CSV = ROOT / "artifacts" / "corl2027" / "replay_value_paper_summary.csv"
DEFAULT_OUTPUT_SVG = ROOT / "artifacts" / "corl2027" / "replay_value_paper_summary.svg"
DEFAULT_OUTPUT_MD = ROOT / "artifacts" / "corl2027" / "replay_value_paper_summary.md"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build canonical CoRL replay-value comparison artifacts from saved experiment JSON files."
    )
    parser.add_argument("--scene-token-json", type=Path, default=DEFAULT_SCENE_TOKEN_JSON)
    parser.add_argument("--replay-value-safe-json", type=Path, default=DEFAULT_REPLAY_VALUE_SAFE_JSON)
    parser.add_argument("--replay-value-balanced-json", type=Path, default=DEFAULT_REPLAY_VALUE_BALANCED_JSON)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-svg", type=Path, default=DEFAULT_OUTPUT_SVG)
    parser.add_argument("--output-markdown", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = build_report(
        scene_token=json.loads(args.scene_token_json.read_text(encoding="utf-8")),
        replay_value_safe=json.loads(args.replay_value_safe_json.read_text(encoding="utf-8")),
        replay_value_balanced=json.loads(args.replay_value_balanced_json.read_text(encoding="utf-8")),
        scene_token_json=str(args.scene_token_json),
        replay_value_safe_json=str(args.replay_value_safe_json),
        replay_value_balanced_json=str(args.replay_value_balanced_json),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.write_text(report_csv(report), encoding="utf-8")
    args.output_svg.parent.mkdir(parents=True, exist_ok=True)
    args.output_svg.write_text(report_svg(report), encoding="utf-8")
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text(report_markdown(report) + "\n", encoding="utf-8")
    print(report_markdown(report))
    return 0


def build_report(
    *,
    scene_token: Mapping[str, Any],
    replay_value_safe: Mapping[str, Any],
    replay_value_balanced: Mapping[str, Any],
    scene_token_json: str,
    replay_value_safe_json: str,
    replay_value_balanced_json: str,
) -> dict[str, Any]:
    proxy_row = _bootstrap_proxy_baseline_row(scene_token)
    scene_row = _selector_row(scene_token, "g1_student_top1")
    safe_row = _selector_row(replay_value_safe, "replay_value")
    balanced_row = _selector_row(replay_value_balanced, "replay_value")
    oracle_row = _selector_row(replay_value_balanced, "replay_oracle")
    rows = [
        _named_row("proxy_top1", "Proxy top-1", proxy_row),
        _named_row("scene_token_student", "Scene-token student", scene_row),
        _named_row("replay_value_safe", "Replay-value, safety-heavy", safe_row),
        _named_row("replay_value_balanced", "Replay-value, balanced", balanced_row),
        _named_row("replay_value_oracle", "Replay-value oracle", oracle_row),
    ]
    return {
        "schema": "corl_replay_value_comparison_v1",
        "sources": {
            "scene_token_json": scene_token_json,
            "replay_value_safe_json": replay_value_safe_json,
            "replay_value_balanced_json": replay_value_balanced_json,
        },
        "rows": rows,
    }


def _bootstrap_proxy_baseline_row(report: Mapping[str, Any]) -> Mapping[str, Any]:
    iterations = list(report.get("iterations", []))
    if iterations:
        first_holdout = iterations[0].get("holdout", {})
        if "g0_proxy_top1" in first_holdout:
            return first_holdout["g0_proxy_top1"]
    return _selector_row(report, "g0_proxy_top1")


def _selector_row(report: Mapping[str, Any], selector_name: str) -> Mapping[str, Any]:
    holdout = report["holdout"]
    if selector_name in holdout:
        return holdout[selector_name]
    raise KeyError(f"selector {selector_name!r} missing from holdout report")


def _named_row(row_id: str, label: str, row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row_id,
        "label": label,
        "replay_fail_count": int(row["selected_replay_infeasible_count"]),
        "replay_fail_rate": float(row["selected_replay_infeasible_rate"]),
        "progress_m": float(row["mean_selected_progress_m"]),
        "ade_3s_m": _float_or_none(row.get("mean_ade_3s_m")),
        "fde_3s_m": _float_or_none(row.get("mean_fde_3s_m")),
        "entropy": float(row["selected_token_entropy"]),
    }


def report_csv(report: Mapping[str, Any]) -> str:
    lines = [
        "id,label,replay_fail_count,replay_fail_rate,progress_m,ade_3s_m,fde_3s_m,entropy",
    ]
    for row in report["rows"]:
        values = [
            row["id"],
            row["label"],
            row["replay_fail_count"],
            f"{float(row['replay_fail_rate']):.6f}",
            f"{float(row['progress_m']):.6f}",
            _csv_metric(row["ade_3s_m"]),
            _csv_metric(row["fde_3s_m"]),
            f"{float(row['entropy']):.6f}",
        ]
        lines.append(",".join(str(value) for value in values))
    return "\n".join(lines) + "\n"


def report_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Expanded-Bank Replay-Value Comparison",
        "",
        "| Selector | Replay fail | Progress (m) | ADE3 (m) | FDE3 (m) | Entropy |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['label']} | {float(row['replay_fail_rate']):.3f} | {float(row['progress_m']):.3f} | "
            f"{_metric(row['ade_3s_m'])} | {_metric(row['fde_3s_m'])} | {float(row['entropy']):.3f} |"
        )
    return "\n".join(lines)


def report_svg(report: Mapping[str, Any]) -> str:
    rows = list(report["rows"])
    width = 900
    height = 520
    left = 84
    right = 36
    top = 54
    bottom = 88
    progress_values = [float(row["progress_m"]) for row in rows]
    fail_values = [float(row["replay_fail_rate"]) for row in rows]
    x_min, x_max = _expanded_range(min(progress_values), max(progress_values), pad_fraction=0.08)
    y_min, y_max = _expanded_range(min(fail_values), max(fail_values), pad_fraction=0.12, floor=0.0, ceiling=1.0)

    def sx(progress: float) -> float:
        return left + ((progress - x_min) / (x_max - x_min)) * (width - left - right)

    def sy(rate: float) -> float:
        return top + ((y_max - rate) / (y_max - y_min)) * (height - top - bottom)

    palette = {
        "proxy_top1": "#8a1538",
        "scene_token_student": "#b85c00",
        "replay_value_safe": "#1d4ed8",
        "replay_value_balanced": "#0f766e",
        "replay_value_oracle": "#4b5563",
    }
    x_ticks = _tick_values(x_min, x_max, 5)
    y_ticks = _tick_values(y_min, y_max, 5)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        ".title{font:700 21px sans-serif;fill:#171717}"
        ".label{font:13px sans-serif;fill:#333}"
        ".tick{font:12px sans-serif;fill:#555}"
        ".grid{stroke:#ddd;stroke-width:1}"
        ".axis{stroke:#222;stroke-width:1.5}"
        ".note{font:12px sans-serif;fill:#555}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#fffdf7"/>',
        f'<text class="title" x="{left}" y="30">Expanded-bank replay-value comparison</text>',
    ]
    for tick in x_ticks:
        x = sx(tick)
        elements.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height - bottom}"/>')
        elements.append(f'<text class="tick" x="{x:.2f}" y="{height - bottom + 24}" text-anchor="middle">{tick:.1f}</text>')
    for tick in y_ticks:
        y = sy(tick)
        elements.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}"/>')
        elements.append(f'<text class="tick" x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{tick:.2f}</text>')
    elements.extend(
        [
            f'<line class="axis" x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>',
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}"/>',
        ]
    )
    for row in rows:
        x = sx(float(row["progress_m"]))
        y = sy(float(row["replay_fail_rate"]))
        color = palette[row["id"]]
        elements.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="{color}" stroke="white" stroke-width="2"/>')
        elements.append(f'<text class="tick" x="{x + 10:.2f}" y="{y - 10:.2f}">{row["label"]}</text>')
    elements.extend(
        [
            f'<text class="label" x="{(left + width - right) / 2:.2f}" y="{height - 30}" text-anchor="middle">'
            "Mean selected progress (m), higher is better</text>",
            f'<text class="label" transform="translate(24 {(top + height - bottom) / 2:.2f}) rotate(-90)" text-anchor="middle">'
            "Replay-infeasible rate, lower is better</text>",
            f'<text class="note" x="{left}" y="{height - 8}">Source JSON files are recorded in the sidecar summary JSON.</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements) + "\n"


def _expanded_range(
    minimum: float,
    maximum: float,
    *,
    pad_fraction: float,
    floor: float | None = None,
    ceiling: float | None = None,
) -> tuple[float, float]:
    if abs(maximum - minimum) < 1.0e-8:
        minimum -= 1.0
        maximum += 1.0
    span = maximum - minimum
    padded_min = minimum - pad_fraction * span
    padded_max = maximum + pad_fraction * span
    if floor is not None:
        padded_min = max(floor, padded_min)
    if ceiling is not None:
        padded_max = min(ceiling, padded_max)
    if padded_max - padded_min < 1.0e-8:
        padded_max = padded_min + 1.0
    return padded_min, padded_max


def _tick_values(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [round(start, 3)]
    step = (end - start) / float(count - 1)
    return [round(start + step * index, 3) for index in range(count)]


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _metric(value: float | None) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _csv_metric(value: float | None) -> str:
    return "" if value is None else f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
