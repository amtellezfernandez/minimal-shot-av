"""Generate a DAgger ablation comparison figure from held-out LHS JSON artifacts.

The figure is written as an SVG so it stays dependency-light and easy to diff.
It compares the 2-step frontier against three 3-step variants:
  - baseline 3-step
  - source-decayed 3-step
  - no-inverse-frequency 3-step
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "artifacts" / "heldout_latin_hypercube"
DEFAULT_OUTPUT = ROOT / "docs" / "images" / "dagger_aggregation_ablation.svg"
SUITES = ("overall", "gauntlet", "adversarial", "hidden")

RUNS = [
    ("2-step Frontier", "results_dagger_iter2_mid.json", "#1b7f5f"),
    ("3-step Baseline", "results_dagger_iter3_mid.json", "#c45a36"),
    ("3-step Source Decay", "results_dagger_iter3_srcdecay_mid.json", "#2c6fb7"),
    ("3-step No Inv Freq", "results_dagger_iter3_noinv_mid.json", "#8c5cc4"),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _load_metrics(input_dir: Path) -> dict[str, dict[str, dict[str, float]]]:
    loaded: dict[str, dict[str, dict[str, float]]] = {}
    for label, filename, color in RUNS:
        payload = json.loads((input_dir / filename).read_text())
        suite_metrics: dict[str, dict[str, float]] = {}
        overall = payload["overall"]["agents"]["Token-DAgger-BC"]
        suite_metrics["overall"] = {
            "pass_rate": float(overall["pass_rate"]),
            "collision_rate": float(overall["collision_rate"]),
        }
        for suite in SUITES[1:]:
            data = payload["by_suite"][suite]["agents"]["Token-DAgger-BC"]
            suite_metrics[suite] = {
                "pass_rate": float(data["pass_rate"]),
                "collision_rate": float(data["collision_rate"]),
            }
        loaded[label] = {"color": color, "metrics": suite_metrics}
    return loaded


def _svg_text(x: float, y: float, text: str, *, size: int = 14, weight: str = "400", fill: str = "#1f2430", anchor: str = "start") -> str:
    safe = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica, Arial, sans-serif" font-size="{size}" font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{safe}</text>'


def _draw_panel(
    title: str,
    metric_key: str,
    max_value: float,
    x0: float,
    y0: float,
    width: float,
    height: float,
    loaded: dict[str, dict[str, dict[str, float]]],
) -> list[str]:
    left = x0 + 56
    top = y0 + 34
    inner_w = width - 76
    inner_h = height - 70
    group_w = inner_w / len(SUITES)
    bar_w = min(22.0, group_w / (len(RUNS) + 1.4))
    gap = bar_w * 0.32
    baseline_y = top + inner_h

    lines = [
        f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{width:.1f}" height="{height:.1f}" rx="8" fill="#ffffff" stroke="#d7dce5"/>',
        _svg_text(x0 + 18, y0 + 24, title, size=15, weight="700"),
        f'<line x1="{left:.1f}" y1="{baseline_y:.1f}" x2="{left + inner_w:.1f}" y2="{baseline_y:.1f}" stroke="#7e8797" stroke-width="1.2"/>',
        f'<line x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{baseline_y:.1f}" stroke="#7e8797" stroke-width="1.2"/>',
    ]

    for tick in range(5):
        value = max_value * tick / 4
        y = baseline_y - inner_h * tick / 4
        lines.append(f'<line x1="{left:.1f}" y1="{y:.1f}" x2="{left + inner_w:.1f}" y2="{y:.1f}" stroke="#edf0f5" stroke-width="1"/>')
        label = f"{value:.0f}%" if metric_key == "pass_rate" else f"{value:.1f}%"
        lines.append(_svg_text(left - 8, y + 4, label, size=11, fill="#697386", anchor="end"))

    suite_labels = {
        "overall": "Overall",
        "gauntlet": "Gauntlet",
        "adversarial": "Adversarial",
        "hidden": "Hidden",
    }
    for suite_idx, suite in enumerate(SUITES):
        gx = left + suite_idx * group_w
        lines.append(_svg_text(gx + group_w * 0.5, baseline_y + 22, suite_labels[suite], size=12, fill="#495468", anchor="middle"))
        total_bar_w = len(RUNS) * bar_w + (len(RUNS) - 1) * gap
        bx = gx + (group_w - total_bar_w) * 0.5
        for run_idx, (label, _, _) in enumerate(RUNS):
            color = loaded[label]["color"]
            value = loaded[label]["metrics"][suite][metric_key]
            h = inner_h * value / max_value
            x = bx + run_idx * (bar_w + gap)
            y = baseline_y - h
            lines.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" rx="3" fill="{color}"/>')
            lines.append(_svg_text(x + bar_w * 0.5, y - 6, f"{value:.2f}", size=10, fill="#354052", anchor="middle"))
    return lines


def build_svg(loaded: dict[str, dict[str, dict[str, float]]]) -> str:
    width = 1240
    height = 760
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="#f6f8fb"/>',
        _svg_text(42, 54, "DAgger Aggregation Ablations on Held-Out Latin-Hypercube Sweep", size=28, weight="700"),
        _svg_text(
            42,
            84,
            "Two-step DAgger remains the frontier. Source decay partially recovers the three-step collapse; removing inverse-frequency weights breaks closed-loop safety.",
            size=15,
            fill="#4e596b",
        ),
    ]

    parts.extend(_draw_panel("Pass Rate", "pass_rate", 100.0, 40, 126, 1160, 290, loaded))
    parts.extend(_draw_panel("Collision Rate", "collision_rate", 4.0, 40, 438, 1160, 244, loaded))

    legend_y = 712
    legend_x = 54
    for label, _, _ in RUNS:
        color = loaded[label]["color"]
        parts.append(f'<rect x="{legend_x:.1f}" y="{legend_y - 12:.1f}" width="18" height="18" rx="3" fill="{color}"/>')
        parts.append(_svg_text(legend_x + 28, legend_y + 2, label, size=13, fill="#364152"))
        legend_x += 265

    parts.append("</svg>")
    return "\n".join(parts)


def main() -> None:
    args = _parse_args()
    loaded = _load_metrics(args.input_dir)
    svg = build_svg(loaded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
