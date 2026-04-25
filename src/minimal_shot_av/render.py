from __future__ import annotations

from pathlib import Path

from .environment import Scenario, interpolate_lane
from .policy import Rollout


def _polyline(points: list[tuple[float, float]], color: str, width: float, dash: str = "") -> str:
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr} />'


def render_svg(path: Path, scenario: Scenario, rollout: Rollout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lane = interpolate_lane(scenario.lane_center)
    trajectory = [(step.x, step.y) for step in rollout.steps]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {scenario.width} {scenario.height}">',
        '<rect width="100%" height="100%" fill="#f5f1e8" />',
        _polyline(lane, "#c8b99a", scenario.lane_half_width * 2.0),
        _polyline(lane, "#3e3a36", 0.8, dash="2 2"),
    ]

    for obstacle in scenario.obstacles:
        svg.append(
            f'<circle cx="{obstacle.x:.2f}" cy="{obstacle.y:.2f}" r="{obstacle.radius:.2f}" fill="#a33d2b" opacity="0.82" />'
        )

    if trajectory:
        svg.append(_polyline(trajectory, "#005f73", 1.4))

    sx, sy = scenario.start
    gx, gy = scenario.goal
    svg.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="1.6" fill="#0a9396" />')
    svg.append(f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="1.8" fill="#ee9b00" />')
    svg.append("</svg>")

    path.write_text("\n".join(svg))

