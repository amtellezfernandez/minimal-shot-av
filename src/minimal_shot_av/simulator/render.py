from __future__ import annotations

import math
from pathlib import Path

from .environment import Actor, Scenario, actor_at_tick, interpolate_lane
from .policy import Rollout


def _polygon(points: list[tuple[float, float]], fill: str, opacity: float = 1.0) -> str:
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{joined}" fill="{fill}" opacity="{opacity:.3f}" />'


def _polyline(points: list[tuple[float, float]], color: str, width: float, dash: str = "") -> str:
    joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<polyline points="{joined}" fill="none" stroke="{color}" stroke-width="{width}"{dash_attr} />'


def _actor_color(kind: str) -> str:
    colors = {
        "vehicle": "#6d597a",
        "pedestrian": "#9d4edd",
        "cyclist": "#2a9d8f",
        "animal": "#7f5539",
        "debris": "#5c677d",
        "worker": "#f77f00",
        "special_vehicle": "#d00000",
    }
    return colors.get(kind, "#6c757d")


def _actor_svg(actor: Actor) -> str:
    color = _actor_color(actor.kind)
    if actor.kind in {"pedestrian", "animal", "debris", "worker"}:
        return f'<circle cx="{actor.x:.2f}" cy="{actor.y:.2f}" r="{actor.radius:.2f}" fill="{color}" opacity="0.9" />'
    width = max(actor.width, 0.6)
    length = max(actor.length, 0.6)
    x = actor.x - length * 0.5
    y = actor.y - width * 0.5
    angle = actor.heading * 180.0 / 3.141592653589793
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{length:.2f}" height="{width:.2f}" '
        f'fill="{color}" opacity="0.85" transform="rotate({angle:.2f} {actor.x:.2f} {actor.y:.2f})" />'
    )


def _map_feature_svg(feature: dict[str, float | int | str | bool]) -> str:
    kind = str(feature.get("kind", ""))
    x = float(feature.get("x", 0.0))
    y = float(feature.get("y", 0.0))
    if kind == "crosswalk":
        width = float(feature.get("width", 12.0))
        return f'<rect x="{x - 1.0:.2f}" y="{y - width / 2:.2f}" width="2.0" height="{width:.2f}" fill="#ffffff" opacity="0.7" />'
    if kind == "conflict_zone":
        radius = float(feature.get("radius", 8.0))
        return f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="#ffb703" opacity="0.16" />'
    if kind == "lane_closure":
        length = float(feature.get("length", 20.0))
        return f'<rect x="{x:.2f}" y="{y - 2.0:.2f}" width="{length:.2f}" height="4.0" fill="#e85d04" opacity="0.22" />'
    if kind == "merge_zone":
        length = float(feature.get("length", 20.0))
        return f'<rect x="{x:.2f}" y="{y - 5.0:.2f}" width="{length:.2f}" height="10.0" fill="#00b4d8" opacity="0.12" />'
    if kind == "avoidance_corridor":
        width = float(feature.get("width", 10.0))
        return f'<rect x="{x - 5.0:.2f}" y="{y - width / 2:.2f}" width="22.0" height="{width:.2f}" fill="#90be6d" opacity="0.14" />'
    return ""


def _lane_count(scenario: Scenario) -> int:
    for feature in scenario.map_features:
        if str(feature.get("kind", "")) == "route_corridor":
            try:
                return max(1, int(feature.get("lane_count", 1)))
            except (TypeError, ValueError):
                return 1
    return 1


def _offset_ribbon(points: list[tuple[float, float]], offset_m: float) -> list[tuple[float, float]]:
    if not points:
        return []
    if len(points) == 1:
        return [points[0]]
    normals: list[tuple[float, float]] = []
    for index, point in enumerate(points):
        prev_point = points[index - 1] if index > 0 else points[index]
        next_point = points[index + 1] if index < len(points) - 1 else points[index]
        dx = next_point[0] - prev_point[0]
        dy = next_point[1] - prev_point[1]
        norm = math.hypot(dx, dy)
        if norm <= 1e-9:
            normals.append((0.0, 1.0))
            continue
        normals.append((-dy / norm, dx / norm))
    return [(point[0] + nx * offset_m, point[1] + ny * offset_m) for point, (nx, ny) in zip(points, normals)]


def _road_surface_svg(scenario: Scenario, lane: list[tuple[float, float]]) -> list[str]:
    lane_count = _lane_count(scenario)
    half_width = float(scenario.lane_half_width)
    road_half_width = half_width * max(1.0, lane_count / 2.0)
    left_edge = _offset_ribbon(lane, road_half_width)
    right_edge = _offset_ribbon(lane, -road_half_width)
    svg = [
        _polygon(left_edge + list(reversed(right_edge)), "#d2c3a5", opacity=1.0),
        _polyline(left_edge, "#4b463f", 0.55),
        _polyline(right_edge, "#4b463f", 0.55),
    ]
    if lane_count > 1:
        lane_width = (road_half_width * 2.0) / lane_count
        for divider in range(1, lane_count):
            offset = -road_half_width + lane_width * divider
            dash = "2.4 2.0"
            color = "#f1ede2" if divider != lane_count // 2 or lane_count % 2 == 0 else "#d8d2c3"
            svg.append(_polyline(_offset_ribbon(lane, offset), color, 0.35, dash=dash))
    else:
        svg.append(_polyline(lane, "#f1ede2", 0.3, dash="2.4 2.0"))
    return svg


def render_svg(path: Path, scenario: Scenario, rollout: Rollout) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    lane = interpolate_lane(scenario.lane_center)
    trajectory = [(step.x, step.y) for step in rollout.steps]

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {scenario.width} {scenario.height}">',
        '<rect width="100%" height="100%" fill="#f5f1e8" />',
        f'<text x="3" y="5" font-size="3.2" fill="#2b2d42">{scenario.cluster} seed={scenario.seed} success={rollout.success}</text>',
    ]
    svg.extend(_road_surface_svg(scenario, lane))

    for feature in scenario.map_features:
        feature_svg = _map_feature_svg(feature)
        if feature_svg:
            svg.append(feature_svg)

    for obstacle in scenario.obstacles:
        svg.append(
            f'<circle cx="{obstacle.x:.2f}" cy="{obstacle.y:.2f}" r="{obstacle.radius:.2f}" fill="#a33d2b" opacity="0.62" />'
        )

    for actor in scenario.actors:
        actor_path = [(actor_at_tick(actor, tick).x, actor_at_tick(actor, tick).y) for tick in (0, 20, 40)]
        svg.append(_polyline(actor_path, _actor_color(actor.kind), 0.45, dash="1 1"))
        svg.append(_actor_svg(actor))

    if trajectory:
        svg.append(_polyline(trajectory, "#005f73", 1.4))

    sx, sy = scenario.start
    gx, gy = scenario.goal
    svg.append(f'<circle cx="{sx:.2f}" cy="{sy:.2f}" r="1.6" fill="#0a9396" />')
    svg.append(f'<circle cx="{gx:.2f}" cy="{gy:.2f}" r="1.8" fill="#ee9b00" />')
    svg.append("</svg>")

    path.write_text("\n".join(svg))
