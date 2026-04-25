from __future__ import annotations

from dataclasses import dataclass
import math

from .environment import Obstacle, Scenario, interpolate_lane, nearest_lane_point


@dataclass
class PerceivedObstacle:
    x: float
    y: float
    radius: float
    signed_distance: float


@dataclass
class ScenePerception:
    lane_index: int
    lane_point: tuple[float, float]
    lane_error: float
    lane_heading: tuple[float, float]
    corridor_margin: float
    free_space_confidence: float
    uncertainty: float
    visible_obstacles: list[PerceivedObstacle]


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _signed_distance(position: tuple[float, float], obstacle: Obstacle) -> float:
    return math.dist(position, (obstacle.x, obstacle.y)) - obstacle.radius


def perceive_scene(scenario: Scenario, position: tuple[float, float], visibility_radius: float = 18.0) -> ScenePerception:
    lane_points = interpolate_lane(scenario.lane_center)
    lane_index, lane_point, lane_error = nearest_lane_point(position, lane_points)
    next_index = min(len(lane_points) - 1, lane_index + 4)
    lane_heading = _normalize(
        (
            lane_points[next_index][0] - lane_point[0],
            lane_points[next_index][1] - lane_point[1],
        )
    )

    visible_obstacles: list[PerceivedObstacle] = []
    nearest_signed_distance = math.inf
    for obstacle in scenario.obstacles:
        signed_distance = _signed_distance(position, obstacle)
        if signed_distance <= visibility_radius:
            visible_obstacles.append(
                PerceivedObstacle(
                    x=obstacle.x,
                    y=obstacle.y,
                    radius=obstacle.radius,
                    signed_distance=signed_distance,
                )
            )
            nearest_signed_distance = min(nearest_signed_distance, signed_distance)

    if math.isinf(nearest_signed_distance):
        nearest_signed_distance = visibility_radius

    corridor_margin = max(0.0, scenario.lane_half_width - lane_error)
    obstacle_pressure = max(0.0, min(1.0, (10.0 - nearest_signed_distance) / 10.0))
    uncertainty = min(1.0, (1.0 - min(1.0, corridor_margin / max(scenario.lane_half_width, 1e-6))) * 0.4 + obstacle_pressure * 0.6)
    free_space_confidence = max(0.0, 1.0 - uncertainty)

    return ScenePerception(
        lane_index=lane_index,
        lane_point=lane_point,
        lane_heading=lane_heading,
        lane_error=lane_error,
        corridor_margin=corridor_margin,
        free_space_confidence=free_space_confidence,
        uncertainty=uncertainty,
        visible_obstacles=visible_obstacles,
    )
