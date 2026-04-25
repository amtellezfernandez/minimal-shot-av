from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import random
from pathlib import Path


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float


@dataclass
class Scenario:
    width: float
    height: float
    lane_center: list[tuple[float, float]]
    lane_half_width: float
    obstacles: list[Obstacle]
    start: tuple[float, float]
    goal: tuple[float, float]
    seed: int


def generate_scenario(seed: int, width: float = 120.0, height: float = 80.0) -> Scenario:
    rng = random.Random(seed)
    lane_half_width = rng.uniform(7.0, 10.0)
    control_points = 8
    lane_center: list[tuple[float, float]] = []

    for index in range(control_points):
        x = 10 + index * (width - 20) / (control_points - 1)
        y = height * 0.5 + rng.uniform(-height * 0.22, height * 0.22)
        lane_center.append((x, y))

    obstacles: list[Obstacle] = []
    for _ in range(rng.randint(7, 12)):
        segment_index = rng.randint(1, control_points - 2)
        anchor_x, anchor_y = lane_center[segment_index]
        lateral_offset = rng.uniform(-lane_half_width * 0.95, lane_half_width * 0.95)
        forward_offset = rng.uniform(-5.0, 5.0)
        obstacles.append(
            Obstacle(
                x=anchor_x + forward_offset,
                y=anchor_y + lateral_offset,
                radius=rng.uniform(1.2, 2.8),
            )
        )

    start = (lane_center[0][0] - 4.0, lane_center[0][1])
    goal = (lane_center[-1][0] + 4.0, lane_center[-1][1])
    return Scenario(width, height, lane_center, lane_half_width, obstacles, start, goal, seed)


def interpolate_lane(centerline: list[tuple[float, float]], samples_per_segment: int = 16) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for first, second in zip(centerline, centerline[1:]):
        for sample in range(samples_per_segment):
            t = sample / samples_per_segment
            x = first[0] + (second[0] - first[0]) * t
            y = first[1] + (second[1] - first[1]) * t
            points.append((x, y))
    points.append(centerline[-1])
    return points


def nearest_lane_point(point: tuple[float, float], lane_points: list[tuple[float, float]]) -> tuple[int, tuple[float, float], float]:
    best_index = 0
    best_point = lane_points[0]
    best_distance = math.inf

    for index, lane_point in enumerate(lane_points):
        distance = math.dist(point, lane_point)
        if distance < best_distance:
            best_index = index
            best_point = lane_point
            best_distance = distance

    return best_index, best_point, best_distance


def write_rollout(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def scenario_to_dict(scenario: Scenario) -> dict:
    return {
        "width": scenario.width,
        "height": scenario.height,
        "lane_center": scenario.lane_center,
        "lane_half_width": scenario.lane_half_width,
        "obstacles": [asdict(obstacle) for obstacle in scenario.obstacles],
        "start": scenario.start,
        "goal": scenario.goal,
        "seed": scenario.seed,
    }

