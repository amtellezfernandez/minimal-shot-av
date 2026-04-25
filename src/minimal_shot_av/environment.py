from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
import math
import random
from pathlib import Path
from typing import Any


@dataclass
class Obstacle:
    x: float
    y: float
    radius: float
    kind: str = "obstacle"
    label: str = "obstacle"


@dataclass
class Actor:
    actor_id: str
    kind: str
    x: float
    y: float
    width: float
    length: float
    heading: float
    speed: float
    vx: float
    vy: float
    behavior: str
    role: str
    active_from: int = 0
    active_until: int = 10_000

    @property
    def radius(self) -> float:
        return max(self.width, self.length) * 0.5


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
    cluster: str = "baseline"
    tags: dict[str, Any] = field(default_factory=dict)
    actors: list[Actor] = field(default_factory=list)
    map_features: list[dict[str, Any]] = field(default_factory=list)
    environment: dict[str, Any] = field(default_factory=dict)


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
        lateral_offset = rng.choice((-1.0, 1.0)) * rng.uniform(lane_half_width * 0.45, lane_half_width * 0.95)
        forward_offset = rng.uniform(-5.0, 5.0)
        obstacles.append(
            Obstacle(
                x=anchor_x + forward_offset,
                y=anchor_y + lateral_offset,
                radius=rng.uniform(1.2, 2.8),
                kind="background_obstacle",
                label="baseline_texture",
            )
        )

    start = (lane_center[0][0] - 4.0, lane_center[0][1])
    goal = (lane_center[-1][0] + 4.0, lane_center[-1][1])
    return Scenario(width, height, lane_center, lane_half_width, obstacles, start, goal, seed)


def actor_at_tick(actor: Actor, tick: int, dt: float = 0.25) -> Actor:
    if tick < actor.active_from:
        return actor
    elapsed = max(0, tick - actor.active_from) * dt
    if actor.behavior in {"cut_in", "swerve"}:
        longitudinal = actor.speed * elapsed
        lateral = min(4.5, 0.38 * elapsed * elapsed)
        lateral *= -1.0 if actor.vy < 0.0 else 1.0
        return replace(
            actor,
            x=actor.x + math.cos(actor.heading) * longitudinal,
            y=actor.y + math.sin(actor.heading) * longitudinal + lateral,
        )
    if actor.behavior in {"darting", "erratic_pedestrian"}:
        pause = 0.4 if int(elapsed * 2.0) % 3 == 0 else 1.0
        wobble = math.sin(elapsed * 3.7) * 0.55
        return replace(actor, x=actor.x + actor.vx * elapsed * pause, y=actor.y + actor.vy * elapsed * pause + wobble)
    if actor.behavior in {"sudden_brake", "hesitating"}:
        moving_time = min(elapsed, 1.2)
        creep_time = max(0.0, elapsed - 1.2)
        distance = actor.speed * moving_time + actor.speed * 0.15 * creep_time
        return replace(actor, x=actor.x + math.cos(actor.heading) * distance, y=actor.y + math.sin(actor.heading) * distance)
    if actor.behavior == "wrong_way":
        return replace(actor, x=actor.x - abs(actor.vx) * elapsed, y=actor.y + actor.vy * elapsed)
    return replace(actor, x=actor.x + actor.vx * elapsed, y=actor.y + actor.vy * elapsed)


def actor_to_obstacle(actor: Actor, tick: int = 0, dt: float = 0.25) -> Obstacle | None:
    if tick < actor.active_from or tick > actor.active_until:
        return None
    projected = actor_at_tick(actor, tick, dt)
    return Obstacle(projected.x, projected.y, projected.radius, kind=projected.kind, label=projected.role)


def obstacles_at_tick(scenario: Scenario, tick: int, dt: float = 0.25) -> list[Obstacle]:
    obstacles = [obstacle for obstacle in scenario.obstacles if obstacle.kind != "ambient"]
    for actor in scenario.actors:
        obstacle = actor_to_obstacle(actor, tick, dt)
        if obstacle is not None:
            obstacles.append(obstacle)
    return obstacles


def scenario_at_tick(scenario: Scenario, tick: int, dt: float = 0.25) -> Scenario:
    return replace(scenario, obstacles=obstacles_at_tick(scenario, tick, dt))


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
        "cluster": scenario.cluster,
        "tags": scenario.tags,
        "actors": [asdict(actor) for actor in scenario.actors],
        "map_features": scenario.map_features,
        "environment": scenario.environment,
    }
