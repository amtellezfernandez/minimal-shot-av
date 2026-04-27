from __future__ import annotations

from dataclasses import dataclass
import math

from .environment import Scenario, interpolate_lane
from .perception import ScenePerception


@dataclass
class WorldState:
    position: tuple[float, float]
    target_point: tuple[float, float]
    progress_fraction: float
    collision_risk: float
    goal_distance: float
    uncertainty: float


def update_world_state(
    scenario: Scenario,
    position: tuple[float, float],
    perception: ScenePerception,
    lookahead: int = 12,
) -> WorldState:
    lane_points = interpolate_lane(scenario.lane_center)
    target_index = min(len(lane_points) - 1, perception.lane_index + lookahead)
    target_point = lane_points[target_index]
    progress_fraction = target_index / max(1, len(lane_points) - 1)
    goal_distance = math.dist(position, scenario.goal)

    nearest_signed_distance = min(
        (obstacle.signed_distance for obstacle in perception.visible_obstacles),
        default=20.0,
    )
    collision_risk = max(0.0, min(1.0, (5.0 - nearest_signed_distance) / 5.0))

    return WorldState(
        position=position,
        target_point=target_point,
        progress_fraction=progress_fraction,
        collision_risk=collision_risk,
        goal_distance=goal_distance,
        uncertainty=perception.uncertainty,
    )

