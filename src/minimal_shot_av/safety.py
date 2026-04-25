from __future__ import annotations

from dataclasses import dataclass
import math

from .perception import ScenePerception
from .planner import PlannedAction
from .world_model import WorldState


@dataclass
class SafeAction:
    direction: tuple[float, float]
    speed: float
    mode: str
    intervention: bool


def apply_safety_filter(action: PlannedAction, world_state: WorldState, perception: ScenePerception) -> SafeAction:
    if world_state.collision_risk > 0.985:
        return SafeAction(direction=(0.0, 0.0), speed=0.0, mode="emergency_stop", intervention=True)

    if perception.uncertainty > 0.78:
        return SafeAction(
            direction=action.direction,
            speed=min(action.speed, 0.5),
            mode="cautious_follow",
            intervention=True,
        )

    if perception.corridor_margin < 0.45:
        return SafeAction(
            direction=_recovery_direction(world_state, perception),
            speed=min(action.speed, 0.75),
            mode="lane_recovery",
            intervention=True,
        )

    return SafeAction(direction=action.direction, speed=action.speed, mode=action.mode, intervention=False)


def _recovery_direction(world_state: WorldState, perception: ScenePerception) -> tuple[float, float]:
    target_vector = (
        world_state.target_point[0] - world_state.position[0],
        world_state.target_point[1] - world_state.position[1],
    )
    target_direction = _normalize(target_vector)
    if target_direction != (0.0, 0.0):
        return target_direction
    return perception.lane_heading


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)
