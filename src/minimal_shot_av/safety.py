from __future__ import annotations

from dataclasses import dataclass

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
            direction=perception.lane_heading,
            speed=min(action.speed, 0.75),
            mode="lane_recovery",
            intervention=True,
        )

    return SafeAction(direction=action.direction, speed=action.speed, mode=action.mode, intervention=False)
