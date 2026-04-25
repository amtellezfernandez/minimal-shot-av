from __future__ import annotations

from dataclasses import dataclass
import math

from .environment import Scenario
from .perception import perceive_scene
from .planner import plan_action
from .safety import apply_safety_filter
from .world_model import update_world_state


@dataclass
class StepRecord:
    t: int
    x: float
    y: float
    lane_error: float
    min_obstacle_distance: float
    uncertainty: float
    collision_risk: float
    action_mode: str
    speed: float
    intervention: bool


@dataclass
class Rollout:
    success: bool
    collision: bool
    reached_goal: bool
    steps: list[StepRecord]


def run_policy(scenario: Scenario, max_steps: int = 220, step_size: float = 1.25) -> Rollout:
    position = scenario.start
    steps: list[StepRecord] = []
    collision = False
    reached_goal = False

    for tick in range(max_steps):
        perception = perceive_scene(scenario, position)
        world_state = update_world_state(scenario, position, perception)
        planned_action = plan_action(position, world_state, perception, nominal_step_size=step_size)
        safe_action = apply_safety_filter(planned_action, world_state, perception)

        position = (
            position[0] + safe_action.direction[0] * safe_action.speed,
            position[1] + safe_action.direction[1] * safe_action.speed,
        )

        for obstacle in scenario.obstacles:
            if math.dist(position, (obstacle.x, obstacle.y)) <= obstacle.radius:
                collision = True
                break

        min_obstacle_distance = min(
            (math.dist(position, (obstacle.x, obstacle.y)) - obstacle.radius for obstacle in scenario.obstacles),
            default=math.inf,
        )
        steps.append(
            StepRecord(
                t=tick,
                x=position[0],
                y=position[1],
                lane_error=perception.lane_error,
                min_obstacle_distance=min_obstacle_distance,
                uncertainty=world_state.uncertainty,
                collision_risk=world_state.collision_risk,
                action_mode=safe_action.mode,
                speed=safe_action.speed,
                intervention=safe_action.intervention,
            )
        )

        if collision:
            break

        if math.dist(position, scenario.goal) < 3.0:
            reached_goal = True
            break

    success = reached_goal and not collision
    return Rollout(success=success, collision=collision, reached_goal=reached_goal, steps=steps)
