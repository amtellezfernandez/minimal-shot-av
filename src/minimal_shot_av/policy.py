from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

from .environment import Scenario, scenario_at_tick
from .perception import ScenePerception, perceive_scene
from .planner import PlannedAction, plan_action
from .safety import SafeAction, apply_safety_filter
from .spotlight_reflex import plan_spotlight_reflex_action
from .world_model import WorldState, update_world_state


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
    goal_distance: float | None = None
    progress: float | None = None
    comfort_cost: float | None = None
    active_actor_count: int | None = None
    stall: bool | None = None
    candidate_count: int | None = None
    selected_maneuver: str | None = None
    rfs_score: float | None = None
    rfs_3s_score: float | None = None
    rfs_5s_score: float | None = None
    rfs_3s_reference: str | None = None
    rfs_5s_reference: str | None = None
    rfs_3s_inside_trust_region: bool | None = None
    rfs_5s_inside_trust_region: bool | None = None


@dataclass
class Rollout:
    success: bool
    collision: bool
    reached_goal: bool
    steps: list[StepRecord]


PolicyPlanner = Callable[
    [Scenario, tuple[float, float], WorldState, ScenePerception, float],
    tuple[PlannedAction, dict[str, Any]],
]


def run_policy(scenario: Scenario, max_steps: int = 220, step_size: float = 1.25) -> Rollout:
    return _run_rollout(scenario, _baseline_planner, max_steps, step_size)


def run_spotlight_reflex_policy(scenario: Scenario, max_steps: int = 220, step_size: float = 1.25) -> Rollout:
    return _run_rollout(
        scenario,
        _spotlight_reflex_planner,
        max_steps,
        step_size,
        preserve_planned_direction=True,
    )


def _baseline_planner(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    step_size: float,
) -> tuple[PlannedAction, dict[str, Any]]:
    del scenario
    return plan_action(position, world_state, perception, nominal_step_size=step_size), {}


def _spotlight_reflex_planner(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    step_size: float,
) -> tuple[PlannedAction, dict[str, Any]]:
    planned_action, selection = plan_spotlight_reflex_action(
        scenario,
        position,
        world_state,
        perception,
        nominal_step_size=step_size,
    )
    return planned_action, selection.to_metadata()


def _run_rollout(
    scenario: Scenario,
    planner: PolicyPlanner,
    max_steps: int,
    step_size: float,
    preserve_planned_direction: bool = False,
) -> Rollout:
    position = scenario.start
    steps: list[StepRecord] = []
    collision = False
    reached_goal = False

    for tick in range(max_steps):
        active_scenario = scenario_at_tick(scenario, tick)
        previous_position = position
        perception = perceive_scene(active_scenario, position)
        world_state = update_world_state(active_scenario, position, perception)
        planned_action, metadata = planner(active_scenario, position, world_state, perception, step_size)
        safe_action = apply_safety_filter(planned_action, world_state, perception)
        if (
            preserve_planned_direction
            and safe_action.direction == (0.0, 0.0)
            and safe_action.speed > 0.0
            and planned_action.direction != (0.0, 0.0)
        ):
            safe_action.direction = planned_action.direction

        position = (
            position[0] + safe_action.direction[0] * safe_action.speed,
            position[1] + safe_action.direction[1] * safe_action.speed,
        )

        for obstacle in active_scenario.obstacles:
            if math.dist(position, (obstacle.x, obstacle.y)) <= obstacle.radius:
                collision = True
                break

        min_obstacle_distance = min(
            (math.dist(position, (obstacle.x, obstacle.y)) - obstacle.radius for obstacle in active_scenario.obstacles),
            default=math.inf,
        )
        previous_goal_distance = math.dist(previous_position, scenario.goal)
        goal_distance = math.dist(position, scenario.goal)
        steps.append(
            _step_record(
                tick,
                position,
                perception,
                world_state,
                safe_action,
                goal_distance,
                previous_goal_distance,
                min_obstacle_distance,
                len(active_scenario.actors),
                steps[-1].speed if steps else None,
                metadata,
            )
        )

        if collision:
            break

        if math.dist(position, scenario.goal) < 3.0:
            reached_goal = True
            break

    success = reached_goal and not collision
    return Rollout(success=success, collision=collision, reached_goal=reached_goal, steps=steps)


def _step_record(
    tick: int,
    position: tuple[float, float],
    perception: ScenePerception,
    world_state: WorldState,
    safe_action: SafeAction,
    goal_distance: float,
    previous_goal_distance: float,
    min_obstacle_distance: float,
    active_actor_count: int,
    previous_speed: float | None,
    metadata: dict[str, Any],
) -> StepRecord:
    return StepRecord(
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
        goal_distance=goal_distance,
        progress=previous_goal_distance - goal_distance,
        comfort_cost=0.0 if previous_speed is None else abs(safe_action.speed - previous_speed),
        active_actor_count=active_actor_count,
        stall=safe_action.speed == 0.0 and goal_distance >= 3.0,
        **_spotlight_step_fields(metadata),
    )


def _spotlight_step_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        "candidate_count": int(metadata["candidate_count"]),
        "selected_maneuver": str(metadata["selected_maneuver"]),
        "rfs_score": float(metadata["rfs_score"]),
        "rfs_3s_score": float(metadata["rfs_3s_score"]),
        "rfs_5s_score": float(metadata["rfs_5s_score"]),
        "rfs_3s_reference": str(metadata["rfs_3s_reference"]),
        "rfs_5s_reference": str(metadata["rfs_5s_reference"]),
        "rfs_3s_inside_trust_region": bool(metadata["rfs_3s_inside_trust_region"]),
        "rfs_5s_inside_trust_region": bool(metadata["rfs_5s_inside_trust_region"]),
    }
