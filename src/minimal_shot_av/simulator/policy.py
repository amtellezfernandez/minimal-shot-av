from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

from .environment import Scenario, scenario_at_tick
from .perception import ScenePerception, perceive_scene
from .planner import PlannedAction, plan_action
from .safety import SafeAction, apply_safety_filter
from .spotlight_reflex import SpotlightReflexConfig, plan_spotlight_reflex_action
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
    selector_score: float | None = None
    selector_3s_score: float | None = None
    selector_5s_score: float | None = None
    selector_3s_reference: str | None = None
    selector_5s_reference: str | None = None
    selector_3s_inside_region: bool | None = None
    selector_5s_inside_region: bool | None = None


@dataclass
class Rollout:
    success: bool
    collision: bool
    reached_goal: bool
    steps: list[StepRecord]


PolicyPlanner = Callable[
    [Scenario, tuple[float, float], WorldState, ScenePerception, "RolloutConfig"],
    tuple[PlannedAction, dict[str, Any]],
]


@dataclass(frozen=True)
class RolloutConfig:
    max_steps: int = 220
    step_size: float = 1.25
    goal_tolerance_m: float = 3.0
    stall_goal_distance_m: float = 3.0
    preserve_planned_direction: bool = False
    spotlight: SpotlightReflexConfig = field(default_factory=SpotlightReflexConfig)


DEFAULT_BASELINE_ROLLOUT_CONFIG = RolloutConfig()
DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG = RolloutConfig(preserve_planned_direction=True)


def run_policy(
    scenario: Scenario,
    max_steps: int = 220,
    step_size: float = 1.25,
    config: RolloutConfig | None = None,
) -> Rollout:
    config = _rollout_config(config, max_steps, step_size, DEFAULT_BASELINE_ROLLOUT_CONFIG)
    return _run_rollout(scenario, _baseline_planner, config)


def run_spotlight_reflex_policy(
    scenario: Scenario,
    max_steps: int = 220,
    step_size: float = 1.25,
    config: RolloutConfig | None = None,
) -> Rollout:
    config = _rollout_config(config, max_steps, step_size, DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG)
    return _run_rollout(
        scenario,
        _spotlight_reflex_planner,
        config,
    )


def _rollout_config(
    config: RolloutConfig | None,
    max_steps: int,
    step_size: float,
    default: RolloutConfig,
) -> RolloutConfig:
    if config is not None:
        return config
    if max_steps == default.max_steps and step_size == default.step_size:
        return default
    return RolloutConfig(
        max_steps=max_steps,
        step_size=step_size,
        goal_tolerance_m=default.goal_tolerance_m,
        stall_goal_distance_m=default.stall_goal_distance_m,
        preserve_planned_direction=default.preserve_planned_direction,
        spotlight=default.spotlight,
    )


def _baseline_planner(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    config: RolloutConfig,
) -> tuple[PlannedAction, dict[str, Any]]:
    del scenario
    return plan_action(position, world_state, perception, nominal_step_size=config.step_size), {}


def _spotlight_reflex_planner(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    config: RolloutConfig,
) -> tuple[PlannedAction, dict[str, Any]]:
    planned_action, selection = plan_spotlight_reflex_action(
        scenario,
        position,
        world_state,
        perception,
        nominal_step_size=config.step_size,
        config=config.spotlight,
    )
    return planned_action, selection.to_metadata()


def _run_rollout(
    scenario: Scenario,
    planner: PolicyPlanner,
    config: RolloutConfig,
) -> Rollout:
    position = scenario.start
    steps: list[StepRecord] = []
    collision = False
    reached_goal = False

    for tick in range(config.max_steps):
        active_scenario = scenario_at_tick(scenario, tick)
        previous_position = position
        perception = perceive_scene(active_scenario, position)
        world_state = update_world_state(active_scenario, position, perception)
        planned_action, metadata = planner(active_scenario, position, world_state, perception, config)
        safe_action = apply_safety_filter(planned_action, world_state, perception)
        if (
            config.preserve_planned_direction
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
                config,
            )
        )

        if collision:
            break

        if math.dist(position, scenario.goal) < config.goal_tolerance_m:
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
    config: RolloutConfig,
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
        stall=safe_action.speed == 0.0 and goal_distance >= config.stall_goal_distance_m,
        **_spotlight_step_fields(metadata),
    )


def _spotlight_step_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    if not metadata:
        return {}
    return {
        "candidate_count": int(metadata["candidate_count"]),
        "selected_maneuver": str(metadata["selected_maneuver"]),
        "selector_score": float(metadata["selector_score"]),
        "selector_3s_score": float(metadata["selector_3s_score"]),
        "selector_5s_score": float(metadata["selector_5s_score"]),
        "selector_3s_reference": str(metadata["selector_3s_reference"]),
        "selector_5s_reference": str(metadata["selector_5s_reference"]),
        "selector_3s_inside_region": bool(metadata["selector_3s_inside_region"]),
        "selector_5s_inside_region": bool(metadata["selector_5s_inside_region"]),
    }
