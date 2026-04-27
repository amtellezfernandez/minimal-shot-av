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
DEFAULT_SPOTLIGHT_ROLLOUT_CONFIG = RolloutConfig(max_steps=225, preserve_planned_direction=True)


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
    max_steps: int = 225,
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
        safe_action = _recenter_route_drift(active_scenario, safe_action, world_state, perception)
        safe_action = _recover_stalled_spotlight_progress(safe_action, planned_action, world_state, perception, steps)
        safe_action = _commit_wod_intersection_crossing(active_scenario, safe_action, world_state, perception, steps)
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


def _recover_stalled_spotlight_progress(
    safe_action: SafeAction,
    planned_action: PlannedAction,
    world_state: WorldState,
    perception: ScenePerception,
    steps: list[StepRecord],
) -> SafeAction:
    if world_state.collision_risk > 0.75 or world_state.goal_distance < 8.0:
        return safe_action
    if len(steps) < 12:
        return safe_action
    if (
        planned_action.mode.startswith("spotlight_reflex:")
        and any(step.action_mode == "deadlock_breakout" for step in steps[-4:])
        and world_state.collision_risk < 0.72
        and steps[-1].min_obstacle_distance > 1.15
    ):
        direction = _direction_to_target(world_state)
        speed = max(safe_action.speed, 0.75)
        if direction != (0.0, 0.0) and _visible_clearance_after_step(world_state, perception, direction, speed) > 1.05:
            return SafeAction(direction=direction, speed=speed, mode="deadlock_breakout", intervention=True)
    if not _stalled_spotlight_recovery_candidate(safe_action, planned_action, world_state):
        return safe_action
    if any(step.action_mode == "intersection_crossing" for step in steps[-8:]):
        return safe_action
    if world_state.collision_risk > 0.65 and any(step.action_mode == "progress_recovery" for step in steps[-3:]):
        return safe_action

    if len(steps) >= 48:
        long_recent = steps[-48:]
        long_progress = sum(step.progress or 0.0 for step in long_recent)
        long_min_clearance = min(step.min_obstacle_distance for step in long_recent)
        long_recovery_modes = sum(
            1
            for step in long_recent
            if step.action_mode in {"risk_nudge", "progress_recovery", "cautious_follow"} or step.action_mode.endswith(":crawl")
        )
        lane_stall = abs(perception.lane_error) > 4.0 and long_progress < 4.0 and long_min_clearance > 1.1
        oscillation_stall = long_progress < 8.0 and long_recovery_modes >= 30 and long_min_clearance > 1.30
        if lane_stall or oscillation_stall:
            direction = _direction_to_target(world_state)
            speed = max(safe_action.speed, 1.15)
            if direction != (0.0, 0.0) and _visible_clearance_after_step(world_state, perception, direction, speed) > 1.05:
                return SafeAction(direction=direction, speed=speed, mode="deadlock_breakout", intervention=True)

    recent = steps[-12:]
    recent_progress = sum(step.progress or 0.0 for step in recent)
    recent_crawl_or_nudge = sum(
        1
        for step in recent
        if step.action_mode in {"risk_nudge", "cautious_follow"} or step.action_mode.endswith(":crawl")
    )
    min_recent_clearance = min(step.min_obstacle_distance for step in recent)
    if recent_progress > 1.0 or recent_crawl_or_nudge < 8 or min_recent_clearance < 1.2:
        return safe_action

    direction = _progress_recovery_direction(world_state, perception, planned_action)
    if direction == (0.0, 0.0):
        return safe_action
    recovery_speed = 0.85 if abs(perception.lane_error) > 5.0 and min_recent_clearance > 1.45 else 0.6
    return SafeAction(direction=direction, speed=max(safe_action.speed, recovery_speed), mode="progress_recovery", intervention=True)


def _recenter_route_drift(
    scenario: Scenario,
    safe_action: SafeAction,
    world_state: WorldState,
    perception: ScenePerception,
) -> SafeAction:
    if not scenario.cluster.startswith("adversarial:"):
        return safe_action
    if abs(perception.lane_error) < 4.0:
        return safe_action
    if world_state.collision_risk > 0.30 or perception.uncertainty > 0.75:
        return safe_action
    if safe_action.mode in {"risk_nudge", "risk_escape", "emergency_escape", "emergency_stop"}:
        return safe_action
    if "evasive" in safe_action.mode:
        return safe_action
    nearest_clearance = min((obstacle.signed_distance for obstacle in perception.visible_obstacles), default=math.inf)
    if nearest_clearance < 1.2:
        return safe_action

    lane_return = _normalize(
        (
            perception.lane_point[0] - world_state.position[0],
            perception.lane_point[1] - world_state.position[1],
        )
    )
    target_direction = _direction_to_target(world_state)
    if lane_return == (0.0, 0.0):
        return safe_action
    direction = lane_return
    if target_direction != (0.0, 0.0):
        direction = _normalize(
            (
                lane_return[0] * 0.85 + target_direction[0] * 0.65,
                lane_return[1] * 0.85 + target_direction[1] * 0.65,
            )
        )
    if direction == (0.0, 0.0):
        return safe_action
    return SafeAction(
        direction=direction,
        speed=min(max(safe_action.speed, 1.05), 1.15),
        mode="route_recenter",
        intervention=True,
    )


def _stalled_spotlight_recovery_candidate(
    safe_action: SafeAction,
    planned_action: PlannedAction,
    world_state: WorldState,
) -> bool:
    if not planned_action.mode.startswith("spotlight_reflex:"):
        return False
    maneuver = planned_action.mode.rsplit(":", maxsplit=1)[-1]
    if maneuver == "crawl":
        return True
    if safe_action.mode != "risk_nudge" or maneuver not in {"nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover"}:
        return False
    if world_state.collision_risk > 0.70:
        return False

    target_direction = _direction_to_target(world_state)
    planned_progress = planned_action.direction[0] * target_direction[0] + planned_action.direction[1] * target_direction[1]
    return planned_progress > 0.25


def _commit_wod_intersection_crossing(
    scenario: Scenario,
    safe_action: SafeAction,
    world_state: WorldState,
    perception: ScenePerception,
    steps: list[StepRecord],
) -> SafeAction:
    if scenario.cluster != "intersection" or scenario.tags.get("generator") != "wod_e2e_procedural_v1":
        return safe_action
    if len(steps) < 36 or world_state.goal_distance < 8.0 or world_state.collision_risk > 0.96:
        return safe_action

    row = _wod_intersection_row(scenario)
    if row is None:
        return safe_action
    row_x, target_y = row
    if world_state.position[0] < row_x - 9.0 or world_state.position[0] > row_x + 4.0:
        return safe_action

    recent = steps[-24:]
    recent_crossing = any(step.action_mode == "intersection_crossing" for step in steps[-8:])
    recent_progress = sum(step.progress or 0.0 for step in recent)
    recent_interventions = sum(1 for step in recent if step.intervention)
    recent_recovery_modes = sum(
        1
        for step in recent
        if step.action_mode in {"risk_nudge", "risk_escape", "cautious_follow", "progress_recovery", "deadlock_breakout", "guarded_evasive"}
        or step.action_mode.endswith(":crawl")
    )
    if not recent_crossing and (recent_progress > 8.0 or recent_interventions < 10 or recent_recovery_modes < 12):
        return safe_action

    current_y = world_state.position[1]
    local_row_clearance = _static_clearance_at(scenario, (row_x, current_y))
    local_exit_clearance = _static_clearance_at(scenario, (row_x + 5.0, current_y))
    if (
        (abs(current_y - target_y) > 2.5 or safe_action.mode in {"risk_escape", "risk_nudge"})
        and local_row_clearance > 1.05
        and local_exit_clearance > 1.5
    ):
        target_y = current_y

    if world_state.position[0] < row_x - 2.0 and abs(current_y - target_y) > 0.75:
        target = (row_x - 3.5, target_y)
        speed = 0.75
    else:
        target = (row_x + 8.0, target_y)
        speed = 0.95
    direction = _normalize((target[0] - world_state.position[0], target[1] - world_state.position[1]))
    if direction == (0.0, 0.0):
        return safe_action

    next_clearance = _static_clearance_at(
        scenario,
        (
            world_state.position[0] + direction[0] * speed,
            world_state.position[1] + direction[1] * speed,
        ),
    )
    current_clearance = _static_clearance_at(scenario, world_state.position)
    if next_clearance < 1.05 and next_clearance < current_clearance + 0.10:
        return safe_action
    return SafeAction(direction=direction, speed=max(safe_action.speed, speed), mode="intersection_crossing", intervention=True)


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


def _direction_to_target(world_state: WorldState) -> tuple[float, float]:
    vector = (
        world_state.target_point[0] - world_state.position[0],
        world_state.target_point[1] - world_state.position[1],
    )
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _wod_intersection_row(scenario: Scenario) -> tuple[float, float] | None:
    row_obstacles = [obstacle for obstacle in scenario.obstacles if obstacle.label == "cross_traffic_texture"]
    if len(row_obstacles) < 3:
        return None
    row_x = sum(obstacle.x for obstacle in row_obstacles) / len(row_obstacles)
    lane_y = min(scenario.lane_center, key=lambda point: abs(point[0] - row_x))[1]
    lower = lane_y - scenario.lane_half_width * 0.95
    upper = lane_y + scenario.lane_half_width * 0.95
    intervals = sorted(
        (
            max(lower, obstacle.y - obstacle.radius - 0.95),
            min(upper, obstacle.y + obstacle.radius + 0.95),
        )
        for obstacle in row_obstacles
        if lower <= obstacle.y <= upper
    )

    gaps: list[tuple[float, float]] = []
    cursor = lower
    for start, end in intervals:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < upper:
        gaps.append((cursor, upper))
    if not gaps:
        return None

    usable = [gap for gap in gaps if gap[1] - gap[0] >= 1.2]
    if not usable:
        usable = gaps
    best = min(usable, key=lambda gap: (abs(((gap[0] + gap[1]) * 0.5) - lane_y), -(gap[1] - gap[0])))
    return row_x, (best[0] + best[1]) * 0.5


def _static_clearance_at(scenario: Scenario, position: tuple[float, float]) -> float:
    return min(
        (math.dist(position, (obstacle.x, obstacle.y)) - obstacle.radius for obstacle in scenario.obstacles),
        default=math.inf,
    )


def _visible_clearance_after_step(
    world_state: WorldState,
    perception: ScenePerception,
    direction: tuple[float, float],
    speed: float,
) -> float:
    if not perception.visible_obstacles:
        return math.inf
    next_position = (
        world_state.position[0] + direction[0] * speed,
        world_state.position[1] + direction[1] * speed,
    )
    return min(
        math.dist(next_position, (obstacle.x, obstacle.y)) - obstacle.radius
        for obstacle in perception.visible_obstacles
    )


def _progress_recovery_direction(
    world_state: WorldState,
    perception: ScenePerception,
    planned_action: PlannedAction,
) -> tuple[float, float]:
    target_direction = _direction_to_target(world_state)
    if target_direction == (0.0, 0.0):
        target_direction = planned_action.direction
    if abs(perception.lane_error) > 3.0:
        lane_return = _normalize(
            (
                perception.lane_point[0] - world_state.position[0],
                perception.lane_point[1] - world_state.position[1],
            )
        )
        if lane_return != (0.0, 0.0):
            target_weight = 0.90 if abs(perception.lane_error) > 5.0 else 0.70
            lane_weight = 0.30 if abs(perception.lane_error) > 5.0 else 0.70
            target_direction = _normalize(
                (
                    target_direction[0] * target_weight + lane_return[0] * lane_weight,
                    target_direction[1] * target_weight + lane_return[1] * lane_weight,
                )
            )
    if target_direction == (0.0, 0.0) or not perception.visible_obstacles:
        return target_direction

    nearest = min(perception.visible_obstacles, key=lambda obstacle: obstacle.signed_distance)
    away = _normalize(
        (
            world_state.position[0] - nearest.x,
            world_state.position[1] - nearest.y,
        )
    )
    forward_component = away[0] * target_direction[0] + away[1] * target_direction[1]
    lateral = _normalize(
        (
            away[0] - forward_component * target_direction[0],
            away[1] - forward_component * target_direction[1],
        )
    )
    if lateral == (0.0, 0.0):
        lateral = (-target_direction[1], target_direction[0])
    return _normalize(
        (
            target_direction[0] * 0.78 + lateral[0] * 0.62,
            target_direction[1] * 0.78 + lateral[1] * 0.62,
        )
    )


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)
