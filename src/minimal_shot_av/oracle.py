from __future__ import annotations

from dataclasses import dataclass
import math

from .environment import Scenario, interpolate_lane, nearest_lane_point, scenario_at_tick
from .perception import perceive_scene
from .policy import Rollout, StepRecord
from .world_model import update_world_state


ORACLE_SPEED_SCALES = (1.0, 0.65, 0.35, 0.0)
ORACLE_ANGLES = (-1.1, -0.55, 0.0, 0.55, 1.1)
ORACLE_HORIZON_STEPS = 5
ORACLE_LANE_SAMPLES_PER_SEGMENT = 12
ORACLE_LOOKAHEAD_SAMPLES = 24


@dataclass(frozen=True)
class OracleCertificate:
    solvable: bool
    reasoning_trace: str
    rollout: Rollout
    min_clearance: float
    intervention_free: bool


def run_oracle_policy(scenario: Scenario, max_steps: int = 220, step_size: float = 1.35) -> OracleCertificate:
    """Run a privileged receding-horizon oracle over full simulator state.

    The oracle is not a submitted driving policy. It sees future actor positions
    over a short horizon and exists to check that a generated scenario has a
    feasible route through the current abstract simulator.
    """

    position = scenario.start
    dense_lane = interpolate_lane(scenario.lane_center, samples_per_segment=ORACLE_LANE_SAMPLES_PER_SEGMENT)
    steps: list[StepRecord] = []
    collision = False
    reached_goal = False

    for tick in range(max_steps):
        active_scenario = scenario_at_tick(scenario, tick)
        previous_position = position
        direction, speed, mode = _choose_privileged_action(scenario, position, tick, step_size, dense_lane)
        position = (position[0] + direction[0] * speed, position[1] + direction[1] * speed)

        for obstacle in active_scenario.obstacles:
            if math.dist(position, (obstacle.x, obstacle.y)) <= obstacle.radius:
                collision = True
                break

        perception = perceive_scene(active_scenario, position)
        world_state = update_world_state(active_scenario, position, perception)
        min_clearance = min(
            (math.dist(position, (obstacle.x, obstacle.y)) - obstacle.radius for obstacle in active_scenario.obstacles),
            default=math.inf,
        )
        previous_goal_distance = math.dist(previous_position, scenario.goal)
        goal_distance = math.dist(position, scenario.goal)
        steps.append(
            StepRecord(
                t=tick,
                x=position[0],
                y=position[1],
                lane_error=perception.lane_error,
                min_obstacle_distance=min_clearance,
                uncertainty=world_state.uncertainty,
                collision_risk=world_state.collision_risk,
                action_mode=f"oracle:{mode}",
                speed=speed,
                intervention=False,
                goal_distance=goal_distance,
                progress=previous_goal_distance - goal_distance,
                comfort_cost=0.0 if not steps else abs(speed - steps[-1].speed),
                active_actor_count=len(active_scenario.actors),
                stall=speed == 0.0 and goal_distance >= 3.0,
            )
        )

        if collision:
            break
        if goal_distance < 3.0:
            reached_goal = True
            break

    rollout = Rollout(success=reached_goal and not collision, collision=collision, reached_goal=reached_goal, steps=steps)
    min_clearance = min((step.min_obstacle_distance for step in steps), default=math.inf)
    trace = oracle_reasoning_trace(scenario, rollout)
    return OracleCertificate(
        solvable=rollout.success and min_clearance > 0.25,
        reasoning_trace=trace,
        rollout=rollout,
        min_clearance=min_clearance,
        intervention_free=True,
    )


def oracle_reasoning_trace(scenario: Scenario, rollout: Rollout | None = None) -> str:
    hazard = scenario.tags.get("primary_hazard_type", "unknown_hazard")
    composition = scenario.tags.get("hazard_composition", hazard)
    topology = scenario.tags.get("topology", scenario.cluster)
    condition = scenario.tags.get("condition", scenario.environment.get("weather", "unknown_condition"))
    decision = scenario.tags.get("intended_decision", "maintain_safe_progress")
    status = "feasibility_check_passed" if rollout is None or rollout.success else "feasibility_check_failed"
    return (
        f"{status}: topology={topology}; condition={condition}; hazards={composition}; "
        f"primary={hazard}; required_decision={decision}; response=preserve clearance, "
        "yield to dynamic conflicts, avoid static blockers, then recover to route."
    )


def _choose_privileged_action(
    scenario: Scenario,
    position: tuple[float, float],
    tick: int,
    step_size: float,
    dense_lane: list[tuple[float, float]],
) -> tuple[tuple[float, float], float, str]:
    target = _lookahead_target(scenario, position, dense_lane)
    base = _normalize((target[0] - position[0], target[1] - position[1]))
    if base == (0.0, 0.0):
        base = _normalize((scenario.goal[0] - position[0], scenario.goal[1] - position[1]))
    future_scenarios = tuple(
        scenario_at_tick(scenario, tick + offset) for offset in range(1, ORACLE_HORIZON_STEPS + 1)
    )
    candidates: list[tuple[float, tuple[float, float], float, str]] = []
    for speed_scale in ORACLE_SPEED_SCALES:
        for angle in ORACLE_ANGLES:
            direction = _normalize(_rotate(base, angle))
            speed = step_size * speed_scale
            score = _horizon_score(scenario, position, direction, speed, future_scenarios)
            mode = _candidate_mode(speed_scale, angle)
            candidates.append((score, direction, speed, mode))
    _, direction, speed, mode = max(candidates, key=lambda item: item[0])
    return direction, speed, mode


def _candidate_mode(speed_scale: float, angle: float) -> str:
    if speed_scale <= 0.5:
        return "yield"
    if angle > 0.2:
        return "avoid_left"
    if angle < -0.2:
        return "avoid_right"
    return "progress"


def _horizon_score(
    scenario: Scenario,
    position: tuple[float, float],
    direction: tuple[float, float],
    speed: float,
    future_scenarios: tuple[Scenario, ...],
) -> float:
    simulated = position
    min_clearance = math.inf
    progress = 0.0
    lane_penalty = 0.0
    for active in future_scenarios:
        previous_goal = math.dist(simulated, scenario.goal)
        simulated = (simulated[0] + direction[0] * speed, simulated[1] + direction[1] * speed)
        progress += previous_goal - math.dist(simulated, scenario.goal)
        for obstacle in active.obstacles:
            clearance = math.dist(simulated, (obstacle.x, obstacle.y)) - obstacle.radius
            min_clearance = min(min_clearance, clearance)
            if clearance < 0.0:
                return -1_000_000.0 + clearance * 10_000.0
        perception = perceive_scene(active, simulated)
        lane_penalty += max(0.0, perception.lane_error - active.lane_half_width * 0.75)
    clearance_score = min(5.0, min_clearance) * 22.0
    return progress * 18.0 + clearance_score - lane_penalty * 12.0 - (0.15 if speed == 0.0 else 0.0)


def _lookahead_target(
    scenario: Scenario,
    position: tuple[float, float],
    dense_lane: list[tuple[float, float]],
) -> tuple[float, float]:
    best_index, _, _ = nearest_lane_point(position, dense_lane)
    if best_index >= len(dense_lane) - ORACLE_LOOKAHEAD_SAMPLES:
        return scenario.goal
    return dense_lane[min(len(dense_lane) - 1, best_index + ORACLE_LOOKAHEAD_SAMPLES)]


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _rotate(vector: tuple[float, float], angle: float) -> tuple[float, float]:
    c = math.cos(angle)
    s = math.sin(angle)
    return (vector[0] * c - vector[1] * s, vector[0] * s + vector[1] * c)
