from __future__ import annotations

from dataclasses import dataclass, field
import math

from .environment import Actor, Obstacle, Scenario
from .perception import ScenePerception
from .planner import PlannedAction
from .trajectory_selector import TrajectoryCandidate, TrajectoryReference, TrajectorySelectorScore, TrajectorySelectorConfig, Trajectory, score_candidate
from .world_model import WorldState


@dataclass(frozen=True)
class ManeuverSpec:
    name: str
    min_speed_mps: float
    speed_scale: float
    lateral_offset_m: float
    lateral_profile: str = "smooth"


@dataclass(frozen=True)
class ReferenceRuleConfig:
    clear_obstacle_pressure_max: float = 0.25
    clear_uncertainty_max: float = 0.45
    clear_corridor_ratio_min: float = 0.35
    obstacle_pressure_min: float = 0.20
    high_uncertainty_min: float = 0.55
    low_corridor_ratio_max: float = 0.25
    clear_maintain_score: float = 92.0
    clear_center_score: float = 88.0
    obstacle_slow_yield_score: float = 86.0
    obstacle_nudge_score: float = 94.0
    obstacle_evasive_score: float = 89.0
    uncertainty_crawl_score: float = 90.0
    uncertainty_stop_score: float = 82.0
    lane_recover_score: float = 93.0
    default_maintain_score: float = 80.0
    default_slow_yield_score: float = 76.0
    lane_center_min_speed_mps: float = 0.75
    lane_recover_min_speed_mps: float = 0.5
    lane_recover_speed_scale: float = 0.65


@dataclass(frozen=True)
class SimulatorBackedScoreConfig:
    use_privileged_actor_forecast: bool = False
    min_action_clearance_m: float = 0.55
    unsafe_action_penalty: float = 1_000.0
    avoid_unnecessary_stop_penalty: float = 250.0
    progress_bonus_min: float = -20.0
    progress_bonus_max: float = 40.0
    goal_progress_weight: float = 2.5
    target_progress_weight: float = 4.0
    moving_speed_bonus_cap: float = 12.0
    moving_speed_bonus_weight: float = 6.0
    near_clearance_target_m: float = 2.0
    near_clearance_penalty_weight: float = 55.0
    horizon_clearance_target_m: float = 0.75
    horizon_clearance_penalty_weight: float = 12.0
    avoidance_side_clearance_delta_m: float = 0.25
    obstacle_ignore_behind_m: float = -1.0
    obstacle_weight_epsilon_m: float = 0.1
    obstacle_pressure_distance_m: float = 10.0


@dataclass(frozen=True)
class TrajectoryGenerationConfig:
    point_count: int = 20
    horizon_seconds: float = 5.0
    action_index: int = 3
    smoothstep_a: float = 3.0
    smoothstep_b: float = 2.0
    goal_heading_distance_m: float = 25.0


@dataclass(frozen=True)
class SpotlightReflexConfig:
    selector: TrajectorySelectorConfig = field(default_factory=TrajectorySelectorConfig)
    references: ReferenceRuleConfig = field(default_factory=ReferenceRuleConfig)
    scoring: SimulatorBackedScoreConfig = field(default_factory=SimulatorBackedScoreConfig)
    trajectory: TrajectoryGenerationConfig = field(default_factory=TrajectoryGenerationConfig)
    maneuvers: tuple[ManeuverSpec, ...] = (
        ManeuverSpec("stop", 0.0, 0.0, 0.0),
        ManeuverSpec("crawl", 0.35, 0.25, 0.0),
        ManeuverSpec("maintain", 0.75, 1.0, 0.0),
        ManeuverSpec("slow_yield", 0.45, 0.55, 0.0),
        ManeuverSpec("nudge_left", 0.65, 0.85, 2.0),
        ManeuverSpec("nudge_right", 0.65, 0.85, -2.0),
        ManeuverSpec("evasive_left", 0.55, 0.70, 8.0, "early"),
        ManeuverSpec("evasive_right", 0.55, 0.70, -8.0, "early"),
        ManeuverSpec("lane_recover", 0.50, 0.65, 0.0),
    )


DEFAULT_SPOTLIGHT_CONFIG = SpotlightReflexConfig()


@dataclass(frozen=True)
class SpotlightSelection:
    candidate: TrajectoryCandidate
    score: TrajectorySelectorScore
    candidate_count: int
    reference_count: int

    def to_metadata(self) -> dict[str, float | int | str | bool]:
        return {
            "candidate_count": self.candidate_count,
            "reference_count": self.reference_count,
            "selected_maneuver": self.candidate.name,
            "selector_score": self.score.combined_score,
            "selector_3s_score": self.score.score_3s,
            "selector_5s_score": self.score.score_5s,
            "selector_3s_reference": self.score.reference_3s_label,
            "selector_5s_reference": self.score.reference_5s_label,
            "selector_3s_inside_region": self.score.inside_3s_region,
            "selector_5s_inside_region": self.score.inside_5s_region,
        }


def generate_maneuver_candidates(
    position: tuple[float, float],
    heading: tuple[float, float],
    speed_mps: float,
    config: SpotlightReflexConfig | None = None,
) -> list[TrajectoryCandidate]:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    forward = _normalize(heading)
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    base_speed = max(0.0, speed_mps)

    candidates: list[TrajectoryCandidate] = []
    for spec in config.maneuvers:
        maneuver_speed = max(spec.min_speed_mps, base_speed * spec.speed_scale)
        trajectory = _trajectory(
            position,
            forward,
            left,
            maneuver_speed,
            spec.lateral_offset_m,
            lateral_profile=spec.lateral_profile,
            config=config.trajectory,
        )
        candidates.append(
            TrajectoryCandidate(
                name=spec.name,
                trajectory=trajectory,
                confidence=1.0,
                metadata={"speed_mps": maneuver_speed, "lateral_offset_m": spec.lateral_offset_m},
            )
        )
    return candidates


def generate_pseudo_references(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    speed_mps: float,
    heading: tuple[float, float] | None = None,
    config: SpotlightReflexConfig | None = None,
) -> list[TrajectoryReference]:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    rules = config.references
    reference_heading = heading if heading is not None else perception.lane_heading
    candidates = {
        candidate.name: candidate
        for candidate in generate_maneuver_candidates(position, reference_heading, speed_mps, config)
    }
    obstacle_pressure = _obstacle_pressure(perception, config.scoring)
    corridor_ratio = perception.corridor_margin / max(scenario.lane_half_width, 1e-6)
    uncertainty = max(world_state.uncertainty, perception.uncertainty)

    references: list[TrajectoryReference] = []
    if (
        obstacle_pressure < rules.clear_obstacle_pressure_max
        and uncertainty < rules.clear_uncertainty_max
        and corridor_ratio > rules.clear_corridor_ratio_min
    ):
        references.append(TrajectoryReference("clear_corridor_maintain", candidates["maintain"].trajectory, rules.clear_maintain_score))
        references.append(
            TrajectoryReference(
                "clear_corridor_center_progress",
                _lane_center_reference(position, world_state, speed_mps, config),
                rules.clear_center_score,
            )
        )

    if obstacle_pressure >= rules.obstacle_pressure_min:
        side = _avoidance_side(position, perception, candidates, scenario, config)
        references.append(TrajectoryReference("obstacle_pressure_slow_yield", candidates["slow_yield"].trajectory, rules.obstacle_slow_yield_score))
        references.append(TrajectoryReference(f"obstacle_pressure_nudge_{side}", candidates[f"nudge_{side}"].trajectory, rules.obstacle_nudge_score))
        references.append(
            TrajectoryReference(f"obstacle_pressure_evasive_{side}", candidates[f"evasive_{side}"].trajectory, rules.obstacle_evasive_score)
        )

    if uncertainty >= rules.high_uncertainty_min:
        references.append(TrajectoryReference("high_uncertainty_crawl", candidates["crawl"].trajectory, rules.uncertainty_crawl_score))
        references.append(TrajectoryReference("high_uncertainty_stop", candidates["stop"].trajectory, rules.uncertainty_stop_score))

    if corridor_ratio < rules.low_corridor_ratio_max:
        references.append(
            TrajectoryReference(
                "low_corridor_margin_lane_recover",
                _lane_recover_reference(position, perception, speed_mps, reference_heading, config),
                rules.lane_recover_score,
            )
        )

    if not references:
        references.append(TrajectoryReference("default_maintain", candidates["maintain"].trajectory, rules.default_maintain_score))
        references.append(TrajectoryReference("default_slow_yield", candidates["slow_yield"].trajectory, rules.default_slow_yield_score))

    return references


def select_maneuver(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    speed_mps: float,
    config: SpotlightReflexConfig | None = None,
) -> SpotlightSelection:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    heading = _planning_heading(position, world_state, perception, scenario, config)
    candidates = generate_maneuver_candidates(position, heading, speed_mps, config)
    references = generate_pseudo_references(scenario, position, world_state, perception, speed_mps, heading, config)

    scored_candidates: list[tuple[float, TrajectoryCandidate, TrajectorySelectorScore]] = []
    moving_candidate_is_safe = any(
        candidate.name != "stop" and _action_clearance(candidate.trajectory, scenario, config) >= config.scoring.min_action_clearance_m
        for candidate in candidates
    )
    for candidate in candidates:
        candidate_score = score_candidate(candidate, references, speed_mps, config.selector)
        effective_score = _simulator_backed_score(
            candidate_score,
            candidate,
            scenario,
            position,
            world_state,
            moving_candidate_is_safe,
            config,
        )
        scored_candidates.append((effective_score, candidate, candidate_score))

    best_effective_score, best_candidate, best_score = scored_candidates[0]
    for effective_score, candidate, candidate_score in scored_candidates[1:]:
        if (effective_score, candidate.confidence) > (best_effective_score, best_candidate.confidence):
            best_candidate = candidate
            best_score = candidate_score
            best_effective_score = effective_score

    return SpotlightSelection(best_candidate, best_score, len(candidates), len(references))


def plan_spotlight_reflex_action(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    nominal_step_size: float = 1.25,
    config: SpotlightReflexConfig | None = None,
) -> tuple[PlannedAction, SpotlightSelection]:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    speed_mps = max(0.0, nominal_step_size)
    selection = select_maneuver(scenario, position, world_state, perception, speed_mps, config)
    next_point = selection.candidate.trajectory[config.trajectory.action_index]
    step_vector = (next_point[0] - position[0], next_point[1] - position[1])
    step_distance = math.hypot(step_vector[0], step_vector[1])
    direction = _normalize(step_vector)
    action = PlannedAction(
        direction=direction,
        speed=step_distance,
        mode=f"spotlight_reflex:{selection.candidate.name}",
        score=selection.score.combined_score,
    )
    return action, selection


def _simulator_backed_score(
    score: TrajectorySelectorScore,
    candidate: TrajectoryCandidate,
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    moving_candidate_is_safe: bool,
    config: SpotlightReflexConfig,
) -> float:
    scoring = config.scoring
    action_clearance = _action_clearance(candidate.trajectory, scenario, config)
    full_clearance = _min_obstacle_clearance_with_config(candidate.trajectory, scenario, config)
    if action_clearance < scoring.min_action_clearance_m:
        return score.combined_score - scoring.unsafe_action_penalty

    if candidate.name == "stop" and moving_candidate_is_safe:
        return score.combined_score - scoring.avoid_unnecessary_stop_penalty

    action_point = candidate.trajectory[min(config.trajectory.action_index, len(candidate.trajectory) - 1)]
    final_point = candidate.trajectory[-1]
    goal_progress = math.dist(position, scenario.goal) - math.dist(final_point, scenario.goal)
    target_progress = math.dist(position, world_state.target_point) - math.dist(action_point, world_state.target_point)
    progress_bonus = max(
        scoring.progress_bonus_min,
        min(
            scoring.progress_bonus_max,
            goal_progress * scoring.goal_progress_weight + target_progress * scoring.target_progress_weight,
        ),
    )
    if candidate.name != "stop":
        progress_bonus += min(
            scoring.moving_speed_bonus_cap,
            float(candidate.metadata.get("speed_mps", 0.0)) * scoring.moving_speed_bonus_weight,
        )
    near_penalty = max(0.0, scoring.near_clearance_target_m - action_clearance) * scoring.near_clearance_penalty_weight
    horizon_penalty = max(0.0, scoring.horizon_clearance_target_m - full_clearance) * scoring.horizon_clearance_penalty_weight
    return score.combined_score + progress_bonus - near_penalty - horizon_penalty


def _action_clearance(trajectory: Trajectory, scenario: Scenario, config: SpotlightReflexConfig | None = None) -> float:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    return _min_obstacle_clearance_with_config(trajectory[: config.trajectory.action_index + 1], scenario, config)


def _min_obstacle_clearance(trajectory: Trajectory, scenario: Scenario) -> float:
    return _min_obstacle_clearance_with_config(trajectory, scenario, DEFAULT_SPOTLIGHT_CONFIG)


def _min_obstacle_clearance_with_config(
    trajectory: Trajectory,
    scenario: Scenario,
    config: SpotlightReflexConfig,
) -> float:
    min_clearance = math.inf
    for point_index, point in enumerate(trajectory):
        point_x, point_y = point
        for obstacle in _trajectory_step_obstacles(scenario, point_index, config):
            clearance = math.hypot(point_x - obstacle.x, point_y - obstacle.y) - obstacle.radius
            if clearance < min_clearance:
                min_clearance = clearance
    return min_clearance


def _trajectory_step_obstacles(
    scenario: Scenario,
    point_index: int,
    config: SpotlightReflexConfig,
) -> list[Obstacle]:
    if not config.scoring.use_privileged_actor_forecast or not scenario.actors:
        return scenario.obstacles

    cache = scenario.environment.setdefault("_forecast_obstacles_by_point_index", {})
    if point_index in cache:
        return cache[point_index]

    static_obstacles, moving_actors = _forecast_static_and_moving_obstacles(scenario)

    current_tick = int(scenario.environment.get("tick", 0))
    actor_obstacles = [
        obstacle
        for actor in moving_actors
        if (obstacle := _project_actor_to_obstacle(actor, current_tick + point_index + 1)) is not None
    ]
    obstacles = static_obstacles + actor_obstacles
    cache[point_index] = obstacles
    return obstacles


def _forecast_static_and_moving_obstacles(scenario: Scenario) -> tuple[list[Obstacle], list[Actor]]:
    cache_key = "_forecast_static_and_moving_obstacles"
    if cache_key in scenario.environment:
        return scenario.environment[cache_key]

    moving_actors = [actor for actor in scenario.actors if _actor_is_moving(actor)]
    if not moving_actors:
        result = (scenario.obstacles, [])
        scenario.environment[cache_key] = result
        return result

    current_tick = int(scenario.environment.get("tick", 0))
    current_actor_obstacles = [
        obstacle
        for actor in moving_actors
        if (obstacle := _project_actor_to_obstacle(actor, current_tick)) is not None
    ]
    static_obstacles = [
        obstacle
        for obstacle in scenario.obstacles
        if not any(_same_obstacle(obstacle, actor_obstacle) for actor_obstacle in current_actor_obstacles)
    ]
    result = (static_obstacles, moving_actors)
    scenario.environment[cache_key] = result
    return result


def _actor_is_moving(actor: Actor) -> bool:
    return any(abs(float(getattr(actor, field_name))) > 1e-9 for field_name in ("speed", "vx", "vy"))


def _same_obstacle(first: Obstacle, second: Obstacle) -> bool:
    return (
        first.kind == second.kind
        and first.label == second.label
        and math.isclose(first.x, second.x, abs_tol=1e-9)
        and math.isclose(first.y, second.y, abs_tol=1e-9)
        and math.isclose(first.radius, second.radius, abs_tol=1e-9)
    )


def _project_actor_to_obstacle(actor: Actor, tick: int, dt: float = 0.25) -> Obstacle | None:
    if tick < actor.active_from or tick > actor.active_until:
        return None
    elapsed = max(0, tick - actor.active_from) * dt
    x = actor.x
    y = actor.y
    if actor.behavior in {"cut_in", "swerve"}:
        longitudinal = actor.speed * elapsed
        lateral = min(4.5, 0.38 * elapsed * elapsed)
        lateral *= -1.0 if actor.vy < 0.0 else 1.0
        x += math.cos(actor.heading) * longitudinal
        y += math.sin(actor.heading) * longitudinal + lateral
    elif actor.behavior in {"darting", "erratic_pedestrian"}:
        pause = 0.4 if int(elapsed * 2.0) % 3 == 0 else 1.0
        wobble = math.sin(elapsed * 3.7) * 0.55
        x += actor.vx * elapsed * pause
        y += actor.vy * elapsed * pause + wobble
    elif actor.behavior in {"sudden_brake", "hesitating"}:
        moving_time = min(elapsed, 1.2)
        creep_time = max(0.0, elapsed - 1.2)
        distance = actor.speed * moving_time + actor.speed * 0.15 * creep_time
        x += math.cos(actor.heading) * distance
        y += math.sin(actor.heading) * distance
    elif actor.behavior == "wrong_way":
        x -= abs(actor.vx) * elapsed
        y += actor.vy * elapsed
    else:
        x += actor.vx * elapsed
        y += actor.vy * elapsed
    return Obstacle(x, y, max(actor.width, actor.length) * 0.5, kind=actor.kind, label=actor.role)


def _trajectory(
    position: tuple[float, float],
    forward: tuple[float, float],
    left: tuple[float, float],
    speed_mps: float,
    final_lateral_offset: float,
    lateral_profile: str = "smooth",
    config: TrajectoryGenerationConfig | None = None,
) -> Trajectory:
    config = config or DEFAULT_SPOTLIGHT_CONFIG.trajectory
    points: Trajectory = []
    for step in range(1, config.point_count + 1):
        t = step / config.point_count
        seconds = t * config.horizon_seconds
        lateral_t = _lateral_interpolation(t, lateral_profile, config)
        forward_distance = speed_mps * seconds
        lateral_distance = final_lateral_offset * lateral_t
        points.append(
            (
                position[0] + forward[0] * forward_distance + left[0] * lateral_distance,
                position[1] + forward[1] * forward_distance + left[1] * lateral_distance,
            )
        )
    return points


def _lateral_interpolation(t: float, profile: str, config: TrajectoryGenerationConfig | None = None) -> float:
    config = config or DEFAULT_SPOTLIGHT_CONFIG.trajectory
    if profile == "smooth":
        return t * t * (config.smoothstep_a - config.smoothstep_b * t)
    if profile == "early":
        return math.sqrt(t)
    raise ValueError(f"unknown lateral trajectory profile: {profile}")


def _lane_center_reference(
    position: tuple[float, float],
    world_state: WorldState,
    speed_mps: float,
    config: SpotlightReflexConfig | None = None,
) -> Trajectory:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    forward = _normalize((world_state.target_point[0] - position[0], world_state.target_point[1] - position[1]))
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    return _trajectory(
        position,
        forward,
        left,
        max(config.references.lane_center_min_speed_mps, speed_mps),
        0.0,
        config=config.trajectory,
    )


def _lane_recover_reference(
    position: tuple[float, float],
    perception: ScenePerception,
    speed_mps: float,
    heading: tuple[float, float] | None = None,
    config: SpotlightReflexConfig | None = None,
) -> Trajectory:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    forward = _normalize(heading if heading is not None else perception.lane_heading)
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    dx = perception.lane_point[0] - position[0]
    dy = perception.lane_point[1] - position[1]
    lateral_offset = dx * left[0] + dy * left[1]
    return _trajectory(
        position,
        forward,
        left,
        max(config.references.lane_recover_min_speed_mps, speed_mps * config.references.lane_recover_speed_scale),
        lateral_offset,
        config=config.trajectory,
    )


def _planning_heading(
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    scenario: Scenario,
    config: SpotlightReflexConfig | None = None,
) -> tuple[float, float]:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    goal_vector = (scenario.goal[0] - position[0], scenario.goal[1] - position[1])
    goal_heading = _normalize(goal_vector)
    lane_heading = _normalize(perception.lane_heading)
    if lane_heading == (0.0, 0.0) or math.dist(position, scenario.goal) < config.trajectory.goal_heading_distance_m:
        return goal_heading
    return lane_heading


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _obstacle_pressure(
    perception: ScenePerception,
    config: SimulatorBackedScoreConfig | None = None,
) -> float:
    config = config or DEFAULT_SPOTLIGHT_CONFIG.scoring
    nearest_signed_distance = min(
        (obstacle.signed_distance for obstacle in perception.visible_obstacles),
        default=config.obstacle_pressure_distance_m * 2.0,
    )
    return max(0.0, min(1.0, (config.obstacle_pressure_distance_m - nearest_signed_distance) / config.obstacle_pressure_distance_m))


def _avoidance_side(
    position: tuple[float, float],
    perception: ScenePerception,
    candidates: dict[str, TrajectoryCandidate],
    scenario: Scenario,
    config: SpotlightReflexConfig | None = None,
) -> str:
    config = config or DEFAULT_SPOTLIGHT_CONFIG
    scoring = config.scoring
    left_clearance = max(
        _min_obstacle_clearance_with_config(candidates["nudge_left"].trajectory, scenario, config),
        _min_obstacle_clearance_with_config(candidates["evasive_left"].trajectory, scenario, config),
    )
    right_clearance = max(
        _min_obstacle_clearance_with_config(candidates["nudge_right"].trajectory, scenario, config),
        _min_obstacle_clearance_with_config(candidates["evasive_right"].trajectory, scenario, config),
    )
    if abs(left_clearance - right_clearance) > scoring.avoidance_side_clearance_delta_m:
        return "left" if left_clearance > right_clearance else "right"

    forward = _normalize(perception.lane_heading)
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    weighted_lateral = 0.0
    total_weight = 0.0
    for obstacle in perception.visible_obstacles:
        dx = obstacle.x - position[0]
        dy = obstacle.y - position[1]
        forward_distance = dx * forward[0] + dy * forward[1]
        if forward_distance < scoring.obstacle_ignore_behind_m:
            continue
        lateral = dx * left[0] + dy * left[1]
        weight = 1.0 / max(
            obstacle.signed_distance + obstacle.radius + scoring.obstacle_weight_epsilon_m,
            scoring.obstacle_weight_epsilon_m,
        )
        weighted_lateral += lateral * weight
        total_weight += weight
    if total_weight == 0.0:
        return "left"
    return "right" if weighted_lateral > 0.0 else "left"
