from __future__ import annotations

from dataclasses import dataclass, field
import math

from .environment import Scenario
from .perception import ScenePerception
from .planner import PlannedAction
from .world_model import WorldState


Trajectory = list[tuple[float, float]]

RFS_3S_INDEX = 11
RFS_5S_INDEX = 19
ACTION_INDEX = 3
RFS_3S_LATERAL_M = 1.0
RFS_3S_LONGITUDINAL_M = 4.0
RFS_5S_LATERAL_M = 1.8
RFS_5S_LONGITUDINAL_M = 7.2
RFS_SCORE_FLOOR = 4.0


@dataclass(frozen=True)
class ManeuverCandidate:
    name: str
    trajectory: Trajectory
    confidence: float = 1.0
    source: str = "deterministic_maneuver_library"
    metadata: dict[str, float | str] = field(default_factory=dict)


@dataclass(frozen=True)
class RfsReference:
    label: str
    trajectory: Trajectory
    score: float


@dataclass(frozen=True)
class RfsScore:
    combined_score: float
    score_3s: float
    score_5s: float
    reference_3s_label: str
    reference_5s_label: str
    inside_3s_trust_region: bool
    inside_5s_trust_region: bool


@dataclass(frozen=True)
class SpotlightSelection:
    candidate: ManeuverCandidate
    score: RfsScore
    candidate_count: int
    reference_count: int

    def to_metadata(self) -> dict[str, float | int | str | bool]:
        return {
            "candidate_count": self.candidate_count,
            "reference_count": self.reference_count,
            "selected_maneuver": self.candidate.name,
            "rfs_score": self.score.combined_score,
            "rfs_3s_score": self.score.score_3s,
            "rfs_5s_score": self.score.score_5s,
            "rfs_3s_reference": self.score.reference_3s_label,
            "rfs_5s_reference": self.score.reference_5s_label,
            "rfs_3s_inside_trust_region": self.score.inside_3s_trust_region,
            "rfs_5s_inside_trust_region": self.score.inside_5s_trust_region,
        }


def speed_scale(speed_mps: float) -> float:
    if speed_mps <= 1.4:
        return 0.5
    if speed_mps >= 11.0:
        return 1.0
    return 0.5 + 0.5 * (speed_mps - 1.4) / (11.0 - 1.4)


def trust_region_score(
    candidate: Trajectory,
    reference: Trajectory,
    reference_score: float,
    speed_mps: float,
    index: int,
) -> tuple[float, bool]:
    if index == RFS_3S_INDEX:
        lateral_threshold = RFS_3S_LATERAL_M
        longitudinal_threshold = RFS_3S_LONGITUDINAL_M
    elif index == RFS_5S_INDEX:
        lateral_threshold = RFS_5S_LATERAL_M
        longitudinal_threshold = RFS_5S_LONGITUDINAL_M
    else:
        raise ValueError(f"unsupported RFS trust-region index: {index}")

    if len(candidate) <= index or len(reference) <= index:
        raise ValueError("candidate and reference trajectories must contain the requested RFS index")

    scale = speed_scale(speed_mps)
    lateral_threshold *= scale
    longitudinal_threshold *= scale

    tangent = _reference_tangent(reference, index)
    normal = (-tangent[1], tangent[0])
    dx = candidate[index][0] - reference[index][0]
    dy = candidate[index][1] - reference[index][1]
    longitudinal_error = abs(dx * tangent[0] + dy * tangent[1])
    lateral_error = abs(dx * normal[0] + dy * normal[1])

    longitudinal_overshoot = max(0.0, longitudinal_error / longitudinal_threshold - 1.0)
    lateral_overshoot = max(0.0, lateral_error / lateral_threshold - 1.0)
    overshoot = max(longitudinal_overshoot, lateral_overshoot)
    inside = overshoot == 0.0
    if inside:
        return reference_score, True
    return max(reference_score * (0.1**overshoot), RFS_SCORE_FLOOR), False


def score_candidate(candidate: ManeuverCandidate, references: list[RfsReference], speed_mps: float) -> RfsScore:
    if not references:
        raise ValueError("at least one RFS reference is required")

    best_3s = _best_reference_score(candidate.trajectory, references, speed_mps, RFS_3S_INDEX)
    best_5s = _best_reference_score(candidate.trajectory, references, speed_mps, RFS_5S_INDEX)
    combined = 0.5 * best_3s[0] + 0.5 * best_5s[0]
    return RfsScore(
        combined_score=combined,
        score_3s=best_3s[0],
        score_5s=best_5s[0],
        reference_3s_label=best_3s[1].label,
        reference_5s_label=best_5s[1].label,
        inside_3s_trust_region=best_3s[2],
        inside_5s_trust_region=best_5s[2],
    )


def generate_maneuver_candidates(
    position: tuple[float, float],
    heading: tuple[float, float],
    speed_mps: float,
) -> list[ManeuverCandidate]:
    forward = _normalize(heading)
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    base_speed = max(0.0, speed_mps)

    specs = [
        ("stop", 0.0, 0.0, "smooth"),
        ("crawl", max(0.35, base_speed * 0.25), 0.0, "smooth"),
        ("maintain", max(0.75, base_speed), 0.0, "smooth"),
        ("slow_yield", max(0.45, base_speed * 0.55), 0.0, "smooth"),
        ("nudge_left", max(0.65, base_speed * 0.85), 2.0, "smooth"),
        ("nudge_right", max(0.65, base_speed * 0.85), -2.0, "smooth"),
        ("evasive_left", max(0.55, base_speed * 0.70), 8.0, "early"),
        ("evasive_right", max(0.55, base_speed * 0.70), -8.0, "early"),
        ("lane_recover", max(0.50, base_speed * 0.65), 0.0, "smooth"),
    ]

    candidates: list[ManeuverCandidate] = []
    for name, maneuver_speed, lateral_offset, lateral_profile in specs:
        trajectory = _trajectory(position, forward, left, maneuver_speed, lateral_offset, lateral_profile=lateral_profile)
        candidates.append(
            ManeuverCandidate(
                name=name,
                trajectory=trajectory,
                confidence=1.0,
                metadata={"speed_mps": maneuver_speed, "lateral_offset_m": lateral_offset},
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
) -> list[RfsReference]:
    reference_heading = heading if heading is not None else perception.lane_heading
    candidates = {candidate.name: candidate for candidate in generate_maneuver_candidates(position, reference_heading, speed_mps)}
    obstacle_pressure = _obstacle_pressure(perception)
    corridor_ratio = perception.corridor_margin / max(scenario.lane_half_width, 1e-6)
    uncertainty = max(world_state.uncertainty, perception.uncertainty)

    references: list[RfsReference] = []
    if obstacle_pressure < 0.25 and uncertainty < 0.45 and corridor_ratio > 0.35:
        references.append(RfsReference("clear_corridor_maintain", candidates["maintain"].trajectory, 92.0))
        references.append(RfsReference("clear_corridor_center_progress", _lane_center_reference(position, world_state, speed_mps), 88.0))

    if obstacle_pressure >= 0.20:
        side = _avoidance_side(position, perception, candidates, scenario)
        references.append(RfsReference("obstacle_pressure_slow_yield", candidates["slow_yield"].trajectory, 86.0))
        references.append(RfsReference(f"obstacle_pressure_nudge_{side}", candidates[f"nudge_{side}"].trajectory, 94.0))
        references.append(RfsReference(f"obstacle_pressure_evasive_{side}", candidates[f"evasive_{side}"].trajectory, 89.0))

    if uncertainty >= 0.55:
        references.append(RfsReference("high_uncertainty_crawl", candidates["crawl"].trajectory, 90.0))
        references.append(RfsReference("high_uncertainty_stop", candidates["stop"].trajectory, 82.0))

    if corridor_ratio < 0.25:
        references.append(
            RfsReference("low_corridor_margin_lane_recover", _lane_recover_reference(position, perception, speed_mps, reference_heading), 93.0)
        )

    if not references:
        references.append(RfsReference("default_maintain", candidates["maintain"].trajectory, 80.0))
        references.append(RfsReference("default_slow_yield", candidates["slow_yield"].trajectory, 76.0))

    return references


def select_maneuver(
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    speed_mps: float,
) -> SpotlightSelection:
    heading = _planning_heading(position, world_state, perception, scenario)
    candidates = generate_maneuver_candidates(position, heading, speed_mps)
    references = generate_pseudo_references(scenario, position, world_state, perception, speed_mps, heading)

    scored_candidates: list[tuple[float, ManeuverCandidate, RfsScore]] = []
    moving_candidate_is_safe = any(
        candidate.name != "stop" and _action_clearance(candidate.trajectory, scenario) >= 0.55 for candidate in candidates
    )
    for candidate in candidates:
        candidate_score = score_candidate(candidate, references, speed_mps)
        effective_score = _simulator_backed_score(
            candidate_score,
            candidate,
            scenario,
            position,
            world_state,
            moving_candidate_is_safe,
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
) -> tuple[PlannedAction, SpotlightSelection]:
    speed_mps = max(0.0, nominal_step_size)
    selection = select_maneuver(scenario, position, world_state, perception, speed_mps)
    next_point = selection.candidate.trajectory[ACTION_INDEX]
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


def _best_reference_score(
    candidate: Trajectory,
    references: list[RfsReference],
    speed_mps: float,
    index: int,
) -> tuple[float, RfsReference, bool]:
    best_score = -math.inf
    best_reference = references[0]
    best_inside = False
    for reference in references:
        score, inside = trust_region_score(candidate, reference.trajectory, reference.score, speed_mps, index)
        if score > best_score:
            best_score = score
            best_reference = reference
            best_inside = inside
    return best_score, best_reference, best_inside


def _simulator_backed_score(
    score: RfsScore,
    candidate: ManeuverCandidate,
    scenario: Scenario,
    position: tuple[float, float],
    world_state: WorldState,
    moving_candidate_is_safe: bool,
) -> float:
    action_clearance = _action_clearance(candidate.trajectory, scenario)
    full_clearance = _min_obstacle_clearance(candidate.trajectory, scenario)
    if action_clearance < 0.55:
        return score.combined_score - 1_000.0

    if candidate.name == "stop" and moving_candidate_is_safe:
        return score.combined_score - 250.0

    action_point = candidate.trajectory[min(ACTION_INDEX, len(candidate.trajectory) - 1)]
    final_point = candidate.trajectory[-1]
    goal_progress = math.dist(position, scenario.goal) - math.dist(final_point, scenario.goal)
    target_progress = math.dist(position, world_state.target_point) - math.dist(action_point, world_state.target_point)
    progress_bonus = max(-20.0, min(40.0, goal_progress * 2.5 + target_progress * 4.0))
    if candidate.name != "stop":
        progress_bonus += min(12.0, float(candidate.metadata.get("speed_mps", 0.0)) * 6.0)
    near_penalty = max(0.0, 2.0 - action_clearance) * 55.0
    horizon_penalty = max(0.0, 0.75 - full_clearance) * 12.0
    return score.combined_score + progress_bonus - near_penalty - horizon_penalty


def _action_clearance(trajectory: Trajectory, scenario: Scenario) -> float:
    return _min_obstacle_clearance(trajectory[: ACTION_INDEX + 1], scenario)


def _min_obstacle_clearance(trajectory: Trajectory, scenario: Scenario) -> float:
    return min(
        (
            math.dist(point, (obstacle.x, obstacle.y)) - obstacle.radius
            for point in trajectory
            for obstacle in scenario.obstacles
        ),
        default=math.inf,
    )


def _trajectory(
    position: tuple[float, float],
    forward: tuple[float, float],
    left: tuple[float, float],
    speed_mps: float,
    final_lateral_offset: float,
    lateral_profile: str = "smooth",
) -> Trajectory:
    points: Trajectory = []
    for step in range(1, 21):
        t = step / 20.0
        seconds = t * 5.0
        lateral_t = _lateral_interpolation(t, lateral_profile)
        forward_distance = speed_mps * seconds
        lateral_distance = final_lateral_offset * lateral_t
        points.append(
            (
                position[0] + forward[0] * forward_distance + left[0] * lateral_distance,
                position[1] + forward[1] * forward_distance + left[1] * lateral_distance,
            )
        )
    return points


def _lateral_interpolation(t: float, profile: str) -> float:
    if profile == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if profile == "early":
        return math.sqrt(t)
    raise ValueError(f"unknown lateral trajectory profile: {profile}")


def _lane_center_reference(
    position: tuple[float, float],
    world_state: WorldState,
    speed_mps: float,
) -> Trajectory:
    forward = _normalize((world_state.target_point[0] - position[0], world_state.target_point[1] - position[1]))
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    return _trajectory(position, forward, left, max(0.75, speed_mps), 0.0)


def _lane_recover_reference(
    position: tuple[float, float],
    perception: ScenePerception,
    speed_mps: float,
    heading: tuple[float, float] | None = None,
) -> Trajectory:
    forward = _normalize(heading if heading is not None else perception.lane_heading)
    if forward == (0.0, 0.0):
        forward = (1.0, 0.0)
    left = (-forward[1], forward[0])
    dx = perception.lane_point[0] - position[0]
    dy = perception.lane_point[1] - position[1]
    lateral_offset = dx * left[0] + dy * left[1]
    return _trajectory(position, forward, left, max(0.5, speed_mps * 0.65), lateral_offset)


def _planning_heading(
    position: tuple[float, float],
    world_state: WorldState,
    perception: ScenePerception,
    scenario: Scenario,
) -> tuple[float, float]:
    goal_vector = (scenario.goal[0] - position[0], scenario.goal[1] - position[1])
    goal_heading = _normalize(goal_vector)
    lane_heading = _normalize(perception.lane_heading)
    if lane_heading == (0.0, 0.0) or math.dist(position, scenario.goal) < 25.0:
        return goal_heading
    return lane_heading


def _reference_tangent(reference: Trajectory, index: int) -> tuple[float, float]:
    if index > 0:
        tangent = (reference[index][0] - reference[index - 1][0], reference[index][1] - reference[index - 1][1])
    else:
        tangent = (reference[1][0] - reference[0][0], reference[1][1] - reference[0][1])
    normalized = _normalize(tangent)
    if normalized == (0.0, 0.0):
        return (1.0, 0.0)
    return normalized


def _normalize(vector: tuple[float, float]) -> tuple[float, float]:
    norm = math.hypot(vector[0], vector[1])
    if norm == 0.0:
        return (0.0, 0.0)
    return (vector[0] / norm, vector[1] / norm)


def _obstacle_pressure(perception: ScenePerception) -> float:
    nearest_signed_distance = min((obstacle.signed_distance for obstacle in perception.visible_obstacles), default=20.0)
    return max(0.0, min(1.0, (10.0 - nearest_signed_distance) / 10.0))


def _avoidance_side(
    position: tuple[float, float],
    perception: ScenePerception,
    candidates: dict[str, ManeuverCandidate],
    scenario: Scenario,
) -> str:
    left_clearance = max(
        _min_obstacle_clearance(candidates["nudge_left"].trajectory, scenario),
        _min_obstacle_clearance(candidates["evasive_left"].trajectory, scenario),
    )
    right_clearance = max(
        _min_obstacle_clearance(candidates["nudge_right"].trajectory, scenario),
        _min_obstacle_clearance(candidates["evasive_right"].trajectory, scenario),
    )
    if abs(left_clearance - right_clearance) > 0.25:
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
        if forward_distance < -1.0:
            continue
        lateral = dx * left[0] + dy * left[1]
        weight = 1.0 / max(obstacle.signed_distance + obstacle.radius + 0.1, 0.1)
        weighted_lateral += lateral * weight
        total_weight += weight
    if total_weight == 0.0:
        return "left"
    return "right" if weighted_lateral > 0.0 else "left"
