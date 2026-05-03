from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from .rfs_metric import Trajectory
from .wod_e2e import WodE2EPreferenceFrame


WOD_FUTURE_WAYPOINTS = 20


def kinematic_candidate_payloads(
    frames: Iterable[WodE2EPreferenceFrame],
    *,
    source: str = "wod_kinematic_non_text",
    profile: str = "base",
) -> list[dict[str, object]]:
    payloads: list[dict[str, object]] = []
    for frame in frames:
        for candidate_index, (candidate_name, trajectory) in enumerate(
            kinematic_trajectories(frame.past_trajectory, profile=profile)
        ):
            payloads.append(
                {
                    "frame_name": frame.frame_name,
                    "source": source,
                    "candidate_name": candidate_name,
                    "candidate_index": candidate_index,
                    "trajectory_20wp_4hz": _json_trajectory(trajectory),
                }
            )
    return payloads


def write_kinematic_candidate_jsonl(
    frames: Iterable[WodE2EPreferenceFrame],
    path: str | Path,
    *,
    source: str = "wod_kinematic_non_text",
    profile: str = "base",
) -> int:
    count = 0
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for payload in kinematic_candidate_payloads(frames, source=source, profile=profile):
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            count += 1
    return count


def kinematic_trajectories(
    past_trajectory: Sequence[tuple[float, float]],
    *,
    profile: str = "base",
) -> list[tuple[str, Trajectory]]:
    if len(past_trajectory) < 2:
        raise ValueError("at least two past trajectory points are required")
    if profile not in {"base", "expanded", "reflex"}:
        raise ValueError(f"unsupported kinematic profile: {profile!r}")
    last_step = _step(past_trajectory[-2], past_trajectory[-1])
    trajectories = [
        ("constant_velocity", _constant_velocity(last_step)),
        ("constant_acceleration", _constant_acceleration(past_trajectory, last_step)),
        ("constant_heading_change", _constant_heading_change(past_trajectory, last_step)),
        ("hold_position", [(0.0, 0.0)] * WOD_FUTURE_WAYPOINTS),
    ]
    if profile in {"expanded", "reflex"}:
        trajectories.extend(
            [
                ("speed_25pct", _constant_velocity(_scale_step(last_step, 0.25))),
                ("speed_50pct", _constant_velocity(_scale_step(last_step, 0.5))),
                ("speed_75pct", _constant_velocity(_scale_step(last_step, 0.75))),
                ("speed_125pct", _constant_velocity(_scale_step(last_step, 1.25))),
                ("stop_by_3s", _decelerate_to_stop(last_step, stop_index=12)),
                ("stop_by_5s", _decelerate_to_stop(last_step, stop_index=20)),
            ]
        )
    if profile == "reflex":
        trajectories.extend(_reflex_trajectories(last_step))
    return trajectories


def _constant_velocity(step: tuple[float, float]) -> Trajectory:
    return [(step[0] * index, step[1] * index) for index in range(1, WOD_FUTURE_WAYPOINTS + 1)]


def _scale_step(step: tuple[float, float], scale: float) -> tuple[float, float]:
    return (step[0] * scale, step[1] * scale)


def _decelerate_to_stop(step: tuple[float, float], *, stop_index: int) -> Trajectory:
    stop = max(1, min(WOD_FUTURE_WAYPOINTS, int(stop_index)))
    trajectory: Trajectory = []
    x = 0.0
    y = 0.0
    for index in range(1, WOD_FUTURE_WAYPOINTS + 1):
        scale = max(0.0, 1.0 - (index - 1) / stop)
        x += step[0] * scale
        y += step[1] * scale
        trajectory.append((x, y))
    return trajectory


def _reflex_trajectories(step: tuple[float, float]) -> list[tuple[str, Trajectory]]:
    """Training-free evasive/yield hypotheses from ego motion only."""
    return [
        ("yield_creep", _decelerate_to_scale(step, final_scale=0.20)),
        ("yield_then_go", _yield_then_go(step, yield_steps=8, resume_scale=0.85)),
        ("lane_offset_left_1m", _lateral_offset(step, lateral_m=1.0)),
        ("lane_offset_right_1m", _lateral_offset(step, lateral_m=-1.0)),
        ("lane_change_left_3m", _lateral_offset(step, lateral_m=3.2)),
        ("lane_change_right_3m", _lateral_offset(step, lateral_m=-3.2)),
        ("avoid_left_return", _avoid_and_return(step, lateral_m=2.2)),
        ("avoid_right_return", _avoid_and_return(step, lateral_m=-2.2)),
    ]


def _decelerate_to_scale(step: tuple[float, float], *, final_scale: float) -> Trajectory:
    scale_end = max(0.0, float(final_scale))
    trajectory: Trajectory = []
    x = 0.0
    y = 0.0
    denominator = max(1, WOD_FUTURE_WAYPOINTS - 1)
    for index in range(WOD_FUTURE_WAYPOINTS):
        ratio = index / denominator
        scale = (1.0 - ratio) + scale_end * ratio
        x += step[0] * scale
        y += step[1] * scale
        trajectory.append((x, y))
    return trajectory


def _yield_then_go(
    step: tuple[float, float],
    *,
    yield_steps: int,
    resume_scale: float,
) -> Trajectory:
    slow_steps = max(1, min(WOD_FUTURE_WAYPOINTS - 1, int(yield_steps)))
    trajectory: Trajectory = []
    x = 0.0
    y = 0.0
    for index in range(WOD_FUTURE_WAYPOINTS):
        if index < slow_steps:
            scale = 0.15
        else:
            ramp = (index - slow_steps + 1) / max(1, WOD_FUTURE_WAYPOINTS - slow_steps)
            scale = 0.15 + (float(resume_scale) - 0.15) * min(1.0, ramp)
        x += step[0] * scale
        y += step[1] * scale
        trajectory.append((x, y))
    return trajectory


def _lateral_offset(step: tuple[float, float], *, lateral_m: float) -> Trajectory:
    forward, lateral = _basis(step)
    trajectory: Trajectory = []
    for index in range(1, WOD_FUTURE_WAYPOINTS + 1):
        progress = _smoothstep(index / WOD_FUTURE_WAYPOINTS)
        x = forward[0] * _step_norm(step) * index + lateral[0] * lateral_m * progress
        y = forward[1] * _step_norm(step) * index + lateral[1] * lateral_m * progress
        trajectory.append((x, y))
    return trajectory


def _avoid_and_return(step: tuple[float, float], *, lateral_m: float) -> Trajectory:
    forward, lateral = _basis(step)
    norm = _step_norm(step)
    trajectory: Trajectory = []
    for index in range(1, WOD_FUTURE_WAYPOINTS + 1):
        phase = index / WOD_FUTURE_WAYPOINTS
        if phase <= 0.55:
            lateral_scale = _smoothstep(phase / 0.55)
        else:
            lateral_scale = 1.0 - _smoothstep((phase - 0.55) / 0.45)
        x = forward[0] * norm * index + lateral[0] * lateral_m * lateral_scale
        y = forward[1] * norm * index + lateral[1] * lateral_m * lateral_scale
        trajectory.append((x, y))
    return trajectory


def _basis(step: tuple[float, float]) -> tuple[tuple[float, float], tuple[float, float]]:
    norm = _step_norm(step)
    if norm <= 1e-8:
        return (1.0, 0.0), (0.0, 1.0)
    forward = (step[0] / norm, step[1] / norm)
    lateral = (-forward[1], forward[0])
    return forward, lateral


def _step_norm(step: tuple[float, float]) -> float:
    import math

    return math.hypot(float(step[0]), float(step[1]))


def _smoothstep(value: float) -> float:
    ratio = min(1.0, max(0.0, float(value)))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def _constant_acceleration(
    past_trajectory: Sequence[tuple[float, float]],
    last_step: tuple[float, float],
) -> Trajectory:
    if len(past_trajectory) < 3:
        return _constant_velocity(last_step)
    previous_step = _step(past_trajectory[-3], past_trajectory[-2])
    acceleration_step = (last_step[0] - previous_step[0], last_step[1] - previous_step[1])
    return [
        (
            last_step[0] * index + 0.5 * acceleration_step[0] * index * (index + 1),
            last_step[1] * index + 0.5 * acceleration_step[1] * index * (index + 1),
        )
        for index in range(1, WOD_FUTURE_WAYPOINTS + 1)
    ]


def _constant_heading_change(
    past_trajectory: Sequence[tuple[float, float]],
    last_step: tuple[float, float],
) -> Trajectory:
    if len(past_trajectory) < 3:
        return _constant_velocity(last_step)
    previous_step = _step(past_trajectory[-3], past_trajectory[-2])
    cross = previous_step[0] * last_step[1] - previous_step[1] * last_step[0]
    dot = previous_step[0] * last_step[0] + previous_step[1] * last_step[1]
    if cross == 0.0 and dot == 0.0:
        return _constant_velocity(last_step)

    import math

    heading_delta = math.atan2(cross, dot)
    step = last_step
    x = 0.0
    y = 0.0
    trajectory: Trajectory = []
    for _ in range(WOD_FUTURE_WAYPOINTS):
        x += step[0]
        y += step[1]
        trajectory.append((x, y))
        cos_delta = math.cos(heading_delta)
        sin_delta = math.sin(heading_delta)
        step = (
            step[0] * cos_delta - step[1] * sin_delta,
            step[0] * sin_delta + step[1] * cos_delta,
        )
    return trajectory


def _step(start: tuple[float, float], end: tuple[float, float]) -> tuple[float, float]:
    return (float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))


def _json_trajectory(trajectory: Trajectory) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in trajectory]
