from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from typing import Any, Sequence

from .trajectory_resampling import resample_64wp_10hz_to_20wp_4hz
from .wod_e2e import WodE2EPreferenceFrame


DEFAULT_QWEN3_VL_MODEL = "Qwen/Qwen3-VL-8B-Instruct"


@dataclass(frozen=True)
class Qwen3VlWodConfig:
    model_name: str = DEFAULT_QWEN3_VL_MODEL
    license: str = "Apache-2.0"
    output_waypoints: int = 64
    output_hz: float = 10.0
    wod_waypoints: int = 20
    wod_hz: float = 4.0
    horizon_s: float = 5.0
    max_prompt_tokens: int = 4096
    use_coc_style_plan: bool = True
    runtime_policy: str = "offline-trained fast trajectory decoder + RFS selector"
    public_model_pretraining: tuple[str, ...] = (DEFAULT_QWEN3_VL_MODEL,)
    grpo_reward_weights: dict[str, float] = field(
        default_factory=lambda: {
            "rfs": 1.0,
            "format": 0.05,
            "smoothness": 0.03,
            "finite": 0.10,
        }
    )

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_coc_style_prompt(frame: WodE2EPreferenceFrame, *, config: Qwen3VlWodConfig | None = None) -> str:
    """Build a concise Chain-of-Causation-style instruction without requiring hidden CoT at inference."""

    config = config or Qwen3VlWodConfig()
    return "\n".join(
        [
            "You are a fast end-to-end driving planner for WOD-E2E.",
            "Use the camera evidence, route intent, ego history, and rater-preference objective.",
            "Do not output private chain-of-thought. Output compact causal fields and trajectory JSON only.",
            "",
            "Required causal fields:",
            "1. critical_objects: visible or likely hazards that constrain the ego path.",
            "2. interaction: why each critical object can affect the next 5 seconds.",
            "3. intent: route command and desired progress.",
            "4. maneuver: one of maintain, slow_yield, stop, nudge_left, nudge_right, evasive_left, evasive_right.",
            "5. trajectory_64wp_10hz: exactly 64 future ego waypoints in ego-frame meters.",
            "",
            "Scoring target:",
            "- Maximize WOD-E2E RFS by matching any human-rated acceptable trajectory.",
            "- Avoid invalid, non-finite, jerky, or collision-prone paths.",
            "- The submitted trajectory will be resampled to 20 waypoints at 4 Hz over 5 seconds.",
            "",
            f"frame_name: {frame.frame_name}",
            f"intent_id: {frame.intent}",
            f"initial_speed_mps: {frame.init_speed_mps:.3f}",
            f"past_trajectory_ego_xy: {json.dumps(_round_trajectory(frame.past_trajectory))}",
            "",
            "Return strict JSON with keys: critical_objects, interaction, intent, maneuver, trajectory_64wp_10hz.",
            f"Model/config: {config.model_name}, output={config.output_waypoints}wp@{config.output_hz:g}Hz.",
        ]
    )


def build_sft_record(frame: WodE2EPreferenceFrame, *, config: Qwen3VlWodConfig | None = None) -> dict[str, Any]:
    config = config or Qwen3VlWodConfig()
    trajectory_64 = upsample_20wp_4hz_to_64wp_10hz(frame.future_trajectory)
    answer = {
        "critical_objects": [],
        "interaction": "Use logged WOD future as supervised trajectory target; visual labels may be added offline.",
        "intent": int(frame.intent),
        "maneuver": "logged_future",
        "trajectory_64wp_10hz": _round_trajectory(trajectory_64),
    }
    return {
        "frame_name": frame.frame_name,
        "model": config.model_name,
        "messages": [
            {"role": "system", "content": "You output compact driving-causation JSON and waypoints."},
            {"role": "user", "content": build_coc_style_prompt(frame, config=config)},
            {"role": "assistant", "content": json.dumps(answer, separators=(",", ":"))},
        ],
        "rfs_references": [
            {"label": reference.label, "score": reference.score, "trajectory": _round_trajectory(reference.trajectory)}
            for reference in frame.references
        ],
        "trajectory_20wp_4hz": _round_trajectory(frame.future_trajectory),
    }


def trajectory_64_to_wod20_from_response(response: str | dict[str, Any]) -> list[tuple[float, float]]:
    payload = json.loads(response) if isinstance(response, str) else response
    trajectory = payload.get("trajectory_64wp_10hz")
    if not isinstance(trajectory, list):
        raise ValueError("response is missing trajectory_64wp_10hz")
    points = [_point_from_json(item) for item in trajectory]
    if len(points) != 64:
        raise ValueError(f"expected 64 trajectory points, got {len(points)}")
    return resample_64wp_10hz_to_20wp_4hz(points)


def upsample_20wp_4hz_to_64wp_10hz(trajectory: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(trajectory) != 20:
        raise ValueError("WOD trajectory must contain 20 waypoints")
    source = [(float(x), float(y)) for x, y in trajectory]
    result: list[tuple[float, float]] = []
    for index in range(64):
        timestamp_s = (index + 1) / 10.0
        source_position = timestamp_s * 4.0 - 1.0
        if source_position <= 0.0:
            result.append(source[0])
        elif source_position >= len(source) - 1:
            result.append(source[-1])
        else:
            lower = int(source_position)
            upper = lower + 1
            fraction = source_position - lower
            result.append(
                (
                    source[lower][0] + (source[upper][0] - source[lower][0]) * fraction,
                    source[lower][1] + (source[upper][1] - source[lower][1]) * fraction,
                )
            )
    return result


def _point_from_json(item: Any) -> tuple[float, float]:
    if isinstance(item, dict):
        return (float(item["x"]), float(item["y"]))
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return (float(item[0]), float(item[1]))
    raise ValueError(f"invalid trajectory point: {item!r}")


def _round_trajectory(trajectory: Sequence[tuple[float, float]]) -> list[list[float]]:
    return [[round(float(x), 4), round(float(y), 4)] for x, y in trajectory]
