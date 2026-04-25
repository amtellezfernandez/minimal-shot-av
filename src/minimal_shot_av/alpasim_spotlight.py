from __future__ import annotations

import json
from typing import Any

import numpy as np
from alpasim_driver.models.base import (
    BaseTrajectoryModel,
    DriveCommand,
    ModelPrediction,
    PredictionInput,
)
from alpasim_driver.schema import ModelConfig

from .environment import Scenario
from .perception import perceive_scene
from .spotlight_reflex import select_maneuver
from .world_model import update_world_state


class SpotlightReflexAlpaSimModel(BaseTrajectoryModel):
    """AlpaSim trajectory-model adapter for the zero-shot Spotlight Reflex policy.

    AlpaSim calls model plugins with camera tensors, route command, speed, acceleration,
    and ego history. This adapter keeps the first bridge deliberately dependency-light:
    it emits the same 20-point, 5-second Spotlight trajectory format in AlpaSim's rig
    frame. Camera-grounded hazard extraction should be layered above this adapter.
    """

    _DEFAULT_CAMERA_IDS = ["camera_front_wide_120fov"]
    _HORIZON_SECONDS = 5.0

    @classmethod
    def from_config(
        cls,
        model_cfg: ModelConfig,
        device: Any,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "SpotlightReflexAlpaSimModel":
        return cls(
            camera_ids=camera_ids,
            context_length=context_length or 1,
            output_frequency_hz=output_frequency_hz,
        )

    def __init__(
        self,
        camera_ids: list[str] | None = None,
        context_length: int = 1,
        output_frequency_hz: int = 4,
    ) -> None:
        self._camera_ids = camera_ids or list(self._DEFAULT_CAMERA_IDS)
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz

    @property
    def camera_ids(self) -> list[str]:
        return self._camera_ids

    @property
    def context_length(self) -> int:
        return self._context_length

    @property
    def output_frequency_hz(self) -> int:
        return self._output_frequency_hz

    def _encode_command(self, command: DriveCommand) -> str:
        return {
            DriveCommand.LEFT: "left",
            DriveCommand.STRAIGHT: "straight",
            DriveCommand.RIGHT: "right",
            DriveCommand.UNKNOWN: "straight",
        }[command]

    def predict(self, prediction_input: PredictionInput) -> ModelPrediction:
        self._validate_cameras(prediction_input.camera_images)
        for camera_id, frames in prediction_input.camera_images.items():
            if len(frames) != self._context_length:
                raise ValueError(
                    f"SpotlightReflexAlpaSimModel expects {self._context_length} "
                    f"frame(s) for {camera_id}, got {len(frames)}"
                )

        command = self._encode_command(prediction_input.command)
        speed_mps = max(0.25, float(prediction_input.speed))
        scenario = _scenario_from_command(command)
        position = scenario.start
        perception = perceive_scene(scenario, position)
        world_state = update_world_state(scenario, position, perception)
        selection = select_maneuver(
            scenario,
            position,
            world_state,
            perception,
            speed_mps=speed_mps,
        )

        trajectory_xy = _resample_to_frequency(
            np.asarray(selection.candidate.trajectory, dtype=np.float32),
            output_frequency_hz=self._output_frequency_hz,
            horizon_seconds=self._HORIZON_SECONDS,
        )
        headings = self._compute_headings_from_trajectory(trajectory_xy)
        reasoning_text = json.dumps(
            {
                "adapter": "minimal_shot_av.spotlight_reflex",
                "command": command,
                "selected_maneuver": selection.candidate.name,
                "candidate_count": selection.candidate_count,
                "reference_count": selection.reference_count,
                "rfs_score": selection.score.combined_score,
                "rfs_3s_score": selection.score.score_3s,
                "rfs_5s_score": selection.score.score_5s,
                "rfs_3s_reference": selection.score.reference_3s_label,
                "rfs_5s_reference": selection.score.reference_5s_label,
                "note": "trajectory-only AlpaSim bridge; camera hazard extraction is a separate layer",
            },
            sort_keys=True,
        )
        return ModelPrediction(
            trajectory_xy=trajectory_xy,
            headings=headings,
            reasoning_text=reasoning_text,
        )


def _scenario_from_command(command: str) -> Scenario:
    lateral_goal = {"left": 16.0, "straight": 0.0, "right": -16.0}[command]
    lane_center = [
        (0.0, 0.0),
        (18.0, lateral_goal * 0.12),
        (38.0, lateral_goal * 0.45),
        (62.0, lateral_goal * 0.82),
        (82.0, lateral_goal),
    ]
    return Scenario(
        width=100.0,
        height=60.0,
        lane_center=lane_center,
        lane_half_width=6.0,
        obstacles=[],
        start=(0.0, 0.0),
        goal=(86.0, lateral_goal),
        seed=0,
        cluster="alpasim_route_command",
        tags={"source": "alpasim_adapter", "route_command": command},
    )


def _resample_to_frequency(
    trajectory_xy: np.ndarray,
    output_frequency_hz: int,
    horizon_seconds: float,
) -> np.ndarray:
    expected_points = max(1, int(round(output_frequency_hz * horizon_seconds)))
    if expected_points == trajectory_xy.shape[0]:
        return trajectory_xy

    source_t = np.linspace(1.0 / trajectory_xy.shape[0], 1.0, trajectory_xy.shape[0])
    target_t = np.linspace(1.0 / expected_points, 1.0, expected_points)
    x = np.interp(target_t, source_t, trajectory_xy[:, 0])
    y = np.interp(target_t, source_t, trajectory_xy[:, 1])
    return np.stack((x, y), axis=1).astype(np.float32)
