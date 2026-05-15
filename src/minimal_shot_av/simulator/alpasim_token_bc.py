from __future__ import annotations

import json
import math
import os
from dataclasses import replace
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - exercised only in non-alpasim installs.
    torch = None
    nn = None

from .alpasim_signal import extract_alpasim_signal, scenario_from_command
from .alpasim_spotlight import BaseTrajectoryModel, DriveCommand, ModelPrediction, PredictionInput, _resample_to_frequency
from .environment import scenario_at_tick
from .perception import perceive_scene
from .spotlight_reflex import DEFAULT_SPOTLIGHT_CONFIG, evaluate_maneuver_candidates, generate_maneuver_candidates, _planning_heading
from .world_model import update_world_state


N_FEATURES = 10
N_TOKENS = 9
HIDDEN = 256
DROPOUT = 0.15
TRAJECTORY_MODES = ("token", "longitudinal_only", "clamped_lateral")
SELECTION_MODES = ("argmax", "hybrid_veto")
TOKEN_ORDER = (
    "stop",
    "crawl",
    "maintain",
    "slow_yield",
    "nudge_left",
    "nudge_right",
    "evasive_left",
    "evasive_right",
    "lane_recover",
)


class _GeomMLP(nn.Module if nn is not None else object):
    def __init__(self, out_dim: int = N_TOKENS) -> None:
        if nn is None:
            raise ImportError("TokenBCAlpaSimModel requires torch; install with the alpasim extra.")
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(N_FEATURES),
            nn.Linear(N_FEATURES, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.GELU(),
            nn.Linear(HIDDEN // 2, out_dim),
        )

    def forward(self, x: Any) -> Any:
        return self.net(x)


class TokenBCAlpaSimModel(BaseTrajectoryModel):
    """AlpaSim adapter for trained token BC/DAgger checkpoints.

    Production configs default to CUDA so a learned AlpaSim run cannot silently fall back
    to CPU. Tests may still pass ``device="cpu"`` explicitly with a toy checkpoint.
    """

    _DEFAULT_CAMERA_IDS = ["camera_front_wide_120fov"]
    _HORIZON_SECONDS = 5.0

    @classmethod
    def from_config(
        cls,
        model_cfg: Any,
        device: Any,
        camera_ids: list[str],
        context_length: int | None,
        output_frequency_hz: int,
    ) -> "TokenBCAlpaSimModel":
        checkpoint_path = _cfg_value(model_cfg, "checkpoint_path", None)
        if not checkpoint_path:
            raise ValueError("TokenBCAlpaSimModel requires model.checkpoint_path")
        cfg_device = _cfg_value(model_cfg, "device", None)
        requested_device = str(cfg_device if cfg_device is not None else "cuda")
        trajectory_mode = os.getenv("MSA_TOKENBC_TRAJECTORY_MODE", str(_cfg_value(model_cfg, "trajectory_mode", "token")))
        max_lateral_offset_m = float(
            os.getenv("MSA_TOKENBC_MAX_LATERAL_OFFSET_M", str(_cfg_value(model_cfg, "max_lateral_offset_m", 2.0)))
        )
        selection_mode = os.getenv("MSA_TOKENBC_SELECTION_MODE", str(_cfg_value(model_cfg, "selection_mode", "argmax")))
        hybrid_top_k = int(os.getenv("MSA_TOKENBC_HYBRID_TOP_K", str(_cfg_value(model_cfg, "hybrid_top_k", 3))))
        hybrid_geo_weight = float(
            os.getenv("MSA_TOKENBC_HYBRID_GEOMETRIC_WEIGHT", str(_cfg_value(model_cfg, "hybrid_geometric_weight", 0.75)))
        )
        hybrid_policy_temperature = float(
            os.getenv(
                "MSA_TOKENBC_HYBRID_POLICY_TEMPERATURE",
                str(_cfg_value(model_cfg, "hybrid_policy_temperature", 1.0)),
            )
        )
        hybrid_veto_margin = float(
            os.getenv("MSA_TOKENBC_HYBRID_VETO_MARGIN", str(_cfg_value(model_cfg, "hybrid_veto_margin", 8.0)))
        )
        hybrid_max_geometric_rank = int(
            os.getenv(
                "MSA_TOKENBC_HYBRID_MAX_GEOMETRIC_RANK",
                str(_cfg_value(model_cfg, "hybrid_max_geometric_rank", 2)),
            )
        )
        selection_log_path = os.getenv("MSA_TOKENBC_SELECTION_LOG_PATH", str(_cfg_value(model_cfg, "selection_log_path", "")))
        return cls(
            checkpoint_path=checkpoint_path,
            device=requested_device,
            camera_ids=camera_ids,
            context_length=context_length or 1,
            output_frequency_hz=output_frequency_hz,
            trajectory_mode=trajectory_mode,
            max_lateral_offset_m=max_lateral_offset_m,
            selection_mode=selection_mode,
            hybrid_top_k=hybrid_top_k,
            hybrid_geometric_weight=hybrid_geo_weight,
            hybrid_policy_temperature=hybrid_policy_temperature,
            hybrid_veto_margin=hybrid_veto_margin,
            hybrid_max_geometric_rank=hybrid_max_geometric_rank,
            selection_log_path=selection_log_path or None,
        )

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        device: str = "cuda",
        camera_ids: list[str] | None = None,
        context_length: int = 1,
        output_frequency_hz: int = 4,
        trajectory_mode: str = "token",
        max_lateral_offset_m: float = 2.0,
        selection_mode: str = "argmax",
        hybrid_top_k: int = 3,
        hybrid_geometric_weight: float = 0.75,
        hybrid_policy_temperature: float = 1.0,
        hybrid_veto_margin: float = 8.0,
        hybrid_max_geometric_rank: int = 2,
        selection_log_path: str | Path | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("TokenBCAlpaSimModel requires torch; install with the alpasim extra.")
        self._camera_ids = camera_ids or list(self._DEFAULT_CAMERA_IDS)
        self._context_length = context_length
        self._output_frequency_hz = output_frequency_hz
        self._checkpoint_path = str(checkpoint_path)
        self._device = _resolve_device(device)
        self._trajectory_mode = _resolve_trajectory_mode(trajectory_mode)
        self._max_lateral_offset_m = max(0.0, float(max_lateral_offset_m))
        self._selection_mode = _resolve_selection_mode(selection_mode)
        self._hybrid_top_k = max(1, int(hybrid_top_k))
        self._hybrid_geometric_weight = float(hybrid_geometric_weight)
        self._hybrid_policy_temperature = max(1e-3, float(hybrid_policy_temperature))
        self._hybrid_veto_margin = max(0.0, float(hybrid_veto_margin))
        self._hybrid_max_geometric_rank = max(1, int(hybrid_max_geometric_rank))
        self._selection_log_path = Path(selection_log_path).resolve() if selection_log_path else None
        self._selection_log_lock = Lock()
        self._prediction_counter = 0
        self._model, self._feat_mean, self._feat_std, self._token_order = _load_checkpoint(
            Path(checkpoint_path),
            device=self._device,
        )

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
                    f"TokenBCAlpaSimModel expects {self._context_length} frame(s) "
                    f"for {camera_id}, got {len(frames)}"
                )

        command = self._encode_command(prediction_input.command)
        speed_mps = max(0.25, float(prediction_input.speed))
        alpasim_signal = extract_alpasim_signal(prediction_input)
        scenario = scenario_from_command(command, alpasim_signal)
        active_scenario = scenario_at_tick(scenario, 0)
        position = active_scenario.start
        perception = perceive_scene(active_scenario, position)
        world_state = update_world_state(active_scenario, position, perception)

        features = _extract_features(world_state, perception, speed_mps)
        norm_features = ((features - self._feat_mean) / self._feat_std).astype(np.float32)
        with torch.no_grad():
            x = torch.from_numpy(norm_features).unsqueeze(0).to(self._device)
            logits = self._model(x).squeeze(0).detach().cpu().numpy()

        heading = _planning_heading(position, world_state, perception, active_scenario, DEFAULT_SPOTLIGHT_CONFIG)
        adapter_config = _adapter_spotlight_config(
            trajectory_mode=self._trajectory_mode,
            max_lateral_offset_m=self._max_lateral_offset_m,
        )
        candidates = _generate_adapter_candidates(
            position,
            heading,
            speed_mps,
            config=adapter_config,
        )
        spotlight_evaluations, reference_count = evaluate_maneuver_candidates(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps=speed_mps,
            config=adapter_config,
        )
        selection_info = _select_token_with_mode(
            logits=logits,
            token_order=self._token_order,
            evaluations=spotlight_evaluations,
            selection_mode=self._selection_mode,
            hybrid_top_k=self._hybrid_top_k,
            hybrid_geometric_weight=self._hybrid_geometric_weight,
            hybrid_policy_temperature=self._hybrid_policy_temperature,
            hybrid_veto_margin=self._hybrid_veto_margin,
            hybrid_max_geometric_rank=self._hybrid_max_geometric_rank,
        )
        chosen_token = str(selection_info["hybrid_token"])
        candidate = candidates.get(chosen_token) or candidates["maintain"]
        trajectory_xy = _resample_to_frequency(
            np.asarray(candidate.trajectory, dtype=np.float32),
            output_frequency_hz=self._output_frequency_hz,
            horizon_seconds=self._HORIZON_SECONDS,
        )
        headings = self._compute_headings_from_trajectory(trajectory_xy)
        reasoning_text = json.dumps(
            {
                "adapter": "minimal_shot_av.simulator.alpasim_token_bc",
                "checkpoint_path": self._checkpoint_path,
                "command": command,
                "selected_maneuver": chosen_token,
                "selection_mode": self._selection_mode,
                "trajectory_mode": self._trajectory_mode,
                "max_lateral_offset_m": self._max_lateral_offset_m,
                "top_logits": _top_logits(logits, self._token_order),
                "spotlight_selected_maneuver": selection_info["spotlight_token"],
                "selection_trace": selection_info,
                "reference_count": reference_count,
                "top_candidate_summaries": [
                    evaluation.explanation.to_summary()
                    for evaluation in sorted(
                        spotlight_evaluations,
                        key=lambda item: item.explanation.effective_score,
                        reverse=True,
                    )[:3]
                ],
                "obstacle_pressure": world_state.obstacle_pressure,
                "route_blockage": world_state.route_blockage,
                "corridor_blocked": world_state.corridor_blocked,
                "left_clearance": world_state.left_clearance,
                "right_clearance": world_state.right_clearance,
                "preferred_escape_side": world_state.preferred_escape_side,
                "alpasim_signal": alpasim_signal,
            },
            sort_keys=True,
        )
        self._append_selection_log(
            prediction_input=prediction_input,
            command=command,
            speed_mps=speed_mps,
            selection_info=selection_info,
            logits=logits,
            spotlight_evaluations=spotlight_evaluations,
            alpasim_signal=alpasim_signal,
        )
        return ModelPrediction(trajectory_xy=trajectory_xy, headings=headings, reasoning_text=reasoning_text)

    def _append_selection_log(
        self,
        *,
        prediction_input: PredictionInput,
        command: str,
        speed_mps: float,
        selection_info: dict[str, Any],
        logits: np.ndarray,
        spotlight_evaluations: list[Any],
        alpasim_signal: dict[str, Any],
    ) -> None:
        if self._selection_log_path is None:
            return
        self._selection_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._prediction_counter += 1
        record = {
            "frame_index": self._prediction_counter,
            "scene_id": _prediction_scene_id(prediction_input),
            "command": command,
            "speed_mps": round(float(speed_mps), 4),
            "selection_mode": self._selection_mode,
            "trajectory_mode": self._trajectory_mode,
            "max_lateral_offset_m": self._max_lateral_offset_m,
            "hybrid_veto_margin": self._hybrid_veto_margin,
            "hybrid_max_geometric_rank": self._hybrid_max_geometric_rank,
            "dagger_argmax_token": selection_info["dagger_argmax_token"],
            "spotlight_token": selection_info["spotlight_token"],
            "hybrid_token": selection_info["hybrid_token"],
            "dagger_argmax_vetoed": selection_info["dagger_argmax_vetoed"],
            "used_fallback_geometric": selection_info["used_fallback_geometric"],
            "hybrid_matches_dagger": selection_info["hybrid_matches_dagger"],
            "hybrid_matches_spotlight": selection_info["hybrid_matches_spotlight"],
            "decision_type": selection_info["decision_type"],
            "dagger_topk_tokens": selection_info["dagger_topk_tokens"],
            "dagger_argmax_geo_gap": selection_info["dagger_argmax_geo_gap"],
            "dagger_argmax_geo_rank": selection_info["dagger_argmax_geo_rank"],
            "veto_margin": selection_info["veto_margin"],
            "max_geometric_rank": selection_info["max_geometric_rank"],
            "veto_reason": selection_info["veto_reason"],
            "vetoed_tokens": selection_info["vetoed_tokens"],
            "top_logits": _top_logits(logits, self._token_order),
            "spotlight_top_candidates": [
                evaluation.explanation.to_summary()
                for evaluation in sorted(
                    spotlight_evaluations,
                    key=lambda item: item.explanation.effective_score,
                    reverse=True,
                )[:3]
            ],
            "alpasim_signal": alpasim_signal,
        }
        with self._selection_log_lock:
            with self._selection_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")


def _load_checkpoint(path: Path, *, device: str) -> tuple[_GeomMLP, np.ndarray, np.ndarray, tuple[str, ...]]:
    if not path.is_file():
        raise FileNotFoundError(f"Token BC checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    token_order = tuple(payload.get("token_names", TOKEN_ORDER))
    if len(token_order) != N_TOKENS or len(set(token_order)) != N_TOKENS:
        raise ValueError(f"expected {N_TOKENS} unique token names, got {list(token_order)}")
    expected_names = set(TOKEN_ORDER)
    if set(token_order) != expected_names:
        missing = sorted(expected_names - set(token_order))
        extra = sorted(set(token_order) - expected_names)
        raise ValueError(f"checkpoint token_names do not match adapter candidates; missing={missing}, extra={extra}")
    model = _GeomMLP(len(token_order)).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    feat_mean = np.asarray(payload["feat_mean"], dtype=np.float32)
    feat_std = np.asarray(payload["feat_std"], dtype=np.float32)
    if feat_mean.shape != (N_FEATURES,) or feat_std.shape != (N_FEATURES,):
        raise ValueError(f"expected feature normalizers of shape {(N_FEATURES,)}, got {feat_mean.shape}/{feat_std.shape}")
    return model, feat_mean, np.maximum(feat_std, 1e-6), token_order


def _resolve_trajectory_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in TRAJECTORY_MODES:
        raise ValueError(f"unknown trajectory_mode={mode!r}; expected one of {TRAJECTORY_MODES}")
    return normalized


def _resolve_selection_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SELECTION_MODES:
        raise ValueError(f"unknown selection_mode={mode!r}; expected one of {SELECTION_MODES}")
    return normalized


def _adapter_spotlight_config(*, trajectory_mode: str, max_lateral_offset_m: float) -> Any:
    if trajectory_mode == "token":
        return DEFAULT_SPOTLIGHT_CONFIG
    adjusted_maneuvers = []
    for spec in DEFAULT_SPOTLIGHT_CONFIG.maneuvers:
        lateral_offset = spec.lateral_offset_m
        if trajectory_mode == "longitudinal_only":
            lateral_offset = 0.0
        elif trajectory_mode == "clamped_lateral":
            lateral_offset = float(np.clip(lateral_offset, -max_lateral_offset_m, max_lateral_offset_m))
        adjusted_maneuvers.append(replace(spec, lateral_offset_m=lateral_offset))
    return replace(DEFAULT_SPOTLIGHT_CONFIG, maneuvers=tuple(adjusted_maneuvers))


def _generate_adapter_candidates(
    position: tuple[float, float],
    heading: tuple[float, float],
    speed_mps: float,
    *,
    config: Any,
) -> dict[str, Any]:
    return {
        candidate.name: candidate
        for candidate in generate_maneuver_candidates(
            position,
            heading,
            speed_mps,
            config,
        )
    }


def _resolve_device(device: str) -> str:
    requested = str(device).strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("CUDA is unavailable; pass device='cpu' explicitly only for local smoke tests.")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for TokenBCAlpaSimModel but is not available.")
    return requested


def _extract_features(world_state: Any, perception: Any, speed_mps: float) -> np.ndarray:
    escape = {"left": -1.0, "right": 1.0, "balanced": 0.0}.get(
        str(getattr(world_state, "preferred_escape_side", "balanced")).lower(),
        0.0,
    )
    min_obstacle_distance = min(
        (
            float(getattr(obstacle, "signed_distance", 20.0))
            for obstacle in getattr(perception, "visible_obstacles", [])
        ),
        default=20.0,
    )
    return np.array(
        [
            float(getattr(world_state, "obstacle_pressure", 0.0)),
            float(getattr(world_state, "route_blockage", 0.0)),
            float(getattr(world_state, "corridor_blocked", False)),
            min(float(getattr(world_state, "left_clearance", 20.0)), 20.0),
            min(float(getattr(world_state, "right_clearance", 20.0)), 20.0),
            escape,
            float(speed_mps),
            float(getattr(perception, "lane_error", 0.0)),
            float(max(getattr(world_state, "uncertainty", 0.0), getattr(perception, "uncertainty", 0.0))),
            min(min_obstacle_distance, 20.0),
        ],
        dtype=np.float32,
    )


def _select_token_with_mode(
    *,
    logits: np.ndarray,
    token_order: tuple[str, ...],
    evaluations: list[Any],
    selection_mode: str,
    hybrid_top_k: int,
    hybrid_geometric_weight: float,
    hybrid_policy_temperature: float,
    hybrid_veto_margin: float,
    hybrid_max_geometric_rank: int,
) -> dict[str, Any]:
    raw_idx = int(np.argmax(logits))
    raw_token = str(token_order[raw_idx])
    spotlight_eval_by_name = {evaluation.candidate.name: evaluation for evaluation in evaluations}
    safe_global = [
        evaluation
        for evaluation in evaluations
        if evaluation.explanation.safety_penalty <= 0.0
    ]
    safe_ordered = sorted(
        safe_global if safe_global else evaluations,
        key=lambda item: (item.explanation.effective_score, item.candidate.confidence),
        reverse=True,
    )
    geometric_ranks = {evaluation.candidate.name: rank + 1 for rank, evaluation in enumerate(safe_ordered)}
    best_geometric_score = float(safe_ordered[0].explanation.effective_score)
    spotlight_best = max(
        evaluations,
        key=lambda item: (item.explanation.effective_score, item.candidate.confidence),
    )
    spotlight_token = str(spotlight_best.candidate.name)
    raw_evaluation = spotlight_eval_by_name.get(raw_token)
    if raw_evaluation is None:
        raise ValueError(f"checkpoint selected token {raw_token!r}, but no matching candidate exists")
    if selection_mode == "argmax":
        return _selection_record(
            token_order=token_order,
            raw_idx=raw_idx,
            chosen_idx=raw_idx,
            spotlight_token=spotlight_token,
            topk_indices=[raw_idx],
            safe_topk_indices=[raw_idx] if raw_evaluation.explanation.safety_penalty <= 0.0 else [],
            used_fallback_geometric=False,
            dagger_argmax_vetoed=False,
            hybrid_policy_scores={raw_token: 0.0},
            hybrid_geometric_scores={raw_token: float(raw_evaluation.explanation.effective_score)},
            dagger_argmax_geo_gap=max(0.0, best_geometric_score - float(raw_evaluation.explanation.effective_score)),
            dagger_argmax_geo_rank=int(geometric_ranks.get(raw_token, len(safe_ordered) + 1)),
            veto_margin=hybrid_veto_margin,
            max_geometric_rank=hybrid_max_geometric_rank,
            veto_reason="none",
            vetoed_tokens=[],
        )

    scaled_logits = logits.astype(np.float64) / hybrid_policy_temperature
    policy_log_probs = scaled_logits - _logsumexp(scaled_logits)
    sorted_indices = list(np.argsort(logits)[::-1])
    topk_indices = sorted_indices[: max(1, min(hybrid_top_k, len(sorted_indices)))]
    safe_topk_indices: list[int] = []
    vetoed_tokens: list[dict[str, Any]] = []
    raw_veto_reason = "none"
    for idx in topk_indices:
        token = str(token_order[idx])
        evaluation = spotlight_eval_by_name.get(token)
        if evaluation is None:
            vetoed_tokens.append({"token": token, "reason": "missing_evaluation"})
            if idx == raw_idx:
                raw_veto_reason = "missing_evaluation"
            continue
        if evaluation.explanation.safety_penalty > 0.0:
            vetoed_tokens.append({"token": token, "reason": "unsafe_action"})
            if idx == raw_idx:
                raw_veto_reason = "unsafe_action"
            continue
        geo_score = float(evaluation.explanation.effective_score)
        geo_gap = max(0.0, best_geometric_score - geo_score)
        geo_rank = int(geometric_ranks.get(token, len(safe_ordered) + 1))
        if geo_gap > hybrid_veto_margin:
            vetoed_tokens.append({"token": token, "reason": "geometric_gap", "geo_gap": round(geo_gap, 4)})
            if idx == raw_idx:
                raw_veto_reason = "geometric_gap"
            continue
        if geo_rank > hybrid_max_geometric_rank:
            vetoed_tokens.append({"token": token, "reason": "geometric_rank", "geo_rank": geo_rank})
            if idx == raw_idx:
                raw_veto_reason = "geometric_rank"
            continue
        safe_topk_indices.append(idx)
    used_fallback_geometric = False
    dagger_argmax_vetoed = raw_idx not in safe_topk_indices
    geometric_scores = {
        evaluation.candidate.name: float(evaluation.explanation.effective_score) for evaluation in evaluations
    }

    if safe_topk_indices:
        geo_values = np.array([geometric_scores[str(token_order[idx])] for idx in safe_topk_indices], dtype=np.float64)
        geo_norm = (geo_values - geo_values.mean()) / max(float(geo_values.std()), 1e-6)
        combined = np.array([policy_log_probs[idx] for idx in safe_topk_indices], dtype=np.float64) + hybrid_geometric_weight * geo_norm
        chosen_idx = int(safe_topk_indices[int(np.argmax(combined))])
    else:
        fallback_eval = max(
            safe_global if safe_global else evaluations,
            key=lambda item: (item.explanation.effective_score, item.candidate.confidence),
        )
        chosen_idx = int(token_order.index(fallback_eval.candidate.name))
        used_fallback_geometric = True

    return _selection_record(
        token_order=token_order,
        raw_idx=raw_idx,
        chosen_idx=chosen_idx,
        spotlight_token=spotlight_token,
        topk_indices=topk_indices,
        safe_topk_indices=safe_topk_indices,
        used_fallback_geometric=used_fallback_geometric,
        dagger_argmax_vetoed=dagger_argmax_vetoed,
        hybrid_policy_scores={str(token_order[idx]): round(float(policy_log_probs[idx]), 4) for idx in topk_indices},
        hybrid_geometric_scores={str(token_order[idx]): round(float(geometric_scores[str(token_order[idx])]), 4) for idx in topk_indices},
        dagger_argmax_geo_gap=max(0.0, best_geometric_score - float(geometric_scores[raw_token])),
        dagger_argmax_geo_rank=int(geometric_ranks.get(raw_token, len(safe_ordered) + 1)),
        veto_margin=hybrid_veto_margin,
        max_geometric_rank=hybrid_max_geometric_rank,
        veto_reason=raw_veto_reason,
        vetoed_tokens=vetoed_tokens,
    )


def _selection_record(
    *,
    token_order: tuple[str, ...],
    raw_idx: int,
    chosen_idx: int,
    spotlight_token: str,
    topk_indices: list[int],
    safe_topk_indices: list[int],
    used_fallback_geometric: bool,
    dagger_argmax_vetoed: bool,
    hybrid_policy_scores: dict[str, float],
    hybrid_geometric_scores: dict[str, float],
    dagger_argmax_geo_gap: float,
    dagger_argmax_geo_rank: int,
    veto_margin: float,
    max_geometric_rank: int,
    veto_reason: str,
    vetoed_tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    dagger_token = str(token_order[raw_idx])
    hybrid_token = str(token_order[chosen_idx])
    if used_fallback_geometric:
        decision_type = "fallback_geometric"
    elif hybrid_token == dagger_token == spotlight_token:
        decision_type = "agreement"
    elif hybrid_token == dagger_token:
        decision_type = "dagger_wins"
    elif hybrid_token == spotlight_token:
        decision_type = "spotlight_wins"
    else:
        decision_type = "compromise"
    return {
        "dagger_argmax_token": dagger_token,
        "spotlight_token": spotlight_token,
        "hybrid_token": hybrid_token,
        "dagger_topk_tokens": [str(token_order[idx]) for idx in topk_indices],
        "safe_topk_tokens": [str(token_order[idx]) for idx in safe_topk_indices],
        "dagger_argmax_vetoed": bool(dagger_argmax_vetoed),
        "used_fallback_geometric": bool(used_fallback_geometric),
        "hybrid_matches_dagger": hybrid_token == dagger_token,
        "hybrid_matches_spotlight": hybrid_token == spotlight_token,
        "decision_type": decision_type,
        "hybrid_policy_scores": hybrid_policy_scores,
        "hybrid_geometric_scores": hybrid_geometric_scores,
        "dagger_argmax_geo_gap": round(float(dagger_argmax_geo_gap), 4),
        "dagger_argmax_geo_rank": int(dagger_argmax_geo_rank),
        "veto_margin": round(float(veto_margin), 4),
        "max_geometric_rank": int(max_geometric_rank),
        "veto_reason": veto_reason,
        "vetoed_tokens": list(vetoed_tokens),
    }


def _prediction_scene_id(prediction_input: Any) -> str | None:
    for field_name in ("scene_id", "clip_id", "clipgt_id"):
        value = getattr(prediction_input, field_name, None)
        if value:
            return str(value)
    metadata = getattr(prediction_input, "session_metadata", None)
    scene_id = getattr(metadata, "scene_id", None)
    if scene_id:
        return str(scene_id)
    return None


def _logsumexp(values: np.ndarray) -> float:
    max_value = float(np.max(values))
    return max_value + float(np.log(np.exp(values - max_value).sum()))


def _top_logits(logits: np.ndarray, token_order: tuple[str, ...], limit: int = 3) -> list[dict[str, float | str]]:
    indices = sorted(range(len(logits)), key=lambda idx: float(logits[idx]), reverse=True)[:limit]
    return [{"token": token_order[idx], "logit": round(float(logits[idx]), 4)} for idx in indices]


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return getattr(config, key, default)
