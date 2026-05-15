"""Held-out generator sweep with Latin-hypercube profile sampling.

This is intended to be a stronger OOD evaluation than the hand-authored
corridor-pressure grid. It samples generator profiles over a stratified set
of scenario axes, then evaluates matched seeds across all current BC baselines
and Spotlight Reflex.

Output:
  artifacts/heldout_latin_hypercube/results.json
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
import multiprocessing
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.compositional_scenarios import (  # noqa: E402
    AmbientGeometryConfig,
    CompositionalScenarioProfile,
    DEFAULT_COMPOSITIONAL_PROFILE,
    EnvironmentConfig,
    HazardGeometryConfig,
)

N_WORKERS = 20
COLLISION_RISK = 0.90
DEFAULT_SEED_START = 1
DEFAULT_SEED_END = 40
DEFAULT_PROFILE_COUNT = 24
DEFAULT_SUITES = ("gauntlet", "adversarial", "hidden")
SUITE_CASES = {
    "gauntlet": 6,
    "adversarial": 2,
    "compositional": 5,
    "hidden": 1,
}
TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class EvalTask:
    profile_idx: int
    suite: str
    case_idx: int
    seed: int


@dataclass(frozen=True)
class EvalResult:
    profile_idx: int
    suite: str
    continuous_passed: bool
    continuous_collision: bool
    token_passed: bool
    token_collision: bool
    token_rnn_passed: bool | None
    token_rnn_collision: bool | None
    token_transformer_passed: bool | None
    token_transformer_collision: bool | None
    token_transformer_dagger_passed: bool | None
    token_transformer_dagger_collision: bool | None
    token_dagger_passed: bool | None
    token_dagger_collision: bool | None
    spotlight_passed: bool
    spotlight_collision: bool


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=int, default=DEFAULT_PROFILE_COUNT)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-end", type=int, default=DEFAULT_SEED_END)
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suites from: compositional, adversarial, gauntlet, hidden",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "heldout_latin_hypercube" / "results.json",
    )
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts" / "bc_models")
    parser.add_argument("--profile-seed", type=int, default=2027)
    return parser.parse_args()


def _latin_hypercube_unit(n: int, dims: int, rng: random.Random) -> list[list[float]]:
    if n <= 0:
        raise ValueError("n must be positive")
    if dims <= 0:
        raise ValueError("dims must be positive")

    columns: list[list[float]] = []
    for _ in range(dims):
        bins = list(range(n))
        rng.shuffle(bins)
        column = [(b + rng.random()) / n for b in bins]
        columns.append(column)
    return [[columns[d][i] for d in range(dims)] for i in range(n)]


def _interp(x: float, lo: float, hi: float) -> float:
    return lo + x * (hi - lo)


def _scale_visibility(ranges: dict[str, tuple[float, float]], scale: float) -> dict[str, tuple[float, float]]:
    scaled: dict[str, tuple[float, float]] = {}
    for key, (lo, hi) in ranges.items():
        scaled[key] = (max(0.18, lo * scale), min(1.0, hi * scale))
    return scaled


def _scale_dict_values(values: dict[str, float], scale: float) -> dict[str, float]:
    return {key: value * scale for key, value in values.items()}


def _profile_from_sample(idx: int, sample: list[float]) -> CompositionalScenarioProfile:
    if len(sample) != 9:
        raise ValueError(f"expected 9-D sample, got {len(sample)}")

    base = DEFAULT_COMPOSITIONAL_PROFILE
    width_lo = _interp(sample[0], 1.4, 4.8)
    width_span = _interp(sample[1], 0.9, 2.6)
    width_hi = min(6.6, max(width_lo + 0.4, width_lo + width_span))
    gauntlet_pressure = _interp(sample[2], 1.35, 3.9)
    adversarial_pressure = _interp(sample[3], 1.0, 2.7)
    hidden_pressure = _interp(sample[4], 0.95, 1.9)
    visibility_scale = _interp(sample[5], 0.58, 1.0)
    intrusion_scale = _interp(sample[6], 0.55, 1.25)
    ambient_count = int(round(_interp(sample[7], 2.0, 10.0)))
    hidden_hazard_count = 1 + int(math.floor(sample[8] * 3.0))

    environment = replace(
        base.environment,
        visibility_ranges=_scale_visibility(base.environment.visibility_ranges, visibility_scale),
    )
    hazards = replace(
        base.hazards,
        crossing_start_scale=_scale_dict_values(base.hazards.crossing_start_scale, intrusion_scale),
        cut_in_start_scale=_scale_dict_values(base.hazards.cut_in_start_scale, intrusion_scale),
        emergency_longitudinal_offset_m=_scale_dict_values(
            base.hazards.emergency_longitudinal_offset_m,
            intrusion_scale,
        ),
    )

    return replace(
        base,
        name=f"heldout-lhs-{idx:03d}",
        gauntlet_lane_half_width_cap=(round(width_lo, 3), round(width_hi, 3)),
        suite_pressure={
            "compositional": base.suite_pressure["compositional"],
            "adversarial": round(adversarial_pressure, 3),
            "gauntlet": round(gauntlet_pressure, 3),
            "hidden": round(hidden_pressure, 3),
        },
        hazard_counts={
            "compositional": (1,),
            "adversarial": (2, 3, 4),
            "gauntlet": (4,),
            "hidden": (hidden_hazard_count,),
        },
        ambient_base_count=ambient_count,
        hazards=hazards,
        environment=environment,
    )


def build_lhs_profiles(count: int, *, seed: int) -> list[CompositionalScenarioProfile]:
    rng = random.Random(seed)
    samples = _latin_hypercube_unit(count, 9, rng)
    return [_profile_from_sample(idx, sample) for idx, sample in enumerate(samples)]


def _init_worker(model_dir_str: str, profiles: list[CompositionalScenarioProfile]) -> None:
    import numpy as np
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)

    class GeomMLP(nn.Module):
        def __init__(self, out_dim: int, n_features: int = 10, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class GeomGRUTokenBC(nn.Module):
        def __init__(self, n_features: int = 10, n_tokens: int = 9, hidden: int = 160) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(n_features)
            self.gru = nn.GRU(
                input_size=n_features,
                hidden_size=hidden,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, n_tokens),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.norm(x)
            _, h = self.gru(x)
            return self.head(h[-1])

    class GeomTransformerTokenBC(nn.Module):
        def __init__(
            self,
            n_features: int = 10,
            n_tokens: int = 9,
            hidden: int = 192,
            n_heads: int = 4,
            n_layers: int = 2,
            history_len: int = 8,
        ) -> None:
            super().__init__()
            self.input_norm = nn.LayerNorm(n_features)
            self.input_proj = nn.Linear(n_features, hidden)
            self.pos_embed = nn.Parameter(torch.zeros(1, history_len, hidden))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=n_heads,
                dim_feedforward=hidden * 2,
                dropout=0.15,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.head = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden),
                nn.GELU(),
                nn.Dropout(0.15),
                nn.Linear(hidden, n_tokens),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.input_norm(x)
            x = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
            x = self.encoder(x)
            return self.head(x[:, -1, :])

    model_dir = Path(model_dir_str)
    tok_ckpt = torch.load(model_dir / "token_bc.pt", map_location="cpu", weights_only=False)
    cont_ckpt = torch.load(model_dir / "continuous_bc.pt", map_location="cpu", weights_only=False)

    tok_model = GeomMLP(9)
    tok_model.load_state_dict(tok_ckpt["state_dict"])
    tok_model.eval()
    cont_model = GeomMLP(2)
    cont_model.load_state_dict(cont_ckpt["state_dict"])
    cont_model.eval()

    _WORKER["torch"] = torch
    _WORKER["tok_model"] = tok_model
    _WORKER["cont_model"] = cont_model
    _WORKER["tok_feat_mean"] = np.array(tok_ckpt["feat_mean"], dtype=np.float32)
    _WORKER["tok_feat_std"] = np.array(tok_ckpt["feat_std"], dtype=np.float32)
    _WORKER["cont_feat_mean"] = np.array(cont_ckpt["feat_mean"], dtype=np.float32)
    _WORKER["cont_feat_std"] = np.array(cont_ckpt["feat_std"], dtype=np.float32)
    _WORKER["cont_mean"] = np.array(cont_ckpt["cont_mean"], dtype=np.float32)
    _WORKER["cont_std"] = np.array(cont_ckpt["cont_std"], dtype=np.float32)
    _WORKER["profiles"] = profiles

    rnn_path = model_dir / "token_rnn_bc.pt"
    if rnn_path.is_file():
        rnn_ckpt = torch.load(rnn_path, map_location="cpu", weights_only=False)
        rnn_model = GeomGRUTokenBC(
            n_features=int(rnn_ckpt.get("n_features", 10)),
            n_tokens=int(rnn_ckpt.get("n_tokens", 9)),
            hidden=int(rnn_ckpt.get("rnn_hidden", 160)),
        )
        rnn_model.load_state_dict(rnn_ckpt["state_dict"])
        rnn_model.eval()
        _WORKER["rnn_model"] = rnn_model
        _WORKER["rnn_feat_mean"] = np.array(rnn_ckpt["feat_mean"], dtype=np.float32)
        _WORKER["rnn_feat_std"] = np.array(rnn_ckpt["feat_std"], dtype=np.float32)
        _WORKER["rnn_history_len"] = int(rnn_ckpt.get("history_len", 8))
    else:
        _WORKER["rnn_model"] = None

    transformer_path = model_dir / "token_transformer_bc.pt"
    if transformer_path.is_file():
        transformer_ckpt = torch.load(transformer_path, map_location="cpu", weights_only=False)
        transformer_model = GeomTransformerTokenBC(
            n_features=int(transformer_ckpt.get("n_features", 10)),
            n_tokens=int(transformer_ckpt.get("n_tokens", 9)),
            hidden=int(transformer_ckpt.get("transformer_hidden", 192)),
            n_heads=int(transformer_ckpt.get("transformer_heads", 4)),
            n_layers=int(transformer_ckpt.get("transformer_layers", 2)),
            history_len=int(transformer_ckpt.get("history_len", 8)),
        )
        transformer_model.load_state_dict(transformer_ckpt["state_dict"])
        transformer_model.eval()
        _WORKER["transformer_model"] = transformer_model
        _WORKER["transformer_feat_mean"] = np.array(transformer_ckpt["feat_mean"], dtype=np.float32)
        _WORKER["transformer_feat_std"] = np.array(transformer_ckpt["feat_std"], dtype=np.float32)
        _WORKER["transformer_history_len"] = int(transformer_ckpt.get("history_len", 8))
    else:
        _WORKER["transformer_model"] = None

    transformer_dagger_path = model_dir / "token_transformer_dagger_bc.pt"
    if transformer_dagger_path.is_file():
        transformer_dagger_ckpt = torch.load(transformer_dagger_path, map_location="cpu", weights_only=False)
        transformer_dagger_model = GeomTransformerTokenBC(
            n_features=int(transformer_dagger_ckpt.get("n_features", 10)),
            n_tokens=int(transformer_dagger_ckpt.get("n_tokens", 9)),
            hidden=int(transformer_dagger_ckpt.get("transformer_hidden", 192)),
            n_heads=int(transformer_dagger_ckpt.get("transformer_heads", 4)),
            n_layers=int(transformer_dagger_ckpt.get("transformer_layers", 2)),
            history_len=int(transformer_dagger_ckpt.get("history_len", 8)),
        )
        transformer_dagger_model.load_state_dict(transformer_dagger_ckpt["state_dict"])
        transformer_dagger_model.eval()
        _WORKER["transformer_dagger_model"] = transformer_dagger_model
        _WORKER["transformer_dagger_feat_mean"] = np.array(transformer_dagger_ckpt["feat_mean"], dtype=np.float32)
        _WORKER["transformer_dagger_feat_std"] = np.array(transformer_dagger_ckpt["feat_std"], dtype=np.float32)
        _WORKER["transformer_dagger_history_len"] = int(transformer_dagger_ckpt.get("history_len", 8))
    else:
        _WORKER["transformer_dagger_model"] = None

    dagger_path = model_dir / "token_dagger_bc.pt"
    if dagger_path.is_file():
        dagger_ckpt = torch.load(dagger_path, map_location="cpu", weights_only=False)
        dagger_model = GeomMLP(9)
        dagger_model.load_state_dict(dagger_ckpt["state_dict"])
        dagger_model.eval()
        _WORKER["dagger_model"] = dagger_model
        _WORKER["dagger_feat_mean"] = np.array(dagger_ckpt["feat_mean"], dtype=np.float32)
        _WORKER["dagger_feat_std"] = np.array(dagger_ckpt["feat_std"], dtype=np.float32)
    else:
        _WORKER["dagger_model"] = None


def _obs_dict(world_state, perception, speed_mps: float) -> dict[str, float | bool | str]:
    return {
        "obstacle_pressure": world_state.obstacle_pressure,
        "route_blockage": world_state.route_blockage,
        "corridor_blocked": world_state.corridor_blocked,
        "left_clearance": getattr(world_state, "left_clearance", 20.0),
        "right_clearance": getattr(world_state, "right_clearance", 20.0),
        "preferred_escape_side": getattr(world_state, "preferred_escape_side", "balanced"),
        "speed": speed_mps,
        "lane_error": perception.lane_error,
        "uncertainty": max(world_state.uncertainty, perception.uncertainty),
        "min_obstacle_distance": min(
            (getattr(o, "signed_distance", 20.0) for o in perception.visible_obstacles),
            default=20.0,
        ),
    }


def _extract_features(obs: dict[str, float | bool | str]):
    import numpy as np

    escape = {"left": -1.0, "right": 1.0, "balanced": 0.0}.get(
        str(obs.get("preferred_escape_side", "balanced")).lower(),
        0.0,
    )
    return np.array(
        [
            float(obs.get("obstacle_pressure", 0.0)),
            float(obs.get("route_blockage", 0.0)),
            float(obs.get("corridor_blocked", False)),
            min(float(obs.get("left_clearance", 20.0)), 20.0),
            min(float(obs.get("right_clearance", 20.0)), 20.0),
            escape,
            float(obs.get("speed", 0.0)),
            float(obs.get("lane_error", 0.0)),
            float(obs.get("uncertainty", 0.0)),
            min(float(obs.get("min_obstacle_distance", 20.0)), 20.0),
        ],
        dtype=np.float32,
    )


def _is_collision(rollout) -> bool:
    return (
        not rollout.success
        and rollout.steps is not None
        and any(getattr(st, "collision_risk", 0) > COLLISION_RISK for st in rollout.steps)
    )


def _run_task(task: EvalTask) -> EvalResult:
    import numpy as np

    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.planner import PlannedAction
    from minimal_shot_av.simulator.policy import RolloutConfig, _run_rollout, run_spotlight_reflex_policy
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG,
        ManeuverSpec,
        SpotlightReflexConfig,
        _planning_heading,
        generate_maneuver_candidates,
    )

    torch = _WORKER["torch"]
    tok_model = _WORKER["tok_model"]
    cont_model = _WORKER["cont_model"]
    rnn_model = _WORKER["rnn_model"]
    transformer_model = _WORKER["transformer_model"]
    transformer_dagger_model = _WORKER["transformer_dagger_model"]
    dagger_model = _WORKER["dagger_model"]
    tok_feat_mean = _WORKER["tok_feat_mean"]
    tok_feat_std = _WORKER["tok_feat_std"]
    cont_feat_mean = _WORKER["cont_feat_mean"]
    cont_feat_std = _WORKER["cont_feat_std"]
    cont_mean = _WORKER["cont_mean"]
    cont_std = _WORKER["cont_std"]
    rnn_feat_mean = _WORKER.get("rnn_feat_mean")
    rnn_feat_std = _WORKER.get("rnn_feat_std")
    rnn_history_len = int(_WORKER.get("rnn_history_len", 8))
    transformer_feat_mean = _WORKER.get("transformer_feat_mean")
    transformer_feat_std = _WORKER.get("transformer_feat_std")
    transformer_history_len = int(_WORKER.get("transformer_history_len", 8))
    transformer_dagger_feat_mean = _WORKER.get("transformer_dagger_feat_mean")
    transformer_dagger_feat_std = _WORKER.get("transformer_dagger_feat_std")
    transformer_dagger_history_len = int(_WORKER.get("transformer_dagger_history_len", 8))
    dagger_feat_mean = _WORKER.get("dagger_feat_mean")
    dagger_feat_std = _WORKER.get("dagger_feat_std")
    profile = _WORKER["profiles"][task.profile_idx]

    def token_candidate_action(
        active_scenario,
        position,
        world_state,
        perception,
        speed_mps,
        chosen_name: str,
        mode_prefix: str,
    ):
        cfg = DEFAULT_SPOTLIGHT_CONFIG
        heading = _planning_heading(position, world_state, perception, active_scenario, cfg)
        candidates = {c.name: c for c in generate_maneuver_candidates(position, heading, speed_mps, cfg)}
        candidate = candidates.get(chosen_name, candidates.get("maintain"))
        next_pt = candidate.trajectory[cfg.trajectory.action_index]
        vec = (next_pt[0] - position[0], next_pt[1] - position[1])
        dist = math.hypot(*vec)
        norm = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)
        return PlannedAction(direction=norm, speed=dist, mode=f"{mode_prefix}:{chosen_name}", score=0.0), {}

    effective_seed = task.seed + task.case_idx * 1_000 + task.profile_idx * 100_000
    scenario = generate_compositional_scenario(effective_seed, task.suite, profile=profile)

    rs = run_spotlight_reflex_policy(scenario)

    def token_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        feats = _extract_features(_obs_dict(world_state, perception, speed_mps))
        x = torch.from_numpy((feats - tok_feat_mean) / tok_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(tok_model(x).argmax(1).item())
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            TOKEN_ORDER[chosen_idx],
            "token_bc",
        )

    def token_dagger_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if dagger_model is None or dagger_feat_mean is None or dagger_feat_std is None:
            raise RuntimeError("Token-DAgger-BC model is not loaded")
        feats = _extract_features(_obs_dict(world_state, perception, speed_mps))
        x = torch.from_numpy((feats - dagger_feat_mean) / dagger_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(dagger_model(x).argmax(1).item())
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            TOKEN_ORDER[chosen_idx],
            "token_dagger_bc",
        )

    rnn_history: list[np.ndarray] = []
    transformer_history: list[np.ndarray] = []
    transformer_dagger_history: list[np.ndarray] = []

    def token_rnn_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if rnn_model is None or rnn_feat_mean is None or rnn_feat_std is None:
            raise RuntimeError("Token-RNN-BC model is not loaded")
        feats = _extract_features(_obs_dict(world_state, perception, speed_mps))
        norm_feats = ((feats - rnn_feat_mean) / rnn_feat_std).astype(np.float32)
        rnn_history.append(norm_feats)
        history = rnn_history[-rnn_history_len:]
        if len(history) < rnn_history_len:
            history = [history[0]] * (rnn_history_len - len(history)) + history
        x = torch.from_numpy(np.stack(history, axis=0)).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(rnn_model(x).argmax(1).item())
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            TOKEN_ORDER[chosen_idx],
            "token_rnn_bc",
        )

    def token_transformer_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if transformer_model is None or transformer_feat_mean is None or transformer_feat_std is None:
            raise RuntimeError("Token-Transformer-BC model is not loaded")
        feats = _extract_features(_obs_dict(world_state, perception, speed_mps))
        norm_feats = ((feats - transformer_feat_mean) / transformer_feat_std).astype(np.float32)
        transformer_history.append(norm_feats)
        history = transformer_history[-transformer_history_len:]
        if len(history) < transformer_history_len:
            history = [history[0]] * (transformer_history_len - len(history)) + history
        x = torch.from_numpy(np.stack(history, axis=0)).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(transformer_model(x).argmax(1).item())
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            TOKEN_ORDER[chosen_idx],
            "token_transformer_bc",
        )

    def token_transformer_dagger_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if (
            transformer_dagger_model is None
            or transformer_dagger_feat_mean is None
            or transformer_dagger_feat_std is None
        ):
            raise RuntimeError("Token-Transformer-DAgger-BC model is not loaded")
        feats = _extract_features(_obs_dict(world_state, perception, speed_mps))
        norm_feats = ((feats - transformer_dagger_feat_mean) / transformer_dagger_feat_std).astype(np.float32)
        transformer_dagger_history.append(norm_feats)
        history = transformer_dagger_history[-transformer_dagger_history_len:]
        if len(history) < transformer_dagger_history_len:
            history = [history[0]] * (transformer_dagger_history_len - len(history)) + history
        x = torch.from_numpy(np.stack(history, axis=0)).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(transformer_dagger_model(x).argmax(1).item())
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            TOKEN_ORDER[chosen_idx],
            "token_transformer_dagger_bc",
        )

    def continuous_planner(active_scenario, position, world_state, perception, speed_mps, config):
        feats = _extract_features(_obs_dict(world_state, perception, speed_mps))
        x = torch.from_numpy((feats - cont_feat_mean) / cont_feat_std).unsqueeze(0)
        with torch.no_grad():
            pred = cont_model(x).squeeze(0).numpy()
        speed_scale = float(np.clip(pred[0] * cont_std[0] + cont_mean[0], 0.0, 1.2))
        lat_offset = float(np.clip(pred[1] * cont_std[1] + cont_mean[1], -12.0, 12.0))
        temp_spec = ManeuverSpec("continuous_bc", 0.0, speed_scale, lat_offset)
        temp_cfg = SpotlightReflexConfig(
            selector=DEFAULT_SPOTLIGHT_CONFIG.selector,
            references=DEFAULT_SPOTLIGHT_CONFIG.references,
            scoring=DEFAULT_SPOTLIGHT_CONFIG.scoring,
            trajectory=DEFAULT_SPOTLIGHT_CONFIG.trajectory,
            maneuvers=(temp_spec,),
        )
        heading = _planning_heading(position, world_state, perception, active_scenario, temp_cfg)
        candidates = generate_maneuver_candidates(position, heading, speed_mps, temp_cfg)
        if not candidates:
            return PlannedAction(direction=(1.0, 0.0), speed=0.0, mode="continuous_bc:fallback", score=0.0), {}
        next_pt = candidates[0].trajectory[DEFAULT_SPOTLIGHT_CONFIG.trajectory.action_index]
        vec = (next_pt[0] - position[0], next_pt[1] - position[1])
        dist = math.hypot(*vec)
        norm = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)
        return PlannedAction(direction=norm, speed=dist, mode="continuous_bc", score=0.0), {}

    def run_with_planner(scenario_obj, planner_fn):
        def wrapped(active_scenario, position, world_state, perception, config):
            return planner_fn(active_scenario, position, world_state, perception, config.step_size, config)

        cfg = RolloutConfig(spotlight=DEFAULT_SPOTLIGHT_CONFIG)
        return _run_rollout(scenario_obj, wrapped, cfg)

    ra = run_with_planner(scenario, continuous_planner)
    rb = run_with_planner(scenario, token_bc_planner)
    rrnn = run_with_planner(scenario, token_rnn_planner) if rnn_model is not None else None
    rtransformer = run_with_planner(scenario, token_transformer_planner) if transformer_model is not None else None
    rtransformer_dagger = (
        run_with_planner(scenario, token_transformer_dagger_planner) if transformer_dagger_model is not None else None
    )
    rdagger = run_with_planner(scenario, token_dagger_planner) if dagger_model is not None else None

    return EvalResult(
        profile_idx=task.profile_idx,
        suite=task.suite,
        continuous_passed=ra.success,
        continuous_collision=_is_collision(ra),
        token_passed=rb.success,
        token_collision=_is_collision(rb),
        token_rnn_passed=rrnn.success if rrnn is not None else None,
        token_rnn_collision=_is_collision(rrnn) if rrnn is not None else None,
        token_transformer_passed=rtransformer.success if rtransformer is not None else None,
        token_transformer_collision=_is_collision(rtransformer) if rtransformer is not None else None,
        token_transformer_dagger_passed=rtransformer_dagger.success if rtransformer_dagger is not None else None,
        token_transformer_dagger_collision=_is_collision(rtransformer_dagger) if rtransformer_dagger is not None else None,
        token_dagger_passed=rdagger.success if rdagger is not None else None,
        token_dagger_collision=_is_collision(rdagger) if rdagger is not None else None,
        spotlight_passed=rs.success,
        spotlight_collision=_is_collision(rs),
    )


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = (z / d) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, c - h), min(1.0, c + h))


def _agent_rows(results: list[EvalResult]) -> dict[str, dict[str, Any]]:
    rows = {
        "Continuous-BC": {
            "passed": [r.continuous_passed for r in results],
            "collision": [r.continuous_collision for r in results],
        },
        "Token-BC": {
            "passed": [r.token_passed for r in results],
            "collision": [r.token_collision for r in results],
        },
        "Spotlight Reflex": {
            "passed": [r.spotlight_passed for r in results],
            "collision": [r.spotlight_collision for r in results],
        },
    }

    if results and all(r.token_rnn_passed is not None for r in results):
        rows["Token-RNN-BC"] = {
            "passed": [bool(r.token_rnn_passed) for r in results],
            "collision": [bool(r.token_rnn_collision) for r in results],
        }
    if results and all(r.token_transformer_passed is not None for r in results):
        rows["Token-Transformer-BC"] = {
            "passed": [bool(r.token_transformer_passed) for r in results],
            "collision": [bool(r.token_transformer_collision) for r in results],
        }
    if results and all(r.token_transformer_dagger_passed is not None for r in results):
        rows["Token-Transformer-DAgger-BC"] = {
            "passed": [bool(r.token_transformer_dagger_passed) for r in results],
            "collision": [bool(r.token_transformer_dagger_collision) for r in results],
        }
    if results and all(r.token_dagger_passed is not None for r in results):
        rows["Token-DAgger-BC"] = {
            "passed": [bool(r.token_dagger_passed) for r in results],
            "collision": [bool(r.token_dagger_collision) for r in results],
        }
    return rows


def _summarize_agent(passed: list[bool], collision: list[bool]) -> dict[str, Any]:
    n = len(passed)
    pass_count = sum(passed)
    coll_count = sum(collision)
    p_lo, p_hi = _wilson_ci(pass_count, n)
    c_lo, c_hi = _wilson_ci(coll_count, n)
    return {
        "passes": pass_count,
        "collisions": coll_count,
        "total": n,
        "pass_rate": round(pass_count / n * 100, 2),
        "collision_rate": round(coll_count / n * 100, 2),
        "pass_ci95": [round(p_lo * 100, 2), round(p_hi * 100, 2)],
        "coll_ci95": [round(c_lo * 100, 2), round(c_hi * 100, 2)],
    }


def _paired_vs_spotlight(agent_passed: list[bool], agent_collision: list[bool], spot_passed: list[bool], spot_collision: list[bool]) -> dict[str, Any]:
    n = len(agent_passed)
    agent_only_pass = sum(a and not s for a, s in zip(agent_passed, spot_passed))
    spotlight_only_pass = sum(s and not a for a, s in zip(agent_passed, spot_passed))
    agent_only_collision = sum(a and not s for a, s in zip(agent_collision, spot_collision))
    spotlight_only_collision = sum(s and not a for a, s in zip(agent_collision, spot_collision))
    pass_delta = (sum(agent_passed) - sum(spot_passed)) / n * 100.0
    coll_delta = (sum(agent_collision) - sum(spot_collision)) / n * 100.0
    return {
        "n": n,
        "agent_only_pass": agent_only_pass,
        "spotlight_only_pass": spotlight_only_pass,
        "agent_only_collision": agent_only_collision,
        "spotlight_only_collision": spotlight_only_collision,
        "pass_delta_pp": round(pass_delta, 2),
        "collision_delta_pp": round(coll_delta, 2),
    }


def _summarize_bucket(results: list[EvalResult]) -> dict[str, Any]:
    rows = _agent_rows(results)
    spotlight = rows["Spotlight Reflex"]
    summary: dict[str, Any] = {"agents": {}}
    for name, row in rows.items():
        summary["agents"][name] = _summarize_agent(row["passed"], row["collision"])
    summary["paired_vs_spotlight"] = {
        name: _paired_vs_spotlight(row["passed"], row["collision"], spotlight["passed"], spotlight["collision"])
        for name, row in rows.items()
        if name != "Spotlight Reflex"
    }
    return summary


def _profile_axes(profile: CompositionalScenarioProfile) -> dict[str, Any]:
    return {
        "gauntlet_lane_half_width_cap": list(profile.gauntlet_lane_half_width_cap),
        "suite_pressure": dict(profile.suite_pressure),
        "ambient_base_count": profile.ambient_base_count,
        "hazard_counts": {key: list(value) for key, value in profile.hazard_counts.items()},
        "visibility_ranges": {key: list(value) for key, value in profile.environment.visibility_ranges.items()},
        "crossing_start_scale": dict(profile.hazards.crossing_start_scale),
        "cut_in_start_scale": dict(profile.hazards.cut_in_start_scale),
        "emergency_longitudinal_offset_m": dict(profile.hazards.emergency_longitudinal_offset_m),
    }


def main() -> None:
    args = _parse_args()
    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    invalid = [suite for suite in suites if suite not in SUITE_CASES]
    if invalid:
        raise ValueError(f"invalid suites: {invalid}")
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")

    profiles = build_lhs_profiles(args.profiles, seed=args.profile_seed)
    tasks = [
        EvalTask(profile_idx=profile_idx, suite=suite, case_idx=case_idx, seed=seed)
        for profile_idx in range(len(profiles))
        for suite in suites
        for case_idx in range(SUITE_CASES[suite])
        for seed in range(args.seed_start, args.seed_end + 1)
    ]

    print(f"Held-out LHS profiles: {len(profiles)}")
    print(f"Suites: {', '.join(suites)}")
    print(f"Tasks: {len(tasks)} scenarios × 5 agents")

    with multiprocessing.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(str(args.model_dir), profiles),
    ) as pool:
        results = pool.map(_run_task, tasks)

    overall = _summarize_bucket(results)
    by_suite = {
        suite: _summarize_bucket([result for result in results if result.suite == suite])
        for suite in suites
    }
    by_profile = []
    for profile_idx, profile in enumerate(profiles):
        profile_results = [result for result in results if result.profile_idx == profile_idx]
        bucket = _summarize_bucket(profile_results)
        by_profile.append(
            {
                "profile_idx": profile_idx,
                "name": profile.name,
                "axes": _profile_axes(profile),
                "summary": bucket,
            }
        )

    output = {
        "schema": "heldout_latin_hypercube_v1",
        "config": {
            "profiles": len(profiles),
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "suites": list(suites),
            "workers": args.workers,
            "profile_seed": args.profile_seed,
            "model_dir": str(args.model_dir),
        },
        "overall": overall,
        "by_suite": by_suite,
        "profiles": by_profile,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print("\nOverall:")
    for name, metrics in overall["agents"].items():
        print(
            f"  {name:<18} pass={metrics['pass_rate']:.2f}% "
            f"coll={metrics['collision_rate']:.2f}%"
        )
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
