"""Phase diagram: corridor tightness × obstacle pressure → BC-vs-Spotlight collision delta.

Sweeps two independent axes to find the operating envelope where BC collapses
while Spotlight Reflex holds at 0% collision:

  X: corridor tightness — gauntlet_lane_half_width_cap shrinks from wide→extreme
  Y: obstacle pressure  — suite_pressure["gauntlet"] scales from baseline→2×

Each cell runs 5 agents (Continuous-BC, Token-BC, Token-RNN-BC,
Token-DAgger-BC, Spotlight) on 6 gauntlet cases × N_SEEDS seeds =
6·N_SEEDS rollouts per agent per cell.

Output: artifacts/stress_phase/results.json
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

N_SEEDS   = 120          # per case per cell (6 cases → 720 rollouts/agent/cell)
N_WORKERS = 20
N_GAUNTLET_CASES = 6
COLLISION_RISK = 0.90

# Corridor half-widths (metres). Total = 2×.
# Standard car ~1.8m wide; need clearance each side.
CORRIDOR_LEVELS = [
    ("wide",    (4.6, 6.2)),   # current gauntlet (9–12 m total) — baseline
    ("medium",  (3.0, 4.0)),   # 6–8 m total — urban arterial
    ("narrow",  (1.8, 2.5)),   # 3.6–5 m total — tight lane
    ("extreme", (1.1, 1.8)),   # 2.2–3.6 m total — barely passable
]

# Pressure multipliers applied to gauntlet baseline (2.35)
PRESSURE_LEVELS = [
    ("1.0×",  1.0),
    ("1.5×",  1.5),
    ("2.0×",  2.0),
]

TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]

# Per-worker model cache, populated by pool initializer
_WORKER: dict = {}


@dataclass
class StressTask:
    corridor_idx: int
    pressure_idx: int
    case_idx: int
    seed: int


@dataclass
class StressResult:
    corridor_idx: int
    pressure_idx: int
    continuous_passed: bool
    continuous_collision: bool
    token_passed: bool
    token_collision: bool
    token_rnn_passed: bool | None
    token_rnn_collision: bool | None
    token_dagger_passed: bool | None
    token_dagger_collision: bool | None
    spotlight_passed: bool
    spotlight_collision: bool


def _init_worker(model_dir_str: str) -> None:
    """Load BC models once per worker process to avoid per-task reload overhead."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    import torch.nn as nn
    import numpy as np

    torch.set_num_threads(1)

    model_dir = Path(model_dir_str)

    class GeomMLP(nn.Module):
        def __init__(self, out_dim: int, n_features: int = 10, hidden: int = 256) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(n_features),
                nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden, hidden),     nn.GELU(), nn.Dropout(0.15),
                nn.Linear(hidden, hidden // 2), nn.GELU(),
                nn.Linear(hidden // 2, out_dim),
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    class GeomGRUTokenBC(nn.Module):
        def __init__(
            self,
            n_features: int = 10,
            n_tokens: int = 9,
            hidden: int = 160,
        ) -> None:
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

    tok_ckpt  = torch.load(model_dir / "token_bc.pt",      map_location="cpu", weights_only=False)
    cont_ckpt = torch.load(model_dir / "continuous_bc.pt", map_location="cpu", weights_only=False)

    tok_model  = GeomMLP(9);  tok_model.load_state_dict(tok_ckpt["state_dict"]);  tok_model.eval()
    cont_model = GeomMLP(2); cont_model.load_state_dict(cont_ckpt["state_dict"]); cont_model.eval()

    _WORKER["torch"]           = torch
    _WORKER["tok_model"]       = tok_model
    _WORKER["cont_model"]      = cont_model
    _WORKER["tok_feat_mean"]   = np.array(tok_ckpt["feat_mean"],  dtype=np.float32)
    _WORKER["tok_feat_std"]    = np.array(tok_ckpt["feat_std"],   dtype=np.float32)
    _WORKER["cont_feat_mean"]  = np.array(cont_ckpt["feat_mean"], dtype=np.float32)
    _WORKER["cont_feat_std"]   = np.array(cont_ckpt["feat_std"],  dtype=np.float32)
    _WORKER["cont_mean"]       = np.array(cont_ckpt["cont_mean"], dtype=np.float32)
    _WORKER["cont_std"]        = np.array(cont_ckpt["cont_std"],  dtype=np.float32)

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


def _run_triple(task: StressTask) -> StressResult:
    import sys, math
    from pathlib import Path
    root = Path(__file__).resolve().parents[4]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import numpy as np

    torch          = _WORKER["torch"]
    tok_model      = _WORKER["tok_model"]
    cont_model     = _WORKER["cont_model"]
    rnn_model      = _WORKER["rnn_model"]
    dagger_model   = _WORKER["dagger_model"]
    tok_feat_mean  = _WORKER["tok_feat_mean"]
    tok_feat_std   = _WORKER["tok_feat_std"]
    cont_feat_mean = _WORKER["cont_feat_mean"]
    cont_feat_std  = _WORKER["cont_feat_std"]
    cont_mean      = _WORKER["cont_mean"]
    cont_std       = _WORKER["cont_std"]
    rnn_feat_mean  = _WORKER.get("rnn_feat_mean")
    rnn_feat_std   = _WORKER.get("rnn_feat_std")
    rnn_history_len = int(_WORKER.get("rnn_history_len", 8))
    dagger_feat_mean = _WORKER.get("dagger_feat_mean")
    dagger_feat_std = _WORKER.get("dagger_feat_std")

    from minimal_shot_av.simulator.compositional_scenarios import (
        generate_compositional_scenario,
        CompositionalScenarioProfile,
        DifficultyConfig,
        TopologyGeometryConfig,
        HazardGeometryConfig,
        AmbientGeometryConfig,
        EnvironmentConfig,
        CorridorClearanceConfig,
    )
    from minimal_shot_av.simulator.policy import (
        RolloutConfig, run_spotlight_reflex_policy, _run_rollout,
    )
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG, SpotlightReflexConfig, ManeuverSpec,
        generate_maneuver_candidates, _planning_heading,
    )
    from minimal_shot_av.simulator.planner import PlannedAction

    CORRIDOR_LEVELS_LOCAL = [
        ("wide",    (4.6, 6.2)),
        ("medium",  (3.0, 4.0)),
        ("narrow",  (1.8, 2.5)),
        ("extreme", (1.1, 1.8)),
    ]
    PRESSURE_LEVELS_LOCAL = [
        ("1.0×",  1.0),
        ("1.5×",  1.5),
        ("2.0×",  2.0),
    ]
    BASELINE_GAUNTLET_PRESSURE = 2.35

    _, half_width_cap = CORRIDOR_LEVELS_LOCAL[task.corridor_idx]
    _, pressure_mult  = PRESSURE_LEVELS_LOCAL[task.pressure_idx]
    scaled_pressure = BASELINE_GAUNTLET_PRESSURE * pressure_mult

    profile = CompositionalScenarioProfile(
        gauntlet_lane_half_width_cap=half_width_cap,
        suite_pressure={
            "compositional": 1.0,
            "adversarial": 1.8,
            "gauntlet": scaled_pressure,
            "hidden": 1.2,
        },
    )

    ESCAPE = {"left": -1.0, "right": 1.0, "balanced": 0.0}

    def extract_features(d: dict) -> np.ndarray:
        escape = ESCAPE.get(str(d.get("preferred_escape_side", "balanced")).lower(), 0.0)
        return np.array([
            float(d.get("obstacle_pressure", 0.0)),
            float(d.get("route_blockage", 0.0)),
            float(d.get("corridor_blocked", False)),
            min(float(d.get("left_clearance", 20.0)), 20.0),
            min(float(d.get("right_clearance", 20.0)), 20.0),
            escape,
            float(d.get("speed", 0.0)),
            float(d.get("lane_error", 0.0)),
            float(d.get("uncertainty", 0.0)),
            min(float(d.get("min_obstacle_distance", 20.0)), 20.0),
        ], dtype=np.float32)

    def is_collision(rollout) -> bool:
        return (
            not rollout.success and rollout.steps is not None
            and any(getattr(st, "collision_risk", 0) > COLLISION_RISK for st in rollout.steps)
        )

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, "gauntlet", profile=profile)

    # Agent C: Spotlight Reflex
    rs = run_spotlight_reflex_policy(scenario)

    TOKEN_ORDER_LOCAL = [
        "stop", "crawl", "maintain", "slow_yield",
        "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
    ]

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

    def token_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        d = {
            "obstacle_pressure":    world_state.obstacle_pressure,
            "route_blockage":       world_state.route_blockage,
            "corridor_blocked":     world_state.corridor_blocked,
            "left_clearance":       getattr(world_state, "left_clearance", 20.0),
            "right_clearance":      getattr(world_state, "right_clearance", 20.0),
            "preferred_escape_side": getattr(world_state, "preferred_escape_side", "balanced"),
            "speed":                speed_mps,
            "lane_error":           perception.lane_error,
            "uncertainty":          max(world_state.uncertainty, perception.uncertainty),
            "min_obstacle_distance": min(
                (getattr(o, "signed_distance", 20.0) for o in perception.visible_obstacles),
                default=20.0,
            ),
        }
        feats = extract_features(d)
        x = torch.from_numpy((feats - tok_feat_mean) / tok_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(tok_model(x).argmax(1).item())
        chosen_name = TOKEN_ORDER_LOCAL[chosen_idx]
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            chosen_name,
            "token_bc",
        )

    def dagger_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if dagger_model is None or dagger_feat_mean is None or dagger_feat_std is None:
            raise RuntimeError("Token-DAgger-BC model is not loaded")
        d = {
            "obstacle_pressure":    world_state.obstacle_pressure,
            "route_blockage":       world_state.route_blockage,
            "corridor_blocked":     world_state.corridor_blocked,
            "left_clearance":       getattr(world_state, "left_clearance", 20.0),
            "right_clearance":      getattr(world_state, "right_clearance", 20.0),
            "preferred_escape_side": getattr(world_state, "preferred_escape_side", "balanced"),
            "speed":                speed_mps,
            "lane_error":           perception.lane_error,
            "uncertainty":          max(world_state.uncertainty, perception.uncertainty),
            "min_obstacle_distance": min(
                (getattr(o, "signed_distance", 20.0) for o in perception.visible_obstacles),
                default=20.0,
            ),
        }
        feats = extract_features(d)
        x = torch.from_numpy((feats - dagger_feat_mean) / dagger_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(dagger_model(x).argmax(1).item())
        chosen_name = TOKEN_ORDER_LOCAL[chosen_idx]
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            chosen_name,
            "token_dagger_bc",
        )

    rnn_history: list[np.ndarray] = []

    def token_rnn_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if rnn_model is None or rnn_feat_mean is None or rnn_feat_std is None:
            raise RuntimeError("Token-RNN-BC model is not loaded")
        d = {
            "obstacle_pressure":    world_state.obstacle_pressure,
            "route_blockage":       world_state.route_blockage,
            "corridor_blocked":     world_state.corridor_blocked,
            "left_clearance":       getattr(world_state, "left_clearance", 20.0),
            "right_clearance":      getattr(world_state, "right_clearance", 20.0),
            "preferred_escape_side": getattr(world_state, "preferred_escape_side", "balanced"),
            "speed":                speed_mps,
            "lane_error":           perception.lane_error,
            "uncertainty":          max(world_state.uncertainty, perception.uncertainty),
            "min_obstacle_distance": min(
                (getattr(o, "signed_distance", 20.0) for o in perception.visible_obstacles),
                default=20.0,
            ),
        }
        feats = extract_features(d)
        norm_feats = ((feats - rnn_feat_mean) / rnn_feat_std).astype(np.float32)
        rnn_history.append(norm_feats)
        history = rnn_history[-rnn_history_len:]
        if len(history) < rnn_history_len:
            history = [history[0]] * (rnn_history_len - len(history)) + history
        x = torch.from_numpy(np.stack(history, axis=0)).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(rnn_model(x).argmax(1).item())
        chosen_name = TOKEN_ORDER_LOCAL[chosen_idx]
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            chosen_name,
            "token_rnn_bc",
        )

    def continuous_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        d = {
            "obstacle_pressure":    world_state.obstacle_pressure,
            "route_blockage":       world_state.route_blockage,
            "corridor_blocked":     world_state.corridor_blocked,
            "left_clearance":       getattr(world_state, "left_clearance", 20.0),
            "right_clearance":      getattr(world_state, "right_clearance", 20.0),
            "preferred_escape_side": getattr(world_state, "preferred_escape_side", "balanced"),
            "speed":                speed_mps,
            "lane_error":           perception.lane_error,
            "uncertainty":          max(world_state.uncertainty, perception.uncertainty),
            "min_obstacle_distance": min(
                (getattr(o, "signed_distance", 20.0) for o in perception.visible_obstacles),
                default=20.0,
            ),
        }
        feats = extract_features(d)
        x = torch.from_numpy((feats - cont_feat_mean) / cont_feat_std).unsqueeze(0)
        with torch.no_grad():
            pred = cont_model(x).squeeze(0).numpy()
        speed_scale = float(np.clip(pred[0] * cont_std[0] + cont_mean[0], 0.0, 1.2))
        lat_offset  = float(np.clip(pred[1] * cont_std[1] + cont_mean[1], -12.0, 12.0))

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

    def run_with_planner(scenario, planner_fn):
        def wrapped(active_scenario, position, world_state, perception, config):
            return planner_fn(active_scenario, position, world_state, perception,
                              config.step_size, config)
        cfg = RolloutConfig(spotlight=DEFAULT_SPOTLIGHT_CONFIG)
        return _run_rollout(scenario, wrapped, cfg)

    rb = run_with_planner(scenario, token_bc_planner)
    ra = run_with_planner(scenario, continuous_bc_planner)
    rrnn = run_with_planner(scenario, token_rnn_bc_planner) if rnn_model is not None else None
    rdagger = run_with_planner(scenario, dagger_bc_planner) if dagger_model is not None else None

    return StressResult(
        corridor_idx=task.corridor_idx,
        pressure_idx=task.pressure_idx,
        continuous_passed=ra.success,   continuous_collision=is_collision(ra),
        token_passed=rb.success,        token_collision=is_collision(rb),
        token_rnn_passed=rrnn.success if rrnn is not None else None,
        token_rnn_collision=is_collision(rrnn) if rrnn is not None else None,
        token_dagger_passed=rdagger.success if rdagger is not None else None,
        token_dagger_collision=is_collision(rdagger) if rdagger is not None else None,
        spotlight_passed=rs.success,    spotlight_collision=is_collision(rs),
    )


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0: return (0.0, 1.0)
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2*n)) / d
    h = (z / d) * math.sqrt(p*(1-p)/n + z**2/(4*n**2))
    return (max(0.0, c-h), min(1.0, c+h))


def main() -> None:
    model_dir = ROOT / "artifacts" / "bc_models"
    tasks = [
        StressTask(corridor_idx=ci, pressure_idx=pi, case_idx=c, seed=s)
        for ci in range(len(CORRIDOR_LEVELS))
        for pi in range(len(PRESSURE_LEVELS))
        for c in range(N_GAUNTLET_CASES)
        for s in range(1, N_SEEDS + 1)
    ]
    n_cells = len(CORRIDOR_LEVELS) * len(PRESSURE_LEVELS)
    n_per_cell = N_GAUNTLET_CASES * N_SEEDS
    n_agents = 5
    print(f"Stress-test phase diagram: {n_cells} cells × {n_per_cell} rollouts × {n_agents} agents")
    print(f"Total: {len(tasks) * n_agents:,} rollouts ({N_WORKERS} workers)…\n")

    with multiprocessing.Pool(
        N_WORKERS,
        initializer=_init_worker,
        initargs=(str(model_dir),),
    ) as pool:
        results = pool.map(_run_triple, tasks)

    output: dict[str, Any] = {}
    hdr = f"{'Corridor':<10} {'Pressure':<7}  {'N':>4}"
    hdr += f"  {'ContBC':>11}  {'TokBC':>11}  {'TokRNN':>11}  {'TokDag':>11}  {'SR':>9}"
    print(hdr)
    print("─" * len(hdr))

    phase: list[dict] = []
    for ci, (cname, cap) in enumerate(CORRIDOR_LEVELS):
        for pi, (pname, pmult) in enumerate(PRESSURE_LEVELS):
            sr = [r for r in results if r.corridor_idx == ci and r.pressure_idx == pi]
            n = len(sr)
            cont_coll = sum(r.continuous_collision for r in sr)
            tok_coll  = sum(r.token_collision     for r in sr)
            rnn_coll  = sum(bool(r.token_rnn_collision) for r in sr if r.token_rnn_collision is not None)
            dagger_coll = sum(bool(r.token_dagger_collision) for r in sr if r.token_dagger_collision is not None)
            spot_coll = sum(r.spotlight_collision  for r in sr)
            cont_pass = sum(r.continuous_passed    for r in sr)
            tok_pass  = sum(r.token_passed         for r in sr)
            rnn_pass  = sum(bool(r.token_rnn_passed) for r in sr if r.token_rnn_passed is not None)
            dagger_pass = sum(bool(r.token_dagger_passed) for r in sr if r.token_dagger_passed is not None)
            spot_pass = sum(r.spotlight_passed     for r in sr)

            cont_c_lo, cont_c_hi = wilson_ci(cont_coll, n)
            tok_c_lo,  tok_c_hi  = wilson_ci(tok_coll,  n)
            rnn_c_lo,  rnn_c_hi  = wilson_ci(rnn_coll,  n)
            dagger_c_lo, dagger_c_hi = wilson_ci(dagger_coll, n)
            spot_c_lo, spot_c_hi = wilson_ci(spot_coll, n)

            delta_tok = tok_coll/n - spot_coll/n  # positive = BC worse

            print(
                f"  {cname:<9} {pname:<7}  {n:>4}"
                f"  {cont_coll/n*100:>5.1f}% [{cont_c_lo*100:.1f},{cont_c_hi*100:.1f}]"
                f"  {tok_coll/n*100:>5.1f}% [{tok_c_lo*100:.1f},{tok_c_hi*100:.1f}]"
                f"  {rnn_coll/n*100:>5.1f}% [{rnn_c_lo*100:.1f},{rnn_c_hi*100:.1f}]"
                f"  {dagger_coll/n*100:>5.1f}% [{dagger_c_lo*100:.1f},{dagger_c_hi*100:.1f}]"
                f"  {spot_coll/n*100:>5.1f}% [{spot_c_lo*100:.1f},{spot_c_hi*100:.1f}]"
            )
            phase.append({
                "corridor": cname, "half_width_cap": list(cap),
                "pressure": pname, "pressure_mult": pmult,
                "n": n,
                "continuous_bc": {
                    "passes": cont_pass, "collisions": cont_coll,
                    "pass_rate": round(cont_pass/n*100, 1),
                    "collision_rate": round(cont_coll/n*100, 1),
                    "coll_ci95": [round(cont_c_lo*100, 1), round(cont_c_hi*100, 1)],
                },
                "token_bc": {
                    "passes": tok_pass, "collisions": tok_coll,
                    "pass_rate": round(tok_pass/n*100, 1),
                    "collision_rate": round(tok_coll/n*100, 1),
                    "coll_ci95": [round(tok_c_lo*100, 1), round(tok_c_hi*100, 1)],
                },
                "token_rnn_bc": {
                    "passes": rnn_pass, "collisions": rnn_coll,
                    "pass_rate": round(rnn_pass/n*100, 1),
                    "collision_rate": round(rnn_coll/n*100, 1),
                    "coll_ci95": [round(rnn_c_lo*100, 1), round(rnn_c_hi*100, 1)],
                },
                "token_dagger_bc": {
                    "passes": dagger_pass, "collisions": dagger_coll,
                    "pass_rate": round(dagger_pass/n*100, 1),
                    "collision_rate": round(dagger_coll/n*100, 1),
                    "coll_ci95": [round(dagger_c_lo*100, 1), round(dagger_c_hi*100, 1)],
                },
                "spotlight": {
                    "passes": spot_pass, "collisions": spot_coll,
                    "pass_rate": round(spot_pass/n*100, 1),
                    "collision_rate": round(spot_coll/n*100, 1),
                    "coll_ci95": [round(spot_c_lo*100, 1), round(spot_c_hi*100, 1)],
                },
                "delta_tok_bc_minus_spotlight_pp": round(delta_tok*100, 2),
                "delta_token_rnn_bc_minus_spotlight_pp": round((rnn_coll/n - spot_coll/n) * 100, 2),
                "delta_token_dagger_bc_minus_spotlight_pp": round((dagger_coll/n - spot_coll/n) * 100, 2),
            })

    output = {
        "corridor_levels": [{"name": n, "half_width_cap": list(c)} for n, c in CORRIDOR_LEVELS],
        "pressure_levels": [{"name": n, "mult": m} for n, m in PRESSURE_LEVELS],
        "n_seeds": N_SEEDS,
        "n_gauntlet_cases": N_GAUNTLET_CASES,
        "cells": phase,
    }
    out_path = ROOT / "artifacts" / "stress_phase" / "results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
