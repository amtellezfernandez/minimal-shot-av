"""Evaluate BC baselines on gauntlet + adversarial + compositional suites.

Agent A: Continuous-BC   — MLP → (speed_scale, lat_offset_m) → on-the-fly trajectory
Agent B: Token-BC        — MLP → argmax over 9 tokens → pre-defined trajectory
Agent C: Token-RNN-BC    — GRU over scalar history → argmax over 9 tokens
Agent D: Token-Transformer-BC — transformer over scalar history → argmax over 9 tokens
Agent E: Token-DAgger-BC — one-iteration on-policy relabelled Token-BC
Agent F: Spotlight Reflex — geometric rules → token selection

Evaluation protocol:
  - Same matched seeds across all agents
  - Gauntlet (6 cases × 80 seeds = 480), adversarial (2×80=160), compositional (5×80=400)
  - Reports pass rate and collision rate with Wilson 95% CI

Model loading: pool initializer loads both models once per worker process (not per task).
"""
from __future__ import annotations

import json
import math
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED_START = 1
SEED_END = 80
N_WORKERS = 20
COLLISION_RISK = 0.90

SUITES = [
    ("gauntlet",      6),
    ("adversarial",   2),
    ("compositional", 5),
]

TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]

# Per-worker model cache, populated by pool initializer
_WORKER = {}


@dataclass
class EvalTask:
    suite: str
    case_idx: int
    seed: int


@dataclass
class TripleResult:
    suite: str
    continuous_passed: bool
    continuous_collision: bool
    token_passed: bool
    token_collision: bool
    token_rnn_passed: bool | None
    token_rnn_collision: bool | None
    token_transformer_passed: bool | None
    token_transformer_collision: bool | None
    token_dagger_passed: bool | None
    token_dagger_collision: bool | None
    spotlight_passed: bool
    spotlight_collision: bool


def _init_worker(model_dir_str: str) -> None:
    """Load models once per worker process."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import torch
    import torch.nn as nn
    import numpy as np

    torch.set_num_threads(1)  # prevent OMP deadlock in pool workers

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

    tok_ckpt  = torch.load(model_dir / "token_bc.pt",      map_location="cpu", weights_only=False)
    cont_ckpt = torch.load(model_dir / "continuous_bc.pt", map_location="cpu", weights_only=False)

    tok_model  = GeomMLP(9);  tok_model.load_state_dict(tok_ckpt["state_dict"]);  tok_model.eval()
    cont_model = GeomMLP(2); cont_model.load_state_dict(cont_ckpt["state_dict"]); cont_model.eval()

    _WORKER["tok_model"]       = tok_model
    _WORKER["cont_model"]      = cont_model
    _WORKER["tok_feat_mean"]   = np.array(tok_ckpt["feat_mean"],  dtype=np.float32)
    _WORKER["tok_feat_std"]    = np.array(tok_ckpt["feat_std"],   dtype=np.float32)
    _WORKER["cont_feat_mean"]  = np.array(cont_ckpt["feat_mean"], dtype=np.float32)
    _WORKER["cont_feat_std"]   = np.array(cont_ckpt["feat_std"],  dtype=np.float32)
    _WORKER["cont_mean"]       = np.array(cont_ckpt["cont_mean"], dtype=np.float32)
    _WORKER["cont_std"]        = np.array(cont_ckpt["cont_std"],  dtype=np.float32)
    _WORKER["torch"]           = torch

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


def _run_triple(task: EvalTask) -> TripleResult:
    import sys, math
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    import numpy as np

    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import (
        RolloutConfig, run_spotlight_reflex_policy, _run_rollout,
    )
    from minimal_shot_av.simulator.spotlight_reflex import (
        DEFAULT_SPOTLIGHT_CONFIG, SpotlightReflexConfig, ManeuverSpec,
        generate_maneuver_candidates, _planning_heading,
    )
    from minimal_shot_av.simulator.planner import PlannedAction

    torch         = _WORKER["torch"]
    tok_model     = _WORKER["tok_model"]
    cont_model    = _WORKER["cont_model"]
    rnn_model     = _WORKER["rnn_model"]
    transformer_model = _WORKER["transformer_model"]
    dagger_model  = _WORKER["dagger_model"]
    tok_feat_mean = _WORKER["tok_feat_mean"]
    tok_feat_std  = _WORKER["tok_feat_std"]
    cont_feat_mean = _WORKER["cont_feat_mean"]
    cont_feat_std  = _WORKER["cont_feat_std"]
    cont_mean     = _WORKER["cont_mean"]
    cont_std      = _WORKER["cont_std"]
    rnn_feat_mean = _WORKER.get("rnn_feat_mean")
    rnn_feat_std  = _WORKER.get("rnn_feat_std")
    rnn_history_len = int(_WORKER.get("rnn_history_len", 8))
    transformer_feat_mean = _WORKER.get("transformer_feat_mean")
    transformer_feat_std = _WORKER.get("transformer_feat_std")
    transformer_history_len = int(_WORKER.get("transformer_history_len", 8))
    dagger_feat_mean = _WORKER.get("dagger_feat_mean")
    dagger_feat_std  = _WORKER.get("dagger_feat_std")

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

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, task.suite)

    # Agent C: Spotlight Reflex
    rs = run_spotlight_reflex_policy(scenario)

    LOCAL_TOKEN_ORDER = [
        "stop", "crawl", "maintain", "slow_yield",
        "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
    ]

    def _obs_dict(world_state, perception, speed_mps) -> dict:
        return {
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

    # Agent B: Token-BC planner
    def token_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        feats = extract_features(_obs_dict(world_state, perception, speed_mps))
        x = torch.from_numpy((feats - tok_feat_mean) / tok_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(tok_model(x).argmax(1).item())
        chosen_name = LOCAL_TOKEN_ORDER[chosen_idx]
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
        feats = extract_features(_obs_dict(world_state, perception, speed_mps))
        x = torch.from_numpy((feats - dagger_feat_mean) / dagger_feat_std).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(dagger_model(x).argmax(1).item())
        chosen_name = LOCAL_TOKEN_ORDER[chosen_idx]
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
    transformer_history: list[np.ndarray] = []

    def token_rnn_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if rnn_model is None or rnn_feat_mean is None or rnn_feat_std is None:
            raise RuntimeError("Token-RNN-BC model is not loaded")
        feats = extract_features(_obs_dict(world_state, perception, speed_mps))
        norm_feats = ((feats - rnn_feat_mean) / rnn_feat_std).astype(np.float32)
        rnn_history.append(norm_feats)
        history = rnn_history[-rnn_history_len:]
        if len(history) < rnn_history_len:
            history = [history[0]] * (rnn_history_len - len(history)) + history
        x = torch.from_numpy(np.stack(history, axis=0)).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(rnn_model(x).argmax(1).item())
        chosen_name = LOCAL_TOKEN_ORDER[chosen_idx]
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            chosen_name,
            "token_rnn_bc",
        )

    def token_transformer_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        if transformer_model is None or transformer_feat_mean is None or transformer_feat_std is None:
            raise RuntimeError("Token-Transformer-BC model is not loaded")
        feats = extract_features(_obs_dict(world_state, perception, speed_mps))
        norm_feats = ((feats - transformer_feat_mean) / transformer_feat_std).astype(np.float32)
        transformer_history.append(norm_feats)
        history = transformer_history[-transformer_history_len:]
        if len(history) < transformer_history_len:
            history = [history[0]] * (transformer_history_len - len(history)) + history
        x = torch.from_numpy(np.stack(history, axis=0)).unsqueeze(0)
        with torch.no_grad():
            chosen_idx = int(transformer_model(x).argmax(1).item())
        chosen_name = LOCAL_TOKEN_ORDER[chosen_idx]
        return token_candidate_action(
            active_scenario,
            position,
            world_state,
            perception,
            speed_mps,
            chosen_name,
            "token_transformer_bc",
        )

    # Agent A: Continuous-BC planner
    def continuous_bc_planner(active_scenario, position, world_state, perception, speed_mps, config):
        feats = extract_features(_obs_dict(world_state, perception, speed_mps))
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
    rtransformer = run_with_planner(scenario, token_transformer_bc_planner) if transformer_model is not None else None
    rdagger = run_with_planner(scenario, dagger_bc_planner) if dagger_model is not None else None

    return TripleResult(
        suite=task.suite,
        continuous_passed=ra.success, continuous_collision=is_collision(ra),
        token_passed=rb.success,      token_collision=is_collision(rb),
        token_rnn_passed=rrnn.success if rrnn is not None else None,
        token_rnn_collision=is_collision(rrnn) if rrnn is not None else None,
        token_transformer_passed=rtransformer.success if rtransformer is not None else None,
        token_transformer_collision=is_collision(rtransformer) if rtransformer is not None else None,
        token_dagger_passed=rdagger.success if rdagger is not None else None,
        token_dagger_collision=is_collision(rdagger) if rdagger is not None else None,
        spotlight_passed=rs.success,  spotlight_collision=is_collision(rs),
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
        EvalTask(suite=suite, case_idx=c, seed=s)
        for suite, n_cases in SUITES
        for c in range(n_cases)
        for s in range(SEED_START, SEED_END + 1)
    ]
    print(f"Evaluating BC baselines on {len(tasks)} scenarios ({N_WORKERS} workers)…")

    with multiprocessing.Pool(
        N_WORKERS,
        initializer=_init_worker,
        initargs=(str(model_dir),),
    ) as pool:
        results = pool.map(_run_triple, tasks)

    output = {}
    print()
    for suite, _ in SUITES:
        sr = [r for r in results if r.suite == suite]
        n  = len(sr)
        rows = {
            "Continuous-BC (A)": (
                sum(r.continuous_passed for r in sr),
                sum(r.continuous_collision for r in sr)),
            "Token-BC (B)":      (
                sum(r.token_passed for r in sr),
                sum(r.token_collision for r in sr)),
        }
        if all(r.token_rnn_passed is not None for r in sr):
            rows["Token-RNN-BC (C)"] = (
                sum(bool(r.token_rnn_passed) for r in sr),
                sum(bool(r.token_rnn_collision) for r in sr),
            )
        if all(r.token_transformer_passed is not None for r in sr):
            rows["Token-Transformer-BC (D)"] = (
                sum(bool(r.token_transformer_passed) for r in sr),
                sum(bool(r.token_transformer_collision) for r in sr),
            )
        if all(r.token_dagger_passed is not None for r in sr):
            rows["Token-DAgger-BC (E)"] = (
                sum(bool(r.token_dagger_passed) for r in sr),
                sum(bool(r.token_dagger_collision) for r in sr),
            )
        rows["Spotlight Reflex (E)"] = (
            sum(r.spotlight_passed for r in sr),
            sum(r.spotlight_collision for r in sr),
        )
        print(f"\n── {suite.upper()}  (n={n}) ──")
        print(f"{'Agent':<24}  {'Pass%':>6} {'95% CI':>13}  {'Coll%':>6} {'95% CI':>11}")
        print("─" * 68)
        suite_out = {}
        for agent_name, (passes, colls) in rows.items():
            p_lo, p_hi = wilson_ci(passes, n)
            c_lo, c_hi = wilson_ci(colls, n)
            print(f"  {agent_name:<22}  {passes/n*100:>5.1f}% [{p_lo*100:.1f},{p_hi*100:.1f}]"
                  f"  {colls/n*100:>5.1f}% [{c_lo*100:.1f},{c_hi*100:.1f}]")
            suite_out[agent_name] = {
                "passes": passes, "collisions": colls, "total": n,
                "pass_rate": round(passes/n*100, 1),
                "collision_rate": round(colls/n*100, 1),
                "pass_ci95": [round(p_lo*100,1), round(p_hi*100,1)],
                "coll_ci95": [round(c_lo*100,1), round(c_hi*100,1)],
            }
        output[suite] = suite_out

    out_path = ROOT / "artifacts" / "bc_vs_spotlight.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
