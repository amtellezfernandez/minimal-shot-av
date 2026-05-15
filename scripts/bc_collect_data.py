"""Collect Spotlight Reflex expert transitions for BC training.

Feature vector (10 geometric scalars, no object identity):
  [obstacle_pressure, route_blockage, corridor_blocked,
   left_clearance, right_clearance, escape_side_encoded,
   speed, lane_error, uncertainty, min_obstacle_distance]

Labels:
  token_idx  : integer in [0, 8] → selected maneuver token
  speed_scale: float              → ManeuverSpec.speed_scale (for continuous BC)
  lat_offset : float              → ManeuverSpec.lateral_offset_m (for continuous BC)

Data source: compositional + adversarial suites, seeds 1-200 each case.
Gauntlet and hidden are held out for evaluation only.
"""
from __future__ import annotations

import json
import multiprocessing
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

SEED_START = 1
SEED_END = 200       # 200 seeds per case for training density
N_WORKERS = 20
TRAIN_SUITES = [("compositional", 5), ("adversarial", 2)]  # gauntlet/hidden held out

TOKEN_ORDER = [
    "stop", "crawl", "maintain", "slow_yield",
    "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
]
TOKEN_INDEX = {name: i for i, name in enumerate(TOKEN_ORDER)}

ESCAPE_ENCODING = {"left": -1.0, "right": 1.0, "balanced": 0.0}


def _extract_features(step: object) -> list[float]:
    d = step if isinstance(step, dict) else vars(step)
    escape_raw = d.get("preferred_escape_side", "balanced")
    escape = ESCAPE_ENCODING.get(str(escape_raw).lower(), 0.0)
    return [
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
    ]


@dataclass
class CollectTask:
    suite: str
    case_idx: int
    seed: int


def _collect_rollout(task: CollectTask) -> list[dict]:
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import run_spotlight_reflex_policy
    from minimal_shot_av.simulator.spotlight_reflex import DEFAULT_SPOTLIGHT_CONFIG

    effective_seed = task.seed + task.case_idx * 1_000
    scenario = generate_compositional_scenario(effective_seed, task.suite)
    rollout = run_spotlight_reflex_policy(scenario)

    token_params = {m.name: (m.speed_scale, m.lateral_offset_m) for m in DEFAULT_SPOTLIGHT_CONFIG.maneuvers}
    token_index = {name: i for i, name in enumerate([
        "stop", "crawl", "maintain", "slow_yield",
        "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
    ])}

    transitions = []
    rollout_id = f"{task.suite}:{task.case_idx}:{task.seed}"
    for step_idx, step in enumerate(rollout.steps):
        d = step if isinstance(step, dict) else vars(step)
        token_name = d.get("selected_maneuver", "maintain")
        if token_name not in token_index:
            continue
        escape_raw = d.get("preferred_escape_side", "balanced")
        escape_enc = {"left": -1.0, "right": 1.0, "balanced": 0.0}.get(str(escape_raw).lower(), 0.0)
        feats = [
            float(d.get("obstacle_pressure", 0.0)),
            float(d.get("route_blockage", 0.0)),
            float(d.get("corridor_blocked", False)),
            min(float(d.get("left_clearance", 20.0)), 20.0),
            min(float(d.get("right_clearance", 20.0)), 20.0),
            escape_enc,
            float(d.get("speed", 0.0)),
            float(d.get("lane_error", 0.0)),
            float(d.get("uncertainty", 0.0)),
            min(float(d.get("min_obstacle_distance", 20.0)), 20.0),
        ]
        speed_scale, lat_offset = token_params.get(token_name, (0.75, 0.0))
        transitions.append({
            "rollout_id": rollout_id,
            "step_idx": step_idx,
            "features": feats,
            "token_idx": token_index[token_name],
            "speed_scale": speed_scale,
            "lat_offset": lat_offset,
        })
    return transitions


def main() -> None:
    tasks = [
        CollectTask(suite=suite, case_idx=c, seed=s)
        for suite, n_cases in TRAIN_SUITES
        for c in range(n_cases)
        for s in range(SEED_START, SEED_END + 1)
    ]
    print(f"Collecting from {len(tasks)} rollouts ({N_WORKERS} workers)…")

    with multiprocessing.Pool(N_WORKERS) as pool:
        all_rollouts = pool.map(_collect_rollout, tasks)

    all_transitions = [t for rollout in all_rollouts for t in rollout]
    print(f"Collected {len(all_transitions):,} transitions")

    # Token distribution
    counts = [0] * 9
    token_names = [
        "stop", "crawl", "maintain", "slow_yield",
        "nudge_left", "nudge_right", "evasive_left", "evasive_right", "lane_recover",
    ]
    for t in all_transitions:
        counts[t["token_idx"]] += 1
    print("Token distribution:")
    for name, count in zip(token_names, counts):
        print(f"  {name:<16}: {count:6d}  ({count/len(all_transitions)*100:.1f}%)")

    features = np.array([t["features"] for t in all_transitions], dtype=np.float32)
    token_idx = np.array([t["token_idx"] for t in all_transitions], dtype=np.int64)
    speed_scale = np.array([t["speed_scale"] for t in all_transitions], dtype=np.float32)
    lat_offset = np.array([t["lat_offset"] for t in all_transitions], dtype=np.float32)
    rollout_ids_text = [str(t["rollout_id"]) for t in all_transitions]
    rollout_vocab = {name: idx for idx, name in enumerate(dict.fromkeys(rollout_ids_text))}
    rollout_id = np.array([rollout_vocab[name] for name in rollout_ids_text], dtype=np.int64)
    step_idx = np.array([t["step_idx"] for t in all_transitions], dtype=np.int64)

    # Normalisation stats (saved with data for inference)
    feat_mean = features.mean(axis=0)
    feat_std  = features.std(axis=0) + 1e-6

    out_dir = ROOT / "artifacts" / "bc_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "features.npy",   features)
    np.save(out_dir / "token_idx.npy",  token_idx)
    np.save(out_dir / "speed_scale.npy", speed_scale)
    np.save(out_dir / "lat_offset.npy", lat_offset)
    np.save(out_dir / "rollout_id.npy", rollout_id)
    np.save(out_dir / "step_idx.npy", step_idx)
    np.save(out_dir / "feat_mean.npy",  feat_mean)
    np.save(out_dir / "feat_std.npy",   feat_std)

    meta = {
        "n_transitions": len(all_transitions),
        "n_features": 10,
        "n_tokens": 9,
        "n_rollouts": len(rollout_vocab),
        "has_sequence_index": True,
        "token_names": token_names,
        "train_suites": [s for s, _ in TRAIN_SUITES],
        "seeds": f"{SEED_START}-{SEED_END}",
        "feature_names": [
            "obstacle_pressure", "route_blockage", "corridor_blocked",
            "left_clearance", "right_clearance", "escape_side",
            "speed", "lane_error", "uncertainty", "min_obstacle_distance",
        ],
        "token_distribution": {n: int(c) for n, c in zip(token_names, counts)},
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nSaved to {out_dir}")


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
