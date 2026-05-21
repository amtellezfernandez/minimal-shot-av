"""Audit DAgger representation drift and closed-loop regression.

This script diagnoses whether later DAgger iterations fail because of a simple
token-distribution shift or because the learned selector loses fidelity across
its mixed nominal/recovery state sources.

It reports three groups of evidence:
  1. Dataset/source drift across expert and DAgger relabel pools.
  2. Per-source checkpoint accuracy for Token-BC and DAgger checkpoints.
  3. Continuous closed-loop metrics on the held-out Latin-hypercube sweep.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import multiprocessing
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eval_heldout_latin_hypercube import DEFAULT_SUITES, SUITE_CASES, TOKEN_ORDER, build_lhs_profiles


N_FEATURES = 10
N_TOKENS = len(TOKEN_ORDER)
HIDDEN = 256
DEFAULT_WORKERS = 12
DEFAULT_PROFILES = 12
DEFAULT_SEED_START = 1
DEFAULT_SEED_END = 10
DEFAULT_OUTPUT = ROOT / "artifacts" / "dagger_representation_audit.json"

FEATURE_NAMES = (
    "obstacle_pressure",
    "route_blockage",
    "corridor_blocked",
    "left_clearance",
    "right_clearance",
    "escape_side",
    "speed",
    "lane_error",
    "uncertainty",
    "min_obstacle_distance",
)

_WORKER: dict[str, Any] = {}


@dataclass(frozen=True)
class MetricTask:
    profile_idx: int
    suite: str
    case_idx: int
    seed: int


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=int, default=DEFAULT_PROFILES)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-end", type=int, default=DEFAULT_SEED_END)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--profile-seed", type=int, default=2027)
    parser.add_argument(
        "--suites",
        type=str,
        default=",".join(DEFAULT_SUITES),
        help="Comma-separated suites from: compositional, adversarial, gauntlet, hidden",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _dataset_specs() -> list[tuple[str, Path]]:
    return [
        ("expert", ROOT / "artifacts" / "bc_data"),
        ("dagger_iter1", ROOT / "artifacts" / "bc_dagger_data"),
        ("dagger_iter2", ROOT / "artifacts" / "bc_dagger_data_iter2"),
        ("dagger_iter3", ROOT / "artifacts" / "bc_dagger_data_iter3"),
    ]


def _checkpoint_specs() -> list[tuple[str, Path]]:
    return [
        ("token_bc", ROOT / "artifacts" / "bc_models" / "token_bc.pt"),
        ("dagger_iter1", ROOT / "artifacts" / "bc_models" / "token_dagger_bc.pt"),
        ("dagger_iter2", ROOT / "artifacts" / "bc_models_iter2" / "token_dagger_bc.pt"),
        ("dagger_iter3", ROOT / "artifacts" / "bc_models_iter3" / "token_dagger_bc.pt"),
    ]


def _load_dataset(data_dir: Path) -> dict[str, Any]:
    features = np.load(data_dir / "features.npy").astype(np.float32)
    token_idx = np.load(data_dir / "token_idx.npy").astype(np.int64)
    meta_path = data_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    counts = np.bincount(token_idx, minlength=N_TOKENS).astype(np.int64)
    fractions = counts / max(len(token_idx), 1)
    feature_mean = features.mean(axis=0)
    feature_std = features.std(axis=0) + 1e-6
    return {
        "path": str(data_dir),
        "meta": meta,
        "features": features,
        "token_idx": token_idx,
        "counts": counts,
        "fractions": fractions,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
    }


def _safe_log2(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-12, None)
    return np.log2(clipped)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(float(p.sum()), 1e-12)
    q = q / max(float(q.sum()), 1e-12)
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(np.where(p > 0, p * (_safe_log2(p) - _safe_log2(m)), 0.0)))
    kl_qm = float(np.sum(np.where(q > 0, q * (_safe_log2(q) - _safe_log2(m)), 0.0)))
    return 0.5 * (kl_pm + kl_qm)


def _dataset_drift_report(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    expert = datasets["expert"]
    report: dict[str, Any] = {"sources": {}, "pairwise_js_divergence": {}}
    previous_name: str | None = None
    previous: dict[str, Any] | None = None
    for name, data in datasets.items():
        feature_shift_vs_expert = (data["feature_mean"] - expert["feature_mean"]) / expert["feature_std"]
        report["sources"][name] = {
            "path": data["path"],
            "n_transitions": int(len(data["token_idx"])),
            "token_counts": {token: int(count) for token, count in zip(TOKEN_ORDER, data["counts"])},
            "token_fractions": {token: round(float(frac), 6) for token, frac in zip(TOKEN_ORDER, data["fractions"])},
            "maintain_fraction": round(float(data["fractions"][TOKEN_ORDER.index("maintain")]), 6),
            "evasive_fraction": round(
                float(
                    data["fractions"][TOKEN_ORDER.index("evasive_left")]
                    + data["fractions"][TOKEN_ORDER.index("evasive_right")]
                ),
                6,
            ),
            "recovery_fraction": round(float(data["fractions"][TOKEN_ORDER.index("lane_recover")]), 6),
            "feature_mean": {feat: round(float(val), 6) for feat, val in zip(FEATURE_NAMES, data["feature_mean"])},
            "feature_mean_shift_vs_expert_z": {
                feat: round(float(val), 4) for feat, val in zip(FEATURE_NAMES, feature_shift_vs_expert)
            },
            "mean_abs_feature_shift_vs_expert_z": round(float(np.abs(feature_shift_vs_expert).mean()), 4),
            "max_abs_feature_shift_vs_expert_z": round(float(np.abs(feature_shift_vs_expert).max()), 4),
        }
        report["pairwise_js_divergence"][f"{name}_vs_expert"] = round(
            _js_divergence(data["fractions"], expert["fractions"]),
            6,
        )
        if previous_name is not None and previous is not None:
            report["pairwise_js_divergence"][f"{name}_vs_{previous_name}"] = round(
                _js_divergence(data["fractions"], previous["fractions"]),
                6,
            )
        previous_name = name
        previous = data
    return report


def _eval_checkpoints(datasets: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import torch
    import torch.nn as nn

    class GeomMLP(nn.Module):
        def __init__(self, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(N_FEATURES),
                nn.Linear(N_FEATURES, HIDDEN), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(),
                nn.Linear(HIDDEN // 2, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    report: dict[str, Any] = {}
    for ckpt_name, ckpt_path in _checkpoint_specs():
        if not ckpt_path.is_file():
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = GeomMLP(N_TOKENS)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        feat_mean = np.asarray(ckpt["feat_mean"], dtype=np.float32)
        feat_std = np.asarray(ckpt["feat_std"], dtype=np.float32)
        ckpt_report: dict[str, Any] = {
            "checkpoint": str(ckpt_path),
            "source_metrics": {},
        }
        for data_name, data in datasets.items():
            features = torch.from_numpy((data["features"] - feat_mean) / feat_std).float()
            labels = torch.from_numpy(data["token_idx"]).long()
            with torch.no_grad():
                pred = model(features).argmax(dim=1)
            correct = pred.eq(labels)
            accuracy = float(correct.float().mean().item())
            per_token_recall = {}
            for token_idx, token_name in enumerate(TOKEN_ORDER):
                mask = labels.eq(token_idx)
                if int(mask.sum().item()) == 0:
                    continue
                recall = float(pred[mask].eq(labels[mask]).float().mean().item())
                per_token_recall[token_name] = round(recall, 4)
            ckpt_report["source_metrics"][data_name] = {
                "accuracy": round(accuracy, 4),
                "error_rate": round(1.0 - accuracy, 4),
                "per_token_recall": per_token_recall,
            }
        report[ckpt_name] = ckpt_report
    return report


def _obs_features(world_state, perception, speed_mps: float) -> np.ndarray:
    escape = {"left": -1.0, "right": 1.0, "balanced": 0.0}.get(
        str(getattr(world_state, "preferred_escape_side", "balanced")).lower(),
        0.0,
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
            min(
                (float(getattr(obstacle, "signed_distance", 20.0)) for obstacle in getattr(perception, "visible_obstacles", [])),
                default=20.0,
            ),
        ],
        dtype=np.float32,
    )


def _init_worker(model_specs: list[tuple[str, str]], profiles) -> None:
    import torch
    import torch.nn as nn

    from minimal_shot_av.simulator.spotlight_reflex import DEFAULT_SPOTLIGHT_CONFIG

    torch.set_num_threads(1)

    class GeomMLP(nn.Module):
        def __init__(self, out_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(N_FEATURES),
                nn.Linear(N_FEATURES, HIDDEN), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(HIDDEN, HIDDEN), nn.GELU(), nn.Dropout(0.15),
                nn.Linear(HIDDEN, HIDDEN // 2), nn.GELU(),
                nn.Linear(HIDDEN // 2, out_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    loaded = {}
    for label, ckpt_path_str in model_specs:
        ckpt = torch.load(ckpt_path_str, map_location="cpu", weights_only=False)
        model = GeomMLP(N_TOKENS)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        loaded[label] = {
            "model": model,
            "feat_mean": np.asarray(ckpt["feat_mean"], dtype=np.float32),
            "feat_std": np.asarray(ckpt["feat_std"], dtype=np.float32),
        }

    _WORKER["torch"] = torch
    _WORKER["models"] = loaded
    _WORKER["profiles"] = profiles
    _WORKER["spotlight_config"] = DEFAULT_SPOTLIGHT_CONFIG


def _planner_action(active_scenario, position, world_state, perception, speed_mps: float, chosen_name: str):
    import math

    from minimal_shot_av.simulator.planner import PlannedAction
    from minimal_shot_av.simulator.spotlight_reflex import _planning_heading, generate_maneuver_candidates

    cfg = _WORKER["spotlight_config"]
    heading = _planning_heading(position, world_state, perception, active_scenario, cfg)
    by_name = {candidate.name: candidate for candidate in generate_maneuver_candidates(position, heading, speed_mps, cfg)}
    candidate = by_name.get(chosen_name) or by_name["maintain"]
    next_point = candidate.trajectory[cfg.trajectory.action_index]
    vec = (next_point[0] - position[0], next_point[1] - position[1])
    dist = math.hypot(*vec)
    direction = (vec[0] / dist, vec[1] / dist) if dist > 1e-8 else (1.0, 0.0)
    return PlannedAction(direction=direction, speed=dist, mode=f"{chosen_name}", score=0.0), {}


def _summarize_rollout(rollout) -> dict[str, Any]:
    steps = rollout.steps or []
    clearances = np.asarray([float(step.min_obstacle_distance) for step in steps], dtype=np.float32)
    speeds = np.asarray([float(step.speed) for step in steps], dtype=np.float32)
    progress = np.asarray([float(step.progress or 0.0) for step in steps], dtype=np.float32)
    interventions = np.asarray([1.0 if step.intervention else 0.0 for step in steps], dtype=np.float32)
    stalls = np.asarray([1.0 if step.stall else 0.0 for step in steps], dtype=np.float32)

    if len(clearances) == 0:
        min_clearance = math.inf
        p05_clearance = math.inf
    else:
        min_clearance = float(clearances.min())
        p05_clearance = float(np.quantile(clearances, 0.05))

    if len(speeds) >= 2:
        accel = np.diff(speeds)
        mean_abs_accel = float(np.mean(np.abs(accel)))
    else:
        accel = np.zeros(0, dtype=np.float32)
        mean_abs_accel = 0.0
    if len(accel) >= 2:
        jerk = np.diff(accel)
        mean_abs_jerk = float(np.mean(np.abs(jerk)))
    else:
        mean_abs_jerk = 0.0

    threshold_idx = np.where(clearances < 1.0)[0]
    first_sub_1m_s = None if len(threshold_idx) == 0 else float(threshold_idx[0])

    return {
        "pass": bool(rollout.success),
        "collision": bool(rollout.collision),
        "reached_goal": bool(rollout.reached_goal),
        "min_clearance_m": round(min_clearance, 4),
        "p05_clearance_m": round(p05_clearance, 4),
        "mean_progress_m": round(float(progress.mean()) if len(progress) else 0.0, 4),
        "mean_abs_accel": round(mean_abs_accel, 4),
        "mean_abs_jerk": round(mean_abs_jerk, 4),
        "intervention_rate": round(float(interventions.mean()) if len(interventions) else 0.0, 4),
        "stall_rate": round(float(stalls.mean()) if len(stalls) else 0.0, 4),
        "first_sub_1m_step": first_sub_1m_s,
        "steps": len(steps),
    }


def _run_metric_task(task: MetricTask) -> dict[str, Any]:
    from minimal_shot_av.simulator.compositional_scenarios import generate_compositional_scenario
    from minimal_shot_av.simulator.policy import RolloutConfig, _run_rollout, run_spotlight_reflex_policy

    torch = _WORKER["torch"]
    profile = _WORKER["profiles"][task.profile_idx]
    scenario = generate_compositional_scenario(task.seed + task.case_idx * 1_000 + task.profile_idx * 100_000, task.suite, profile=profile)
    cfg = RolloutConfig(spotlight=_WORKER["spotlight_config"])

    agent_results: dict[str, Any] = {}
    spotlight_rollout = run_spotlight_reflex_policy(scenario)
    agent_results["spotlight"] = _summarize_rollout(spotlight_rollout)

    for label, payload in _WORKER["models"].items():
        model = payload["model"]
        feat_mean = payload["feat_mean"]
        feat_std = payload["feat_std"]

        def planner(active_scenario, position, world_state, perception, config):
            features = _obs_features(world_state, perception, config.step_size)
            x = torch.from_numpy((features - feat_mean) / feat_std).unsqueeze(0)
            with torch.no_grad():
                chosen_idx = int(model(x).argmax(dim=1).item())
            return _planner_action(active_scenario, position, world_state, perception, config.step_size, TOKEN_ORDER[chosen_idx])

        rollout = _run_rollout(scenario, planner, cfg)
        agent_results[label] = _summarize_rollout(rollout)

    return {
        "profile_idx": task.profile_idx,
        "suite": task.suite,
        "seed": task.seed,
        "case_idx": task.case_idx,
        "agents": agent_results,
    }


def _metric_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {"overall": {}, "by_suite": {}}
    if not rows:
        return report

    agent_names = sorted(rows[0]["agents"].keys())
    metrics = (
        "pass",
        "collision",
        "min_clearance_m",
        "p05_clearance_m",
        "mean_progress_m",
        "mean_abs_accel",
        "mean_abs_jerk",
        "intervention_rate",
        "stall_rate",
    )

    def summarize(subset: list[dict[str, Any]]) -> dict[str, Any]:
        bucket: dict[str, Any] = {}
        for agent_name in agent_names:
            agent_rows = [row["agents"][agent_name] for row in subset]
            bucket[agent_name] = {
                metric: round(float(np.mean([float(ar[metric]) for ar in agent_rows])), 4)
                for metric in metrics
            }
        return bucket

    report["overall"] = summarize(rows)
    for suite in sorted({row["suite"] for row in rows}):
        report["by_suite"][suite] = summarize([row for row in rows if row["suite"] == suite])
    return report


def _closed_loop_metrics(args: argparse.Namespace) -> dict[str, Any]:
    suites = tuple(part.strip() for part in args.suites.split(",") if part.strip())
    invalid = [suite for suite in suites if suite not in SUITE_CASES]
    if invalid:
        raise ValueError(f"invalid suites: {invalid}")
    profiles = build_lhs_profiles(args.profiles, seed=args.profile_seed)
    tasks = [
        MetricTask(profile_idx=profile_idx, suite=suite, case_idx=case_idx, seed=seed)
        for profile_idx in range(len(profiles))
        for suite in suites
        for case_idx in range(SUITE_CASES[suite])
        for seed in range(args.seed_start, args.seed_end + 1)
    ]
    model_specs = [
        (label, str(path))
        for label, path in _checkpoint_specs()
        if label.startswith("dagger_") and path.is_file()
    ]
    with multiprocessing.Pool(
        args.workers,
        initializer=_init_worker,
        initargs=(model_specs, profiles),
    ) as pool:
        rows = pool.map(_run_metric_task, tasks)
    return {
        "config": {
            "profiles": args.profiles,
            "seed_start": args.seed_start,
            "seed_end": args.seed_end,
            "suites": list(suites),
            "workers": args.workers,
            "profile_seed": args.profile_seed,
            "tasks": len(tasks),
        },
        "summary": _metric_summary(rows),
    }


def main() -> None:
    args = _parse_args()
    datasets = {name: _load_dataset(path) for name, path in _dataset_specs() if path.is_dir()}
    report = {
        "schema": "dagger_representation_audit_v1",
        "dataset_drift": _dataset_drift_report(datasets),
        "checkpoint_source_accuracy": _eval_checkpoints(datasets),
        "closed_loop_metrics": _closed_loop_metrics(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
