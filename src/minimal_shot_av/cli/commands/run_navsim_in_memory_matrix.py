#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from minimal_shot_av.model.navsim_maneuver_token_agent import (
    NAVSIM_AGENT_VARIANTS,
    ManeuverTokenNavsimAgent,
)


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "corl2027" / "navsim_matrix"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ManeuverToken variants on NAVSIM using official in-memory PDM scoring. "
            "This avoids NAVSIM metric-cache serialization, which can be slow on small machines."
        )
    )
    parser.add_argument("--navsim-devkit-root", type=Path, default=_env_path("NAVSIM_DEVKIT_ROOT"))
    parser.add_argument("--openscene-data-root", type=Path, default=_env_path("OPENSCENE_DATA_ROOT"))
    parser.add_argument("--nuplan-maps-root", type=Path, default=_env_path("NUPLAN_MAPS_ROOT"))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--split", default="mini")
    parser.add_argument("--max-scenes", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    run_matrix(args)


def run_matrix(args: argparse.Namespace) -> None:
    _require_path(args.navsim_devkit_root, "NAVSIM devkit root")
    _require_path(args.openscene_data_root, "OpenScene data root")
    _require_path(args.nuplan_maps_root, "nuPlan maps root")
    _set_env(args)

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from navsim.common.dataclasses import Scene, SensorConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.metric_caching.metric_cache_processor import MetricCacheProcessor
    from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario

    config_dir = args.navsim_devkit_root / "navsim" / "planning" / "script" / "config" / "pdm_scoring"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="default_run_pdm_score",
            overrides=[
                f"train_test_split={args.split}",
                f"train_test_split.scene_filter.max_scenes={args.max_scenes}",
                "worker=sequential",
                "agent=maneuver_token_agent",
                "metric_cache_path=/tmp/navsim_in_memory_unused",
                "experiment_name=maneuver_token_in_memory",
            ],
        )

    scene_loader = SceneLoader(
        synthetic_sensor_path=None,
        original_sensor_path=None,
        data_path=Path(cfg.navsim_log_path),
        synthetic_scenes_path=Path(cfg.synthetic_scenes_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    processor = MetricCacheProcessor(
        cache_path="/tmp/navsim_in_memory_unused",
        force_feature_computation=True,
        proposal_sampling=instantiate(cfg.proposal_sampling),
    )
    simulator = instantiate(cfg.simulator)
    scorer = instantiate(cfg.scorer)
    trajectory_sampling = instantiate(cfg.proposal_sampling)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    variant_rows: dict[str, list[dict[str, Any]]] = {variant: [] for variant in NAVSIM_AGENT_VARIANTS}
    tokens = scene_loader.tokens_stage_one
    for scene_index, token in enumerate(tokens, start=1):
        scene_dict = scene_loader.scene_frames_dicts[token]
        scene = Scene.from_scene_dict_list(
            scene_dict,
            None,
            num_history_frames=cfg.train_test_split.scene_filter.num_history_frames,
            num_future_frames=cfg.train_test_split.scene_filter.num_future_frames,
            sensor_config=SensorConfig.build_no_sensors(),
        )
        scenario = NavSimScenario(scene, map_root=str(args.nuplan_maps_root), map_version="nuplan-maps-v1.0")
        print(
            f"scene {scene_index}/{len(tokens)} token={token} "
            f"log={scene.scene_metadata.log_name} map={scene.scene_metadata.map_name}",
            flush=True,
        )
        metric_cache = processor.compute_metric_cache(scenario)
        agent_input = scene_loader.get_agent_input_from_token(token)

        for variant in NAVSIM_AGENT_VARIANTS:
            traffic_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
            agent = ManeuverTokenNavsimAgent(variant=variant, trajectory_sampling=trajectory_sampling)
            trajectory = agent.compute_trajectory(agent_input)
            score_row, _ego_simulated_states = pdm_score(
                metric_cache=metric_cache,
                model_trajectory=trajectory,
                future_sampling=simulator.proposal_sampling,
                simulator=simulator,
                scorer=scorer,
                traffic_agents_policy=traffic_policy,
            )
            row = score_row.iloc[0].to_dict()
            row["score"] = _final_score(row)
            row["scene_id"] = token
            row["variant"] = variant
            all_rows.append(row)
            variant_rows[variant].append(row)
            print(f"  {variant}: score={row.get('score')} progress={row.get('ego_progress')}", flush=True)

    for variant, rows in variant_rows.items():
        pd.DataFrame(rows).to_csv(args.output_dir / f"{variant}.csv", index=False)
    pd.DataFrame(all_rows).to_csv(args.output_dir / "all_variants.csv", index=False)


def _final_score(row: dict[str, Any]) -> float:
    weighted_metrics = np.asarray(row["weighted_metrics"], dtype=float)
    weights = np.asarray(row["weighted_metrics_array"], dtype=float)
    denominator = float(weights.sum())
    if denominator <= 0.0:
        return float("nan")
    weighted = float((weighted_metrics * weights).sum() / denominator)
    return float(row["multiplicative_metrics_prod"] * weighted)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def _require_path(path: Path | None, label: str) -> None:
    if path is None or not path.exists():
        raise RuntimeError(f"{label} is required and must exist")


def _set_env(args: argparse.Namespace) -> None:
    os.environ["NAVSIM_DEVKIT_ROOT"] = str(args.navsim_devkit_root)
    os.environ["OPENSCENE_DATA_ROOT"] = str(args.openscene_data_root)
    os.environ["NUPLAN_MAPS_ROOT"] = str(args.nuplan_maps_root)
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")


if __name__ == "__main__":
    main()
