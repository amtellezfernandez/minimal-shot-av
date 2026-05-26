#!/usr/bin/env python3
import argparse
import json
import math
import statistics
from itertools import combinations
from pathlib import Path


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def canonicalize_trajectory(trajectory):
    trajectory = [list(point) for point in (trajectory or [])]
    if trajectory and len(trajectory) < 8:
        trajectory.extend([list(trajectory[-1])] * (8 - len(trajectory)))
    trajectory = trajectory[:8]
    return tuple(round(value, 3) for point in trajectory for value in point)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweeps-dir", type=Path, required=True)
    parser.add_argument("--seed-list", default="0,1,2,3")
    parser.add_argument("--stem-template", default="holdout100_hybrid_k4_seed{seed}_t08_p095")
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    seeds = [int(seed.strip()) for seed in args.seed_list.split(",") if seed.strip()]
    rows = []
    for seed in seeds:
        stem = args.stem_template.format(seed=seed)
        oracle = load_json(args.sweeps_dir / f"{stem}_oracle.json")
        data = load_json(args.sweeps_dir / f"{stem}_candidates.json")
        duplicate_scenes = 0
        duplicate_pairs = 0
        total_pairs = 0
        pair_distances = []
        winner_not_top1 = 0
        for per_scene, item in zip(oracle["per_scene"], data):
            trajectories = [canonicalize_trajectory(candidate.get("trajectory")) for candidate in item["candidates"]]
            if len(set(trajectories)) < len(trajectories):
                duplicate_scenes += 1
            if per_scene["oracle_candidate"] != 0:
                winner_not_top1 += 1
            for traj_a, traj_b in combinations(trajectories, 2):
                total_pairs += 1
                if traj_a == traj_b:
                    duplicate_pairs += 1
                pair_distances.append(math.sqrt(sum((a - b) ** 2 for a, b in zip(traj_a, traj_b))) / len(traj_a))
        rows.append(
            {
                "seed": seed,
                "oracle_at_k_score": oracle["oracle_at_k_score"],
                "oracle_gap": oracle["oracle_gap"],
                "winner_not_top1_scenes": winner_not_top1,
                "winner_not_top1_rate": winner_not_top1 / oracle["scene_count"],
                "scene_duplicate_rate": duplicate_scenes / oracle["scene_count"],
                "pair_duplicate_rate": duplicate_pairs / total_pairs if total_pairs else 0.0,
                "mean_pair_distance": sum(pair_distances) / len(pair_distances),
            }
        )

    summary = {
        "rows": rows,
        "oracle_mean": sum(row["oracle_at_k_score"] for row in rows) / len(rows),
        "oracle_std": statistics.pstdev(row["oracle_at_k_score"] for row in rows),
        "gap_mean": sum(row["oracle_gap"] for row in rows) / len(rows),
        "gap_std": statistics.pstdev(row["oracle_gap"] for row in rows),
        "winner_not_top1_rate_mean": sum(row["winner_not_top1_rate"] for row in rows) / len(rows),
        "scene_duplicate_rate_mean": sum(row["scene_duplicate_rate"] for row in rows) / len(rows),
        "pair_duplicate_rate_mean": sum(row["pair_duplicate_rate"] for row in rows) / len(rows),
        "mean_pair_distance_mean": sum(row["mean_pair_distance"] for row in rows) / len(rows),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
