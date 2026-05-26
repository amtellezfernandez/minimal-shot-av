#!/usr/bin/env python3
import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def image_path_from_item(item: dict) -> str:
    if item.get("images"):
        return item["images"][0]
    return item["messages"][0]["content"][0]["image"]


def candidate_list_from_item(item: dict):
    if "candidates" not in item:
        raise ValueError("candidate JSON must contain a candidates list")
    return item["candidates"]


def map_tokens(subset_path: Path, navsim_logs_dir: Path, output_path: Path) -> dict:
    data = load_json(subset_path)
    by_log = {}
    for item in data:
        image_path = image_path_from_item(item)
        path = Path(image_path)
        log_name = path.parts[-3]
        stem = path.stem
        by_log.setdefault(log_name, set()).add(stem)

    result = {}
    for log_name, stems in by_log.items():
        with (navsim_logs_dir / f"{log_name}.pkl").open("rb") as f:
            frames = pickle.load(f)
        for frame in frames:
            stem = Path(frame["cams"]["CAM_F0"]["data_path"]).stem
            if stem in stems:
                result[stem] = {"token": frame["token"], "log_name": log_name}

    missing = sorted(set().union(*by_log.values()) - set(result))
    if missing:
        raise RuntimeError(f"missing {len(missing)} image stems in navsim logs: {missing[:10]}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


def pad_or_trim_trajectory(traj):
    traj = [list(point) for point in traj]
    if not traj:
        return None
    if len(traj) < 8:
        last = list(traj[-1])
        while len(traj) < 8:
            traj.append(list(last))
    elif len(traj) > 8:
        traj = traj[:8]
    return traj


def build_submission_pickles(
    candidate_json_path: Path,
    token_map_path: Path,
    output_dir: Path,
    team_name: str,
) -> list[Path]:
    from navsim.common.dataclasses import Trajectory

    data = load_json(candidate_json_path)
    token_map = load_json(token_map_path)
    candidate_count = max(len(candidate_list_from_item(item)) for item in data)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for candidate_index in range(candidate_count):
        preds = {}
        for item in data:
            stem = Path(image_path_from_item(item)).stem
            token = token_map[stem]["token"]
            traj = candidate_list_from_item(item)[candidate_index].get("trajectory") or []
            traj = pad_or_trim_trajectory(traj)
            if traj is None:
                continue
            preds[token] = Trajectory(np.array(traj, dtype=np.float32))

        payload = {
            "team_name": team_name,
            "authors": ["Codex"],
            "email": "none@example.com",
            "institution": "local",
            "country / region": "FR",
            "predictions": [preds],
        }
        out_path = output_dir / f"candidate_{candidate_index}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f)
        outputs.append(out_path)
    return outputs


def run_cmd(cmd: list[str], env: dict[str, str]):
    subprocess.run(cmd, check=True, env=env)


def run_metric_cache(
    python_bin: str,
    navsim_repo: Path,
    cache_path: Path,
    log_names: list[str],
    tokens: list[str],
    env: dict[str, str],
):
    cmd = [
        python_bin,
        "-m",
        "navsim.planning.script.run_metric_caching",
        "train_test_split=test",
        "train_test_split.scene_filter.frame_interval=1",
        f"train_test_split.scene_filter.log_names=[{','.join(log_names)}]",
        f"train_test_split.scene_filter.tokens=[{','.join(tokens)}]",
        "worker=sequential",
        f"cache.cache_path={cache_path}",
        f"output_dir={cache_path / 'metadata'}",
    ]
    run_cmd(cmd, {**env, "PYTHONPATH": str(navsim_repo)})


def run_candidate_scores(
    python_bin: str,
    navsim_repo: Path,
    submission_dir: Path,
    score_dir: Path,
    metric_cache_path: Path,
    env: dict[str, str],
):
    score_dir.mkdir(parents=True, exist_ok=True)
    for submission_path in sorted(submission_dir.glob("candidate_*.pkl")):
        candidate_name = submission_path.stem
        candidate_score_dir = score_dir / candidate_name
        candidate_score_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            "-m",
            "navsim.planning.script.run_pdm_score_from_submission",
            "train_test_split=test",
            f"submission_file_path={submission_path}",
            f"metric_cache_path={metric_cache_path}",
            f"output_dir={candidate_score_dir}",
        ]
        run_cmd(cmd, {**env, "PYTHONPATH": str(navsim_repo)})


def load_score_rows(score_dir: Path):
    rows_by_candidate = {}
    avg_by_candidate = {}
    for candidate_dir in sorted(score_dir.glob("candidate_*")):
        csv_files = sorted(candidate_dir.glob("*.csv"))
        if not csv_files:
            continue
        with csv_files[-1].open(newline="") as f:
            rows = list(csv.DictReader(f))
        candidate_index = int(candidate_dir.name.split("_")[-1])
        rows_by_candidate[candidate_index] = [row for row in rows if row["token"] != "average"]
        avg_by_candidate[candidate_index] = next(row for row in rows if row["token"] == "average")
    return rows_by_candidate, avg_by_candidate


def build_oracle_summary(score_dir: Path, output_path: Path):
    rows_by_candidate, avg_by_candidate = load_score_rows(score_dir)
    tokens = [row["token"] for row in rows_by_candidate[min(rows_by_candidate)]]
    per_scene = []
    for token in tokens:
        scores = [
            float(next(row for row in rows_by_candidate[idx] if row["token"] == token)["score"])
            for idx in sorted(rows_by_candidate)
        ]
        oracle_score = max(scores)
        oracle_candidate = scores.index(oracle_score)
        per_scene.append(
            {
                "token": token,
                "scores": scores,
                "oracle_score": oracle_score,
                "oracle_candidate": oracle_candidate,
            }
        )

    candidate_scores = {
        str(idx): float(avg_by_candidate[idx]["score"]) for idx in sorted(avg_by_candidate)
    }
    top1_score = candidate_scores["0"]
    best_fixed_candidate = max(candidate_scores, key=candidate_scores.get)
    best_fixed_score = candidate_scores[best_fixed_candidate]
    oracle_score = sum(row["oracle_score"] for row in per_scene) / len(per_scene)
    summary = {
        "scene_count": len(tokens),
        "top1_candidate": 0,
        "top1_score": top1_score,
        "best_fixed_candidate": int(best_fixed_candidate),
        "best_fixed_score": best_fixed_score,
        "oracle_at_4_score": oracle_score,
        "oracle_gap": oracle_score - top1_score,
        "candidate_scores": candidate_scores,
        "per_scene": per_scene,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-json", type=Path, required=True)
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--navsim-repo", type=Path, required=True)
    parser.add_argument("--navsim-logs-dir", type=Path, required=True)
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--token-map-out", type=Path, required=True)
    parser.add_argument("--oracle-summary-out", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--team-name", default="codex-onevl")
    args = parser.parse_args()

    env = os.environ.copy()
    token_map = map_tokens(args.subset_json, args.navsim_logs_dir, args.token_map_out)
    build_submission_pickles(
        args.candidate_json,
        args.token_map_out,
        args.submission_dir,
        args.team_name,
    )
    unique_logs = sorted({item["log_name"] for item in token_map.values()})
    unique_tokens = [item["token"] for item in token_map.values()]
    run_metric_cache(
        args.python_bin,
        args.navsim_repo,
        args.metric_cache_path,
        unique_logs,
        unique_tokens,
        env,
    )
    run_candidate_scores(
        args.python_bin,
        args.navsim_repo,
        args.submission_dir,
        args.score_dir,
        args.metric_cache_path,
        env,
    )
    summary = build_oracle_summary(args.score_dir, args.oracle_summary_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
