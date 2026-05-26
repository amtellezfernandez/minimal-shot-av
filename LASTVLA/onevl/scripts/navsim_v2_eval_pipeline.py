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


SUMMARY_TOKENS = {
    "extended_pdm_score_stage_one",
    "extended_pdm_score_stage_two",
    "extended_pdm_score_combined",
}


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def candidate_list_from_item(item: dict):
    if "candidates" not in item:
        raise ValueError("candidate JSON must contain a candidates list")
    return item["candidates"]


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


def candidate_count(data: list[dict]) -> int:
    return max(len(candidate_list_from_item(item)) for item in data)


def build_submission_pickles(candidate_json_path: Path, output_dir: Path, team_name: str) -> list[Path]:
    from navsim.common.dataclasses import Trajectory

    data = load_json(candidate_json_path)
    count = candidate_count(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    for candidate_index in range(count):
        first_stage_preds = {}
        second_stage_preds = {}
        for item in data:
            token = item.get("token")
            if not token:
                raise ValueError("NAVSIM v2 candidate items must contain a token field")
            stage = item.get("stage", "first")
            candidates = candidate_list_from_item(item)
            if candidate_index >= len(candidates):
                continue
            traj = candidates[candidate_index].get("trajectory") or []
            traj = pad_or_trim_trajectory(traj)
            if traj is None:
                continue
            pred = Trajectory(np.array(traj, dtype=np.float32))
            if stage == "second":
                second_stage_preds[token] = pred
            else:
                first_stage_preds[token] = pred

        payload = {
            "team_name": team_name,
            "authors": ["Codex"],
            "email": "none@example.com",
            "institution": "local",
            "country / region": "FR",
            "first_stage_predictions": [first_stage_preds],
            "second_stage_predictions": [second_stage_preds],
        }
        out_path = output_dir / f"candidate_{candidate_index}.pkl"
        with out_path.open("wb") as f:
            pickle.dump(payload, f)
        outputs.append(out_path)
    return outputs


def run_cmd(cmd: list[str], env: dict[str, str]):
    subprocess.run(cmd, check=True, env=env)


def navsim_env(navsim_repo: Path, openscene_data_root: Path, navsim_exp_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(navsim_repo)
    env["OPENSCENE_DATA_ROOT"] = str(openscene_data_root)
    env["NAVSIM_EXP_ROOT"] = str(navsim_exp_root)
    return env


def run_metric_cache(
    python_bin: str,
    navsim_repo: Path,
    openscene_data_root: Path,
    navsim_exp_root: Path,
    cache_path: Path,
    split: str,
):
    cmd = [
        python_bin,
        "-m",
        "navsim.planning.script.run_metric_caching",
        f"train_test_split={split}",
        "worker=sequential",
        f"metric_cache_path={cache_path}",
        f"output_dir={cache_path / 'metadata'}",
    ]
    run_cmd(cmd, navsim_env(navsim_repo, openscene_data_root, navsim_exp_root))


def run_candidate_scores(
    python_bin: str,
    navsim_repo: Path,
    openscene_data_root: Path,
    navsim_exp_root: Path,
    submission_dir: Path,
    score_dir: Path,
    metric_cache_path: Path,
    split: str,
):
    score_dir.mkdir(parents=True, exist_ok=True)
    env = navsim_env(navsim_repo, openscene_data_root, navsim_exp_root)
    for submission_path in sorted(submission_dir.glob("candidate_*.pkl")):
        candidate_name = submission_path.stem
        candidate_score_dir = score_dir / candidate_name
        candidate_score_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            python_bin,
            "-m",
            "navsim.planning.script.run_pdm_score_from_submission",
            f"train_test_split={split}",
            f"submission_file_path={submission_path}",
            f"metric_cache_path={metric_cache_path}",
            f"output_dir={candidate_score_dir}",
        ]
        run_cmd(cmd, env)


def latest_csv(candidate_dir: Path) -> Path | None:
    csv_files = sorted(candidate_dir.glob("*.csv"))
    return csv_files[-1] if csv_files else None


def load_score_rows(score_dir: Path):
    rows_by_candidate = {}
    combined_by_candidate = {}
    summary_by_candidate = {}
    for candidate_dir in sorted(score_dir.glob("candidate_*")):
        csv_path = latest_csv(candidate_dir)
        if csv_path is None:
            continue
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        candidate_index = int(candidate_dir.name.split("_")[-1])
        rows_by_candidate[candidate_index] = [row for row in rows if row["token"] not in SUMMARY_TOKENS]
        summaries = {row["token"]: row for row in rows if row["token"] in SUMMARY_TOKENS}
        summary_by_candidate[candidate_index] = summaries
        combined_by_candidate[candidate_index] = summaries["extended_pdm_score_combined"]
    return rows_by_candidate, combined_by_candidate, summary_by_candidate


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_oracle_summary(score_dir: Path, output_path: Path):
    rows_by_candidate, combined_by_candidate, summary_by_candidate = load_score_rows(score_dir)
    candidate_ids = sorted(rows_by_candidate)
    if not candidate_ids:
        raise RuntimeError(f"no candidate score CSVs found under {score_dir}")

    candidate_scores = {str(idx): safe_float(combined_by_candidate[idx]["score"]) for idx in candidate_ids}
    stage_one_scores = {
        str(idx): safe_float(summary_by_candidate[idx]["extended_pdm_score_stage_one"]["score"]) for idx in candidate_ids
    }
    stage_two_scores = {
        str(idx): safe_float(summary_by_candidate[idx]["extended_pdm_score_stage_two"]["score"]) for idx in candidate_ids
    }
    top1_score = candidate_scores["0"]
    best_fixed_candidate = max(candidate_scores, key=candidate_scores.get)

    token_to_rows = {}
    for candidate_idx, rows in rows_by_candidate.items():
        for row in rows:
            token_to_rows.setdefault(row["token"], {})[candidate_idx] = row

    per_token = []
    for token, candidate_rows in sorted(token_to_rows.items()):
        scores = [safe_float(candidate_rows[idx]["score"]) for idx in candidate_ids if idx in candidate_rows]
        if not scores:
            continue
        oracle_score = max(scores)
        per_token.append(
            {
                "token": token,
                "scores": scores,
                "oracle_score": oracle_score,
                "oracle_candidate": scores.index(oracle_score),
            }
        )

    token_oracle_proxy = (
        sum(row["oracle_score"] for row in per_token) / len(per_token) if per_token else float("nan")
    )
    summary = {
        "scene_count": len(per_token),
        "candidate_count": len(candidate_ids),
        "top1_candidate": 0,
        "top1_epdms": top1_score,
        "best_fixed_candidate": int(best_fixed_candidate),
        "best_fixed_epdms": candidate_scores[best_fixed_candidate],
        "fixed_candidate_epdms": candidate_scores,
        "fixed_candidate_stage_one_epdms": stage_one_scores,
        "fixed_candidate_stage_two_epdms": stage_two_scores,
        "token_oracle_proxy_at_k_score": token_oracle_proxy,
        "token_oracle_proxy_gap": token_oracle_proxy - top1_score,
        "per_token": per_token,
    }
    output_path.write_text(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--navsim-repo", type=Path, required=True)
    parser.add_argument("--openscene-data-root", type=Path, required=True)
    parser.add_argument("--navsim-exp-root", type=Path, required=True)
    parser.add_argument("--metric-cache-path", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--oracle-summary-out", type=Path, required=True)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--team-name", default="codex-onevl-v2")
    parser.add_argument("--split", default="navhard_two_stage")
    parser.add_argument("--skip-metric-cache", action="store_true")
    args = parser.parse_args()

    sys.path.insert(0, str(args.navsim_repo))
    build_submission_pickles(args.candidate_json, args.submission_dir, args.team_name)
    if not args.skip_metric_cache:
        run_metric_cache(
            args.python_bin,
            args.navsim_repo,
            args.openscene_data_root,
            args.navsim_exp_root,
            args.metric_cache_path,
            args.split,
        )
    run_candidate_scores(
        args.python_bin,
        args.navsim_repo,
        args.openscene_data_root,
        args.navsim_exp_root,
        args.submission_dir,
        args.score_dir,
        args.metric_cache_path,
        args.split,
    )
    summary = build_oracle_summary(args.score_dir, args.oracle_summary_out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
