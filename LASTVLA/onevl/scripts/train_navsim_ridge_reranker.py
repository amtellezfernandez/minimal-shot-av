#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def image_path_from_item(item: dict) -> str:
    if item.get("images"):
        return item["images"][0]
    return item["messages"][0]["content"][0]["image"]


def prompt_text_from_item(item: dict) -> str:
    content = item["messages"][0]["content"]
    if isinstance(content, str):
        return content
    return content[1]["text"]


def parse_prompt_stats(text: str):
    def pair(label):
        match = re.search(label + r":\s*\[([^\]]+)\]", text)
        if not match:
            return [0.0, 0.0]
        vals = [float(x.strip()) for x in match.group(1).split(",")[:2]]
        return vals if len(vals) == 2 else [0.0, 0.0]

    def hist_last():
        match = re.search(
            r"Historical trajectory:\s*\[\[([^\]]+)\],\s*\[([^\]]+)\],\s*\[([^\]]+)\]\]",
            text,
        )
        if not match:
            return [0.0, 0.0, 0.0]
        vals = [float(x.strip()) for x in match.group(3).split(",")[:3]]
        return vals if len(vals) == 3 else [0.0, 0.0, 0.0]

    vel = pair("Velocity")
    acc = pair("Acceleration")
    hist = hist_last()
    return {
        "cmd_forward": 1.0 if "MOVE FORWARD" in text else 0.0,
        "vel_x": vel[0],
        "vel_y": vel[1],
        "acc_x": acc[0],
        "acc_y": acc[1],
        "hist_x": hist[0],
        "hist_y": hist[1],
        "hist_h": hist[2],
    }


def trajectory_features(traj, orig_len: int):
    arr = np.asarray(traj, dtype=float)
    if len(arr) < 2:
        arr = np.vstack([arr, arr]) if len(arr) == 1 else np.zeros((2, 3))
    diffs = np.diff(arr[:, :2], axis=0)
    step_d = np.linalg.norm(diffs, axis=1)
    headings = arr[:, 2]
    heading_d = np.diff(headings)
    curv = np.divide(np.abs(heading_d), step_d + 1e-6)
    acc2 = np.diff(step_d) if len(step_d) > 1 else np.array([0.0])
    jerk = np.diff(acc2) if len(acc2) > 1 else np.array([0.0])
    final = arr[-1]
    return {
        "orig_len": float(orig_len),
        "is_len8": 1.0 if orig_len == 8 else 0.0,
        "is_padded": 1.0 if orig_len < 8 else 0.0,
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "final_h": float(final[2]),
        "max_abs_y": float(np.max(np.abs(arr[:, 1]))),
        "mean_abs_y": float(np.mean(np.abs(arr[:, 1]))),
        "path_len": float(np.sum(step_d)),
        "mean_step": float(np.mean(step_d)),
        "min_step": float(np.min(step_d)),
        "max_step": float(np.max(step_d)),
        "std_step": float(np.std(step_d)),
        "progress_ratio": float(final[0] / (np.sum(step_d) + 1e-6)),
        "heading_change": float(headings[-1] - headings[0]),
        "mean_abs_heading_rate": float(np.mean(np.abs(heading_d))) if len(heading_d) else 0.0,
        "max_abs_heading_rate": float(np.max(np.abs(heading_d))) if len(heading_d) else 0.0,
        "mean_curv": float(np.mean(curv)) if len(curv) else 0.0,
        "max_curv": float(np.max(curv)) if len(curv) else 0.0,
        "mean_acc2": float(np.mean(acc2)) if len(acc2) else 0.0,
        "max_abs_acc2": float(np.max(np.abs(acc2))) if len(acc2) else 0.0,
        "max_abs_jerk": float(np.max(np.abs(jerk))) if len(jerk) else 0.0,
        "backward_steps": float(np.sum(diffs[:, 0] < -1e-3)),
    }


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


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def fit_ridge(X, y, alpha=1.0):
    Xs, mean, std = standardize(X)
    Xb = np.concatenate([np.ones((len(Xs), 1)), Xs], axis=1)
    reg = np.eye(Xb.shape[1]) * alpha
    reg[0, 0] = 0.0
    weights = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return weights, mean, std


def predict_ridge(model, X):
    weights, mean, std = model
    Xs, _, _ = standardize(X, mean, std)
    Xb = np.concatenate([np.ones((len(Xs), 1)), Xs], axis=1)
    return Xb @ weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--token-map", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated ridge alphas",
    )
    args = parser.parse_args()

    data = load_json(args.candidate_json)
    token_map = load_json(args.token_map)
    rows_by_candidate, avg_by_candidate = load_score_rows(args.score_dir)
    score_lookup = {}
    for candidate_index, rows in rows_by_candidate.items():
        for row in rows:
            score_lookup.setdefault(row["token"], {})[candidate_index] = float(row["score"])

    examples = []
    for item in data:
        prompt_stats = parse_prompt_stats(prompt_text_from_item(item))
        stem = Path(image_path_from_item(item)).stem
        token = token_map[stem]["token"]
        if token not in score_lookup:
            continue
        for candidate in item["candidates"]:
            candidate_index = candidate["candidate_id"]
            avg_log_prob = candidate.get("avg_log_prob")
            if avg_log_prob is None or not math.isfinite(avg_log_prob):
                avg_log_prob = -50.0
                logprob_finite = 0.0
            else:
                logprob_finite = 1.0
            feat = {
                **prompt_stats,
                **trajectory_features(candidate["trajectory"], len(candidate["trajectory"])),
                "candidate_id": float(candidate_index),
                "avg_entropy": float(candidate.get("avg_entropy", 0.0) or 0.0),
                "seq_confidence": float(candidate.get("seq_confidence", 0.0) or 0.0),
                "avg_log_prob": float(avg_log_prob),
                "logprob_finite": logprob_finite,
                "source_top1": 1.0 if candidate.get("source") == "onevl_top1" else 0.0,
            }
            examples.append(
                {
                    "token": token,
                    "candidate_id": candidate_index,
                    "features": feat,
                    "score": score_lookup[token][candidate_index],
                }
            )

    feature_names = sorted(examples[0]["features"].keys())
    by_token = {}
    for ex in examples:
        by_token.setdefault(ex["token"], []).append(ex)

    def matrix(exs):
        X = np.array([[ex["features"][name] for name in feature_names] for ex in exs], dtype=float)
        y = np.array([ex["score"] for ex in exs], dtype=float)
        return X, y

    alphas = [float(v) for v in args.alphas.split(",") if v]
    cv_rows = []
    for alpha in alphas:
        selected_scores = []
        selected_ids = []
        for token, exs in by_token.items():
            train = [ex for train_token, xs in by_token.items() if train_token != token for ex in xs]
            Xtr, ytr = matrix(train)
            model = fit_ridge(Xtr, ytr, alpha=alpha)
            Xte, _ = matrix(exs)
            preds = predict_ridge(model, Xte)
            best = int(np.argmax(preds))
            selected_scores.append(exs[best]["score"])
            selected_ids.append(exs[best]["candidate_id"])
        cv_rows.append(
            {
                "alpha": alpha,
                "mean_score": float(np.mean(selected_scores)),
                "candidate_hist": {str(i): selected_ids.count(i) for i in range(4)},
            }
        )

    best_cv = max(cv_rows, key=lambda row: row["mean_score"])
    alpha = best_cv["alpha"]
    selected_scores = []
    selected_ids = []
    for token, exs in by_token.items():
        train = [ex for train_token, xs in by_token.items() if train_token != token for ex in xs]
        Xtr, ytr = matrix(train)
        model = fit_ridge(Xtr, ytr, alpha=alpha)
        Xte, _ = matrix(exs)
        preds = predict_ridge(model, Xte)
        best = int(np.argmax(preds))
        selected_scores.append(exs[best]["score"])
        selected_ids.append(exs[best]["candidate_id"])

    candidate_scores = {str(idx): float(avg_by_candidate[idx]["score"]) for idx in sorted(avg_by_candidate)}
    top1_score = candidate_scores["0"]
    best_fixed_score = max(candidate_scores.values())
    oracle_at_4_score = np.mean([max(ex["score"] for ex in exs) for exs in by_token.values()])
    summary = {
        "scene_count": len(by_token),
        "feature_count": len(feature_names),
        "alphas": cv_rows,
        "selected_alpha": alpha,
        "top1_score": top1_score,
        "best_fixed_score": best_fixed_score,
        "oracle_at_4_score": float(oracle_at_4_score),
        "reranker_loso_score": float(np.mean(selected_scores)),
        "gap_closed_vs_top1": float((np.mean(selected_scores) - top1_score) / (oracle_at_4_score - top1_score))
        if oracle_at_4_score > top1_score
        else 0.0,
        "selected_candidate_hist": {str(i): selected_ids.count(i) for i in range(4)},
    }
    args.summary_out.write_text(json.dumps(summary, indent=2))

    if args.model_out:
        X, y = matrix(examples)
        weights, mean, std = fit_ridge(X, y, alpha=alpha)
        ranked = sorted(zip(feature_names, weights[1:]), key=lambda kv: abs(kv[1]), reverse=True)
        args.model_out.write_text(
            json.dumps(
                {
                    "alpha": alpha,
                    "intercept": float(weights[0]),
                    "feature_names": feature_names,
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "weights": weights[1:].tolist(),
                    "top_abs_weights": [[name, float(weight)] for name, weight in ranked[:20]],
                },
                indent=2,
            )
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
