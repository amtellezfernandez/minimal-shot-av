#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path

import numpy as np

from train_navsim_ridge_reranker import (
    fit_ridge,
    image_path_from_item,
    load_json,
    load_score_rows,
    parse_prompt_stats,
    predict_ridge,
    prompt_text_from_item,
    select_feature_names,
    trajectory_features,
)


def build_examples(candidate_json: Path, token_map_path: Path, score_dir: Path):
    data = load_json(candidate_json)
    token_map = load_json(token_map_path)
    rows_by_candidate, avg_by_candidate = load_score_rows(score_dir)
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

    candidate_scores = {str(idx): float(avg_by_candidate[idx]["score"]) for idx in sorted(avg_by_candidate)}
    return examples, candidate_scores


def matrix(examples, feature_names):
    X = np.array([[ex["features"][name] for name in feature_names] for ex in examples], dtype=float)
    y = np.array([ex["score"] for ex in examples], dtype=float)
    return X, y


def evaluate_examples(examples, feature_names, model):
    by_token = {}
    for ex in examples:
        by_token.setdefault(ex["token"], []).append(ex)

    selected_scores = []
    selected_ids = []
    for token, exs in by_token.items():
        Xte, _ = matrix(exs, feature_names)
        preds = predict_ridge(model, Xte)
        best = int(np.argmax(preds))
        selected_scores.append(exs[best]["score"])
        selected_ids.append(exs[best]["candidate_id"])
    return by_token, selected_scores, selected_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-candidate-json", type=Path, nargs="+", required=True)
    parser.add_argument("--train-token-map", type=Path, nargs="+", required=True)
    parser.add_argument("--train-score-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--test-candidate-json", type=Path, required=True)
    parser.add_argument("--test-token-map", type=Path, required=True)
    parser.add_argument("--test-score-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument(
        "--alphas",
        default="0.01,0.1,1.0,10.0,100.0",
        help="Comma-separated ridge alphas",
    )
    parser.add_argument(
        "--drop-order-features",
        action="store_true",
        help="Remove candidate_id and source_top1 from the probe feature set.",
    )
    args = parser.parse_args()

    if not (
        len(args.train_candidate_json) == len(args.train_token_map) == len(args.train_score_dir)
    ):
        raise SystemExit("train argument counts must match")

    train_examples = []
    train_source_count = 0
    for cand_json, token_map, score_dir in zip(
        args.train_candidate_json, args.train_token_map, args.train_score_dir
    ):
        exs, _ = build_examples(cand_json, token_map, score_dir)
        train_examples.extend(exs)
        train_source_count += 1

    test_examples, test_candidate_scores = build_examples(
        args.test_candidate_json, args.test_token_map, args.test_score_dir
    )

    feature_names = select_feature_names(train_examples, args.drop_order_features)
    train_by_token = {}
    for ex in train_examples:
        train_by_token.setdefault(ex["token"], []).append(ex)

    alphas = [float(v) for v in args.alphas.split(",") if v]
    cv_rows = []
    for alpha in alphas:
        selected_scores = []
        selected_ids = []
        for token, exs in train_by_token.items():
            train_fold = [ex for train_token, xs in train_by_token.items() if train_token != token for ex in xs]
            Xtr, ytr = matrix(train_fold, feature_names)
            model = fit_ridge(Xtr, ytr, alpha=alpha)
            Xte, _ = matrix(exs, feature_names)
            preds = predict_ridge(model, Xte)
            best = int(np.argmax(preds))
            selected_scores.append(exs[best]["score"])
            selected_ids.append(exs[best]["candidate_id"])
        cv_rows.append(
            {
                "alpha": alpha,
                "mean_score": float(np.mean(selected_scores)),
                "candidate_hist": {str(i): selected_ids.count(i) for i in sorted(set(selected_ids))},
            }
        )

    best_cv = max(cv_rows, key=lambda row: row["mean_score"])
    alpha = best_cv["alpha"]

    Xtr, ytr = matrix(train_examples, feature_names)
    model = fit_ridge(Xtr, ytr, alpha=alpha)
    test_by_token, selected_scores, selected_ids = evaluate_examples(test_examples, feature_names, model)

    candidate_ids = sorted({ex["candidate_id"] for ex in test_examples})
    top1_score = test_candidate_scores["0"]
    best_fixed_score = max(test_candidate_scores.values())
    oracle_at_k_score = np.mean([max(ex["score"] for ex in exs) for exs in test_by_token.values()])
    transfer_score = float(np.mean(selected_scores))

    summary = {
        "train_source_count": train_source_count,
        "train_scene_count": len(train_by_token),
        "test_scene_count": len(test_by_token),
        "candidate_count": len(candidate_ids),
        "feature_count": len(feature_names),
        "drop_order_features": args.drop_order_features,
        "alphas": cv_rows,
        "selected_alpha": alpha,
        "top1_score": top1_score,
        "best_fixed_score": best_fixed_score,
        "oracle_at_k_score": float(oracle_at_k_score),
        "transfer_reranker_score": transfer_score,
        "gap_closed_vs_top1": float((transfer_score - top1_score) / (oracle_at_k_score - top1_score))
        if oracle_at_k_score > top1_score
        else 0.0,
        "selected_candidate_hist": {str(i): selected_ids.count(i) for i in candidate_ids},
    }
    args.summary_out.write_text(json.dumps(summary, indent=2))

    if args.model_out:
        weights, mean, std = model
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
