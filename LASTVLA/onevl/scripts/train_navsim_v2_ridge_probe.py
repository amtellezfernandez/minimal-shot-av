#!/usr/bin/env python3
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from navsim_v2_eval_pipeline import SUMMARY_TOKENS
from train_navsim_ridge_reranker import (
    ORDER_FEATURES,
    fit_ridge,
    parse_prompt_stats,
    predict_ridge,
    prompt_text_from_item,
    select_feature_names,
    trajectory_features,
)


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def latest_csv(candidate_dir: Path) -> Path | None:
    csv_files = sorted(candidate_dir.glob("*.csv"))
    return csv_files[-1] if csv_files else None


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_v2_score_rows(score_dir: Path):
    rows_by_candidate = {}
    official_combined = {}
    stage_one = {}
    stage_two = {}
    for candidate_dir in sorted(score_dir.glob("candidate_*")):
        csv_path = latest_csv(candidate_dir)
        if csv_path is None:
            continue
        candidate_index = int(candidate_dir.name.split("_")[-1])
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        summaries = {row["token"]: row for row in rows if row["token"] in SUMMARY_TOKENS}
        rows_by_candidate[candidate_index] = [row for row in rows if row["token"] not in SUMMARY_TOKENS]
        official_combined[candidate_index] = safe_float(summaries["extended_pdm_score_combined"]["score"])
        stage_one[candidate_index] = safe_float(summaries["extended_pdm_score_stage_one"]["score"])
        stage_two[candidate_index] = safe_float(summaries["extended_pdm_score_stage_two"]["score"])
    return rows_by_candidate, official_combined, stage_one, stage_two


def candidate_features(item: dict, candidate: dict) -> dict:
    avg_log_prob = candidate.get("avg_log_prob")
    if avg_log_prob is None or not math.isfinite(avg_log_prob):
        avg_log_prob = -50.0
        logprob_finite = 0.0
    else:
        logprob_finite = 1.0
    return {
        **parse_prompt_stats(prompt_text_from_item(item)),
        **trajectory_features(candidate["trajectory"], len(candidate["trajectory"])),
        "candidate_id": float(candidate["candidate_id"]),
        "avg_entropy": float(candidate.get("avg_entropy", 0.0) or 0.0),
        "seq_confidence": float(candidate.get("seq_confidence", 0.0) or 0.0),
        "avg_log_prob": float(avg_log_prob),
        "logprob_finite": logprob_finite,
        "source_top1": 1.0 if candidate.get("source") == "onevl_top1" else 0.0,
    }


def build_examples(candidate_json: Path, score_dir: Path):
    data = load_json(candidate_json)
    rows_by_candidate, official_combined, stage_one, stage_two = load_v2_score_rows(score_dir)
    score_lookup = {}
    for candidate_index, rows in rows_by_candidate.items():
        for row in rows:
            score_lookup.setdefault(row["token"], {})[candidate_index] = safe_float(row["score"])

    examples = []
    for item in data:
        token = item.get("token")
        if token not in score_lookup:
            continue
        for candidate in item["candidates"]:
            candidate_index = int(candidate["candidate_id"])
            if candidate_index not in score_lookup[token]:
                continue
            examples.append(
                {
                    "token": token,
                    "stage": item.get("stage", "first"),
                    "candidate_id": candidate_index,
                    "features": candidate_features(item, candidate),
                    "score": score_lookup[token][candidate_index],
                }
            )
    return examples, official_combined, stage_one, stage_two


def matrix(examples, feature_names):
    X = np.array([[ex["features"][name] for name in feature_names] for ex in examples], dtype=float)
    y = np.array([ex["score"] for ex in examples], dtype=float)
    return X, y


def group_by_token(examples):
    by_token = {}
    for ex in examples:
        by_token.setdefault(ex["token"], []).append(ex)
    return by_token


def proxy_fixed_scores(by_token):
    candidate_ids = sorted({ex["candidate_id"] for exs in by_token.values() for ex in exs})
    scores = {}
    for candidate_id in candidate_ids:
        vals = []
        for exs in by_token.values():
            match = [ex["score"] for ex in exs if ex["candidate_id"] == candidate_id]
            if match:
                vals.append(match[0])
        scores[str(candidate_id)] = float(np.mean(vals)) if vals else float("nan")
    return scores


def evaluate_loso(by_token, feature_names, alpha):
    selected_scores = []
    selected_ids = []
    for token, exs in by_token.items():
        train = [ex for train_token, xs in by_token.items() if train_token != token for ex in xs]
        Xtr, ytr = matrix(train, feature_names)
        model = fit_ridge(Xtr, ytr, alpha=alpha)
        Xte, _ = matrix(exs, feature_names)
        preds = predict_ridge(model, Xte)
        best = int(np.argmax(preds))
        selected_scores.append(exs[best]["score"])
        selected_ids.append(exs[best]["candidate_id"])
    return selected_scores, selected_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-json", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--model-out", type=Path)
    parser.add_argument("--alphas", default="0.01,0.1,1.0,10.0,100.0")
    parser.add_argument("--drop-order-features", action="store_true")
    args = parser.parse_args()

    examples, official_combined, stage_one, stage_two = build_examples(args.candidate_json, args.score_dir)
    if not examples:
        raise SystemExit("no examples built")

    feature_names = select_feature_names(examples, args.drop_order_features)
    by_token = group_by_token(examples)
    candidate_ids = sorted({ex["candidate_id"] for ex in examples})
    fixed_proxy = proxy_fixed_scores(by_token)
    top1_proxy = fixed_proxy["0"]
    best_fixed_proxy_candidate = max(fixed_proxy, key=fixed_proxy.get)
    oracle_proxy = float(np.mean([max(ex["score"] for ex in exs) for exs in by_token.values()]))

    cv_rows = []
    for alpha in [float(v) for v in args.alphas.split(",") if v]:
        selected_scores, selected_ids = evaluate_loso(by_token, feature_names, alpha)
        cv_rows.append(
            {
                "alpha": alpha,
                "loso_proxy_score": float(np.mean(selected_scores)),
                "gap_closed_vs_top1_proxy": float((np.mean(selected_scores) - top1_proxy) / (oracle_proxy - top1_proxy))
                if oracle_proxy > top1_proxy
                else 0.0,
                "selected_candidate_hist": {str(i): selected_ids.count(i) for i in candidate_ids},
            }
        )

    best_cv = max(cv_rows, key=lambda row: row["loso_proxy_score"])
    selected_scores, selected_ids = evaluate_loso(by_token, feature_names, best_cv["alpha"])

    official_combined_s = {str(k): float(v) for k, v in official_combined.items()}
    summary = {
        "candidate_json": str(args.candidate_json),
        "score_dir": str(args.score_dir),
        "scene_count": len(by_token),
        "candidate_count": len(candidate_ids),
        "feature_count": len(feature_names),
        "drop_order_features": args.drop_order_features,
        "official_combined_epdms_by_fixed_candidate": official_combined_s,
        "official_stage_one_epdms_by_fixed_candidate": {str(k): float(v) for k, v in stage_one.items()},
        "official_stage_two_epdms_by_fixed_candidate": {str(k): float(v) for k, v in stage_two.items()},
        "official_top1_epdms": official_combined_s["0"],
        "official_best_fixed_candidate": int(max(official_combined_s, key=official_combined_s.get)),
        "official_best_fixed_epdms": float(max(official_combined_s.values())),
        "proxy_fixed_scores": fixed_proxy,
        "proxy_top1_score": float(top1_proxy),
        "proxy_best_fixed_candidate": int(best_fixed_proxy_candidate),
        "proxy_best_fixed_score": float(fixed_proxy[best_fixed_proxy_candidate]),
        "proxy_oracle_at_k_score": oracle_proxy,
        "alphas": cv_rows,
        "selected_alpha": best_cv["alpha"],
        "loso_proxy_ridge_score": float(np.mean(selected_scores)),
        "gap_closed_vs_top1_proxy": float((np.mean(selected_scores) - top1_proxy) / (oracle_proxy - top1_proxy))
        if oracle_proxy > top1_proxy
        else 0.0,
        "selected_candidate_hist": {str(i): selected_ids.count(i) for i in candidate_ids},
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")

    if args.model_out:
        X, y = matrix(examples, feature_names)
        weights, mean, std = fit_ridge(X, y, alpha=best_cv["alpha"])
        ranked = sorted(zip(feature_names, weights[1:]), key=lambda kv: abs(kv[1]), reverse=True)
        args.model_out.parent.mkdir(parents=True, exist_ok=True)
        args.model_out.write_text(
            json.dumps(
                {
                    "alpha": best_cv["alpha"],
                    "intercept": float(weights[0]),
                    "feature_names": feature_names,
                    "excluded_order_features": sorted(ORDER_FEATURES - set(feature_names)),
                    "mean": mean.tolist(),
                    "std": std.tolist(),
                    "weights": weights[1:].tolist(),
                    "top_abs_weights": [[name, float(weight)] for name, weight in ranked[:20]],
                },
                indent=2,
            )
            + "\n"
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
