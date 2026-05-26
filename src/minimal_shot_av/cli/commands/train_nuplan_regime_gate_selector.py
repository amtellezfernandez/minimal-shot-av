#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping

import numpy as np

from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import scene_feature_names
from minimal_shot_av.cli.commands.train_nuplan_bootstrap_student_generator import scene_feature_row
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _candidate_by_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_replay_oracle_token
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _select_with_score_model
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import _token_is_replay_safe
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import build_replay_calibration_examples
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import evaluate_selector_policy
from minimal_shot_av.cli.commands.train_nuplan_replay_calibrated_selector import split_replay_report_by_db
from minimal_shot_av.cli.commands.train_nuplan_replay_value_selector import build_replay_value_examples
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_calibrated_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import fit_replay_value_mlp_selector
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_feature_row
from minimal_shot_av.model.nuplan_maneuver_token_selector import selector_from_payload


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_INPUT = ROOT / "artifacts" / "corl2027" / "nuplan_public_replay_study_interaction1000_expert" / "replay.json"
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "nuplan_regime_gate_selector.json"
DEFAULT_REPORT_JSON = ROOT / "artifacts" / "corl2027" / "nuplan_regime_gate_selector_eval.json"
DEFAULT_REPORT_MD = ROOT / "artifacts" / "corl2027" / "nuplan_regime_gate_selector_eval.md"
MODEL_TYPE = "nuplan_regime_gate_selector_v1"
REGIME_ORDER = ("keep_proxy", "marginal_correction", "intervention")
HEAD_FEATURE_NAMES = (
    "differs_from_proxy",
    "proxy_safe",
    "min_proxy_clearance_m",
    "final_progress_m",
    "progress_retention",
    "score",
    "model_score",
    "minus_proxy_clearance_m",
    "minus_proxy_progress_m",
    "minus_proxy_score",
)
CROSS_HEAD_FEATURE_NAMES = (
    "heads_same_token",
    "intervention_minus_marginal_progress_m",
    "intervention_minus_marginal_retention",
    "intervention_minus_marginal_model_score",
    "intervention_minus_marginal_clearance_m",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a regime gate over keep, marginal-correction, and intervention recovery heads."
    )
    parser.add_argument("--input-replay-json", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-markdown", type=Path, default=DEFAULT_REPORT_MD)
    parser.add_argument("--near-miss-threshold-m", type=float, default=1.0)
    parser.add_argument("--holdout-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--gate-kind", choices=("sequential_binary", "multiclass"), default="sequential_binary")
    parser.add_argument("--marginal-threshold", type=float, default=0.5)
    parser.add_argument("--intervention-threshold", type=float, default=0.5)
    parser.add_argument("--marginal-progress-retention", type=float, default=0.5)
    parser.add_argument("--marginal-fail-penalty", type=float, default=100.0)
    parser.add_argument("--intervention-fail-penalty", type=float, default=250.0)
    parser.add_argument("--value-progress-weight", type=float, default=1.0)
    parser.add_argument("--value-ade-weight", type=float, default=2.0)
    parser.add_argument("--value-hidden-dim", type=int, default=48)
    parser.add_argument("--value-epochs", type=int, default=700)
    parser.add_argument("--value-learning-rate", type=float, default=0.03)
    parser.add_argument("--risk-threshold", type=float, default=0.5)
    parser.add_argument("--risk-hidden-dim", type=int, default=48)
    parser.add_argument("--risk-epochs", type=int, default=700)
    parser.add_argument("--risk-learning-rate", type=float, default=0.03)
    parser.add_argument("--risk-weight", type=float, default=4.0)
    parser.add_argument("--marginal-weight", type=float, default=4.0)
    parser.add_argument("--intervention-weight", type=float, default=4.0)
    parser.add_argument("--keep-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    replay_report = json.loads(args.input_replay_json.read_text(encoding="utf-8"))
    train_report, holdout_report, split = split_replay_report_by_db(
        replay_report,
        holdout_fraction=float(args.holdout_fraction),
        seed=int(args.seed),
    )
    marginal_selector, marginal_fit = fit_value_head(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.marginal_fail_penalty),
        objective_progress_weight=float(args.value_progress_weight),
        objective_ade_weight=float(args.value_ade_weight),
        hidden_dim=int(args.value_hidden_dim),
        epochs=int(args.value_epochs),
        learning_rate=float(args.value_learning_rate),
        seed=int(args.seed),
    )
    intervention_selector, intervention_fit = fit_value_head(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        fail_penalty=float(args.intervention_fail_penalty),
        objective_progress_weight=float(args.value_progress_weight),
        objective_ade_weight=float(args.value_ade_weight),
        hidden_dim=int(args.value_hidden_dim),
        epochs=int(args.value_epochs),
        learning_rate=float(args.value_learning_rate),
        seed=int(args.seed) + 991,
    )
    risk_selector, risk_fit = fit_risk_head(
        train_report,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        hidden_dim=int(args.risk_hidden_dim),
        epochs=int(args.risk_epochs),
        learning_rate=float(args.risk_learning_rate),
        seed=int(args.seed) + 9973,
        risk_weight=float(args.risk_weight),
    )
    marginal_token_fn = lambda scene: select_marginal_head_token(
        scene,
        marginal_selector,
        min_progress_retention=float(args.marginal_progress_retention),
    )
    intervention_token_fn = lambda scene: _select_with_score_model(scene, intervention_selector)
    train_examples = build_regime_gate_examples(
        train_report,
        marginal_token_fn=marginal_token_fn,
        intervention_token_fn=intervention_token_fn,
        marginal_selector=marginal_selector,
        intervention_selector=intervention_selector,
        near_miss_threshold_m=float(args.near_miss_threshold_m),
        marginal_weight=float(args.marginal_weight),
        intervention_weight=float(args.intervention_weight),
        keep_weight=float(args.keep_weight),
    )
    if str(args.gate_kind) == "multiclass":
        gate_model, fit_metrics = fit_regime_gate_mlp(
            train_examples,
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
        )
    else:
        gate_model, fit_metrics = fit_sequential_regime_gate_mlps(
            train_examples,
            hidden_dim=int(args.hidden_dim),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            seed=int(args.seed),
            marginal_positive_weight=float(args.marginal_weight),
            intervention_positive_weight=float(args.intervention_weight),
            marginal_threshold=float(args.marginal_threshold),
            intervention_threshold=float(args.intervention_threshold),
        )
    config = {
        "marginal_progress_retention": float(args.marginal_progress_retention),
        "marginal_fail_penalty": float(args.marginal_fail_penalty),
        "intervention_fail_penalty": float(args.intervention_fail_penalty),
        "value_progress_weight": float(args.value_progress_weight),
        "value_ade_weight": float(args.value_ade_weight),
        "marginal_weight": float(args.marginal_weight),
        "intervention_weight": float(args.intervention_weight),
        "keep_weight": float(args.keep_weight),
        "gate_kind": str(args.gate_kind),
        "marginal_threshold": float(args.marginal_threshold),
        "intervention_threshold": float(args.intervention_threshold),
        "risk_threshold": float(args.risk_threshold),
    }
    report = {
        "schema": "nuplan_regime_gate_selector_eval_v1",
        "input_replay_json": str(args.input_replay_json),
        "split": split,
        "near_miss_threshold_m": float(args.near_miss_threshold_m),
        "config": config,
        "head_fit_metrics": {
            "marginal": marginal_fit,
            "intervention": intervention_fit,
            "risk": risk_fit,
        },
        "train_fit_metrics": fit_metrics,
        "train_example_count": len(train_examples),
        "train": evaluate_regime_gate_family(
            train_report,
            gate_model=gate_model,
            marginal_token_fn=marginal_token_fn,
            intervention_token_fn=intervention_token_fn,
            marginal_selector=marginal_selector,
            intervention_selector=intervention_selector,
            risk_selector=risk_selector,
            risk_threshold=float(args.risk_threshold),
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
        "holdout": evaluate_regime_gate_family(
            holdout_report,
            gate_model=gate_model,
            marginal_token_fn=marginal_token_fn,
            intervention_token_fn=intervention_token_fn,
            marginal_selector=marginal_selector,
            intervention_selector=intervention_selector,
            risk_selector=risk_selector,
            risk_threshold=float(args.risk_threshold),
            near_miss_threshold_m=float(args.near_miss_threshold_m),
        ),
    }
    payload = {
        **regime_gate_payload(gate_model),
        "training": {
            "input_replay_json": str(args.input_replay_json),
            "split": split,
            "config": config,
            "train_fit_metrics": fit_metrics,
            "train_example_count": len(train_examples),
            "marginal_selector_payload": marginal_selector.to_payload(),
            "intervention_selector_payload": intervention_selector.to_payload(),
            "risk_selector_payload": risk_selector.to_payload(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.report_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.report_markdown.write_text(markdown_regime_report(report) + "\n", encoding="utf-8")
    print(markdown_regime_report(report))
    return 0


def fit_value_head(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    fail_penalty: float,
    objective_progress_weight: float,
    objective_ade_weight: float,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[Any, dict[str, float]]:
    examples = build_replay_value_examples(
        replay_report,
        near_miss_threshold_m=near_miss_threshold_m,
        fail_penalty=fail_penalty,
        objective_progress_weight=objective_progress_weight,
        objective_ade_weight=objective_ade_weight,
    )
    selector, metrics = fit_replay_value_mlp_selector(
        examples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
    )
    return selector, {
        **metrics,
        "example_count": float(len(examples)),
        "fail_penalty": float(fail_penalty),
        "objective_progress_weight": float(objective_progress_weight),
        "objective_ade_weight": float(objective_ade_weight),
    }


def fit_risk_head(
    replay_report: Mapping[str, Any],
    *,
    near_miss_threshold_m: float,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    risk_weight: float,
) -> tuple[Any, dict[str, float]]:
    examples = build_replay_calibration_examples(
        replay_report,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    selector, metrics = fit_replay_calibrated_selector(
        examples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        risk_weight=risk_weight,
        proxy_score_weight=1.0,
        progress_weight=0.02,
        unsafe_penalty=2.0,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    return selector, {
        **metrics,
        "example_count": float(len(examples)),
        "risk_weight": float(risk_weight),
    }


def load_regime_gate_policy(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    training = payload.get("training", {}) if isinstance(payload.get("training"), dict) else {}
    marginal_payload = training.get("marginal_selector_payload")
    intervention_payload = training.get("intervention_selector_payload")
    risk_payload = training.get("risk_selector_payload")
    if not isinstance(marginal_payload, dict) or not isinstance(intervention_payload, dict):
        raise ValueError(f"regime gate policy {path} does not contain embedded head selector payloads")
    config = training.get("config", {}) if isinstance(training.get("config"), dict) else {}
    return {
        "gate_model": payload,
        "marginal_selector": selector_from_payload(marginal_payload),
        "intervention_selector": selector_from_payload(intervention_payload),
        "risk_selector": selector_from_payload(risk_payload) if isinstance(risk_payload, dict) else None,
        "marginal_progress_retention": float(config.get("marginal_progress_retention", 0.5)),
        "risk_threshold": float(config.get("risk_threshold", 0.5)),
    }


def select_with_regime_gate_policy(scene: Mapping[str, Any], policy: Mapping[str, Any]) -> str:
    marginal_token_fn = lambda row: select_marginal_head_token(
        row,
        policy["marginal_selector"],
        min_progress_retention=float(policy.get("marginal_progress_retention", 0.5)),
    )
    intervention_token_fn = lambda row: _select_with_score_model(row, policy["intervention_selector"])
    return select_with_regime_gate(
        scene,
        gate_model=policy["gate_model"],
        marginal_token_fn=marginal_token_fn,
        intervention_token_fn=intervention_token_fn,
        marginal_selector=policy["marginal_selector"],
        intervention_selector=policy["intervention_selector"],
    )


def select_marginal_head_token(
    scene: Mapping[str, Any],
    selector: Any,
    *,
    min_progress_retention: float,
) -> str:
    candidates = list(scene.get("candidates", []))
    if not candidates:
        return str(scene.get("selected_token", ""))
    proxy_candidate = _candidate_by_token(scene).get(str(scene.get("selected_token", "")))
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0)) if proxy_candidate is not None else 0.0
    eligible = [
        candidate
        for candidate in candidates
        if progress_retention(scene, str(candidate.get("token", ""))) >= float(min_progress_retention)
    ]
    if not eligible:
        eligible = list(candidates)
    selected = max(
        eligible,
        key=lambda candidate: (
            selector.predict_score(
                candidate.get("features") if "features" in candidate else selector_feature_row(scene, candidate)
            ),
            float(candidate.get("final_progress_m", 0.0)) - proxy_progress,
            float(candidate.get("min_proxy_clearance_m", 0.0)),
        ),
    )
    return str(selected.get("token", scene.get("selected_token", "")))


def build_regime_gate_examples(
    replay_report: Mapping[str, Any],
    *,
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    marginal_selector: Any,
    intervention_selector: Any,
    near_miss_threshold_m: float,
    marginal_weight: float,
    intervention_weight: float,
    keep_weight: float,
) -> list[dict[str, Any]]:
    examples = []
    for scene in replay_report.get("scenes", []):
        marginal_token = str(marginal_token_fn(scene))
        intervention_token = str(intervention_token_fn(scene))
        label = oracle_regime_label(
            scene,
            marginal_token=marginal_token,
            intervention_token=intervention_token,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        examples.append(
            {
                "scene_id": str(scene.get("scene_id", "")),
                "source_db_file": str(scene.get("source_db_file", "")),
                "label": label,
                "label_index": REGIME_ORDER.index(label),
                "example_weight": regime_example_weight(
                    label,
                    marginal_weight=marginal_weight,
                    intervention_weight=intervention_weight,
                    keep_weight=keep_weight,
                ),
                "proxy_token": str(scene.get("selected_token", "")),
                "marginal_token": marginal_token,
                "intervention_token": intervention_token,
                "features": regime_gate_feature_row(
                    scene,
                    marginal_token=marginal_token,
                    intervention_token=intervention_token,
                    marginal_selector=marginal_selector,
                    intervention_selector=intervention_selector,
                ),
            }
        )
    if not examples:
        raise ValueError("cannot train regime gate on an empty replay report")
    return examples


def regime_example_weight(
    label: str,
    *,
    marginal_weight: float,
    intervention_weight: float,
    keep_weight: float,
) -> float:
    if label == "marginal_correction":
        return float(marginal_weight)
    if label == "intervention":
        return float(intervention_weight)
    return float(keep_weight)


def oracle_regime_label(
    scene: Mapping[str, Any],
    *,
    marginal_token: str,
    intervention_token: str,
    near_miss_threshold_m: float,
) -> str:
    proxy_token = str(scene.get("selected_token", ""))
    if _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m):
        return "keep_proxy"
    if _token_is_replay_safe(scene, marginal_token, near_miss_threshold_m):
        return "marginal_correction"
    if _token_is_replay_safe(scene, intervention_token, near_miss_threshold_m):
        return "intervention"
    return "keep_proxy"


def fit_regime_gate_mlp(
    examples: list[Mapping[str, Any]],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float]]:
    feature_names = regime_gate_feature_names()
    x = np.asarray(
        [[float(example["features"][name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    y = np.asarray([int(example["label_index"]) for example in examples], dtype=np.int64)
    weights = np.asarray([float(example.get("example_weight", 1.0)) for example in examples], dtype=np.float64)
    weights = weights / max(float(weights.mean()), 1.0e-8)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.12, size=(x_norm.shape[1], hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.12, size=(hidden_dim, len(REGIME_ORDER)))
    b2 = np.zeros((len(REGIME_ORDER),), dtype=np.float64)
    y_onehot = np.eye(len(REGIME_ORDER), dtype=np.float64)[y]
    weight_matrix = weights.reshape(-1, 1)
    for _ in range(max(1, epochs)):
        z1 = x_norm @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        probabilities = _softmax(logits)
        grad_logits = weight_matrix * (probabilities - y_onehot) / x_norm.shape[0]
        grad_w2 = h1.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_h1 = grad_logits @ w2.T
        grad_z1 = grad_h1 * (z1 > 0.0)
        grad_w1 = x_norm.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
    model = {
        "model_type": MODEL_TYPE,
        "feature_names": list(feature_names),
        "regime_order": list(REGIME_ORDER),
        "hidden_dim": int(hidden_dim),
        "feature_mean": tuple(float(value) for value in feature_mean),
        "feature_scale": tuple(float(value) for value in feature_scale),
        "w1": tuple(tuple(float(value) for value in row) for row in w1),
        "b1": tuple(float(value) for value in b1),
        "w2": tuple(tuple(float(value) for value in row) for row in w2),
        "b2": tuple(float(value) for value in b2),
    }
    probabilities = np.asarray([predict_regime_probabilities(model, example["features"]) for example in examples])
    predicted = np.argmax(probabilities, axis=1)
    losses = -np.log(np.clip(probabilities[np.arange(len(y)), y], 1.0e-8, 1.0))
    label_counts = Counter(str(example["label"]) for example in examples)
    predicted_counts = Counter(REGIME_ORDER[int(index)] for index in predicted)
    return model, {
        "example_count": float(len(examples)),
        "regime_accuracy": float(np.mean(predicted == y)),
        "regime_log_loss": float(np.mean(losses)),
        "label_counts": dict(label_counts),
        "predicted_counts": dict(predicted_counts),
    }


def fit_sequential_regime_gate_mlps(
    examples: list[Mapping[str, Any]],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    marginal_positive_weight: float,
    intervention_positive_weight: float,
    marginal_threshold: float,
    intervention_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    marginal_examples = [
        {**example, "label": 1.0 if str(example["label"]) == "marginal_correction" else 0.0}
        for example in examples
    ]
    intervention_examples = [
        {**example, "label": 1.0 if str(example["label"]) == "intervention" else 0.0}
        for example in examples
    ]
    marginal_model, marginal_metrics = fit_binary_regime_gate_mlp(
        marginal_examples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed,
        positive_weight=marginal_positive_weight,
    )
    intervention_model, intervention_metrics = fit_binary_regime_gate_mlp(
        intervention_examples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        learning_rate=learning_rate,
        seed=seed + 211,
        positive_weight=intervention_positive_weight,
    )
    model = {
        "model_type": MODEL_TYPE,
        "gate_kind": "sequential_binary",
        "regime_order": list(REGIME_ORDER),
        "marginal_threshold": float(marginal_threshold),
        "intervention_threshold": float(intervention_threshold),
        "marginal_model": marginal_model,
        "intervention_model": intervention_model,
    }
    predicted = [predict_regime(model, example["features"]) for example in examples]
    labels = [str(example["label"]) for example in examples]
    label_counts = Counter(labels)
    predicted_counts = Counter(predicted)
    return model, {
        "example_count": float(len(examples)),
        "gate_kind": "sequential_binary",
        "regime_accuracy": float(sum(p == y for p, y in zip(predicted, labels)) / max(1, len(labels))),
        "label_counts": dict(label_counts),
        "predicted_counts": dict(predicted_counts),
        "marginal_binary": marginal_metrics,
        "intervention_binary": intervention_metrics,
    }


def fit_binary_regime_gate_mlp(
    examples: list[Mapping[str, Any]],
    *,
    hidden_dim: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    positive_weight: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    feature_names = regime_gate_feature_names()
    x = np.asarray(
        [[float(example["features"][name]) for name in feature_names] for example in examples],
        dtype=np.float64,
    )
    y = np.asarray([float(example["label"]) for example in examples], dtype=np.float64).reshape(-1, 1)
    weights = np.asarray(
        [float(positive_weight) if float(example["label"]) > 0.5 else 1.0 for example in examples],
        dtype=np.float64,
    ).reshape(-1, 1)
    weights = weights / max(float(weights.mean()), 1.0e-8)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.12, size=(x_norm.shape[1], hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.12, size=(hidden_dim, 1))
    b2 = np.zeros((1,), dtype=np.float64)
    for _ in range(max(1, epochs)):
        z1 = x_norm @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        grad_logits = weights * (probabilities - y) / x_norm.shape[0]
        grad_w2 = h1.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_h1 = grad_logits @ w2.T
        grad_z1 = grad_h1 * (z1 > 0.0)
        grad_w1 = x_norm.T @ grad_z1
        grad_b1 = grad_z1.sum(axis=0)
        w1 -= learning_rate * grad_w1
        b1 -= learning_rate * grad_b1
        w2 -= learning_rate * grad_w2
        b2 -= learning_rate * grad_b2
    model = {
        "feature_names": list(feature_names),
        "hidden_dim": int(hidden_dim),
        "feature_mean": tuple(float(value) for value in feature_mean),
        "feature_scale": tuple(float(value) for value in feature_scale),
        "w1": tuple(tuple(float(value) for value in row) for row in w1),
        "b1": tuple(float(value) for value in b1),
        "w2": tuple(float(value) for value in w2[:, 0]),
        "b2": float(b2[0]),
    }
    probabilities = np.asarray([predict_binary_probability(model, example["features"]) for example in examples])
    labels = y[:, 0]
    loss = -np.mean(
        labels * np.log(np.clip(probabilities, 1.0e-8, 1.0))
        + (1.0 - labels) * np.log(np.clip(1.0 - probabilities, 1.0e-8, 1.0))
    )
    return model, {
        "example_count": float(len(examples)),
        "positive_rate": float(labels.mean()),
        "binary_accuracy": float(np.mean((probabilities >= 0.5) == (labels >= 0.5))),
        "binary_log_loss": float(loss),
    }


def evaluate_regime_gate_family(
    replay_report: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    marginal_selector: Any,
    intervention_selector: Any,
    risk_selector: Any,
    risk_threshold: float,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    regime_fn = lambda scene: select_with_regime_gate(
        scene,
        gate_model=gate_model,
        marginal_token_fn=marginal_token_fn,
        intervention_token_fn=intervention_token_fn,
        marginal_selector=marginal_selector,
        intervention_selector=intervention_selector,
    )
    rows = {
        "proxy_selector": evaluate_selector_policy(
            replay_report,
            selector_name="proxy_selector",
            selected_token_fn=lambda scene: str(scene.get("selected_token", "")),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "marginal_head": evaluate_selector_policy(
            replay_report,
            selector_name="marginal_head",
            selected_token_fn=marginal_token_fn,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "intervention_head": evaluate_selector_policy(
            replay_report,
            selector_name="intervention_head",
            selected_token_fn=intervention_token_fn,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "oracle_regime_gate": evaluate_selector_policy(
            replay_report,
            selector_name="oracle_regime_gate",
            selected_token_fn=lambda scene: select_with_oracle_regime_gate(
                scene,
                marginal_token_fn=marginal_token_fn,
                intervention_token_fn=intervention_token_fn,
                near_miss_threshold_m=near_miss_threshold_m,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "regime_gate": evaluate_selector_policy(
            replay_report,
            selector_name="regime_gate",
            selected_token_fn=regime_fn,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "risk_regime_gate": evaluate_selector_policy(
            replay_report,
            selector_name="risk_regime_gate",
            selected_token_fn=lambda scene: select_with_risk_regime_gate(
                scene,
                marginal_token_fn=marginal_token_fn,
                intervention_token_fn=intervention_token_fn,
                risk_selector=risk_selector,
                risk_threshold=risk_threshold,
            ),
            near_miss_threshold_m=near_miss_threshold_m,
        ),
        "replay_oracle": evaluate_selector_policy(
            replay_report,
            selector_name="replay_oracle",
            selected_token_fn=_select_replay_oracle_token,
            near_miss_threshold_m=near_miss_threshold_m,
        ),
    }
    rows["regime_gate"]["regime_diagnostics"] = regime_gate_diagnostics(
        replay_report,
        gate_model=gate_model,
        marginal_token_fn=marginal_token_fn,
        intervention_token_fn=intervention_token_fn,
        marginal_selector=marginal_selector,
        intervention_selector=intervention_selector,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    rows["oracle_regime_gate"]["regime_diagnostics"] = oracle_regime_diagnostics(
        replay_report,
        marginal_token_fn=marginal_token_fn,
        intervention_token_fn=intervention_token_fn,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    return rows


def select_with_regime_gate(
    scene: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    marginal_selector: Any,
    intervention_selector: Any,
) -> str:
    marginal_token = str(marginal_token_fn(scene))
    intervention_token = str(intervention_token_fn(scene))
    features = regime_gate_feature_row(
        scene,
        marginal_token=marginal_token,
        intervention_token=intervention_token,
        marginal_selector=marginal_selector,
        intervention_selector=intervention_selector,
    )
    regime = predict_regime(gate_model, features)
    if regime == "marginal_correction":
        return marginal_token
    if regime == "intervention":
        return intervention_token
    return str(scene.get("selected_token", ""))


def select_with_risk_regime_gate(
    scene: Mapping[str, Any],
    *,
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    risk_selector: Any,
    risk_threshold: float,
) -> str:
    proxy_token = str(scene.get("selected_token", ""))
    marginal_token = str(marginal_token_fn(scene))
    intervention_token = str(intervention_token_fn(scene))
    scored = [
        ("keep_proxy", proxy_token, predicted_candidate_risk(scene, proxy_token, risk_selector)),
        ("marginal_correction", marginal_token, predicted_candidate_risk(scene, marginal_token, risk_selector)),
        ("intervention", intervention_token, predicted_candidate_risk(scene, intervention_token, risk_selector)),
    ]
    if scored[0][2] <= float(risk_threshold):
        return proxy_token
    for _regime, token, risk in scored[1:]:
        if risk <= float(risk_threshold):
            return token
    return min(scored, key=lambda row: (row[2], 0 if row[0] == "marginal_correction" else 1))[1]


def predicted_candidate_risk(scene: Mapping[str, Any], token: str, risk_selector: Any) -> float:
    candidate = _candidate_by_token(scene).get(str(token))
    if candidate is None:
        return 1.0
    features = candidate.get("features") if "features" in candidate else selector_feature_row(scene, candidate)
    return float(risk_selector.predict_risk(features))


def select_with_oracle_regime_gate(
    scene: Mapping[str, Any],
    *,
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
) -> str:
    marginal_token = str(marginal_token_fn(scene))
    intervention_token = str(intervention_token_fn(scene))
    label = oracle_regime_label(
        scene,
        marginal_token=marginal_token,
        intervention_token=intervention_token,
        near_miss_threshold_m=near_miss_threshold_m,
    )
    if label == "marginal_correction":
        return marginal_token
    if label == "intervention":
        return intervention_token
    return str(scene.get("selected_token", ""))


def regime_gate_diagnostics(
    replay_report: Mapping[str, Any],
    *,
    gate_model: Mapping[str, Any],
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    marginal_selector: Any,
    intervention_selector: Any,
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    label_counts: Counter[str] = Counter()
    predicted_counts: Counter[str] = Counter()
    confusion: dict[str, Counter[str]] = {label: Counter() for label in REGIME_ORDER}
    recovered_by_regime: Counter[str] = Counter()
    introduced_failure_count = 0
    scene_count = 0
    for scene in replay_report.get("scenes", []):
        scene_count += 1
        proxy_token = str(scene.get("selected_token", ""))
        marginal_token = str(marginal_token_fn(scene))
        intervention_token = str(intervention_token_fn(scene))
        oracle = oracle_regime_label(
            scene,
            marginal_token=marginal_token,
            intervention_token=intervention_token,
            near_miss_threshold_m=near_miss_threshold_m,
        )
        features = regime_gate_feature_row(
            scene,
            marginal_token=marginal_token,
            intervention_token=intervention_token,
            marginal_selector=marginal_selector,
            intervention_selector=intervention_selector,
        )
        predicted = predict_regime(gate_model, features)
        selected_token = proxy_token
        if predicted == "marginal_correction":
            selected_token = marginal_token
        elif predicted == "intervention":
            selected_token = intervention_token
        label_counts[oracle] += 1
        predicted_counts[predicted] += 1
        confusion[oracle][predicted] += 1
        if oracle != "keep_proxy" and _token_is_replay_safe(scene, selected_token, near_miss_threshold_m):
            recovered_by_regime[oracle] += 1
        if _token_is_replay_safe(scene, proxy_token, near_miss_threshold_m) and not _token_is_replay_safe(
            scene,
            selected_token,
            near_miss_threshold_m,
        ):
            introduced_failure_count += 1
    return {
        "scene_count": scene_count,
        "oracle_regime_counts": dict(label_counts),
        "predicted_regime_counts": dict(predicted_counts),
        "confusion_matrix": {
            label: {predicted: int(count) for predicted, count in row.items()}
            for label, row in confusion.items()
        },
        "marginal_recovery_count": int(recovered_by_regime["marginal_correction"]),
        "intervention_recovery_count": int(recovered_by_regime["intervention"]),
        "introduced_failure_count": int(introduced_failure_count),
    }


def oracle_regime_diagnostics(
    replay_report: Mapping[str, Any],
    *,
    marginal_token_fn: Callable[[Mapping[str, Any]], str],
    intervention_token_fn: Callable[[Mapping[str, Any]], str],
    near_miss_threshold_m: float,
) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for scene in replay_report.get("scenes", []):
        label = oracle_regime_label(
            scene,
            marginal_token=str(marginal_token_fn(scene)),
            intervention_token=str(intervention_token_fn(scene)),
            near_miss_threshold_m=near_miss_threshold_m,
        )
        counts[label] += 1
    return {
        "scene_count": sum(counts.values()),
        "oracle_regime_counts": dict(counts),
    }


def regime_gate_feature_names() -> tuple[str, ...]:
    names = list(scene_feature_names())
    for prefix in ("marginal", "intervention"):
        names.extend(f"{prefix}_{name}" for name in HEAD_FEATURE_NAMES)
    names.extend(CROSS_HEAD_FEATURE_NAMES)
    return tuple(names)


def regime_gate_feature_row(
    scene: Mapping[str, Any],
    *,
    marginal_token: str,
    intervention_token: str,
    marginal_selector: Any,
    intervention_selector: Any,
) -> dict[str, float]:
    row = dict(scene_feature_row(scene))
    marginal = _head_feature_row(scene, marginal_token, marginal_selector)
    intervention = _head_feature_row(scene, intervention_token, intervention_selector)
    for name, value in marginal.items():
        row[f"marginal_{name}"] = value
    for name, value in intervention.items():
        row[f"intervention_{name}"] = value
    row["heads_same_token"] = 1.0 if str(marginal_token) == str(intervention_token) else 0.0
    row["intervention_minus_marginal_progress_m"] = (
        intervention["final_progress_m"] - marginal["final_progress_m"]
    )
    row["intervention_minus_marginal_retention"] = (
        intervention["progress_retention"] - marginal["progress_retention"]
    )
    row["intervention_minus_marginal_model_score"] = intervention["model_score"] - marginal["model_score"]
    row["intervention_minus_marginal_clearance_m"] = (
        intervention["min_proxy_clearance_m"] - marginal["min_proxy_clearance_m"]
    )
    return row


def _head_feature_row(scene: Mapping[str, Any], token: str, selector: Any) -> dict[str, float]:
    candidate_by_token = _candidate_by_token(scene)
    proxy_token = str(scene.get("selected_token", ""))
    proxy_candidate = candidate_by_token.get(proxy_token)
    candidate = candidate_by_token.get(str(token))
    proxy_clearance = float(proxy_candidate.get("min_proxy_clearance_m", 0.0)) if proxy_candidate is not None else 0.0
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0)) if proxy_candidate is not None else 0.0
    proxy_score = float(proxy_candidate.get("score", 0.0)) if proxy_candidate is not None else 0.0
    if candidate is None:
        return {
            "differs_from_proxy": 1.0 if str(token) != proxy_token else 0.0,
            "proxy_safe": 0.0,
            "min_proxy_clearance_m": 0.0,
            "final_progress_m": 0.0,
            "progress_retention": 0.0,
            "score": 0.0,
            "model_score": 0.0,
            "minus_proxy_clearance_m": -proxy_clearance,
            "minus_proxy_progress_m": -proxy_progress,
            "minus_proxy_score": -proxy_score,
        }
    features = candidate.get("features") if "features" in candidate else selector_feature_row(scene, candidate)
    progress = float(candidate.get("final_progress_m", 0.0))
    score = float(candidate.get("score", 0.0))
    clearance = float(candidate.get("min_proxy_clearance_m", 0.0))
    return {
        "differs_from_proxy": 1.0 if str(token) != proxy_token else 0.0,
        "proxy_safe": 1.0 if bool(candidate.get("proxy_safe", False)) else 0.0,
        "min_proxy_clearance_m": clearance,
        "final_progress_m": progress,
        "progress_retention": progress_retention(scene, str(token)),
        "score": score,
        "model_score": float(selector.predict_score(features)),
        "minus_proxy_clearance_m": clearance - proxy_clearance,
        "minus_proxy_progress_m": progress - proxy_progress,
        "minus_proxy_score": score - proxy_score,
    }


def progress_retention(scene: Mapping[str, Any], token: str) -> float:
    candidate_by_token = _candidate_by_token(scene)
    proxy_candidate = candidate_by_token.get(str(scene.get("selected_token", "")))
    candidate = candidate_by_token.get(str(token))
    if candidate is None:
        return 0.0
    progress = float(candidate.get("final_progress_m", 0.0))
    if proxy_candidate is None:
        return 1.0 if progress >= 0.0 else 0.0
    proxy_progress = float(proxy_candidate.get("final_progress_m", 0.0))
    if proxy_progress <= 1.0e-6:
        return 1.0 if progress >= proxy_progress else 0.0
    return float(progress / proxy_progress)


def predict_regime(model: Mapping[str, Any], features: Mapping[str, float]) -> str:
    if str(model.get("gate_kind", "multiclass")) == "sequential_binary":
        marginal_probability = predict_binary_probability(model["marginal_model"], features)
        if marginal_probability >= float(model.get("marginal_threshold", 0.5)):
            return "marginal_correction"
        intervention_probability = predict_binary_probability(model["intervention_model"], features)
        if intervention_probability >= float(model.get("intervention_threshold", 0.5)):
            return "intervention"
        return "keep_proxy"
    probabilities = predict_regime_probabilities(model, features)
    regime_order = tuple(str(name) for name in model.get("regime_order", REGIME_ORDER))
    return regime_order[int(np.argmax(probabilities))]


def predict_binary_probability(model: Mapping[str, Any], features: Mapping[str, float]) -> float:
    feature_names = tuple(str(name) for name in model["feature_names"])
    vector = np.asarray([float(features[name]) for name in feature_names], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (vector - mean) / scale
    hidden = np.maximum(
        normalized @ np.asarray(model["w1"], dtype=np.float64) + np.asarray(model["b1"], dtype=np.float64),
        0.0,
    )
    logit = float(hidden @ np.asarray(model["w2"], dtype=np.float64) + float(model["b2"]))
    return float(1.0 / (1.0 + np.exp(-max(-40.0, min(40.0, logit)))))


def predict_regime_probabilities(model: Mapping[str, Any], features: Mapping[str, float]) -> np.ndarray:
    feature_names = tuple(str(name) for name in model["feature_names"])
    vector = np.asarray([[float(features[name]) for name in feature_names]], dtype=np.float64)
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    normalized = (vector - mean) / scale
    hidden = np.maximum(
        normalized @ np.asarray(model["w1"], dtype=np.float64) + np.asarray(model["b1"], dtype=np.float64),
        0.0,
    )
    logits = hidden @ np.asarray(model["w2"], dtype=np.float64) + np.asarray(model["b2"], dtype=np.float64)
    return _softmax(logits)[0]


def regime_gate_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    if str(model.get("gate_kind", "multiclass")) == "sequential_binary":
        return {
            "model_type": MODEL_TYPE,
            "gate_kind": "sequential_binary",
            "regime_order": list(model.get("regime_order", REGIME_ORDER)),
            "marginal_threshold": float(model.get("marginal_threshold", 0.5)),
            "intervention_threshold": float(model.get("intervention_threshold", 0.5)),
            "marginal_model": binary_gate_payload(model["marginal_model"]),
            "intervention_model": binary_gate_payload(model["intervention_model"]),
        }
    return {
        "model_type": MODEL_TYPE,
        "gate_kind": "multiclass",
        "feature_names": list(model["feature_names"]),
        "regime_order": list(model.get("regime_order", REGIME_ORDER)),
        "hidden_dim": int(model["hidden_dim"]),
        "feature_mean": list(model["feature_mean"]),
        "feature_scale": list(model["feature_scale"]),
        "w1": [list(row) for row in model["w1"]],
        "b1": list(model["b1"]),
        "w2": [list(row) for row in model["w2"]],
        "b2": list(model["b2"]),
    }


def binary_gate_payload(model: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "feature_names": list(model["feature_names"]),
        "hidden_dim": int(model["hidden_dim"]),
        "feature_mean": list(model["feature_mean"]),
        "feature_scale": list(model["feature_scale"]),
        "w1": [list(row) for row in model["w1"]],
        "b1": list(model["b1"]),
        "w2": list(model["w2"]),
        "b2": float(model["b2"]),
    }


def markdown_regime_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# nuPlan Regime-Gated Selector",
        "",
        f"- Input replay JSON: `{report['input_replay_json']}`",
        f"- Split unit: `{report['split']['split_unit']}`",
        f"- Train scenes: `{report['split']['train_scene_count']}`",
        f"- Holdout scenes: `{report['split']['holdout_scene_count']}`",
        f"- Marginal retention floor: `{report['config']['marginal_progress_retention']:.3f}`",
        f"- Marginal fail penalty: `{report['config']['marginal_fail_penalty']:.1f}`",
        f"- Intervention fail penalty: `{report['config']['intervention_fail_penalty']:.1f}`",
        f"- Train regime accuracy: `{report['train_fit_metrics']['regime_accuracy']:.3f}`",
        "",
    ]
    for split_name in ("train", "holdout"):
        lines.extend(
            [
                f"## {split_name.title()}",
                "",
                "| Selector | Replay-infeasible | Rate | Recovery | Progress | ADE3 | FDE3 | Entropy |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for selector_name, metrics in report[split_name].items():
            lines.append(
                f"| {selector_name} | {metrics['selected_replay_infeasible_count']} | "
                f"{metrics['selected_replay_infeasible_rate']:.3f} | "
                f"{metrics['recoverable_proxy_optimism_fixed_count']}/"
                f"{metrics['recoverable_proxy_optimism_count']} | "
                f"{metrics['mean_selected_progress_m']:.3f} | "
                f"{_optional_metric(metrics['mean_ade_3s_m'])} | "
                f"{_optional_metric(metrics['mean_fde_3s_m'])} | "
                f"{metrics['selected_token_entropy']:.3f} |"
            )
        regime_metrics = report[split_name]["regime_gate"].get("regime_diagnostics", {})
        lines.extend(
            [
                "",
                "### Regime Diagnostics",
                "",
                f"- Oracle regimes: `{json.dumps(regime_metrics.get('oracle_regime_counts', {}), sort_keys=True)}`",
                "- Predicted regimes: "
                f"`{json.dumps(regime_metrics.get('predicted_regime_counts', {}), sort_keys=True)}`",
                f"- Introduced failures: `{regime_metrics.get('introduced_failure_count', 0)}`",
                "",
                "### Token Histograms",
                "",
            ]
        )
        for selector_name, metrics in report[split_name].items():
            lines.append(f"- `{selector_name}`: `{json.dumps(metrics['selected_token_histogram'], sort_keys=True)}`")
        lines.append("")
    return "\n".join(lines)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_values = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp_values / np.clip(exp_values.sum(axis=1, keepdims=True), 1.0e-8, None)


def _optional_metric(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    raise SystemExit(main())
