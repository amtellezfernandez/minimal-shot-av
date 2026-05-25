from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
from typing import Mapping

import numpy as np

from .nuplan_maneuver_token_adapter import ManeuverTokenCandidate
from .nuplan_maneuver_token_adapter import TOKEN_ORDER
from .nuplan_maneuver_token_adapter import build_maneuver_token_candidates
from .nuplan_maneuver_token_adapter import build_scene_diagnostic_record


SELECTOR_FEATURE_NAMES = (
    "obstacle_pressure",
    "route_blockage",
    "corridor_blocked",
    "left_clearance_m",
    "right_clearance_m",
    "escape_side_float",
    "heading_error_rad",
    "lane_offset_m",
    "route_remaining_m",
    "nearest_actor_distance_m",
    "leading_actor_distance_m",
    "rear_closing_actor_count",
    "crossing_actor_count",
    "candidate_speed_scale",
    "candidate_lateral_offset_m",
    "candidate_proxy_safe",
    "candidate_min_proxy_clearance_m",
    "candidate_final_progress_m",
    "candidate_score_heuristic",
)


@dataclass(frozen=True)
class NuPlanMlpSelector:
    feature_names: tuple[str, ...]
    hidden_dim: int
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    w1: tuple[tuple[float, ...], ...]
    b1: tuple[float, ...]
    w2: tuple[float, ...]
    b2: float

    def predict_score(self, features: Mapping[str, float]) -> float:
        vector = np.asarray([float(features[name]) for name in self.feature_names], dtype=np.float64)
        mean = np.asarray(self.feature_mean, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        normalized = (vector - mean) / scale
        hidden = np.maximum(
            normalized @ np.asarray(self.w1, dtype=np.float64) + np.asarray(self.b1, dtype=np.float64),
            0.0,
        )
        return float(hidden @ np.asarray(self.w2, dtype=np.float64) + float(self.b2))

    def select_record(self, scene_record: Mapping[str, Any]) -> dict[str, Any]:
        candidates = list(scene_record["candidates"])
        scored = [
            (
                self.predict_score(selector_feature_row(scene_record, candidate)),
                candidate,
            )
            for candidate in candidates
        ]
        _, selected = max(
            scored,
            key=lambda item: (
                item[0],
                item[1]["min_proxy_clearance_m"],
                item[1]["final_progress_m"],
            ),
        )
        return dict(selected)

    def to_payload(self) -> dict[str, Any]:
        return {
            "model_type": "nuplan_maneuvertoken_selector_mlp_v1",
            "feature_names": list(self.feature_names),
            "hidden_dim": int(self.hidden_dim),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "w1": [list(row) for row in self.w1],
            "b1": list(self.b1),
            "w2": list(self.w2),
            "b2": float(self.b2),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> NuPlanMlpSelector:
        return cls(
            feature_names=tuple(str(name) for name in payload["feature_names"]),
            hidden_dim=int(payload["hidden_dim"]),
            feature_mean=tuple(float(value) for value in payload["feature_mean"]),
            feature_scale=tuple(float(value) for value in payload["feature_scale"]),
            w1=tuple(tuple(float(value) for value in row) for row in payload["w1"]),
            b1=tuple(float(value) for value in payload["b1"]),
            w2=tuple(float(value) for value in payload["w2"]),
            b2=float(payload["b2"]),
        )


def load_selector(path: Path) -> NuPlanMlpSelector:
    return NuPlanMlpSelector.from_payload(json.loads(path.read_text(encoding="utf-8")))


def selector_feature_row(scene_record: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, float]:
    six_scalar_state = scene_record["six_scalar_state"]
    route = scene_record["route_features"]
    actor_summary = scene_record["actor_summary"]
    return {
        "obstacle_pressure": float(six_scalar_state["obstacle_pressure"]),
        "route_blockage": float(six_scalar_state["route_blockage"]),
        "corridor_blocked": 1.0 if bool(six_scalar_state["corridor_blocked"]) else 0.0,
        "left_clearance_m": float(six_scalar_state["left_clearance_m"]),
        "right_clearance_m": float(six_scalar_state["right_clearance_m"]),
        "escape_side_float": float(six_scalar_state["vector"][5]),
        "heading_error_rad": float(route["heading_error_rad"]),
        "lane_offset_m": float(route["lane_offset_m"]),
        "route_remaining_m": float(route["route_remaining_m"]),
        "nearest_actor_distance_m": float(actor_summary["nearest_actor_distance_m"]),
        "leading_actor_distance_m": float(actor_summary["leading_actor_distance_m"]),
        "rear_closing_actor_count": float(actor_summary["rear_closing_actor_count"]),
        "crossing_actor_count": float(actor_summary["crossing_actor_count"]),
        "candidate_speed_scale": float(candidate["speed_scale"]),
        "candidate_lateral_offset_m": float(candidate["lateral_offset_m"]),
        "candidate_proxy_safe": 1.0 if bool(candidate["proxy_safe"]) else 0.0,
        "candidate_min_proxy_clearance_m": float(candidate["min_proxy_clearance_m"]),
        "candidate_final_progress_m": float(candidate["final_progress_m"]),
        "candidate_score_heuristic": float(candidate["score"]),
    }


def build_training_examples(scenes: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    examples = []
    for scene in scenes:
        scene_record = build_scene_diagnostic_record(scene)
        expert_trajectory = list(scene.get("expert_trajectory", []))
        if not expert_trajectory:
            raise ValueError(f"scene {scene_record['scene_id']!r} is missing expert_trajectory")
        best_token = best_supervision_token(scene)
        for candidate in scene_record["candidates"]:
            examples.append(
                {
                    "scene_id": scene_record["scene_id"],
                    "token": candidate["token"],
                    "label": 1.0 if candidate["token"] == best_token else 0.0,
                    "supervision_score": supervision_score(scene_record, candidate, expert_trajectory),
                    "features": selector_feature_row(scene_record, candidate),
                }
            )
    return examples


def best_supervision_token(scene: Mapping[str, Any]) -> str:
    scene_record = build_scene_diagnostic_record(scene)
    expert_trajectory = list(scene.get("expert_trajectory", []))
    if not expert_trajectory:
        raise ValueError(f"scene {scene_record['scene_id']!r} is missing expert_trajectory")
    scored = [
        (
            supervision_score(scene_record, candidate, expert_trajectory),
            candidate["token"],
        )
        for candidate in scene_record["candidates"]
    ]
    return max(scored, key=lambda item: (item[0], -TOKEN_ORDER.index(item[1])))[1]


def supervision_score(
    scene_record: Mapping[str, Any],
    candidate: Mapping[str, Any],
    expert_trajectory: list[Mapping[str, Any] | list[float] | tuple[float, ...]],
) -> float:
    imitation = imitation_distance(candidate["poses"], expert_trajectory)
    proxy_safe_bonus = 2.5 if bool(candidate["proxy_safe"]) else -2.5
    clearance_bonus = min(2.0, float(candidate["min_proxy_clearance_m"]) / 2.0)
    route_penalty = abs(float(scene_record["route_features"]["lane_offset_m"])) * 0.15
    return -imitation + proxy_safe_bonus + clearance_bonus - route_penalty


def imitation_distance(
    candidate_poses: list[list[float] | tuple[float, ...]],
    expert_trajectory: list[Mapping[str, Any] | list[float] | tuple[float, ...]],
) -> float:
    count = min(len(candidate_poses), len(expert_trajectory))
    if count <= 0:
        return float("inf")
    total = 0.0
    for index in range(count):
        candidate = candidate_poses[index]
        expert = expert_trajectory[index]
        cx = float(candidate[0])
        cy = float(candidate[1])
        ex = float(expert["x_m"]) if isinstance(expert, Mapping) else float(expert[0])
        ey = float(expert["y_m"]) if isinstance(expert, Mapping) else float(expert[1])
        total += math.hypot(cx - ex, cy - ey)
    return total / count


def fit_selector_mlp(
    examples: list[Mapping[str, Any]],
    *,
    hidden_dim: int = 16,
    epochs: int = 250,
    learning_rate: float = 0.05,
    seed: int = 0,
) -> tuple[NuPlanMlpSelector, dict[str, float]]:
    if not examples:
        raise ValueError("cannot train selector on an empty example set")
    feature_names = tuple(SELECTOR_FEATURE_NAMES)
    x = np.asarray(
        [
            [float(example["features"][name]) for name in feature_names]
            for example in examples
        ],
        dtype=np.float64,
    )
    y = np.asarray([float(example["label"]) for example in examples], dtype=np.float64).reshape(-1, 1)
    feature_mean = x.mean(axis=0)
    feature_scale = x.std(axis=0)
    feature_scale[feature_scale < 1.0e-8] = 1.0
    x_norm = (x - feature_mean) / feature_scale
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, 0.15, size=(x_norm.shape[1], hidden_dim))
    b1 = np.zeros((hidden_dim,), dtype=np.float64)
    w2 = rng.normal(0.0, 0.15, size=(hidden_dim, 1))
    b2 = np.zeros((1,), dtype=np.float64)
    for _ in range(max(1, epochs)):
        z1 = x_norm @ w1 + b1
        h1 = np.maximum(z1, 0.0)
        logits = h1 @ w2 + b2
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        grad_logits = (probs - y) / x_norm.shape[0]
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
    selector = NuPlanMlpSelector(
        feature_names=feature_names,
        hidden_dim=int(hidden_dim),
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        w1=tuple(tuple(float(value) for value in row) for row in w1),
        b1=tuple(float(value) for value in b1),
        w2=tuple(float(value) for value in w2[:, 0]),
        b2=float(b2[0]),
    )
    metrics = evaluate_selector(selector, examples)
    return selector, metrics


def evaluate_selector(
    selector: NuPlanMlpSelector,
    examples: list[Mapping[str, Any]],
) -> dict[str, float]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for example in examples:
        grouped.setdefault(str(example["scene_id"]), []).append(example)
    correct = 0
    for rows in grouped.values():
        predicted = max(
            rows,
            key=lambda row: (
                selector.predict_score(row["features"]),
                float(row["supervision_score"]),
            ),
        )
        target = max(
            rows,
            key=lambda row: (float(row["label"]), float(row["supervision_score"])),
        )
        if str(predicted["token"]) == str(target["token"]):
            correct += 1
    logits = np.asarray(
        [selector.predict_score(row["features"]) for row in examples],
        dtype=np.float64,
    )
    labels = np.asarray([float(row["label"]) for row in examples], dtype=np.float64)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
    loss = -np.mean(
        labels * np.log(np.clip(probs, 1.0e-8, 1.0))
        + (1.0 - labels) * np.log(np.clip(1.0 - probs, 1.0e-8, 1.0))
    )
    return {
        "scene_accuracy": correct / max(1, len(grouped)),
        "example_log_loss": float(loss),
        "scene_count": float(len(grouped)),
        "example_count": float(len(examples)),
    }


def select_with_model(
    selector: NuPlanMlpSelector,
    scene: Mapping[str, Any],
) -> tuple[ManeuverTokenCandidate, float]:
    candidates = build_maneuver_token_candidates(scene)
    scene_record = build_scene_diagnostic_record(scene)
    best_candidate = None
    best_score = float("-inf")
    for candidate, candidate_record in zip(candidates, scene_record["candidates"]):
        score = selector.predict_score(selector_feature_row(scene_record, candidate_record))
        if best_candidate is None or (score, candidate.min_proxy_clearance_m, candidate.final_progress_m) > (
            best_score,
            best_candidate.min_proxy_clearance_m,
            best_candidate.final_progress_m,
        ):
            best_candidate = candidate
            best_score = score
    if best_candidate is None:
        raise AssertionError("selector saw no candidates")
    return best_candidate, float(best_score)
