from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_NUMERIC_FEATURES = [
    "intent",
    "init_speed_mps",
    "endpoint_distance",
    "total_distance",
    "mean_step_distance",
    "max_step_distance",
    "min_step_distance",
    "final_lateral_abs",
    "max_lateral_abs",
    "lateral_range",
    "forward_progress",
    "mean_speed_mps",
    "max_speed_mps",
    "final_speed_mps",
    "mean_abs_accel_mps2",
    "max_abs_accel_mps2",
    "mean_abs_lateral_step",
    "max_abs_lateral_step",
    "signed_lateral_5s",
    "intent_turn_alignment",
    "mean_abs_heading_change",
    "max_abs_heading_change",
    "x_1s",
    "y_1s",
    "x_2s",
    "y_2s",
    "x_3s",
    "y_3s",
    "x_4s",
    "y_4s",
    "x_5s",
    "y_5s",
]


@dataclass(frozen=True)
class WodPreferenceRanker:
    numeric_features: list[str]
    candidate_names: list[str]
    candidate_families: list[str]
    feature_mean: Sequence[float]
    feature_scale: Sequence[float]
    weights: Sequence[float]
    bias: float

    @classmethod
    def load(cls, path: str | Path) -> WodPreferenceRanker:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            numeric_features=list(payload["numeric_features"]),
            candidate_names=list(payload["candidate_names"]),
            candidate_families=list(payload.get("candidate_families", [])),
            feature_mean=_float_list(payload["feature_mean"]),
            feature_scale=_float_list(payload["feature_scale"]),
            weights=_float_list(payload["weights"]),
            bias=float(payload["bias"]),
        )

    def predict_row(self, row: dict[str, Any]) -> float:
        raw = raw_features(row, self.numeric_features, self.candidate_names, self.candidate_families)
        if not (len(raw) == len(self.feature_mean) == len(self.feature_scale) == len(self.weights)):
            raise ValueError("ranker feature vector and model parameter lengths do not match")
        score = self.bias
        for value, mean, scale, weight in zip(raw, self.feature_mean, self.feature_scale, self.weights):
            safe_scale = scale if abs(scale) > 1e-12 else 1.0
            score += ((value - mean) / safe_scale) * weight
        return float(score)

    def select_row(self, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            raise ValueError("at least one candidate row is required")
        return max(
            rows,
            key=lambda row: (
                self.predict_row(row),
                -int(row.get("candidate_index", 0)),
                str(row["candidate_name"]),
            ),
        )


def raw_features(
    row: dict[str, Any],
    numeric_features: Sequence[str],
    candidate_names: Sequence[str],
    candidate_families: Sequence[str] = (),
) -> list[float]:
    features = row["features"]
    values = [float(features[name]) for name in numeric_features]
    candidate_name = str(row["candidate_name"])
    candidate_family = str(features.get("candidate_family", candidate_name))
    values.extend(1.0 if candidate_name == name else 0.0 for name in candidate_names)
    values.extend(1.0 if candidate_family == family else 0.0 for family in candidate_families)
    return values


def _float_list(values: Sequence[Any]) -> list[float]:
    return [float(value) for value in values]
