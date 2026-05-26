#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
from typing import Mapping


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = ROOT / "artifacts" / "corl2027" / "onevl_candidate_sets.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert OneVL inference outputs into the generic candidate_set_v1 schema."
    )
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--generator-name", default="onevl")
    parser.add_argument("--generator-iteration", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = json.loads(args.input_json.read_text(encoding="utf-8"))
    candidate_sets = convert_onevl_rows_to_candidate_sets(
        rows,
        generator_name=str(args.generator_name),
        generator_iteration=int(args.generator_iteration),
    )
    payload = {
        "schema": "candidate_set_v1",
        "source_format": "onevl_inference_json",
        "input_json": str(args.input_json),
        "candidate_sets": candidate_sets,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {len(candidate_sets)} candidate sets -> {args.output_json}")
    return 0


def convert_onevl_rows_to_candidate_sets(
    rows: list[Mapping[str, Any]],
    *,
    generator_name: str,
    generator_iteration: int,
) -> list[dict[str, Any]]:
    candidate_sets = []
    for index, row in enumerate(rows):
        candidates = [
            convert_onevl_candidate(candidate)
            for candidate in row.get("candidates", [])
            if candidate.get("trajectory") is not None
        ]
        if not candidates:
            continue
        top1_id = str(candidates[0]["candidate_id"])
        candidate_sets.append(
            {
                "scene_id": infer_scene_id(row, index),
                "source": str(row.get("source") or row.get("log_name") or ""),
                "scenario_type": str(row.get("scenario_type", "")),
                "generator": generator_name,
                "generator_iteration": int(generator_iteration),
                "top1_candidate_id": top1_id,
                "metadata": {
                    "candidate_mode": row.get("candidate_mode"),
                    "num_candidates": row.get("num_candidates"),
                    "latency": row.get("latency"),
                    "gt": row.get("GT", ""),
                },
                "candidates": candidates,
            }
        )
    return candidate_sets


def convert_onevl_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = candidate.get("trajectory")
    return {
        "candidate_id": str(candidate.get("candidate_id", "")),
        "trajectory": trajectory,
        "raw_text": candidate.get("raw_text", ""),
        "metrics": {
            "raw_score": optional_float(candidate.get("seq_confidence")),
            "avg_entropy": optional_float(candidate.get("avg_entropy")),
            "avg_log_prob": optional_float(candidate.get("avg_log_prob")),
        },
    }


def infer_scene_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("scene_id", "token", "scenario_token", "sample_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    return f"onevl_sample_{index:06d}"


def optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
