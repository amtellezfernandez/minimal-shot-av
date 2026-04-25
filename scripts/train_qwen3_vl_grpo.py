#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.qwen3_vl_stack import Qwen3VlWodConfig, trajectory_64_to_wod20_from_response
from minimal_shot_av.spotlight_reflex import RfsReference, score_candidate, ManeuverCandidate


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Qwen3-VL WOD-E2E GRPO launcher and reward smoke test."
    )
    parser.add_argument("--dataset", type=Path, default=ROOT / "artifacts" / "qwen3_vl_wod_sft.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "qwen3_vl_grpo")
    parser.add_argument("--model-name", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--dry-run", action="store_true", help="Validate data/reward plumbing without importing TRL.")
    parser.add_argument("--max-records", type=int, default=8)
    args = parser.parse_args()

    config = Qwen3VlWodConfig(model_name=args.model_name)
    records = _load_records(args.dataset, max_records=args.max_records)
    if args.dry_run:
        rewards = [_reward_record(record) for record in records]
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "model_name": config.model_name,
            "records": len(records),
            "mean_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "note": "dry-run validates trajectory parsing, resampling, and local RFS reward plumbing",
        }
        (args.output_dir / "dry_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return 0

    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import transformers  # noqa: F401
        import trl  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Qwen3-VL GRPO dependencies are not installed. Run "
            "`python -m pip install -e '.[qwen-grpo]'` or `scripts/setup_qwen3_vl_stack.sh`."
        ) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = {
        "status": "dependencies_available",
        "model_name": config.model_name,
        "dataset": str(args.dataset),
        "output_dir": str(args.output_dir),
        "implementation_note": (
            "Use TRL GRPOTrainer with reward functions wrapping _reward_record. "
            "Keep Qwen3-VL offline for SFT/GRPO; deploy the distilled trajectory decoder + RFS selector."
        ),
    }
    (args.output_dir / "launch_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(json.dumps(plan, indent=2))
    return 0


def _load_records(path: Path, *, max_records: int) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path}")
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                records.append(json.loads(line))
            if len(records) >= max_records:
                break
    if not records:
        raise ValueError(f"dataset is empty: {path}")
    return records


def _reward_record(record: dict[str, Any]) -> float:
    assistant = record["messages"][-1]["content"]
    trajectory = trajectory_64_to_wod20_from_response(assistant)
    references = [
        RfsReference(label=item["label"], trajectory=[tuple(point) for point in item["trajectory"]], score=float(item["score"]))
        for item in record.get("rfs_references", [])
    ]
    if not references:
        return _format_reward(assistant)
    rfs = score_candidate(ManeuverCandidate("qwen3_vl", trajectory), references, speed_mps=0.0).combined_score
    return float(rfs / 10.0 + _format_reward(assistant) * 0.05 + _smoothness_reward(trajectory) * 0.03)


def _format_reward(response: str) -> float:
    try:
        payload = json.loads(response)
        trajectory = payload["trajectory_64wp_10hz"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return 0.0
    return 1.0 if isinstance(trajectory, list) and len(trajectory) == 64 else 0.0


def _smoothness_reward(trajectory: list[tuple[float, float]]) -> float:
    if len(trajectory) < 3:
        return 0.0
    jerk_like = 0.0
    for index in range(2, len(trajectory)):
        ax = trajectory[index][0] - 2 * trajectory[index - 1][0] + trajectory[index - 2][0]
        ay = trajectory[index][1] - 2 * trajectory[index - 1][1] + trajectory[index - 2][1]
        jerk_like += abs(ax) + abs(ay)
    return max(0.0, 1.0 - jerk_like / max(1, len(trajectory) - 2))


if __name__ == "__main__":
    raise SystemExit(main())
