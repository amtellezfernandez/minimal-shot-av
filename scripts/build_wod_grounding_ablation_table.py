#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

ABLATIONS = (
    (
        "Scalar / geometry only",
        ROOT / "benchmarks" / "current" / "wod_contextual_r175_speed_router_local_cv.json",
        "Linear contextual selector over non-visual candidate features.",
    ),
    (
        "InternVLA only",
        ROOT / "artifacts" / "wod_push_internvla_s2_embeddings_only_champion_cv_official.json",
        "Frozen InternVLA System-2 embedding attached to the selector.",
    ),
    (
        "Cosmos + InternVLA linear fusion",
        ROOT / "artifacts" / "wod_push_fused_cosmos_internvla_symbolic_champion_cv_official.json",
        "Frozen world-model plus VLA symbolic fusion under a non-direct selector.",
    ),
    (
        "Cosmos 64d nonlinear head",
        ROOT / "benchmarks" / "current" / "wod_champion_v20_5fold_mlp_gpu_64d.json",
        "GPU MLP direct policy on frozen Cosmos tokenizer embeddings.",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact WOD grounding ablation table from tracked result files."
    )
    parser.add_argument("--output-json", type=Path, default=ROOT / "artifacts" / "wod_grounding_ablation_table.json")
    parser.add_argument(
        "--output-markdown",
        type=Path,
        default=ROOT / "artifacts" / "wod_grounding_ablation_table.md",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    rows = [_row(name, path, note) for name, path, note in ABLATIONS]
    payload = {
        "schema": "wod_grounding_ablation_table_v1",
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_markdown.write_text(_markdown_table(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _row(name: str, path: Path, note: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "metrics" in payload and isinstance(payload["metrics"], dict):
        payload = {
            **payload.get("metadata", {}),
            **{key: value.get("value") if isinstance(value, dict) else value for key, value in payload["metrics"].items()},
        }
    return {
        "name": name,
        "file": str(path),
        "combined_ranker_mean_rfs": float(payload["combined_ranker_mean_rfs"]),
        "combined_oracle_mean_rfs": float(payload["combined_oracle_mean_rfs"]),
        "combined_ranker_regret_to_oracle": float(payload["combined_ranker_regret_to_oracle"]),
        "combined_ranker_top1_oracle_match_rate": float(payload["combined_ranker_top1_oracle_match_rate"]),
        "embedding_source": payload.get("embedding_source"),
        "direct_policy_enabled": bool(payload.get("direct_policy_enabled")) if "direct_policy_enabled" in payload else False,
        "note": note,
    }


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Ablation | RFS | Oracle | Regret | Oracle match | Grounding signal |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {rfs:.3f} | {oracle:.3f} | {regret:.3f} | {match:.3f} | {signal} |".format(
                name=row["name"],
                rfs=row["combined_ranker_mean_rfs"],
                oracle=row["combined_oracle_mean_rfs"],
                regret=row["combined_ranker_regret_to_oracle"],
                match=row["combined_ranker_top1_oracle_match_rate"],
                signal=_signal_label(row),
            )
        )
    return "\n".join(lines) + "\n"


def _signal_label(row: dict[str, Any]) -> str:
    if row["name"] == "Scalar / geometry only":
        return "none"
    if row["name"] == "InternVLA only":
        return "InternVLA"
    if row["name"] == "Cosmos + InternVLA linear fusion":
        return "Cosmos + InternVLA"
    if row["name"] == "Cosmos 64d nonlinear head":
        return "Cosmos + nonlinear head"
    return str(row.get("embedding_source") or "unknown")


if __name__ == "__main__":
    raise SystemExit(main())
