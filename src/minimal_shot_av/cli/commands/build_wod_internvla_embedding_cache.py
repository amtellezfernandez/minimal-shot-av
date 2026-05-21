#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.world_model import load_external_embedding_cache, write_external_embedding_cache


DEFAULT_COSMOS_CACHE = ROOT / "artifacts" / "cosmos_predict25_wan21_tokenizer_val479.json"
DEFAULT_INTERNVLA_JSONL = ROOT / "artifacts" / "internvla_wod_val.jsonl"
VARIANTS = (
    "vla_unit",
    "concat_cosmos_vla",
    "concat_cosmos_symbolic",
    "concat_cosmos_vla_symbolic",
    "overlay_vla",
    "overlay_symbolic",
    "overlay_vla_symbolic",
)
SYMBOLIC_FEATURE_NAMES = (
    "action_look_down",
    "action_turn_left",
    "action_turn_right",
    "text_forward",
    "text_stop",
    "intent_1",
    "intent_2",
    "intent_3",
    "intent_2_left",
    "intent_3_right",
    "intent_1_look_down",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fuse offline InternVLA System-2 rows with Cosmos WOD external embeddings."
    )
    parser.add_argument("--cosmos-cache", type=Path, default=DEFAULT_COSMOS_CACHE)
    parser.add_argument("--internvla-jsonl", type=Path, default=DEFAULT_INTERNVLA_JSONL)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--source-prefix", default="cosmos-predict25+internvla")
    parser.add_argument("--vla-weight", type=float, default=0.10)
    parser.add_argument("--symbolic-weight", type=float, default=0.20)
    args = parser.parse_args()

    cosmos = load_external_embedding_cache(args.cosmos_cache)
    internvla_rows = load_internvla_rows(args.internvla_jsonl)
    cache = build_internvla_embedding_cache(
        cosmos,
        internvla_rows,
        variant=args.variant,
        vla_weight=args.vla_weight,
        symbolic_weight=args.symbolic_weight,
    )
    dimension = len(next(iter(cache.values())))
    source = source_name(
        args.source_prefix,
        args.variant,
        dimension=dimension,
        vla_weight=args.vla_weight,
        symbolic_weight=args.symbolic_weight,
    )
    count = write_external_embedding_cache(cache, args.output, source=source)
    print(
        json.dumps(
            {
                "cached_frames": count,
                "dimension": dimension,
                "internvla_rows": len(internvla_rows),
                "output": str(args.output),
                "source": source,
                "symbolic_feature_names": list(SYMBOLIC_FEATURE_NAMES),
                "variant": args.variant,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def load_internvla_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            frame_name = str(row["frame_name"])
            if frame_name in rows:
                raise ValueError(f"duplicate InternVLA frame {frame_name!r} at line {line_number}")
            rows[frame_name] = row
    if not rows:
        raise ValueError("InternVLA JSONL contains no rows")
    return rows


def build_internvla_embedding_cache(
    cosmos: dict[str, list[float]],
    internvla_rows: dict[str, dict[str, Any]],
    *,
    variant: str,
    vla_weight: float = 0.10,
    symbolic_weight: float = 0.20,
) -> dict[str, list[float]]:
    if variant not in VARIANTS:
        raise ValueError(f"unsupported InternVLA embedding variant: {variant}")
    _validate_frame_match(cosmos, internvla_rows)
    cache: dict[str, list[float]] = {}
    for frame_name in sorted(cosmos):
        cosmos_vector = [float(value) for value in cosmos[frame_name]]
        internvla = internvla_rows[frame_name]
        vla_vector = _unit_vector(_embedding(internvla, frame_name=frame_name))
        symbolic = symbolic_features(internvla)
        if variant == "vla_unit":
            vector = vla_vector
        elif variant == "concat_cosmos_vla":
            vector = [*cosmos_vector, *vla_vector]
        elif variant == "concat_cosmos_symbolic":
            vector = [*cosmos_vector, *symbolic]
        elif variant == "concat_cosmos_vla_symbolic":
            vector = [*cosmos_vector, *vla_vector, *symbolic]
        elif variant == "overlay_vla":
            vector = _overlay(cosmos_vector, vla_vector, weight=vla_weight)
        elif variant == "overlay_symbolic":
            vector = _overlay(cosmos_vector, symbolic, weight=symbolic_weight)
        else:
            vector = _overlay(
                _overlay(cosmos_vector, vla_vector, weight=vla_weight),
                symbolic,
                weight=symbolic_weight,
            )
        cache[frame_name] = vector
    return cache


def symbolic_features(row: dict[str, Any]) -> list[float]:
    action = str(row.get("action", "")).upper()
    text = str(row.get("model_text", row.get("text", "")))
    upper_text = text.upper()
    intent = int(row.get("intent", 0))
    look_down = 1.0 if action == "LOOK_DOWN" else 0.0
    left = 1.0 if action == "TURN_LEFT" else 0.0
    right = 1.0 if action == "TURN_RIGHT" else 0.0
    forward = 1.0 if "↑" in text or "FORWARD" in upper_text else 0.0
    stop = 1.0 if "STOP" in upper_text or "HALT" in upper_text else 0.0
    intent_1 = 1.0 if intent == 1 else 0.0
    intent_2 = 1.0 if intent == 2 else 0.0
    intent_3 = 1.0 if intent == 3 else 0.0
    return [
        look_down,
        left,
        right,
        forward,
        stop,
        intent_1,
        intent_2,
        intent_3,
        intent_2 * left,
        intent_3 * right,
        intent_1 * look_down,
    ]


def source_name(
    prefix: str,
    variant: str,
    *,
    dimension: int,
    vla_weight: float,
    symbolic_weight: float,
) -> str:
    if variant.startswith("overlay_"):
        return (
            f"{prefix}-{variant}{dimension}"
            f"-vla{_compact_float(vla_weight)}-sym{_compact_float(symbolic_weight)}"
        )
    return f"{prefix}-{variant}{dimension}"


def _validate_frame_match(cosmos: dict[str, list[float]], internvla_rows: dict[str, dict[str, Any]]) -> None:
    cosmos_names = set(cosmos)
    internvla_names = set(internvla_rows)
    if cosmos_names != internvla_names:
        missing_in_internvla = sorted(cosmos_names - internvla_names)[:3]
        extra_in_internvla = sorted(internvla_names - cosmos_names)[:3]
        raise ValueError(
            "Cosmos and InternVLA frame sets differ: "
            f"missing_in_internvla={missing_in_internvla}, extra_in_internvla={extra_in_internvla}"
        )


def _embedding(row: dict[str, Any], *, frame_name: str) -> list[float]:
    raw = row.get("embedding")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"InternVLA row {frame_name!r} is missing a non-empty embedding")
    values = [float(value) for value in raw]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"InternVLA row {frame_name!r} has a non-finite embedding value")
    return values


def _unit_vector(values: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 1e-12:
        return [0.0 for _ in values]
    return [value / norm for value in values]


def _overlay(base: list[float], residual: list[float], *, weight: float) -> list[float]:
    vector = list(base)
    for index, value in enumerate(residual[: len(vector)]):
        vector[index] += float(weight) * float(value)
    return vector


def _compact_float(value: float) -> str:
    return f"{float(value):.6g}".replace(".", "p").replace("-", "m")


if __name__ == "__main__":
    raise SystemExit(main())
