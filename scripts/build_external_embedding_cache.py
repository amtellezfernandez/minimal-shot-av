#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.world_model import write_external_embedding_cache


EXTERNAL_FRAME_EMBEDDING_CACHE_SCHEMA = "external_frame_embeddings_v1"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert exported per-frame embeddings into the WOD external embedding cache schema."
    )
    parser.add_argument("--input", type=Path, required=True, help="JSON or JSONL embedding export.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", required=True, help="Human-readable embedding source, e.g. cosmos-reason2.")
    parser.add_argument("--frame-key", default="frame_name", help="Row field containing the WOD frame name.")
    parser.add_argument("--embedding-key", default="embedding", help="Row field containing the numeric vector.")
    parser.add_argument(
        "--allow-restamp",
        action="store_true",
        help="Allow rewriting an existing external_frame_embeddings_v1 cache with a different source.",
    )
    args = parser.parse_args()

    source = source_from_embedding_export(args.input)
    if source is not None and source != args.source and not args.allow_restamp:
        raise ValueError(
            "input cache source differs from --source; pass --allow-restamp to rewrite provenance"
        )
    cache = load_embedding_export(args.input, frame_key=args.frame_key, embedding_key=args.embedding_key)
    count = write_external_embedding_cache(cache, args.output, source=args.source)
    dimension = len(next(iter(cache.values())))
    print(json.dumps({"cached_frames": count, "dimension": dimension, "path": str(args.output)}, indent=2))
    return 0


def load_embedding_export(
    path: Path,
    *,
    frame_key: str = "frame_name",
    embedding_key: str = "embedding",
) -> dict[str, list[float]]:
    if path.suffix.lower() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _cache_from_rows(rows, frame_key=frame_key, embedding_key=embedding_key)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("schema") == EXTERNAL_FRAME_EMBEDDING_CACHE_SCHEMA:
        frames = payload.get("frames", {})
        if not isinstance(frames, dict):
            raise ValueError("external_frame_embeddings_v1 frames must be an object")
        return _cache_from_mapping(frames)
    if isinstance(payload, dict) and isinstance(payload.get("frames"), dict):
        return _cache_from_mapping(payload["frames"])
    if isinstance(payload, dict) and isinstance(payload.get("embeddings"), dict):
        return _cache_from_mapping(payload["embeddings"])
    if isinstance(payload, dict):
        if not all(_looks_like_vector(value) for value in payload.values()):
            raise ValueError("JSON object exports must be a frame-to-vector mapping or contain frames/embeddings")
        return _cache_from_mapping(payload)
    if isinstance(payload, list):
        return _cache_from_rows(payload, frame_key=frame_key, embedding_key=embedding_key)
    raise ValueError(f"unsupported embedding export payload type: {type(payload).__name__}")


def source_from_embedding_export(path: Path) -> str | None:
    if path.suffix.lower() == ".jsonl":
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("schema") == EXTERNAL_FRAME_EMBEDDING_CACHE_SCHEMA:
        source = payload.get("source")
        return str(source) if source is not None else None
    return None


def _cache_from_mapping(payload: dict[object, object]) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for frame_name, values in payload.items():
        key = str(frame_name)
        if key in cache:
            raise ValueError(f"duplicate embedding frame {key!r}")
        cache[key] = _embedding_vector(values, frame_name=key)
    return cache


def _cache_from_rows(
    rows: Sequence[object],
    *,
    frame_key: str,
    embedding_key: str,
) -> dict[str, list[float]]:
    cache: dict[str, list[float]] = {}
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"embedding row {index} must be an object")
        if frame_key not in row:
            raise ValueError(f"embedding row {index} is missing {frame_key!r}")
        if embedding_key not in row:
            raise ValueError(f"embedding row {index} is missing {embedding_key!r}")
        frame_name = str(row[frame_key])
        if frame_name in cache:
            raise ValueError(f"duplicate embedding frame {frame_name!r}")
        cache[frame_name] = _embedding_vector(row[embedding_key], frame_name=frame_name)
    return cache


def _embedding_vector(values: object, *, frame_name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"embedding row {frame_name!r} must be a JSON array")
    return [float(value) for value in values]


def _looks_like_vector(value: object) -> bool:
    return isinstance(value, list)


if __name__ == "__main__":
    raise SystemExit(main())
