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
        "--tag-key",
        default="tags",
        help="Row/object field containing symbolic reasoning tags to append as one-hot features.",
    )
    parser.add_argument(
        "--no-tags",
        action="store_true",
        help="Ignore symbolic tags even when the export contains --tag-key.",
    )
    parser.add_argument(
        "--tag-vocab",
        default="",
        help="Comma-separated tag vocabulary. If omitted, the vocabulary is inferred and sorted from the export.",
    )
    parser.add_argument(
        "--scalar-key",
        action="append",
        default=[],
        help="Numeric reasoning score field to append after tags. May be repeated, e.g. --scalar-key risk.",
    )
    parser.add_argument(
        "--selector-width",
        type=int,
        help="Crop or zero-pad final vectors to this width. Use 64 for the WOD external_contextual selector.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        help="Optional sidecar JSON documenting tag/scalar expansion and vector width.",
    )
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
    cache, metadata = load_reasoning_export(
        args.input,
        frame_key=args.frame_key,
        embedding_key=args.embedding_key,
        tag_key=args.tag_key,
        include_tags=not args.no_tags,
        tag_vocab=_parse_tag_vocab(args.tag_vocab),
        scalar_keys=args.scalar_key,
        selector_width=args.selector_width,
    )
    count = write_external_embedding_cache(cache, args.output, source=args.source)
    dimension = len(next(iter(cache.values()))) if cache else 0
    result = {"cached_frames": count, "dimension": dimension, "path": str(args.output), **metadata}
    if args.metadata_output is not None:
        args.metadata_output.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def load_embedding_export(
    path: Path,
    *,
    frame_key: str = "frame_name",
    embedding_key: str = "embedding",
) -> dict[str, list[float]]:
    cache, _metadata = load_reasoning_export(
        path,
        frame_key=frame_key,
        embedding_key=embedding_key,
        include_tags=False,
        scalar_keys=(),
    )
    return cache


def load_reasoning_export(
    path: Path,
    *,
    frame_key: str = "frame_name",
    embedding_key: str = "embedding",
    tag_key: str = "tags",
    include_tags: bool = True,
    tag_vocab: Sequence[str] | None = None,
    scalar_keys: Sequence[str] = (),
    selector_width: int | None = None,
) -> tuple[dict[str, list[float]], dict[str, object]]:
    rows = _normalized_rows(path, frame_key=frame_key, embedding_key=embedding_key, tag_key=tag_key)
    resolved_tag_vocab = list(tag_vocab) if tag_vocab is not None else _infer_tag_vocab(rows, tag_key=tag_key)
    if not include_tags:
        resolved_tag_vocab = []

    cache: dict[str, list[float]] = {}
    for index, row in enumerate(rows, start=1):
        frame_name = str(row[frame_key])
        if frame_name in cache:
            raise ValueError(f"duplicate embedding frame {frame_name!r}")
        vector = _embedding_vector(row[embedding_key], frame_name=frame_name)
        vector.extend(_tag_features(row.get(tag_key), resolved_tag_vocab))
        vector.extend(_scalar_features(row, scalar_keys=scalar_keys, row_index=index))
        cache[frame_name] = _fit_width(vector, selector_width, frame_name=frame_name)

    metadata = {
        "base_dimension": _base_dimension(rows, embedding_key=embedding_key),
        "tag_key": tag_key,
        "tag_vocab": resolved_tag_vocab,
        "scalar_keys": [str(value) for value in scalar_keys],
        "selector_width": selector_width,
    }
    return cache, metadata


def _normalized_rows(
    path: Path,
    *,
    frame_key: str,
    embedding_key: str,
    tag_key: str,
) -> list[dict[str, object]]:
    if path.suffix.lower() == ".jsonl":
        raw_rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return _rows_from_sequence(raw_rows, frame_key=frame_key, embedding_key=embedding_key)

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("schema") == EXTERNAL_FRAME_EMBEDDING_CACHE_SCHEMA:
        frames = payload.get("frames", {})
        if not isinstance(frames, dict):
            raise ValueError("external_frame_embeddings_v1 frames must be an object")
        return _rows_from_mapping(frames, frame_key=frame_key, embedding_key=embedding_key, tag_key=tag_key)
    if isinstance(payload, dict) and isinstance(payload.get("frames"), dict):
        return _rows_from_mapping(payload["frames"], frame_key=frame_key, embedding_key=embedding_key, tag_key=tag_key)
    if isinstance(payload, dict) and isinstance(payload.get("embeddings"), dict):
        return _rows_from_mapping(
            payload["embeddings"],
            frame_key=frame_key,
            embedding_key=embedding_key,
            tag_key=tag_key,
        )
    if isinstance(payload, dict):
        if not all(_looks_like_vector(value) for value in payload.values()):
            raise ValueError("JSON object exports must be a frame-to-vector mapping or contain frames/embeddings")
        return _rows_from_mapping(payload, frame_key=frame_key, embedding_key=embedding_key, tag_key=tag_key)
    if isinstance(payload, list):
        return _rows_from_sequence(payload, frame_key=frame_key, embedding_key=embedding_key)
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


def _rows_from_mapping(
    payload: dict[object, object],
    *,
    frame_key: str,
    embedding_key: str,
    tag_key: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for frame_name, values in payload.items():
        key = str(frame_name)
        if key in seen:
            raise ValueError(f"duplicate embedding frame {key!r}")
        seen.add(key)
        if isinstance(values, dict):
            row = dict(values)
            if embedding_key not in row and "embedding" in row:
                row[embedding_key] = row["embedding"]
            if embedding_key not in row and "vector" in row:
                row[embedding_key] = row["vector"]
            if tag_key not in row and "tags" in row:
                row[tag_key] = row["tags"]
            row[frame_key] = key
            rows.append(row)
            continue
        rows.append({frame_key: key, embedding_key: values})
    return rows


def _rows_from_sequence(
    rows: Sequence[object],
    *,
    frame_key: str,
    embedding_key: str,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"embedding row {index} must be an object")
        if frame_key not in row:
            raise ValueError(f"embedding row {index} is missing {frame_key!r}")
        if embedding_key not in row:
            raise ValueError(f"embedding row {index} is missing {embedding_key!r}")
        normalized.append(dict(row))
    return normalized


def _infer_tag_vocab(rows: Sequence[dict[str, object]], *, tag_key: str) -> list[str]:
    tags: set[str] = set()
    for row in rows:
        tags.update(_tag_values(row.get(tag_key)))
    return sorted(tags)


def _tag_values(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    if isinstance(values, list):
        return [str(value) for value in values]
    raise ValueError("reasoning tags must be a string or JSON array")


def _tag_features(values: object, tag_vocab: Sequence[str]) -> list[float]:
    active = set(_tag_values(values))
    return [1.0 if tag in active else 0.0 for tag in tag_vocab]


def _scalar_features(
    row: dict[str, object],
    *,
    scalar_keys: Sequence[str],
    row_index: int,
) -> list[float]:
    features: list[float] = []
    for key in scalar_keys:
        value = row.get(key, 0.0)
        try:
            features.append(float(value))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"embedding row {row_index} scalar {key!r} must be numeric") from exc
    return features


def _fit_width(values: list[float], width: int | None, *, frame_name: str) -> list[float]:
    if width is None:
        return values
    if width <= 0:
        raise ValueError("--selector-width must be positive")
    if len(values) > width:
        return values[:width]
    if len(values) < width:
        return [*values, *[0.0 for _ in range(width - len(values))]]
    return values


def _base_dimension(rows: Sequence[dict[str, object]], *, embedding_key: str) -> int:
    if not rows:
        return 0
    return len(_embedding_vector(rows[0][embedding_key], frame_name=str(rows[0].get("frame_name", "<row-1>"))))


def _parse_tag_vocab(value: str) -> list[str] | None:
    if not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _embedding_vector(values: object, *, frame_name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"embedding row {frame_name!r} must be a JSON array")
    return [float(value) for value in values]


def _looks_like_vector(value: object) -> bool:
    return isinstance(value, list)


if __name__ == "__main__":
    raise SystemExit(main())
