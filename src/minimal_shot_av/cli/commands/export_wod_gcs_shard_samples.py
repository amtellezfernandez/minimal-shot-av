#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.neural_trajectory_model import save_neural_training_frame_cache  # noqa: E402
from minimal_shot_av.model.wod_e2e import load_preference_frames  # noqa: E402


DEFAULT_SHARD_GLOB = "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0/training_*.tfrecord-*"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export shallow compact samples from many WOD-E2E GCS shards.")
    parser.add_argument("--shard-glob", default=DEFAULT_SHARD_GLOB)
    parser.add_argument("--shard-start", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=8)
    parser.add_argument("--records-per-shard", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "cache" / "wod_train_compact.jsonl")
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT / "artifacts" / "cache" / "wod_train_compact_manifest.json",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()

    if args.shard_start < 0:
        raise ValueError("shard_start must be non-negative")
    if args.shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if args.records_per_shard <= 0:
        raise ValueError("records_per_shard must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for shard_offset in range(args.shard_count):
        shard_index = args.shard_start + shard_offset
        before_rows = _line_count(args.output)
        row: dict[str, Any] = {
            "shard_index": shard_index,
            "records_requested": args.records_per_shard,
            "rows_before": before_rows,
        }
        try:
            frames = load_preference_frames(
                "",
                shard_glob=args.shard_glob,
                shard_start=shard_index,
                max_shards=1,
                max_records=args.records_per_shard,
                include_camera_images=False,
                require_preferences=False,
            )
            exported_rows = save_neural_training_frame_cache(frames, args.output, append=True)
            after_rows = _line_count(args.output)
            row.update(
                {
                    "status": "ok",
                    "exported_rows": exported_rows,
                    "rows_after": after_rows,
                    "rows_added_by_file_count": after_rows - before_rows,
                }
            )
        except Exception as exc:  # noqa: BLE001 - manifest should record remote I/O failures.
            after_rows = _line_count(args.output)
            row.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "rows_after": after_rows,
                    "rows_added_by_file_count": after_rows - before_rows,
                }
            )
            rows.append(row)
            _write_manifest(args, rows)
            if not args.continue_on_error:
                print(json.dumps(_manifest(args, rows), indent=2, sort_keys=True))
                return 1
            continue
        rows.append(row)
        _write_manifest(args, rows)

    manifest = _manifest(args, rows)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(row["status"] == "ok" for row in rows) else 1


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _manifest(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema": "wod_gcs_shard_sample_export_manifest_v1",
        "shard_glob": args.shard_glob,
        "shard_start": args.shard_start,
        "shard_count": args.shard_count,
        "records_per_shard": args.records_per_shard,
        "output": str(args.output),
        "manifest_output": str(args.manifest_output),
        "total_rows": _line_count(args.output),
        "successful_shards": sum(1 for row in rows if row["status"] == "ok"),
        "failed_shards": sum(1 for row in rows if row["status"] != "ok"),
        "shards": rows,
    }


def _write_manifest(args: argparse.Namespace, rows: list[dict[str, Any]]) -> None:
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(_manifest(args, rows), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
