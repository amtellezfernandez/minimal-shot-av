#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Deduplicate a JSONL file by one JSON object key.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--key", default="frame_name")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output_path = args.output or args.path.with_suffix(args.path.suffix + ".deduped")
    seen: set[object] = set()
    total = 0
    kept = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.path.open("r", encoding="utf-8") as source, output_path.open("w", encoding="utf-8") as output:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            total += 1
            payload = json.loads(line)
            if args.key not in payload:
                raise KeyError(f"line {line_number} missing key {args.key!r}")
            value = payload[args.key]
            if value in seen:
                continue
            seen.add(value)
            output.write(json.dumps(payload, sort_keys=True) + "\n")
            kept += 1
    print(
        json.dumps(
            {
                "path": str(args.path),
                "output": str(output_path),
                "key": args.key,
                "total": total,
                "kept": kept,
                "duplicates": total - kept,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
