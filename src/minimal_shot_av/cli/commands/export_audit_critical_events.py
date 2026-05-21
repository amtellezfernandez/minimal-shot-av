from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_shot_av.audit import critical_event_bundle, load_audit_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a critical-events bundle from a normalized audit log.")
    parser.add_argument("audit_root", type=Path, help="Audit log directory containing manifest.json and frames.jsonl")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path for the critical-events bundle")
    parser.add_argument("--context-radius", type=int, default=2, help="Frames to keep before and after each bookmark")
    args = parser.parse_args()

    manifest, frames = load_audit_log(args.audit_root)
    bundle = critical_event_bundle(manifest, frames, context_radius=max(0, args.context_radius))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "bookmark_count": bundle["bookmark_count"], "critical_frame_count": bundle["critical_frame_count"]}, indent=2))


if __name__ == "__main__":
    main()
