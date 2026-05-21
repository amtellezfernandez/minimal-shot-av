from __future__ import annotations

import argparse
import json
from pathlib import Path

from minimal_shot_av.audit import load_audit_log, view_audit_log_with_rerun


def main() -> None:
    parser = argparse.ArgumentParser(description="View a normalized audit log in Rerun.")
    parser.add_argument("audit_root", type=Path, help="Audit log directory containing manifest.json and frames.jsonl")
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer.")
    parser.add_argument("--summary-only", action="store_true", help="Print manifest/summary without opening Rerun.")
    args = parser.parse_args()

    if args.summary_only:
        manifest, frames = load_audit_log(args.audit_root)
        print(json.dumps({"manifest": manifest, "frame_count": len(frames)}, indent=2))
        return
    manifest = view_audit_log_with_rerun(args.audit_root, spawn=args.spawn)
    print(f"Opened Rerun view for {manifest['scenario_cluster']} seed={manifest['seed']}")


if __name__ == "__main__":
    main()
