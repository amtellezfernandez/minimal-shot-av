from __future__ import annotations

import argparse
from pathlib import Path

from minimal_shot_av.audit import export_internal_audit_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an internal rollout JSON into a normalized audit log.")
    parser.add_argument("--rollout-json", type=Path, required=True, help="Path to latest_rollout.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the normalized audit log")
    args = parser.parse_args()
    manifest = export_internal_audit_log(args.rollout_json, args.output_dir)
    print(f"Wrote audit log to {args.output_dir} ({manifest['frame_count']} frames)")


if __name__ == "__main__":
    main()
