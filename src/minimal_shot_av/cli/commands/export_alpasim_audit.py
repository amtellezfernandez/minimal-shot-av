from __future__ import annotations

import argparse
from pathlib import Path

from minimal_shot_av.audit import export_alpasim_audit_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Export an AlpaSim run directory into a normalized audit log.")
    parser.add_argument("--run-dir", type=Path, required=True, help="AlpaSim run directory with launch metadata and driver logs")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write the normalized audit log")
    args = parser.parse_args()
    manifest = export_alpasim_audit_log(args.run_dir, args.output_dir)
    print(f"Wrote AlpaSim audit log to {args.output_dir} ({manifest['frame_count']} frames)")


if __name__ == "__main__":
    main()
