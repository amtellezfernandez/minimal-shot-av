#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_submission import load_frame_names, validate_submission_tar


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a packaged WOD-E2E submission tar.gz.")
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--frame-list", type=Path, help="Optional challenge JSON listing required test frame names.")
    args = parser.parse_args()

    required_frame_names = load_frame_names(args.frame_list) if args.frame_list else None
    report = validate_submission_tar(args.submission, required_frame_names=required_frame_names)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
