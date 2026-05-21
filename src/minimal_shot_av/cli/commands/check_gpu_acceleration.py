#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
from minimal_shot_av.cli.commands.check_cuda_preflight import cuda_preflight_report


def main() -> int:
    report = cuda_preflight_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("torch_cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
