#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from check_cuda_preflight import cuda_preflight_report  # noqa: E402


def main() -> int:
    report = cuda_preflight_report()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("torch_cuda_available") else 1


if __name__ == "__main__":
    raise SystemExit(main())
