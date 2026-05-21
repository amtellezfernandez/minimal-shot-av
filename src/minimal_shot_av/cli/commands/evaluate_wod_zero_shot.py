#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.zero_shot_eval import main


if __name__ == "__main__":
    raise SystemExit(main())
