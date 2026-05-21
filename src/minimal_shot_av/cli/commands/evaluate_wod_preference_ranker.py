#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.wod_ranker import WodPreferenceRanker


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a saved WOD-E2E preference ranker on JSONL candidate rows.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--ranker", type=Path, required=True)
    parser.add_argument("--limit-frames", type=int, default=None)
    args = parser.parse_args()

    ranker = WodPreferenceRanker.load(args.ranker)
    rows = _load_rows(args.input)
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_frame[str(row["frame_name"])].append(row)

    selected_scores: list[float] = []
    logged_scores: list[float] = []
    oracle_scores: list[float] = []
    top1 = 0

    for index, (frame_name, frame_rows) in enumerate(sorted(by_frame.items()), start=1):
        if args.limit_frames is not None and index > args.limit_frames:
            break
        selected = ranker.select_row(frame_rows)
        logged = next(row for row in frame_rows if row["candidate_name"] == "logged_future")
        oracle = max(frame_rows, key=lambda row: float(row["rfs_score"]))
        selected_score = float(selected["rfs_score"])
        logged_score = float(logged["rfs_score"])
        oracle_score = float(oracle["rfs_score"])
        selected_scores.append(selected_score)
        logged_scores.append(logged_score)
        oracle_scores.append(oracle_score)
        if selected_score == oracle_score:
            top1 += 1
        print(
            f"{index:04d} frame={frame_name} selected={selected['candidate_name']} "
            f"selected_rfs={selected_score:.3f} logged={logged_score:.3f} "
            f"oracle={oracle_score:.3f}"
        )

    if not selected_scores:
        raise RuntimeError("no frames evaluated")

    count = len(selected_scores)
    print(
        "summary "
        f"frames={count} "
        f"selected_mean={sum(selected_scores) / count:.3f} "
        f"logged_mean={sum(logged_scores) / count:.3f} "
        f"oracle_mean={sum(oracle_scores) / count:.3f} "
        f"regret={sum(o - s for o, s in zip(oracle_scores, selected_scores)) / count:.3f} "
        f"top1={top1 / count:.3f}"
    )
    return 0


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
