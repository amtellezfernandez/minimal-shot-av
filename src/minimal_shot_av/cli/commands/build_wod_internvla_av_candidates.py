#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.internvla_av_bridge import (  # noqa: E402
    load_internvla_navigation_cues,
    write_internvla_av_candidate_jsonl,
)
from minimal_shot_av.model.rfs_metric import RfsReference  # noqa: E402
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame  # noqa: E402


DEFAULT_FRAME_CACHE = ROOT / "artifacts" / "wod_preference_frames_val479.json"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert offline InternVLA pixel-goal/STOP annotations into WOD AV trajectory candidates."
    )
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--internvla-annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="internvla")
    parser.add_argument(
        "--profile",
        choices=("standard", "dense"),
        default="standard",
        help="Candidate template profile to emit. dense adds more action-specific hypotheses.",
    )
    args = parser.parse_args()

    frames = _load_frame_cache(args.frame_cache)
    cues = load_internvla_navigation_cues(args.internvla_annotations)
    count = write_internvla_av_candidate_jsonl(
        frames,
        cues,
        args.output,
        source=args.source,
        profile=args.profile,
    )
    print(
        json.dumps(
            {
                "frames": len(frames),
                "internvla_cues": len(cues),
                "candidate_count": count,
                "output": str(args.output),
                "source": args.source,
                "profile": args.profile,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _load_frame_cache(path: Path) -> list[WodE2EPreferenceFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") not in {None, "wod_preference_frames_v1"}:
        raise ValueError(f"unsupported WOD frame cache schema: {payload.get('schema')!r}")
    rows = payload.get("frames", [])
    if not isinstance(rows, list):
        raise ValueError("WOD frame cache frames must be a list")
    return [_frame_from_payload(row) for row in rows]


def _frame_from_payload(payload: dict[str, object]) -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name=str(payload["frame_name"]),
        past_trajectory=_trajectory(payload["past_trajectory"]),
        future_trajectory=_trajectory(payload["future_trajectory"]),
        intent=int(payload["intent"]),
        init_speed_mps=float(payload["init_speed_mps"]),
        references=[
            RfsReference(
                label=str(reference["label"]),
                trajectory=_trajectory(reference["trajectory"]),
                score=float(reference["score"]),
            )
            for reference in payload.get("references", [])
        ],
    )


def _trajectory(raw: object) -> list[tuple[float, float]]:
    if not isinstance(raw, list):
        raise ValueError("trajectory must be a list")
    points: list[tuple[float, float]] = []
    for item in raw:
        if isinstance(item, dict):
            points.append((float(item["x"]), float(item["y"])))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            points.append((float(item[0]), float(item[1])))
        else:
            raise ValueError(f"invalid trajectory point: {item!r}")
    return points


if __name__ == "__main__":
    raise SystemExit(main())
