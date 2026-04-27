from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.neutral.benchmark_compare import compare_reports, load_metric_report, parse_metric_report
from minimal_shot_av.neutral.benchmark_reports import alpasim_metrics_report_to_metric_report


PUBLISHED_METADATA: dict[str, Any] = {
    "evaluation_contract": "physicalai_nurec_alpasim_closed_loop",
    "scenario_set": "physicalai_av_nurec_910",
    "score_backend": "alpasim",
    "sensor_contract": "published_alpamayo_1_5_model_card",
    "camera_ids": "published_multi_camera_rgb",
    "context_length": "published_alpamayo_1_5",
    "ego_history_hz": 10,
    "output_horizon": "6.4s_64_waypoints_10hz",
    "route_command_source": "navigation_guidance",
    "alpasim_version": "published_model_card_unspecified",
}

FRONT_CAMERA_METADATA: dict[str, Any] = {
    "evaluation_contract": "physicalai_nurec_alpasim_closed_loop",
    "scenario_set": "physicalai_av_nurec_910",
    "score_backend": "alpasim",
    "sensor_contract": "front_wide_120fov_only",
    "camera_ids": ["camera_front_wide_120fov"],
    "context_length": 1,
    "ego_history_hz": 10,
    "output_horizon": "5.0s_20_waypoints_4hz",
    "route_command_source": "waypoint_commands",
    "alpasim_version": "published_model_card_unspecified",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce comparator-ready AlpaSim reports for the published and front-camera tracks."
    )
    parser.add_argument(
        "--published-ours-run",
        required=True,
        help="Run dir or metrics file for our published-contract run.",
    )
    parser.add_argument(
        "--front-ours-run",
        required=True,
        help="Run dir or metrics file for our front-camera-only run.",
    )
    parser.add_argument(
        "--front-alpamayo-run",
        required=True,
        help="Run dir or metrics file for Alpamayo front-camera-only run.",
    )
    parser.add_argument("--output-dir", default=str(ROOT / "benchmarks" / "current"))
    parser.add_argument(
        "--published-baseline",
        default=str(ROOT / "benchmarks" / "baselines" / "alpamayo_1_5_alpasim.json"),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons_dir = output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    published_ours = _write_report(
        args.published_ours_run,
        output_dir / "spotlight_reflex_alpasim_published_contract.json",
        system="spotlight_reflex_alpasim_published_contract",
        metadata=PUBLISHED_METADATA,
    )
    front_ours = _write_report(
        args.front_ours_run,
        output_dir / "spotlight_reflex_alpasim_front_camera.json",
        system="spotlight_reflex_alpasim_front_camera",
        metadata=FRONT_CAMERA_METADATA,
    )
    front_alpamayo = _write_report(
        args.front_alpamayo_run,
        output_dir / "alpamayo_1_5_alpasim_front_camera.json",
        system="alpamayo_1_5_front_camera",
        metadata=FRONT_CAMERA_METADATA,
    )

    published_comparison = compare_reports(published_ours, load_metric_report(args.published_baseline)).to_dict()
    front_comparison = compare_reports(front_ours, front_alpamayo).to_dict()
    _write_json(comparisons_dir / "spotlight_reflex_vs_alpamayo_1_5_published_contract.json", published_comparison)
    _write_json(comparisons_dir / "spotlight_reflex_vs_alpamayo_1_5_front_camera.json", front_comparison)

    print(
        json.dumps(
            {"published_contract": published_comparison, "front_camera": front_comparison},
            indent=2,
            sort_keys=True,
        )
    )


def _write_report(source: str, output: Path, *, system: str, metadata: dict[str, Any]):
    report = alpasim_metrics_report_to_metric_report(
        source,
        system=system,
        suite="physicalai_nurec_alpasim",
        required_metrics=("alpasim_score",),
        metadata=metadata,
    )
    _write_json(output, report)
    return parse_metric_report(report)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
