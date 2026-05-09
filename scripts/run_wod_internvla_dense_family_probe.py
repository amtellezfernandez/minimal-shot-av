#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_internvla_dense_family_probe"
DEFAULT_FRAME_CACHE = ROOT / "artifacts" / "wod_preference_frames_val479.json"
DEFAULT_EXTERNAL_CACHE = ROOT / "artifacts" / "cosmos_predict25_wan21_tokenizer_val479.json"
DEFAULT_NEURAL_MODELS = ",".join(
    str(path)
    for path in (
        ROOT / "artifacts" / "wod_neural_holdout" / "neural_anchor16_h128_e25.json",
        ROOT / "artifacts" / "wod_neural_holdout" / "neural_anchor16_h128_e35_seed31.json",
        ROOT / "artifacts" / "wod_neural_holdout" / "neural_anchor16_h128_e35_seed73.json",
    )
)
DEFAULT_EXTERNAL_CANDIDATES = ROOT / "artifacts" / "wod_internvla_av_candidates_dense.jsonl"


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe dense InternVLA AV candidates around the best family selector.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--external-embedding-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--external-candidate-jsonl", type=Path, default=DEFAULT_EXTERNAL_CANDIDATES)
    parser.add_argument("--neural-candidate-models", default=DEFAULT_NEURAL_MODELS)
    parser.add_argument("--waymo-src", type=Path, required=True)
    parser.add_argument("--official-frames", type=int, default=20)
    parser.add_argument("--official-folds", type=int, default=2)
    parser.add_argument("--skip-full-official", action="store_true")
    args = parser.parse_args()

    if not args.external_candidate_jsonl.is_file():
        raise FileNotFoundError(args.external_candidate_jsonl)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    official_reports = _run_variants(args)
    best_official = _best_report(official_reports)

    full_official = None
    if not args.skip_full_official:
        full_output = args.output_dir / f"full_official_{best_official['label']}.json"
        command = _build_command(
            args,
            output=full_output,
            variant=best_official["variant"],
            frames=479,
            folds=2,
        )
        status = _run(command)
        if status != 0:
            return status
        full_official = json.loads(full_output.read_text(encoding="utf-8"))

    summary = {
        "schema": "wod_internvla_dense_family_probe_summary_v1",
        "best_official": best_official,
        "full_official": full_official,
        "official_reports": official_reports,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _variants() -> list[dict[str, Any]]:
    return [
        {
            "label": "base_dense",
            "selector_family_calibration": "off",
            "selector_family_calibration_min_count": 8,
            "selector_source_calibration": "off",
            "selector_source_calibration_scale": 1.0,
            "learned_reliability_gate_features": False,
            "learned_reliability_model": "ridge",
            "learned_reliability_ridge": 30.0,
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
        },
        {
            "label": "familycal_dense",
            "selector_family_calibration": "speed_source_family",
            "selector_family_calibration_min_count": 6,
            "selector_source_calibration": "off",
            "selector_source_calibration_scale": 1.0,
            "learned_reliability_gate_features": False,
            "learned_reliability_model": "ridge",
            "learned_reliability_ridge": 30.0,
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
        },
        {
            "label": "learned_gate_dense",
            "selector_family_calibration": "speed_source_family",
            "selector_family_calibration_min_count": 6,
            "selector_source_calibration": "off",
            "selector_source_calibration_scale": 1.0,
            "learned_reliability_gate_features": True,
            "learned_reliability_model": "boosted_stumps",
            "learned_reliability_ridge": 30.0,
            "source_gate_ridge": 0.10,
            "source_gate_max_rate": 0.25,
        },
    ]


def _run_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    reports = []
    for variant in _variants():
        output = args.output_dir / f"official_{variant['label']}.json"
        command = _build_command(
            args,
            output=output,
            variant=variant,
            frames=args.official_frames,
            folds=args.official_folds,
        )
        status = _run(command)
        if status != 0:
            raise SystemExit(status)
        reports.append(
            {
                "label": variant["label"],
                "output": str(output),
                "variant": variant,
                "command": command,
                "report": json.loads(output.read_text(encoding="utf-8")),
            }
        )
    return reports


def _build_command(
    args: argparse.Namespace,
    *,
    output: Path,
    variant: dict[str, Any],
    frames: int,
    folds: int,
) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"),
        "--frame-cache",
        str(args.frame_cache),
        "--output",
        str(output),
        "--rfs-backend",
        "official",
        "--waymo-src",
        str(args.waymo_src),
        "--progress-every-fold",
        "--folds",
        str(folds),
        "--max-preference-frames",
        str(frames),
        "--feature-set",
        "base",
        "--aux-feature-set",
        "temporal_summary",
        "--scene-aux-feature-set",
        "external_embeddings",
        "--external-embedding-cache",
        str(args.external_embedding_cache),
        "--external-candidate-jsonl",
        str(args.external_candidate_jsonl),
        "--neural-candidate-models",
        str(args.neural_candidate_models),
        "--residual-modes",
        "3",
        "--kinematic-profile",
        "internnav",
        "--selector-features",
        "family_reliability_contextual",
        "--selector-target",
        "frame_delta",
        "--selector-model",
        "linear",
        "--selector-ridge",
        "1000",
        "--selector-fallback-source-options",
        "kinematic;kinematic,temporal",
        "--selector-fallback-local-selector",
        "--selector-family-calibration",
        str(variant["selector_family_calibration"]),
        "--selector-family-calibration-min-count",
        str(variant["selector_family_calibration_min_count"]),
        "--selector-source-calibration",
        str(variant["selector_source_calibration"]),
        "--selector-source-calibration-scale",
        str(variant["selector_source_calibration_scale"]),
        "--source-gate",
        "independent_train_margin",
        "--source-gate-sources",
        "internnav,internvla",
        "--source-gate-candidate-prefixes",
        "internnav_s2_waypoint_cautious,internnav_s1_stop_progress,internvla_",
        "--source-gate-router",
        "speed",
        "--source-gate-ridge",
        str(variant["source_gate_ridge"]),
        "--source-gate-max-rate",
        str(variant["source_gate_max_rate"]),
        "--source-gate-min-precision",
        "0.0",
        "--source-gate-local-selector",
        "--scene-gate",
        "train_margin",
        "--scene-gate-router",
        "speed",
        "--scene-gate-ridge",
        "2.0",
        "--scene-gate-max-rate",
        "0.16",
        "--selector-postprocess",
        "off",
    ]
    if variant["learned_reliability_gate_features"]:
        command.append("--learned-reliability-gate-features")
        command.extend(["--learned-reliability-model", str(variant["learned_reliability_model"])])
        command.extend(["--learned-reliability-ridge", str(variant["learned_reliability_ridge"])])
    return command


def _best_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    return max(
        reports,
        key=lambda entry: (
            float(entry["report"].get("combined_ranker_mean_rfs", float("-inf"))),
            int(entry["report"].get("source_gate_override_count", 0) or 0),
            float(entry["report"].get("scene_gate_precision", 0.0) or 0.0),
        ),
    )


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
