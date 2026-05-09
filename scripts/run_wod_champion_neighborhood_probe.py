#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_champion_neighborhood_probe"
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Probe a tight source/scene gate neighborhood around the best official champion recipe."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--external-embedding-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--neural-candidate-models", default=DEFAULT_NEURAL_MODELS)
    parser.add_argument("--local-frames", type=int, default=80)
    parser.add_argument("--local-folds", type=int, default=2)
    parser.add_argument("--official-frames", type=int, default=20)
    parser.add_argument("--official-folds", type=int, default=2)
    parser.add_argument("--waymo-src", type=Path, required=True)
    parser.add_argument("--skip-full-official", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    local_reports = _run_variants(
        args,
        backend="local",
        variants=_local_variants(args),
        prefix="local",
    )
    best_local = _best_report(local_reports)
    official_reports = _run_variants(
        args,
        backend="official",
        variants=_official_variants(args),
        prefix="official",
    )
    best_official = _best_report(official_reports)

    full_official = None
    if not args.skip_full_official:
        full_output = args.output_dir / f"full_official_{best_official['label']}.json"
        command = _build_command(
            args,
            backend="official",
            output=full_output,
            source_gate_ridge=best_official["report"]["source_gate_ridge"],
            source_gate_max_rate=best_official["report"]["source_gate_max_rate"],
            scene_gate_ridge=best_official["report"]["scene_gate_ridge"],
            scene_gate_max_rate=best_official["report"]["scene_gate_max_rate"],
            selector_postprocess="off",
        )
        status = _run(command)
        if status != 0:
            return status
        full_official = json.loads(full_output.read_text(encoding="utf-8"))

    summary = {
        "schema": "wod_champion_neighborhood_probe_summary_v1",
        "best_local": best_local,
        "best_official": best_official,
        "full_official": full_official,
        "local_reports": local_reports,
        "official_reports": official_reports,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _local_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "label": "base",
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
            "scene_gate_ridge": 2.0,
            "scene_gate_max_rate": 0.16,
        },
        {
            "label": "source_tighter",
            "source_gate_ridge": 0.10,
            "source_gate_max_rate": 0.28,
            "scene_gate_ridge": 2.0,
            "scene_gate_max_rate": 0.16,
        },
        {
            "label": "source_looser",
            "source_gate_ridge": 0.14,
            "source_gate_max_rate": 0.32,
            "scene_gate_ridge": 2.0,
            "scene_gate_max_rate": 0.16,
        },
        {
            "label": "scene_tighter",
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
            "scene_gate_ridge": 1.6,
            "scene_gate_max_rate": 0.14,
        },
        {
            "label": "scene_looser",
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
            "scene_gate_ridge": 2.4,
            "scene_gate_max_rate": 0.18,
        },
        {
            "label": "balanced",
            "source_gate_ridge": 0.10,
            "source_gate_max_rate": 0.28,
            "scene_gate_ridge": 1.6,
            "scene_gate_max_rate": 0.14,
        },
    ]


def _official_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    return [
        {
            "label": "base",
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
            "scene_gate_ridge": 2.0,
            "scene_gate_max_rate": 0.16,
        },
        {
            "label": "source_tighter",
            "source_gate_ridge": 0.10,
            "source_gate_max_rate": 0.28,
            "scene_gate_ridge": 2.0,
            "scene_gate_max_rate": 0.16,
        },
        {
            "label": "scene_tighter",
            "source_gate_ridge": 0.12,
            "source_gate_max_rate": 0.30,
            "scene_gate_ridge": 1.6,
            "scene_gate_max_rate": 0.14,
        },
        {
            "label": "balanced",
            "source_gate_ridge": 0.10,
            "source_gate_max_rate": 0.28,
            "scene_gate_ridge": 1.6,
            "scene_gate_max_rate": 0.14,
        },
    ]


def _run_variants(
    args: argparse.Namespace,
    *,
    backend: str,
    variants: list[dict[str, Any]],
    prefix: str,
) -> list[dict[str, Any]]:
    reports = []
    for variant in variants:
        output = args.output_dir / f"{prefix}_{variant['label']}.json"
        command = _build_command(
            args,
            backend=backend,
            output=output,
            source_gate_ridge=variant["source_gate_ridge"],
            source_gate_max_rate=variant["source_gate_max_rate"],
            scene_gate_ridge=variant["scene_gate_ridge"],
            scene_gate_max_rate=variant["scene_gate_max_rate"],
            selector_postprocess="off",
        )
        status = _run(command)
        if status != 0:
            raise SystemExit(status)
        reports.append(
            {
                "label": variant["label"],
                "output": str(output),
                "command": command,
                "report": json.loads(output.read_text(encoding="utf-8")),
            }
        )
    return reports


def _build_command(
    args: argparse.Namespace,
    *,
    backend: str,
    output: Path,
    source_gate_ridge: float,
    source_gate_max_rate: float,
    scene_gate_ridge: float,
    scene_gate_max_rate: float,
    selector_postprocess: str,
) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"),
        "--frame-cache",
        str(args.frame_cache),
        "--output",
        str(output),
        "--rfs-backend",
        backend,
        "--feature-set",
        "base",
        "--aux-feature-set",
        "temporal_summary",
        "--scene-aux-feature-set",
        "external_embeddings",
        "--external-embedding-cache",
        str(args.external_embedding_cache),
        "--residual-modes",
        "3",
        "--kinematic-profile",
        "internnav",
        "--selector-features",
        "contextual",
        "--selector-target",
        "frame_delta",
        "--selector-ridge",
        "1000",
        "--selector-model",
        "linear",
        "--selector-fallback-source-options",
        "kinematic;kinematic,temporal",
        "--selector-fallback-local-selector",
        "--source-gate",
        "independent_train_margin",
        "--source-gate-sources",
        "internnav,learned",
        "--source-gate-candidate-prefixes",
        "internnav_s2_waypoint_cautious,internnav_s1_stop_progress,ensemble",
        "--source-gate-router",
        "speed",
        "--source-gate-ridge",
        str(source_gate_ridge),
        "--source-gate-max-rate",
        str(source_gate_max_rate),
        "--source-gate-min-precision",
        "0.0",
        "--source-gate-local-selector",
        "--scene-gate",
        "train_margin",
        "--scene-gate-router",
        "speed",
        "--scene-gate-ridge",
        str(scene_gate_ridge),
        "--scene-gate-max-rate",
        str(scene_gate_max_rate),
        "--scene-aux-feature-set",
        "external_embeddings",
        "--neural-candidate-models",
        str(args.neural_candidate_models),
        "--selector-postprocess",
        selector_postprocess,
    ]
    if backend == "official":
        command.extend(
            [
                "--waymo-src",
                str(args.waymo_src),
                "--progress-every-fold",
                "--folds",
                str(args.official_folds),
                "--max-preference-frames",
                str(args.official_frames),
            ]
        )
    else:
        command.extend(["--folds", str(args.local_folds), "--max-preference-frames", str(args.local_frames)])
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
