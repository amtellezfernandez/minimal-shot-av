#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_source_gate_probe"
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
        description="Probe a small source-gate neighborhood around the best official champion recipe."
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
    local_reports = []
    for label, command in _local_variants(args):
        output = args.output_dir / f"local_{label}.json"
        command = _replace_output(command, output)
        status = _run(command)
        if status != 0:
            return status
        local_reports.append(
            {
                "label": label,
                "output": str(output),
                "command": command,
                "report": json.loads(output.read_text(encoding="utf-8")),
            }
        )
    best_local = _best_report(local_reports)

    official_reports = []
    for label, command in _official_variants(args, best_local["label"]):
        output = args.output_dir / f"official_{label}.json"
        command = _replace_output(command, output)
        status = _run(command)
        if status != 0:
            return status
        official_reports.append(
            {
                "label": label,
                "output": str(output),
                "command": command,
                "report": json.loads(output.read_text(encoding="utf-8")),
            }
        )
    best_official = _best_report(official_reports)

    full_official = None
    if not args.skip_full_official:
        full_output = args.output_dir / f"full_official_{best_official['label']}.json"
        command = _replace_output(
            list(best_official["command"]),
            full_output,
        )
        command = _replace_arg(command, "--rfs-backend", "official")
        command = _replace_arg(command, "--waymo-src", str(args.waymo_src))
        command = _replace_arg(command, "--max-preference-frames", "479")
        command = _replace_arg(command, "--folds", "3")
        if "--progress-every-fold" not in command:
            command.append("--progress-every-fold")
        status = _run(command)
        if status != 0:
            return status
        full_official = json.loads(full_output.read_text(encoding="utf-8"))

    summary = {
        "schema": "wod_source_gate_probe_summary_v1",
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


def _local_variants(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    base = _base_command(
        args,
        output=args.output_dir / "local_base.json",
        backend="local",
        source_gate_ridge=0.12,
        source_gate_max_rate=0.30,
        scene_gate_ridge=2.0,
        scene_gate_max_rate=0.16,
    )
    return [
        ("base", base),
        (
            "source_tight",
            _base_command(
                args,
                output=args.output_dir / "local_source_tight.json",
                backend="local",
                source_gate_ridge=0.10,
                source_gate_max_rate=0.28,
                scene_gate_ridge=2.0,
                scene_gate_max_rate=0.16,
            ),
        ),
        (
            "source_loose",
            _base_command(
                args,
                output=args.output_dir / "local_source_loose.json",
                backend="local",
                source_gate_ridge=0.14,
                source_gate_max_rate=0.32,
                scene_gate_ridge=2.0,
                scene_gate_max_rate=0.16,
            ),
        ),
        (
            "scene_tight",
            _base_command(
                args,
                output=args.output_dir / "local_scene_tight.json",
                backend="local",
                source_gate_ridge=0.12,
                source_gate_max_rate=0.30,
                scene_gate_ridge=1.5,
                scene_gate_max_rate=0.14,
            ),
        ),
        (
            "scene_loose",
            _base_command(
                args,
                output=args.output_dir / "local_scene_loose.json",
                backend="local",
                source_gate_ridge=0.12,
                source_gate_max_rate=0.30,
                scene_gate_ridge=2.5,
                scene_gate_max_rate=0.18,
            ),
        ),
        (
            "balanced",
            _base_command(
                args,
                output=args.output_dir / "local_balanced.json",
                backend="local",
                source_gate_ridge=0.10,
                source_gate_max_rate=0.28,
                scene_gate_ridge=1.5,
                scene_gate_max_rate=0.14,
            ),
        ),
    ]


def _official_variants(args: argparse.Namespace, _best_local_label: str) -> list[tuple[str, list[str]]]:
    return [
        (
            "base",
            _base_command(
                args,
                output=args.output_dir / "official_base.json",
                backend="official",
                source_gate_ridge=0.12,
                source_gate_max_rate=0.30,
                scene_gate_ridge=2.0,
                scene_gate_max_rate=0.16,
            ),
        ),
        (
            "source_tight",
            _base_command(
                args,
                output=args.output_dir / "official_source_tight.json",
                backend="official",
                source_gate_ridge=0.10,
                source_gate_max_rate=0.28,
                scene_gate_ridge=2.0,
                scene_gate_max_rate=0.16,
            ),
        ),
        (
            "scene_tight",
            _base_command(
                args,
                output=args.output_dir / "official_scene_tight.json",
                backend="official",
                source_gate_ridge=0.12,
                source_gate_max_rate=0.30,
                scene_gate_ridge=1.5,
                scene_gate_max_rate=0.14,
            ),
        ),
        (
            "balanced",
            _base_command(
                args,
                output=args.output_dir / "official_balanced.json",
                backend="official",
                source_gate_ridge=0.10,
                source_gate_max_rate=0.28,
                scene_gate_ridge=1.5,
                scene_gate_max_rate=0.14,
            ),
        ),
    ]


def _base_command(
    args: argparse.Namespace,
    *,
    output: Path,
    backend: str,
    source_gate_ridge: float,
    source_gate_max_rate: float,
    scene_gate_ridge: float,
    scene_gate_max_rate: float,
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
        "family_reliability_contextual",
        "--selector-target",
        "frame_delta",
        "--selector-ridge",
        "1000",
        "--selector-model",
        "linear",
        "--neural-top-k",
        "1",
        "--selector-kinematic-fallback",
        "train_margin",
        "--selector-fallback-router",
        "speed",
        "--scene-gate",
        "train_margin",
        "--scene-gate-ridge",
        str(scene_gate_ridge),
        "--scene-gate-max-rate",
        str(scene_gate_max_rate),
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
        "--selector-postprocess",
        "off",
        "--neural-candidate-models",
        str(args.neural_candidate_models),
        "--max-preference-frames",
        str(args.local_frames if backend == "local" else args.official_frames),
        "--folds",
        str(args.local_folds if backend == "local" else args.official_folds),
    ]
    if backend == "official":
        command.extend(["--waymo-src", str(args.waymo_src)])
        command.append("--progress-every-fold")
    return command


def _replace_arg(command: list[str], key: str, value: str) -> list[str]:
    if key in command:
        idx = command.index(key)
        command[idx + 1] = value
        return command
    command.extend([key, value])
    return command


def _replace_output(command: list[str], output: Path) -> list[str]:
    return _replace_arg(command, "--output", str(output))


def _best_report(reports: list[dict[str, Any]]) -> dict[str, Any]:
    best = max(
        reports,
        key=lambda entry: (
            float(entry["report"].get("combined_ranker_mean_rfs", float("-inf"))),
            int(entry["report"].get("direct_policy_override_count", 0) or 0),
            float(entry["report"].get("source_gate_precision", 0.0) or 0.0),
        ),
    )
    return best


def _run(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
