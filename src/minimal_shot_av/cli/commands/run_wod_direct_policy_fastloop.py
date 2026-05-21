#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_direct_policy_fastloop"
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
        description="Run a short direct-policy local tune, then one official preflight on the best candidate."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--study-name", default="wod_direct_policy_fastloop")
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--trial-timeout", type=float, default=1800.0)
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--external-embedding-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--neural-candidate-models", default=DEFAULT_NEURAL_MODELS)
    parser.add_argument("--local-frames", type=int, default=20)
    parser.add_argument("--local-folds", type=int, default=2)
    parser.add_argument("--official-frames", type=int, default=20)
    parser.add_argument("--official-folds", type=int, default=2)
    parser.add_argument("--waymo-src", type=Path, required=True)
    parser.add_argument(
        "--skip-official-preflight",
        action="store_true",
        help="Skip the short official sweep after the local tune.",
    )
    parser.add_argument("--full-official-output", type=Path)
    parser.add_argument("--gain-reward", type=float, default=4.0)
    parser.add_argument("--false-positive-penalty", type=float, default=1.0)
    parser.add_argument("--min-direct-policy-overrides", type=int, default=5)
    parser.add_argument("--min-direct-policy-selected-rate", type=float, default=0.0)
    parser.add_argument("--underactive-penalty", type=float, default=8.0)
    parser.add_argument("--progress-every-fold", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tune_output_dir = args.output_dir / "optuna"
    tune_output_dir.mkdir(parents=True, exist_ok=True)

    tune_command = _tuning_command(args, tune_output_dir)
    tune_status = _run_command(tune_command)
    if tune_status != 0:
        return tune_status

    manifest_path = tune_output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    best = manifest.get("best")
    if not best:
        raise SystemExit("No successful trial was produced by the local tune.")

    official_preflight = None
    if not args.skip_official_preflight:
        best_command = list(best["user_attrs"]["command"])
        official_variants = _official_sweep_commands(
            best_command,
            output_dir=args.output_dir,
            waymo_src=args.waymo_src,
            frames=args.official_frames,
            folds=args.official_folds,
            progress_every_fold=args.progress_every_fold,
        )
        official_reports = []
        for label, official_command, official_output in official_variants:
            status = _run_command(official_command)
            if status != 0:
                return status
            report = json.loads(official_output.read_text(encoding="utf-8"))
            official_reports.append({"label": label, "output": str(official_output), "report": report})
        official_preflight = _best_official_report(official_reports)

    full_official = None
    if args.full_official_output is not None:
        if official_preflight is None:
            best_command = list(best["user_attrs"]["command"])
            official_variants = _official_sweep_commands(
                best_command,
                output_dir=args.output_dir,
                waymo_src=args.waymo_src,
                frames=args.official_frames,
                folds=args.official_folds,
                progress_every_fold=args.progress_every_fold,
            )
            official_reports = []
            for label, official_command, official_output in official_variants:
                status = _run_command(official_command)
                if status != 0:
                    return status
                report = json.loads(official_output.read_text(encoding="utf-8"))
                official_reports.append({"label": label, "output": str(official_output), "report": report})
            official_preflight = _best_official_report(official_reports)
        full_command = _full_official_command(
            list(best["user_attrs"]["command"]),
            output=args.full_official_output,
            waymo_src=args.waymo_src,
            frames=479,
            folds=3,
            progress_every_fold=True,
        )
        full_status = _run_command(full_command)
        if full_status != 0:
            return full_status
        full_official = json.loads(args.full_official_output.read_text(encoding="utf-8"))

    summary = {
        "schema": "wod_direct_policy_fastloop_summary_v1",
        "tune_output_dir": str(tune_output_dir),
        "best": best,
        "official_preflight": official_preflight,
        "full_official": full_official,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _tuning_command(args: argparse.Namespace, output_dir: Path) -> list[str]:
    command = [
        args.python,
        str(ROOT / "scripts" / "tune_wod_direct_policy_optuna.py"),
        "--output-dir",
        str(output_dir),
        "--study-name",
        str(args.study_name),
        "--trials",
        str(args.trials),
        "--trial-timeout",
        str(args.trial_timeout),
        "--frame-cache",
        str(args.frame_cache),
        "--external-embedding-cache",
        str(args.external_embedding_cache),
        "--neural-candidate-models",
        str(args.neural_candidate_models),
        "--rfs-backend",
        "local",
        "--max-preference-frames",
        str(args.local_frames),
        "--folds",
        str(args.local_folds),
        "--gain-reward",
        str(args.gain_reward),
        "--false-positive-penalty",
        str(args.false_positive_penalty),
        "--min-direct-policy-overrides",
        str(args.min_direct_policy_overrides),
        "--min-direct-policy-selected-rate",
        str(args.min_direct_policy_selected_rate),
        "--underactive-penalty",
        str(args.underactive_penalty),
        "--identity-mode",
        "full",
    ]
    return command


def _official_preflight_command(
    command: list[str],
    *,
    output: Path,
    waymo_src: Path,
    frames: int,
    folds: int,
    progress_every_fold: bool,
) -> list[str]:
    command = list(command)
    command = _replace_command_arg(command, "--output", str(output))
    command = _replace_command_arg(command, "--rfs-backend", "official")
    command = _replace_command_arg(command, "--waymo-src", str(waymo_src))
    command = _replace_command_arg(command, "--max-preference-frames", str(frames))
    command = _replace_command_arg(command, "--folds", str(folds))
    if progress_every_fold and "--progress-every-fold" not in command:
        command.append("--progress-every-fold")
    return command


def _official_sweep_commands(
    command: list[str],
    *,
    output_dir: Path,
    waymo_src: Path,
    frames: int,
    folds: int,
    progress_every_fold: bool,
) -> list[tuple[str, list[str], Path]]:
    variants: list[tuple[str, list[str], Path]] = []
    base_variants = [
        ("base", None, None, None, None),
        ("relaxed", 0.0, 0.85, None, None),
        ("permissive", 0.0, 0.95, None, None),
        ("mild", 0.05, 0.65, None, None),
        ("tight", 0.15, 0.45, None, None),
        ("stricter", 0.25, 0.30, None, None),
        ("ultra_tight", 0.35, 0.20, None, None),
        ("risk_off", None, None, 0.0, None),
        ("risk_mid", None, None, 0.75, None),
        ("risk_high", None, None, 1.5, None),
        ("margin_light", None, None, None, 0.25),
        ("margin_safety", None, None, None, 0.5),
    ]
    for label, min_precision, max_rate, risk_weight, min_margin in base_variants:
        output = output_dir / f"official_{label}.json"
        variant = _official_preflight_command(
            command,
            output=output,
            waymo_src=waymo_src,
            frames=frames,
            folds=folds,
            progress_every_fold=progress_every_fold,
        )
        if min_precision is not None:
            variant = _replace_command_arg(variant, "--direct-policy-min-precision", str(min_precision))
        if max_rate is not None:
            variant = _replace_command_arg(variant, "--direct-policy-max-rate", str(max_rate))
        if risk_weight is not None:
            variant = _replace_command_arg(variant, "--direct-policy-risk-weight", str(risk_weight))
        if min_margin is not None:
            variant = _replace_command_arg(variant, "--direct-policy-min-margin", str(min_margin))
        variants.append((label, variant, output))
    return variants


def _best_official_report(official_reports: list[dict[str, Any]]) -> dict[str, Any]:
    def _score(entry: dict[str, Any]) -> tuple[float, int, float]:
        report = entry["report"]
        return (
            float(report.get("combined_ranker_mean_rfs", float("-inf"))),
            int(report.get("direct_policy_override_count", 0)),
            float(report.get("direct_policy_gain_sum", float("-inf"))),
        )

    best = max(official_reports, key=_score)
    return {"label": best["label"], "output": best["output"], "report": best["report"]}


def _full_official_command(
    command: list[str],
    *,
    output: Path,
    waymo_src: Path,
    frames: int,
    folds: int,
    progress_every_fold: bool,
) -> list[str]:
    command = _official_preflight_command(
        command,
        output=output,
        waymo_src=waymo_src,
        frames=frames,
        folds=folds,
        progress_every_fold=progress_every_fold,
    )
    return command


def _replace_command_arg(command: list[str], key: str, value: str) -> list[str]:
    if key in command:
        index = command.index(key)
        if index + 1 >= len(command):
            raise ValueError(f"Missing value for {key}")
        command[index + 1] = value
        return command
    command.extend([key, value])
    return command


def _run_command(command: list[str]) -> int:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
