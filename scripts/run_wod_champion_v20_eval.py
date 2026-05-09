#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv-v20" / "bin" / "python"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_champion_v20_eval"
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

CHAMPION_GATE_CONFIG = {
    "scene_gate_ridge": "0.544749451434533",
    "scene_gate_max_rate": "0.08107063418500374",
    "source_gate_ridge": "0.1543977714129997",
    "source_gate_max_rate": "0.34700297701760474",
    "source_gate_min_precision": "0.0025615214667816244",
}

GATE_RELIABILITY_CONFIG = {
    "world_prior": "latent_predictive",
    "world_prior_mask_mode": "mixed",
    "scene_gate_margin": "2.0",
    "scene_gate_ridge": "0.7548790522503577",
    "scene_gate_max_rate": "0.09289856895135577",
    "source_gate_ridge": "0.14095835657161485",
    "source_gate_model": "boosted_stumps",
    "source_gate_stump_iterations": "160",
    "source_gate_stump_lr": "0.04",
    "source_gate_stump_thresholds": "24",
    "source_gate_max_rate": "0.16",
    "source_gate_min_precision": "0.10",
    "source_gate_conformal_alpha": "0.05",
}

CHAMPION_DIRECT_CONFIG = {
    "direct_policy_model": "random_fourier",
    "direct_policy_identity_features": "none",
    "direct_policy_router": "intent_speed",
    "direct_policy_feature_mode": "relative_contextual",
    "direct_policy_ridge": "0.011262642523293596",
    "direct_policy_min_margin": "0.011207385771324851",
    "direct_policy_max_rate": "0.34540201206743354",
    "direct_policy_min_precision": "0.46965646524171056",
    "direct_policy_risk_weight": "0.2247686778279483",
    "direct_policy_baseline_sources": "temporal,kinematic",
    "direct_policy_candidate_sources": "learned,temporal",
    "direct_policy_rff_dim": "512",
    "direct_policy_rff_scale": "7.8582433589975516",
}

JEPA_PRECEDENT_DIRECT_CONFIG = {
    **CHAMPION_DIRECT_CONFIG,
    "direct_policy_feature_mode": "relative_precedent_latent_contextual",
    "direct_policy_ridge": "0.1770166205973593",
    "direct_policy_min_margin": "0.011207385771324851",
    "direct_policy_max_rate": "0.18406284941792822",
    "direct_policy_min_precision": "0.13843032653917806",
    "direct_policy_risk_weight": "0.4990763341567168",
    "direct_policy_baseline_sources": "kinematic",
    "direct_policy_candidate_sources": "learned",
    "direct_policy_rff_dim": "256",
    "direct_policy_rff_scale": "0.549433420360358",
}

HGB_DIRECT_CONFIG = {
    "direct_policy_model": "sklearn_hist_gradient_boosting",
    "direct_policy_identity_features": "none",
    "direct_policy_router": "intent_speed",
    "direct_policy_feature_mode": "relative_world_prior_contextual",
    "direct_policy_ridge": "0.2348657517157915",
    "direct_policy_min_margin": "0.08414638396927607",
    "direct_policy_max_rate": "0.20500627855326356",
    "direct_policy_min_precision": "0.5205182844931998",
    "direct_policy_risk_weight": "0.2235531600853261",
    "direct_policy_conformal_alpha": "0.0",
    "direct_policy_baseline_sources": "temporal,kinematic",
    "direct_policy_candidate_sources": "learned,internnav",
    "direct_policy_stump_iterations": "202",
    "direct_policy_stump_lr": "0.071368706596259",
    "direct_policy_stump_thresholds": "24",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the current WOD v20 champion direct policy and ablations."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else sys.executable))
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--external-embedding-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--neural-candidate-models", default=DEFAULT_NEURAL_MODELS)
    parser.add_argument(
        "--frame-counts",
        default="120,240,479",
        help="Comma-separated max-preference-frame counts to evaluate.",
    )
    parser.add_argument("--folds", type=int, default=2)
    parser.add_argument("--rfs-backend", choices=("local", "official"), default="local")
    parser.add_argument("--waymo-src", type=Path)
    parser.add_argument(
        "--variants",
        default="hgb_direct,clean_gate_reliability,no_direct,champion,jepa_precedent",
        help="Comma-separated variants: hgb_direct,clean_gate_reliability,champion,no_direct,jepa_precedent.",
    )
    parser.add_argument("--progress-every-fold", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()

    frame_counts = _parse_frame_counts(args.frame_counts)
    variants = _parse_variants(args.variants)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for frames in frame_counts:
        for variant in variants:
            output = args.output_dir / f"{variant}_{frames}.json"
            command = _command_for_variant(args, variant=variant, frames=frames, output=output)
            if args.rerun_existing or not output.is_file():
                status = _run_command(command)
                if status != 0:
                    return status
            report = json.loads(output.read_text(encoding="utf-8"))
            rows.append(_summary_row(variant=variant, frames=frames, output=output, report=report))

    rows = _add_ablation_deltas(rows)
    summary = {
        "schema": "wod_champion_v20_eval_summary_v1",
        "frame_counts": frame_counts,
        "variants": variants,
        "rows": rows,
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _parse_frame_counts(value: str) -> list[int]:
    counts = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not counts:
        raise ValueError("--frame-counts must include at least one positive integer")
    invalid = [count for count in counts if count <= 0]
    if invalid:
        raise ValueError("--frame-counts values must be positive")
    return counts


def _parse_variants(value: str) -> list[str]:
    variants = [item.strip() for item in str(value).split(",") if item.strip()]
    allowed = {"hgb_direct", "clean_gate_reliability", "champion", "no_direct", "jepa_precedent"}
    invalid = sorted(set(variants) - allowed)
    if invalid:
        raise ValueError(f"unsupported variant(s): {', '.join(invalid)}")
    if not variants:
        raise ValueError("--variants must include at least one variant")
    return variants


def _base_command(
    args: argparse.Namespace,
    *,
    frames: int,
    output: Path,
    gate_config: dict[str, str] = CHAMPION_GATE_CONFIG,
) -> list[str]:
    command = [
        str(args.python),
        str(ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"),
        "--frame-cache",
        str(args.frame_cache),
        "--output",
        str(output),
        "--rfs-backend",
        str(args.rfs_backend),
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
        gate_config["scene_gate_ridge"],
        "--scene-gate-max-rate",
        gate_config["scene_gate_max_rate"],
        "--source-gate",
        "independent_train_margin",
        "--source-gate-sources",
        "internnav,learned",
        "--source-gate-candidate-prefixes",
        "internnav_s2_waypoint_cautious,internnav_s1_stop_progress,internnav_s2_waypoint_progress,ensemble",
        "--source-gate-router",
        "speed",
        "--source-gate-ridge",
        gate_config["source_gate_ridge"],
        "--source-gate-max-rate",
        gate_config["source_gate_max_rate"],
        "--source-gate-min-precision",
        gate_config["source_gate_min_precision"],
        "--source-gate-local-selector",
        "--neural-candidate-models",
        str(args.neural_candidate_models),
        "--max-preference-frames",
        str(frames),
        "--folds",
        str(args.folds),
    ]
    if args.rfs_backend == "official":
        if args.waymo_src is None:
            raise ValueError("--waymo-src is required for --rfs-backend official")
        command.extend(["--waymo-src", str(args.waymo_src)])
    if "scene_gate_margin" in gate_config:
        command.extend(["--scene-gate-margin", gate_config["scene_gate_margin"]])
    for key in (
        "source_gate_model",
        "source_gate_stump_iterations",
        "source_gate_stump_lr",
        "source_gate_stump_thresholds",
        "source_gate_conformal_alpha",
    ):
        if key in gate_config:
            command.extend([f"--{key.replace('_', '-')}", gate_config[key]])
    if gate_config.get("source_gate_local_confidence_features") == "true":
        command.append("--source-gate-local-confidence-features")
    if gate_config.get("world_prior") == "latent_predictive":
        command.extend(
            [
                "--world-prior",
                "latent_predictive",
                "--world-prior-mask-mode",
                gate_config.get("world_prior_mask_mode", "mixed"),
            ]
        )
    if args.progress_every_fold:
        command.append("--progress-every-fold")
    return command


def _command_for_variant(
    args: argparse.Namespace,
    *,
    variant: str,
    frames: int,
    output: Path,
) -> list[str]:
    gate_config = GATE_RELIABILITY_CONFIG if variant in {"hgb_direct", "clean_gate_reliability"} else CHAMPION_GATE_CONFIG
    command = _base_command(args, frames=frames, output=output, gate_config=gate_config)
    if variant == "clean_gate_reliability":
        return [
            *command,
            "--learned-reliability-gate-features",
            "--learned-reliability-gate-feature-mode",
            "learned_reliability_confidence_contextual",
            "--selector-postprocess",
            "off",
        ]
    if variant == "hgb_direct":
        return [
            *_append_direct_policy(command, HGB_DIRECT_CONFIG),
            "--learned-reliability-gate-features",
            "--learned-reliability-gate-feature-mode",
            "learned_reliability_confidence_contextual",
        ]
    if variant == "no_direct":
        return [*command, "--selector-postprocess", "off"]
    if variant == "champion":
        return _append_direct_policy(command, CHAMPION_DIRECT_CONFIG)
    if variant == "jepa_precedent":
        return [
            *_append_direct_policy(command, JEPA_PRECEDENT_DIRECT_CONFIG),
            "--world-prior",
            "latent_predictive",
            "--world-prior-mask-mode",
            "mixed",
        ]
    raise ValueError(f"unsupported variant: {variant}")


def _append_direct_policy(command: list[str], config: dict[str, str]) -> list[str]:
    result = [*command, "--selector-postprocess", "direct_policy"]
    for key, value in config.items():
        result.extend([f"--{key.replace('_', '-')}", value])
    return result


def _run_command(command: list[str]) -> int:
    return int(subprocess.run(command, cwd=ROOT, check=False).returncode)


def _summary_row(*, variant: str, frames: int, output: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant": variant,
        "frames": int(frames),
        "output": str(output),
        "combined_ranker_mean_rfs": report.get("combined_ranker_mean_rfs"),
        "combined_ranker_regret_to_oracle": report.get("combined_ranker_regret_to_oracle"),
        "combined_oracle_mean_rfs": report.get("combined_oracle_mean_rfs"),
        "direct_policy_override_count": report.get("direct_policy_override_count"),
        "direct_policy_true_positive_count": report.get("direct_policy_true_positive_count"),
        "direct_policy_false_positive_count": report.get("direct_policy_false_positive_count"),
        "direct_policy_precision": report.get("direct_policy_precision"),
        "direct_policy_gain_sum": report.get("direct_policy_gain_sum"),
        "direct_policy_false_positive_loss_sum": report.get("direct_policy_false_positive_loss_sum"),
        "direct_policy_conformal_alpha": report.get("direct_policy_conformal_alpha"),
        "direct_policy_model": report.get("direct_policy_model"),
        "direct_policy_feature_mode": report.get("direct_policy_feature_mode"),
        "direct_policy_router": report.get("direct_policy_router"),
        "learned_reliability_gate_features": report.get("learned_reliability_gate_features"),
        "learned_reliability_resolved_gate_feature_mode": report.get(
            "learned_reliability_resolved_gate_feature_mode"
        ),
        "world_prior": report.get("world_prior"),
        "world_prior_mask_mode": report.get("world_prior_mask_mode"),
        "source_gate_conformal_alpha": report.get("source_gate_conformal_alpha"),
        "source_gate_local_confidence_features": report.get("source_gate_local_confidence_features"),
        "source_gate_override_count": report.get("source_gate_override_count"),
        "source_gate_true_positive_count": report.get("source_gate_true_positive_count"),
        "source_gate_false_positive_count": report.get("source_gate_false_positive_count"),
        "source_gate_precision": report.get("source_gate_precision"),
        "source_gate_gain_sum": report.get("source_gate_gain_sum"),
        "source_gate_false_positive_loss_sum": report.get("source_gate_false_positive_loss_sum"),
        "scene_gate_override_count": report.get("scene_gate_override_count"),
        "scene_gate_true_positive_count": report.get("scene_gate_true_positive_count"),
        "scene_gate_false_positive_count": report.get("scene_gate_false_positive_count"),
        "scene_gate_precision": report.get("scene_gate_precision"),
        "scene_gate_gain_sum": report.get("scene_gate_gain_sum"),
        "scene_gate_false_positive_loss_sum": report.get("scene_gate_false_positive_loss_sum"),
    }


def _add_ablation_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline_by_frames = {
        int(row["frames"]): row
        for row in rows
        if row["variant"] == "no_direct" and row.get("combined_ranker_mean_rfs") is not None
    }
    updated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        baseline = baseline_by_frames.get(int(row["frames"]))
        if baseline is not None and row.get("combined_ranker_mean_rfs") is not None:
            item["delta_vs_no_direct_rfs"] = float(row["combined_ranker_mean_rfs"]) - float(
                baseline["combined_ranker_mean_rfs"]
            )
        updated.append(item)
    return updated


if __name__ == "__main__":
    raise SystemExit(main())
