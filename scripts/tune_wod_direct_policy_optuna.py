#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv-v20" / "bin" / "python"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_optuna_direct_policy"
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
DEFAULT_DIRECT_POLICY_FEATURE_MODES = ",".join(
    (
        "selector",
        "contextual",
        "relative_contextual",
        "family_reliability_contextual",
        "relative_confidence_contextual",
        "learned_reliability_contextual",
        "learned_reliability_confidence_contextual",
    )
)
WORLD_PRIOR_DIRECT_POLICY_FEATURE_MODES = {
    "world_prior_contextual",
    "relative_world_prior_contextual",
    "relative_retrieval_latent_contextual",
    "relative_precedent_latent_contextual",
}
DIRECT_POLICY_FEATURE_MODE_CHOICES = {
    "selector",
    "linear",
    "contextual",
    "intent_contextual",
    "world_contextual",
    "world_prior_contextual",
    "geometry_contextual",
    "relative_contextual",
    "relative_world_prior_contextual",
    "relative_confidence_contextual",
    "relative_retrieval_latent_contextual",
    "relative_precedent_latent_contextual",
    "family_reliability_contextual",
    "external_contextual",
    "contextual_external",
    "learned_reliability_signal",
    "learned_reliability_signal_external",
    "learned_reliability_contextual",
    "learned_reliability_confidence_contextual",
    "learned_reliability_external",
    "camera_contextual",
    "image_contextual",
}
SOURCE_SET_CHOICES = {
    "anchor",
    "internnav",
    "internvla",
    "kinematic",
    "learned",
    "memory",
    "scene",
    "system2",
    "temporal",
    "world",
}
CHAMPION_GATE_CONFIG = {
    "scene_gate_ridge": "0.544749451434533",
    "scene_gate_max_rate": "0.08107063418500374",
    "source_gate_ridge": "0.1543977714129997",
    "source_gate_max_rate": "0.34700297701760474",
    "source_gate_min_precision": "0.0025615214667816244",
}
RELIABILITY_GATE_CONFIG = {
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
    "source_gate_candidate_prefixes": (
        "internnav_s2_waypoint_cautious,"
        "internnav_s1_stop_progress,"
        "internnav_s2_waypoint_progress,"
        "ensemble"
    ),
}
LEARNED_RELIABILITY_GATE_FEATURE_MODES = {
    "contextual_external",
    "relative_contextual",
    "relative_confidence_contextual",
    "learned_reliability_signal",
    "learned_reliability_signal_external",
    "learned_reliability_contextual",
    "learned_reliability_confidence_contextual",
    "learned_reliability_external",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune WOD direct-policy and gate parameters with an Optuna study."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--study-name", default="wod_direct_policy")
    parser.add_argument("--storage", default="", help="Optuna storage URL. Defaults to sqlite in --output-dir.")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=None)
    parser.add_argument("--trial-timeout", type=float, default=None)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else sys.executable))
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--external-embedding-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--neural-candidate-models", default=DEFAULT_NEURAL_MODELS)
    parser.add_argument("--rfs-backend", choices=("local", "official"), default="local")
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--progress-every-fold", action="store_true")
    parser.add_argument("--identity-mode", choices=("full", "none", "both"), default="both")
    parser.add_argument(
        "--model-choices",
        default="linear,random_fourier,boosted_stumps,logistic,sklearn_hist_gradient_boosting",
        help="Comma-separated direct policy model families to sample.",
    )
    parser.add_argument(
        "--direct-policy-ridge-range",
        default="0.01,30.0",
        help="Low,high log-uniform range for --direct-policy-ridge.",
    )
    parser.add_argument(
        "--direct-policy-max-rate-range",
        default="0.02,0.35",
        help="Low,high uniform range for --direct-policy-max-rate.",
    )
    parser.add_argument(
        "--direct-policy-min-margin-range",
        default="0.0,0.75",
        help="Low,high uniform range for --direct-policy-min-margin.",
    )
    parser.add_argument(
        "--direct-policy-min-precision-range",
        default="0.0,0.65",
        help="Low,high uniform range for --direct-policy-min-precision.",
    )
    parser.add_argument(
        "--direct-policy-risk-weight-range",
        default="0.0,1.5",
        help="Low,high uniform range for --direct-policy-risk-weight.",
    )
    parser.add_argument(
        "--direct-policy-stump-iterations-range",
        default="80,800",
        help="Low,high log-uniform integer range for stump/HGB direct policies.",
    )
    parser.add_argument(
        "--direct-policy-stump-lr-range",
        default="0.01,0.2",
        help="Low,high log-uniform range for stump/HGB/logistic direct policy learning rate.",
    )
    parser.add_argument(
        "--direct-policy-stump-thresholds-choices",
        default="8,16,24,32",
        help="Comma-separated threshold-count choices for stump/HGB direct policies.",
    )
    parser.add_argument(
        "--direct-policy-conformal-alpha-choices",
        default="0.0,0.03,0.05,0.08",
        help=(
            "Comma-separated choices for --direct-policy-conformal-alpha. "
            "Values must be in [0, 0.5)."
        ),
    )
    parser.add_argument(
        "--direct-policy-min-route-observations-choices",
        default="0,6,12,24",
        help=(
            "Comma-separated integer choices for --direct-policy-min-route-observations. "
            "Nonzero values suppress sparse routed thresholds."
        ),
    )
    parser.add_argument(
        "--direct-policy-min-route-positives-choices",
        default="0,1,2,3",
        help=(
            "Comma-separated integer choices for --direct-policy-min-route-positives. "
            "Nonzero values require route-local evidence before direct-policy overrides."
        ),
    )
    parser.add_argument(
        "--direct-policy-feature-modes",
        default=DEFAULT_DIRECT_POLICY_FEATURE_MODES,
        help="Comma-separated --direct-policy-feature-mode values to sample.",
    )
    parser.add_argument(
        "--include-world-prior-direct-policy-features",
        action="store_true",
        help=(
            "Allow JEPA-style world-prior feature modes and enable "
            "--world-prior latent_predictive when one is sampled."
        ),
    )
    parser.add_argument(
        "--world-prior-mask-modes",
        default="tail,mixed",
        help="Comma-separated --world-prior-mask-mode values to sample for world-prior feature modes.",
    )
    parser.add_argument(
        "--direct-policy-candidate-source-sets",
        default="all",
        help=(
            "Semicolon-separated candidate source allowlists to sample. Each option is a comma-separated "
            "source list, or all for no allowlist."
        ),
    )
    parser.add_argument(
        "--direct-policy-baseline-source-sets",
        default="all",
        help=(
            "Semicolon-separated selected-baseline source allowlists to sample. Each option is a comma-separated "
            "source list, or all for no allowlist."
        ),
    )
    parser.add_argument(
        "--gate-recipe",
        choices=("search", "champion", "reliability"),
        default="search",
        help=(
            "Base scene/source gate recipe. search samples legacy gate parameters; champion freezes "
            "the pre-v20 champion gate; reliability freezes the JEPA-style v20 reliability gate."
        ),
    )
    parser.add_argument(
        "--learned-reliability-gate-features",
        action="store_true",
        help=(
            "Append learned visual/state reliability features to scene/source gates. "
            "The reliability gate recipe enables this automatically."
        ),
    )
    parser.add_argument(
        "--learned-reliability-gate-feature-mode",
        choices=tuple(sorted(LEARNED_RELIABILITY_GATE_FEATURE_MODES)),
        default="learned_reliability_confidence_contextual",
    )
    parser.add_argument("--include-torch", action="store_true", help="Also sample torch_mlp direct policies.")
    parser.add_argument(
        "--false-positive-penalty",
        type=float,
        default=0.0,
        help="Subtract this times direct-policy false-positive loss per frame from the reported RFS.",
    )
    parser.add_argument(
        "--false-positive-count-penalty",
        type=float,
        default=0.0,
        help="Subtract this times direct-policy false-positive override count per frame from the objective.",
    )
    parser.add_argument(
        "--gain-reward",
        type=float,
        default=0.0,
        help="Add this times direct-policy realized gain per frame to the reported RFS objective.",
    )
    parser.add_argument(
        "--min-direct-policy-overrides",
        type=int,
        default=0,
        help="Minimum realized positive direct-policy overrides before applying --underactive-penalty.",
    )
    parser.add_argument(
        "--min-direct-policy-selected-rate",
        type=float,
        default=0.0,
        help="Minimum realized direct-policy selected rate before applying --underactive-penalty.",
    )
    parser.add_argument(
        "--underactive-penalty",
        type=float,
        default=0.0,
        help="Subtract this per missing required direct-policy override fraction.",
    )
    parser.add_argument(
        "--fold-rfs-spread-penalty",
        type=float,
        default=0.0,
        help="Subtract this times the max-min fold RFS spread to prefer split-stable policies.",
    )
    parser.add_argument(
        "--max-fold-false-positive-penalty",
        type=float,
        default=0.0,
        help=(
            "Subtract this times the worst fold-level direct-policy false-positive loss per frame. "
            "This catches policies that look good in aggregate but fail one split."
        ),
    )
    parser.add_argument(
        "--min-fold-direct-policy-precision",
        type=float,
        default=0.0,
        help=(
            "Minimum fold-level direct-policy precision to reward when a fold has enough overrides. "
            "Used with --fold-precision-penalty."
        ),
    )
    parser.add_argument(
        "--fold-precision-penalty",
        type=float,
        default=0.0,
        help="Subtract this times average active-fold precision shortfall.",
    )
    parser.add_argument(
        "--fold-precision-min-overrides",
        type=int,
        default=1,
        help="Ignore fold precision penalties for folds with fewer direct-policy overrides than this.",
    )
    parser.add_argument("--rerun-existing", action="store_true")
    args = parser.parse_args()
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.false_positive_penalty < 0.0:
        parser.error("--false-positive-penalty must be non-negative")
    if args.false_positive_count_penalty < 0.0:
        parser.error("--false-positive-count-penalty must be non-negative")
    if args.gain_reward < 0.0:
        parser.error("--gain-reward must be non-negative")
    if args.min_direct_policy_overrides < 0:
        parser.error("--min-direct-policy-overrides must be non-negative")
    if args.min_direct_policy_selected_rate < 0.0 or args.min_direct_policy_selected_rate > 1.0:
        parser.error("--min-direct-policy-selected-rate must be in [0, 1]")
    if args.underactive_penalty < 0.0:
        parser.error("--underactive-penalty must be non-negative")
    if args.fold_rfs_spread_penalty < 0.0:
        parser.error("--fold-rfs-spread-penalty must be non-negative")
    if args.max_fold_false_positive_penalty < 0.0:
        parser.error("--max-fold-false-positive-penalty must be non-negative")
    if args.min_fold_direct_policy_precision < 0.0 or args.min_fold_direct_policy_precision > 1.0:
        parser.error("--min-fold-direct-policy-precision must be in [0, 1]")
    if args.fold_precision_penalty < 0.0:
        parser.error("--fold-precision-penalty must be non-negative")
    if args.fold_precision_min_overrides < 0:
        parser.error("--fold-precision-min-overrides must be non-negative")
    try:
        _int_choices(
            args.direct_policy_min_route_observations_choices,
            name="--direct-policy-min-route-observations-choices",
        )
        _int_choices(
            args.direct_policy_min_route_positives_choices,
            name="--direct-policy-min-route-positives-choices",
        )
        _float_choices(
            args.direct_policy_conformal_alpha_choices,
            name="--direct-policy-conformal-alpha-choices",
            non_negative=True,
            less_than=0.5,
        )
        _int_range(args.direct_policy_stump_iterations_range, name="--direct-policy-stump-iterations-range")
        _float_range(args.direct_policy_stump_lr_range, name="--direct-policy-stump-lr-range", positive=True)
        _int_choices(args.direct_policy_stump_thresholds_choices, name="--direct-policy-stump-thresholds-choices")
    except ValueError as exc:
        parser.error(str(exc))

    optuna = _load_optuna()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{args.output_dir / 'study.db'}"
    sampler = optuna.samplers.TPESampler(seed=2029)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(lambda trial: _objective(trial, args), n_trials=args.trials, timeout=args.timeout)
    manifest = _study_manifest(study, args, storage)
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def _load_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is not installed. Install it with `python3 -m pip install optuna` "
            "or `python3 -m pip install '.[tuning]'`."
        ) from exc
    return optuna


def _objective(trial, args: argparse.Namespace) -> float:
    config = _suggest_config(trial, args)
    output = args.output_dir / f"trial_{int(trial.number):04d}.json"
    command = _command_for_trial(config, output, args)
    fingerprint = _trial_fingerprint(config, command)
    metadata_path = _trial_metadata_path(output)
    trial.set_user_attr("output", str(output))
    trial.set_user_attr("command", command)
    trial.set_user_attr("fingerprint", fingerprint)
    if args.rerun_existing or not _trial_artifact_matches(output, metadata_path, fingerprint):
        try:
            result = _run_command(command, output.with_suffix(".stdout.log"), output.with_suffix(".stderr.log"), args)
        except subprocess.TimeoutExpired:
            trial.set_user_attr("timed_out", True)
            trial.set_user_attr("returncode", 124)
            return -1.0e9
        trial.set_user_attr("returncode", int(result.returncode))
        if result.returncode != 0:
            return -1.0e9
        metadata_path.write_text(
            json.dumps(
                {
                    "schema": "wod_direct_policy_trial_metadata_v1",
                    "fingerprint": fingerprint,
                    "returncode": int(result.returncode),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    report = _load_report(output)
    score = _objective_score(
        report,
        false_positive_penalty=getattr(args, "false_positive_penalty", 0.0),
        false_positive_count_penalty=getattr(args, "false_positive_count_penalty", 0.0),
        gain_reward=getattr(args, "gain_reward", 0.0),
        min_direct_policy_overrides=getattr(args, "min_direct_policy_overrides", 0),
        min_direct_policy_selected_rate=getattr(args, "min_direct_policy_selected_rate", 0.0),
        underactive_penalty=getattr(args, "underactive_penalty", 0.0),
        fold_rfs_spread_penalty=getattr(args, "fold_rfs_spread_penalty", 0.0),
        max_fold_false_positive_penalty=getattr(args, "max_fold_false_positive_penalty", 0.0),
        min_fold_direct_policy_precision=getattr(args, "min_fold_direct_policy_precision", 0.0),
        fold_precision_penalty=getattr(args, "fold_precision_penalty", 0.0),
        fold_precision_min_overrides=getattr(args, "fold_precision_min_overrides", 1),
    )
    metrics = _metrics(report, score)
    for key, value in metrics.items():
        trial.set_user_attr(key, value)
    return score


def _suggest_config(trial, args: argparse.Namespace) -> dict[str, Any]:
    model_choices = _model_choices(args)
    identity_choices = _identity_choices(str(args.identity_mode))
    feature_mode_choices = _direct_policy_feature_mode_choices(args)
    candidate_source_choices = _source_set_choices(
        getattr(args, "direct_policy_candidate_source_sets", "all"),
        name="--direct-policy-candidate-source-sets",
    )
    baseline_source_choices = _source_set_choices(
        getattr(args, "direct_policy_baseline_source_sets", "all"),
        name="--direct-policy-baseline-source-sets",
    )
    model = trial.suggest_categorical("direct_policy_model", model_choices)
    direct_policy_feature_mode = trial.suggest_categorical(
        "direct_policy_feature_mode",
        feature_mode_choices,
    )
    direct_policy_ridge_low, direct_policy_ridge_high = _float_range(
        getattr(args, "direct_policy_ridge_range", "0.01,30.0"),
        name="--direct-policy-ridge-range",
        positive=True,
    )
    direct_policy_max_rate_low, direct_policy_max_rate_high = _float_range(
        getattr(args, "direct_policy_max_rate_range", "0.02,0.35"),
        name="--direct-policy-max-rate-range",
        bounded_unit=True,
    )
    direct_policy_min_margin_low, direct_policy_min_margin_high = _float_range(
        getattr(args, "direct_policy_min_margin_range", "0.0,0.75"),
        name="--direct-policy-min-margin-range",
        non_negative=True,
    )
    direct_policy_min_precision_low, direct_policy_min_precision_high = _float_range(
        getattr(args, "direct_policy_min_precision_range", "0.0,0.65"),
        name="--direct-policy-min-precision-range",
        bounded_unit=True,
    )
    direct_policy_risk_weight_low, direct_policy_risk_weight_high = _float_range(
        getattr(args, "direct_policy_risk_weight_range", "0.0,1.5"),
        name="--direct-policy-risk-weight-range",
        non_negative=True,
    )
    min_route_observation_choices = _int_choices(
        getattr(args, "direct_policy_min_route_observations_choices", "0,6,12,24"),
        name="--direct-policy-min-route-observations-choices",
    )
    min_route_positive_choices = _int_choices(
        getattr(args, "direct_policy_min_route_positives_choices", "0,1,2,3"),
        name="--direct-policy-min-route-positives-choices",
    )
    conformal_alpha_choices = _float_choices(
        getattr(args, "direct_policy_conformal_alpha_choices", "0.0,0.03,0.05,0.08"),
        name="--direct-policy-conformal-alpha-choices",
        non_negative=True,
        less_than=0.5,
    )
    gate_recipe = str(getattr(args, "gate_recipe", "search"))
    config: dict[str, Any] = {
        "gate_recipe": gate_recipe,
        "direct_policy_model": model,
        "direct_policy_identity_features": trial.suggest_categorical(
            "direct_policy_identity_features", identity_choices
        ),
        "direct_policy_router": trial.suggest_categorical(
            "direct_policy_router", ["speed", "speed_fine", "intent_speed"]
        ),
        "direct_policy_feature_mode": direct_policy_feature_mode,
        "direct_policy_ridge": trial.suggest_float(
            "direct_policy_ridge",
            direct_policy_ridge_low,
            direct_policy_ridge_high,
            log=True,
        ),
        "direct_policy_max_rate": trial.suggest_float(
            "direct_policy_max_rate",
            direct_policy_max_rate_low,
            direct_policy_max_rate_high,
        ),
        "direct_policy_min_margin": trial.suggest_float(
            "direct_policy_min_margin",
            direct_policy_min_margin_low,
            direct_policy_min_margin_high,
        ),
        "direct_policy_min_precision": trial.suggest_float(
            "direct_policy_min_precision",
            direct_policy_min_precision_low,
            direct_policy_min_precision_high,
        ),
        "direct_policy_min_route_observations": trial.suggest_categorical(
            "direct_policy_min_route_observations",
            min_route_observation_choices,
        ),
        "direct_policy_min_route_positives": trial.suggest_categorical(
            "direct_policy_min_route_positives",
            min_route_positive_choices,
        ),
        "direct_policy_risk_weight": trial.suggest_float(
            "direct_policy_risk_weight",
            direct_policy_risk_weight_low,
            direct_policy_risk_weight_high,
        ),
        "direct_policy_conformal_alpha": trial.suggest_categorical(
            "direct_policy_conformal_alpha",
            conformal_alpha_choices,
        ),
    }
    if gate_recipe == "search":
        config.update(
            {
                "scene_gate_ridge": trial.suggest_float("scene_gate_ridge", 0.5, 6.0, log=True),
                "scene_gate_max_rate": trial.suggest_float("scene_gate_max_rate", 0.08, 0.22),
                "source_gate_ridge": trial.suggest_float("source_gate_ridge", 0.03, 0.5, log=True),
                "source_gate_max_rate": trial.suggest_float("source_gate_max_rate", 0.12, 0.35),
                "source_gate_min_precision": trial.suggest_float("source_gate_min_precision", 0.0, 0.45),
            }
        )
    else:
        config.update(_gate_recipe_config(gate_recipe))
    candidate_sources = trial.suggest_categorical(
        "direct_policy_candidate_sources",
        candidate_source_choices,
    )
    baseline_sources = trial.suggest_categorical(
        "direct_policy_baseline_sources",
        baseline_source_choices,
    )
    if candidate_sources:
        config["direct_policy_candidate_sources"] = candidate_sources
    if baseline_sources:
        config["direct_policy_baseline_sources"] = baseline_sources
    if model == "random_fourier":
        config.update(
            {
                "direct_policy_rff_dim": trial.suggest_categorical("direct_policy_rff_dim", [64, 128, 256, 512]),
                "direct_policy_rff_scale": trial.suggest_float("direct_policy_rff_scale", 0.5, 8.0, log=True),
            }
        )
    if model in {"boosted_stumps", "logistic", "sklearn_hist_gradient_boosting"}:
        stump_iterations_low, stump_iterations_high = _int_range(
            getattr(args, "direct_policy_stump_iterations_range", "80,800"),
            name="--direct-policy-stump-iterations-range",
        )
        stump_lr_low, stump_lr_high = _float_range(
            getattr(args, "direct_policy_stump_lr_range", "0.01,0.2"),
            name="--direct-policy-stump-lr-range",
            positive=True,
        )
        config.update(
            {
                "direct_policy_stump_iterations": trial.suggest_int(
                    "direct_policy_stump_iterations",
                    stump_iterations_low,
                    stump_iterations_high,
                    log=True,
                ),
                "direct_policy_stump_lr": trial.suggest_float(
                    "direct_policy_stump_lr",
                    stump_lr_low,
                    stump_lr_high,
                    log=True,
                ),
            }
        )
    if model in {"boosted_stumps", "sklearn_hist_gradient_boosting"}:
        config["direct_policy_stump_thresholds"] = trial.suggest_categorical(
            "direct_policy_stump_thresholds",
            _int_choices(
                getattr(args, "direct_policy_stump_thresholds_choices", "8,16,24,32"),
                name="--direct-policy-stump-thresholds-choices",
            ),
        )
    if model == "torch_mlp":
        config.update(
            {
                "direct_policy_torch_epochs": trial.suggest_categorical(
                    "direct_policy_torch_epochs", [10, 20, 40]
                ),
                "direct_policy_torch_lr": trial.suggest_float("direct_policy_torch_lr", 1.0e-4, 1.0e-2, log=True),
                "direct_policy_torch_hidden": trial.suggest_categorical(
                    "direct_policy_torch_hidden", [32, 64, 128]
                ),
                "direct_policy_torch_batch_size": trial.suggest_categorical(
                    "direct_policy_torch_batch_size", [256, 512, 1024]
                ),
                "direct_policy_device": "auto",
            }
        )
    if direct_policy_feature_mode in WORLD_PRIOR_DIRECT_POLICY_FEATURE_MODES:
        config["world_prior_mask_mode"] = trial.suggest_categorical(
            "world_prior_mask_mode",
            _world_prior_mask_mode_choices(args),
        )
    return config


def _model_choices(args: argparse.Namespace) -> list[str]:
    choices = [item.strip() for item in str(args.model_choices).split(",") if item.strip()]
    if bool(getattr(args, "include_torch", False)) and "torch_mlp" not in choices:
        choices.append("torch_mlp")
    allowed = {
        "linear",
        "random_fourier",
        "boosted_stumps",
        "logistic",
        "sklearn_hist_gradient_boosting",
        "torch_mlp",
    }
    invalid = sorted(set(choices) - allowed)
    if invalid:
        raise ValueError(f"unsupported direct policy model choice(s): {', '.join(invalid)}")
    if not choices:
        raise ValueError("at least one direct policy model choice is required")
    return choices


def _gate_recipe_config(recipe: str) -> dict[str, Any]:
    if recipe == "champion":
        return dict(CHAMPION_GATE_CONFIG)
    if recipe == "reliability":
        return dict(RELIABILITY_GATE_CONFIG)
    if recipe == "search":
        return {}
    raise ValueError(f"unsupported gate recipe: {recipe}")


def _identity_choices(identity_mode: str) -> list[str]:
    if identity_mode == "both":
        return ["full", "none"]
    if identity_mode in {"full", "none"}:
        return [identity_mode]
    raise ValueError(f"unsupported identity mode: {identity_mode}")


def _direct_policy_feature_mode_choices(args: argparse.Namespace) -> list[str]:
    raw_value = str(getattr(args, "direct_policy_feature_modes", DEFAULT_DIRECT_POLICY_FEATURE_MODES))
    choices = [
        item.strip()
        for item in raw_value.split(",")
        if item.strip()
    ]
    if bool(getattr(args, "include_world_prior_direct_policy_features", False)) and raw_value == DEFAULT_DIRECT_POLICY_FEATURE_MODES:
        choices.extend(
            choice for choice in sorted(WORLD_PRIOR_DIRECT_POLICY_FEATURE_MODES) if choice not in choices
        )
    invalid = sorted(set(choices) - DIRECT_POLICY_FEATURE_MODE_CHOICES)
    if invalid:
        raise ValueError(f"unsupported direct policy feature mode choice(s): {', '.join(invalid)}")
    if not bool(getattr(args, "include_world_prior_direct_policy_features", False)):
        world_prior_choices = sorted(set(choices) & WORLD_PRIOR_DIRECT_POLICY_FEATURE_MODES)
        if world_prior_choices:
            raise ValueError(
                "world-prior direct policy feature modes require "
                "--include-world-prior-direct-policy-features: "
                + ", ".join(world_prior_choices)
            )
    if not choices:
        raise ValueError("at least one direct policy feature mode is required")
    return choices


def _world_prior_mask_mode_choices(args: argparse.Namespace) -> list[str]:
    choices = [
        item.strip() for item in str(getattr(args, "world_prior_mask_modes", "tail,mixed")).split(",") if item.strip()
    ]
    allowed = {"off", "tail", "random", "mixed"}
    invalid = sorted(set(choices) - allowed)
    if invalid:
        raise ValueError(f"unsupported world prior mask mode choice(s): {', '.join(invalid)}")
    if not choices:
        raise ValueError("at least one world prior mask mode is required")
    return choices


def _source_set_choices(value: str, *, name: str) -> list[str]:
    raw_choices = [item.strip() for item in str(value).split(";")]
    choices: list[str] = []
    for raw_choice in raw_choices:
        if raw_choice in {"", "all", "*"}:
            normalized = ""
        else:
            sources = [source.strip() for source in raw_choice.split(",") if source.strip()]
            invalid = sorted(set(sources) - SOURCE_SET_CHOICES)
            if invalid:
                raise ValueError(f"unsupported source in {name}: {', '.join(invalid)}")
            normalized = ",".join(dict.fromkeys(sources))
        if normalized not in choices:
            choices.append(normalized)
    if not choices:
        raise ValueError(f"{name} must include at least one source set")
    return choices


def _int_choices(value: str, *, name: str) -> list[int]:
    choices: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed = int(item)
        if parsed < 0:
            raise ValueError(f"{name} values must be non-negative")
        if parsed not in choices:
            choices.append(parsed)
    if not choices:
        raise ValueError(f"{name} must include at least one integer")
    return choices


def _int_range(value: str, *, name: str) -> tuple[int, int]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as low,high")
    low, high = int(parts[0]), int(parts[1])
    if low <= 0 or high <= 0:
        raise ValueError(f"{name} values must be positive")
    if low > high:
        raise ValueError(f"{name} low must be <= high")
    return low, high


def _float_choices(
    value: str,
    *,
    name: str,
    non_negative: bool = False,
    less_than: float | None = None,
) -> list[float]:
    choices: list[float] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        parsed = float(item)
        if non_negative and parsed < 0.0:
            raise ValueError(f"{name} values must be non-negative")
        if less_than is not None and parsed >= less_than:
            raise ValueError(f"{name} values must be < {less_than}")
        if parsed not in choices:
            choices.append(parsed)
    if not choices:
        raise ValueError(f"{name} must include at least one float")
    return choices


def _float_range(
    value: str,
    *,
    name: str,
    positive: bool = False,
    non_negative: bool = False,
    bounded_unit: bool = False,
) -> tuple[float, float]:
    parts = [item.strip() for item in str(value).split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"{name} must be formatted as low,high")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise ValueError(f"{name} low must be <= high")
    if positive and low <= 0.0:
        raise ValueError(f"{name} values must be positive")
    if non_negative and low < 0.0:
        raise ValueError(f"{name} values must be non-negative")
    if bounded_unit and (low < 0.0 or high > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return low, high


def _command_for_trial(config: dict[str, Any], output: Path, args: argparse.Namespace) -> list[str]:
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
        str(config["scene_gate_ridge"]),
        "--scene-gate-max-rate",
        str(config["scene_gate_max_rate"]),
        "--source-gate",
        "independent_train_margin",
        "--source-gate-sources",
        "internnav,learned",
        "--source-gate-candidate-prefixes",
        str(
            config.get(
                "source_gate_candidate_prefixes",
                "internnav_s2_waypoint_cautious,internnav_s1_stop_progress,ensemble",
            )
        ),
        "--source-gate-router",
        "speed",
        "--source-gate-ridge",
        str(config["source_gate_ridge"]),
        "--source-gate-max-rate",
        str(config["source_gate_max_rate"]),
        "--source-gate-min-precision",
        str(config["source_gate_min_precision"]),
        "--selector-postprocess",
        "direct_policy",
    ]
    if "scene_gate_margin" in config:
        command.extend(["--scene-gate-margin", str(config["scene_gate_margin"])])
    for key in (
        "source_gate_model",
        "source_gate_stump_iterations",
        "source_gate_stump_lr",
        "source_gate_stump_thresholds",
        "source_gate_conformal_alpha",
    ):
        if key in config:
            command.extend([f"--{key.replace('_', '-')}", str(config[key])])
    if bool(config.get("source_gate_local_selector", True)):
        command.append("--source-gate-local-selector")
    if bool(getattr(args, "learned_reliability_gate_features", False)) or config.get("gate_recipe") == "reliability":
        command.extend(
            [
                "--learned-reliability-gate-features",
                "--learned-reliability-gate-feature-mode",
                str(
                    getattr(
                        args,
                        "learned_reliability_gate_feature_mode",
                        "learned_reliability_confidence_contextual",
                    )
                ),
            ]
        )
    if str(args.neural_candidate_models):
        command.extend(["--neural-candidate-models", str(args.neural_candidate_models)])
    for key in (
        "direct_policy_model",
        "direct_policy_identity_features",
        "direct_policy_router",
        "direct_policy_feature_mode",
        "direct_policy_ridge",
        "direct_policy_min_margin",
        "direct_policy_max_rate",
        "direct_policy_min_precision",
        "direct_policy_min_route_observations",
        "direct_policy_min_route_positives",
        "direct_policy_risk_weight",
        "direct_policy_conformal_alpha",
        "direct_policy_baseline_sources",
        "direct_policy_candidate_sources",
        "direct_policy_rff_dim",
        "direct_policy_rff_scale",
        "direct_policy_stump_iterations",
        "direct_policy_stump_lr",
        "direct_policy_stump_thresholds",
        "direct_policy_torch_epochs",
        "direct_policy_torch_lr",
        "direct_policy_torch_hidden",
        "direct_policy_torch_batch_size",
        "direct_policy_device",
    ):
        if key in config:
            command.extend([f"--{key.replace('_', '-')}", str(config[key])])
    if config.get("world_prior") == "latent_predictive" or (
        config.get("direct_policy_feature_mode") in WORLD_PRIOR_DIRECT_POLICY_FEATURE_MODES
    ):
        command.extend(
            [
                "--world-prior",
                "latent_predictive",
                "--world-prior-mask-mode",
                str(config.get("world_prior_mask_mode", "mixed")),
            ]
        )
    if args.max_preference_frames is not None:
        command.extend(["--max-preference-frames", str(args.max_preference_frames)])
    if args.folds is not None:
        command.extend(["--folds", str(args.folds)])
    if args.progress_every_fold:
        command.append("--progress-every-fold")
    return command


def _run_command(
    command: list[str],
    stdout_path: Path,
    stderr_path: Path,
    args: argparse.Namespace,
) -> subprocess.CompletedProcess:
    with (
        stdout_path.open("w", encoding="utf-8") as stdout_file,
        stderr_path.open("w", encoding="utf-8") as stderr_file,
    ):
        return subprocess.run(
            command,
            cwd=ROOT,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            timeout=args.trial_timeout,
            check=False,
        )


def _trial_metadata_path(output: Path) -> Path:
    return output.with_suffix(".meta.json")


def _trial_fingerprint(config: dict[str, Any], command: list[str]) -> str:
    payload = json.dumps({"command": command, "config": config}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trial_artifact_matches(output: Path, metadata_path: Path, fingerprint: str) -> bool:
    if not output.is_file() or not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return metadata.get("fingerprint") == fingerprint


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload


def _objective_score(
    report: dict[str, Any],
    *,
    false_positive_penalty: float,
    false_positive_count_penalty: float = 0.0,
    gain_reward: float = 0.0,
    min_direct_policy_overrides: int = 0,
    min_direct_policy_selected_rate: float = 0.0,
    underactive_penalty: float = 0.0,
    fold_rfs_spread_penalty: float = 0.0,
    max_fold_false_positive_penalty: float = 0.0,
    min_fold_direct_policy_precision: float = 0.0,
    fold_precision_penalty: float = 0.0,
    fold_precision_min_overrides: int = 1,
) -> float:
    rfs = float(report.get("combined_ranker_mean_rfs", float("-inf")))
    frames = max(1.0, float(report.get("frames", 1.0)))
    score = rfs
    if gain_reward > 0.0:
        gain_sum = float(report.get("direct_policy_gain_sum", 0.0))
        score += float(gain_reward) * (gain_sum / frames)
    loss_sum = float(report.get("direct_policy_false_positive_loss_sum", 0.0))
    if false_positive_penalty > 0.0:
        score -= float(false_positive_penalty) * (loss_sum / frames)
    if false_positive_count_penalty > 0.0:
        false_positive_count = float(report.get("direct_policy_false_positive_count", 0.0))
        score -= float(false_positive_count_penalty) * (false_positive_count / frames)
    required_overrides = max(
        int(min_direct_policy_overrides),
        int(math.ceil(float(min_direct_policy_selected_rate) * frames)),
    )
    useful_override_count = int(
        report.get(
            "direct_policy_true_positive_count",
            report.get("direct_policy_override_count", 0),
        )
    )
    if underactive_penalty > 0.0 and useful_override_count < required_overrides:
        score -= float(underactive_penalty) * float(required_overrides - useful_override_count) / frames
    if fold_rfs_spread_penalty > 0.0:
        score -= float(fold_rfs_spread_penalty) * _fold_rfs_spread(report)
    if max_fold_false_positive_penalty > 0.0:
        score -= float(max_fold_false_positive_penalty) * _max_fold_direct_policy_false_positive_loss_rate(report)
    if fold_precision_penalty > 0.0 and min_fold_direct_policy_precision > 0.0:
        shortfall = _active_fold_direct_policy_precision_shortfall(
            report,
            min_precision=float(min_fold_direct_policy_precision),
            min_overrides=int(fold_precision_min_overrides),
        )
        score -= float(fold_precision_penalty) * shortfall
    return score


def _fold_reports(report: dict[str, Any]) -> list[dict[str, Any]]:
    folds = report.get("folds_detail", [])
    if not isinstance(folds, list):
        return []
    return [fold for fold in folds if isinstance(fold, dict)]


def _fold_rfs_spread(report: dict[str, Any]) -> float:
    values = [
        float(fold["combined_ranker_mean_rfs"])
        for fold in _fold_reports(report)
        if fold.get("combined_ranker_mean_rfs") is not None
    ]
    if len(values) < 2:
        return 0.0
    return float(max(values) - min(values))


def _max_fold_direct_policy_false_positive_loss_rate(report: dict[str, Any]) -> float:
    rates: list[float] = []
    for fold in _fold_reports(report):
        frames = max(1.0, float(fold.get("frames", 1.0)))
        loss_sum = float(fold.get("direct_policy_false_positive_loss_sum", 0.0))
        rates.append(loss_sum / frames)
    return float(max(rates)) if rates else 0.0


def _active_fold_direct_policy_precision_shortfall(
    report: dict[str, Any],
    *,
    min_precision: float,
    min_overrides: int,
) -> float:
    shortfalls: list[float] = []
    for fold in _fold_reports(report):
        override_count = int(fold.get("direct_policy_override_count", 0))
        if override_count < min_overrides:
            continue
        precision = float(fold.get("direct_policy_precision", 0.0))
        shortfalls.append(max(0.0, float(min_precision) - precision))
    if not shortfalls:
        return 0.0
    return float(sum(shortfalls) / len(shortfalls))


def _metrics(report: dict[str, Any], objective: float) -> dict[str, Any]:
    keys = [
        "combined_ranker_mean_rfs",
        "direct_policy_override_count",
        "direct_policy_true_positive_count",
        "direct_policy_precision",
        "direct_policy_mean_gain",
        "direct_policy_false_positive_loss",
        "direct_policy_gain_sum",
        "direct_policy_oracle_positive_rate",
        "direct_policy_oracle_gain_sum",
        "direct_policy_oracle_mean_gain_per_frame",
        "scene_gate_precision",
        "source_gate_precision",
    ]
    return {
        "objective": float(objective),
        "fold_rfs_spread": _fold_rfs_spread(report),
        "max_fold_direct_policy_false_positive_loss_rate": _max_fold_direct_policy_false_positive_loss_rate(report),
        "active_fold_direct_policy_precision_shortfall_050": _active_fold_direct_policy_precision_shortfall(
            report,
            min_precision=0.50,
            min_overrides=1,
        ),
        **{key: report.get(key) for key in keys},
    }


def _study_manifest(study, args: argparse.Namespace, storage: str) -> dict[str, Any]:
    best = None
    try:
        best_trial = study.best_trial
    except ValueError:
        best_trial = None
    if best_trial is not None:
        best = {
            "number": int(best_trial.number),
            "value": float(best_trial.value),
            "params": dict(best_trial.params),
            "user_attrs": dict(best_trial.user_attrs),
        }
    return {
        "schema": "wod_direct_policy_optuna_manifest_v1",
        "study_name": args.study_name,
        "storage": storage,
        "direction": "maximize",
        "trials_requested": int(args.trials),
        "rfs_backend": args.rfs_backend,
        "direct_policy_feature_modes": getattr(
            args,
            "direct_policy_feature_modes",
            DEFAULT_DIRECT_POLICY_FEATURE_MODES,
        ),
        "include_world_prior_direct_policy_features": bool(
            getattr(args, "include_world_prior_direct_policy_features", False)
        ),
        "world_prior_mask_modes": getattr(args, "world_prior_mask_modes", "tail,mixed"),
        "gate_recipe": getattr(args, "gate_recipe", "search"),
        "direct_policy_conformal_alpha_choices": getattr(
            args,
            "direct_policy_conformal_alpha_choices",
            "0.0,0.03,0.05,0.08",
        ),
        "direct_policy_stump_iterations_range": getattr(args, "direct_policy_stump_iterations_range", "80,800"),
        "direct_policy_stump_lr_range": getattr(args, "direct_policy_stump_lr_range", "0.01,0.2"),
        "direct_policy_stump_thresholds_choices": getattr(args, "direct_policy_stump_thresholds_choices", "8,16,24,32"),
        "learned_reliability_gate_features": bool(getattr(args, "learned_reliability_gate_features", False)),
        "learned_reliability_gate_feature_mode": getattr(
            args,
            "learned_reliability_gate_feature_mode",
            "learned_reliability_confidence_contextual",
        ),
        "direct_policy_candidate_source_sets": getattr(args, "direct_policy_candidate_source_sets", "all"),
        "direct_policy_baseline_source_sets": getattr(args, "direct_policy_baseline_source_sets", "all"),
        "direct_policy_min_route_observations_choices": getattr(
            args,
            "direct_policy_min_route_observations_choices",
            "0,6,12,24",
        ),
        "direct_policy_min_route_positives_choices": getattr(
            args,
            "direct_policy_min_route_positives_choices",
            "0,1,2,3",
        ),
        "false_positive_penalty": float(args.false_positive_penalty),
        "false_positive_count_penalty": float(getattr(args, "false_positive_count_penalty", 0.0)),
        "gain_reward": float(args.gain_reward),
        "min_direct_policy_overrides": int(args.min_direct_policy_overrides),
        "min_direct_policy_selected_rate": float(args.min_direct_policy_selected_rate),
        "underactive_penalty": float(args.underactive_penalty),
        "fold_rfs_spread_penalty": float(getattr(args, "fold_rfs_spread_penalty", 0.0)),
        "max_fold_false_positive_penalty": float(getattr(args, "max_fold_false_positive_penalty", 0.0)),
        "min_fold_direct_policy_precision": float(getattr(args, "min_fold_direct_policy_precision", 0.0)),
        "fold_precision_penalty": float(getattr(args, "fold_precision_penalty", 0.0)),
        "fold_precision_min_overrides": int(getattr(args, "fold_precision_min_overrides", 1)),
        "best": best,
        "trial_count": len(study.trials),
    }


if __name__ == "__main__":
    raise SystemExit(main())
