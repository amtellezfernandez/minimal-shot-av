#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts" / "wod_breakthrough"
DEFAULT_FRAME_CACHE = ROOT / "artifacts" / "wod_preference_frames_val479.json"
DEFAULT_EXTERNAL_CACHE = ROOT / "artifacts" / "cosmos_predict25_wan21_tokenizer_val479.json"
DEFAULT_PROMOTION_BASELINE = ROOT / "artifacts" / "wod_fastkin_gate_ridge175_scene020_cv_official.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the fixed overnight-local WOD breakthrough experiment matrix.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--external-embedding-cache", type=Path, default=DEFAULT_EXTERNAL_CACHE)
    parser.add_argument("--neural-candidate-model", type=Path)
    parser.add_argument("--transformer-candidate-model", type=Path)
    parser.add_argument("--promotion-baseline", type=Path, default=DEFAULT_PROMOTION_BASELINE)
    parser.add_argument("--min-rfs-gain", type=float, default=0.0)
    parser.add_argument("--smoke", action="store_true", help="Run a tiny local-RFS smoke matrix.")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Run the full candidate matrix on a local-RFS pilot slice.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args()
    if args.smoke and args.pilot:
        parser.error("--smoke and --pilot are mutually exclusive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _clear_derived_reports(args.output_dir)
    runs = _experiment_matrix(args)
    manifest_rows = []
    for run in runs:
        output = args.output_dir / f"{run['name']}.json"
        command = _command_for_run(run, output, args)
        status = _run_command(command, continue_on_error=args.continue_on_error)
        row = {"name": run["name"], "output": str(output), "command": command, "returncode": status}
        if output.is_file():
            row.update(_metrics(output))
        manifest_rows.append(row)
        if status != 0 and not args.continue_on_error:
            break
    mode = _mode(args)
    promotion_baseline = args.promotion_baseline if mode == "official" else None
    _add_baseline_comparisons(manifest_rows, baseline_report=promotion_baseline)
    best = _best_successful_run(manifest_rows)
    promoted = _validation_candidate_run(manifest_rows, mode=mode, min_rfs_gain=args.min_rfs_gain)
    pilot_promoted = _promoted_run(manifest_rows) if mode != "official" else None
    manifest = {
        "schema": "wod_breakthrough_experiment_manifest_v1",
        "smoke": bool(args.smoke),
        "pilot": bool(args.pilot),
        "mode": mode,
        "promotion_baseline": str(promotion_baseline) if promotion_baseline is not None else "baseline_recheck",
        "runs": manifest_rows,
        "best_observed": best,
        "validation_candidate": promoted,
        "pilot_promoted": pilot_promoted,
        "promoted": promoted,
        "best": best,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if best:
        best_path = args.output_dir / "best_report.json"
        observed_path = args.output_dir / "best_observed_report.json"
        best_payload = {
            "schema": "wod_breakthrough_best_observed_report_v1",
            "report_role": "best_observed_not_necessarily_promoted",
            "promoted": bool(promoted and promoted.get("output") == best.get("output")),
            "pilot_promoted": bool(pilot_promoted and pilot_promoted.get("output") == best.get("output")),
            "best_report_path": best["output"],
            **best,
        }
        observed_path.write_text(
            json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        best_path.write_text(
            json.dumps(best_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if promoted:
        promoted_path = args.output_dir / "promoted_report.json"
        promoted_payload = {
            "schema": "wod_validation_breakthrough_candidate_report_v1",
            "report_role": "validation_cv_candidate",
            "promoted": True,
            "report_path": promoted["output"],
            **promoted,
        }
        promoted_path.write_text(
            json.dumps(promoted_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if pilot_promoted:
        pilot_path = args.output_dir / "pilot_promoted_report.json"
        pilot_payload = {
            "schema": "wod_breakthrough_pilot_promoted_report_v1",
            "report_role": "pilot_candidate_requires_full_validation_cv",
            "promoted": False,
            "pilot_promoted": True,
            "report_path": pilot_promoted["output"],
            **pilot_promoted,
        }
        pilot_path.write_text(
            json.dumps(pilot_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if best else 1


def _clear_derived_reports(output_dir: Path) -> None:
    for name in (
        "manifest.json",
        "best_report.json",
        "best_observed_report.json",
        "promoted_report.json",
        "pilot_promoted_report.json",
    ):
        path = output_dir / name
        if path.exists():
            path.unlink()


def _experiment_matrix(args: argparse.Namespace) -> list[dict[str, Any]]:
    base_runs = [
        {
            "name": "baseline_recheck",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
        },
        {
            "name": "champion_fastkin_scene_gate_rate020",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "kinematic_profile": "base",
            "aux_feature_set": "temporal_summary",
            "residual_modes": "3",
            "residual_grouping": "off",
            "selector_fallback_source_options": "kinematic;kinematic,temporal",
            "source_gate": "train_margin",
            "source_gate_sources": "kinematic",
            "source_gate_router": "speed",
            "source_gate_max_rate": "0.20",
            "source_gate_route_allowlist": "speed:fast",
            "scene_gate": "train_margin",
            "scene_gate_router": "speed",
            "scene_gate_max_rate": "0.12",
            "scene_aux_feature_set": "external_embeddings",
        },
        {
            "name": "champion_fastkin_scene_gate_rate024",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "kinematic_profile": "base",
            "aux_feature_set": "temporal_summary",
            "residual_modes": "3",
            "residual_grouping": "off",
            "selector_fallback_source_options": "kinematic;kinematic,temporal",
            "source_gate": "train_margin",
            "source_gate_sources": "kinematic",
            "source_gate_router": "speed",
            "source_gate_max_rate": "0.24",
            "source_gate_route_allowlist": "speed:fast",
            "scene_gate": "train_margin",
            "scene_gate_router": "speed",
            "scene_gate_max_rate": "0.12",
            "scene_aux_feature_set": "external_embeddings",
        },
        {
            "name": "source_policy_speed_oracle_prior",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "selector_source_policy": "oracle_source",
        },
        {
            "name": "source_calibrated_family_calibrated",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "selector_source_calibration": "speed",
            "selector_family_calibration": "speed_source_family",
            "selector_family_calibration_min_count": "6",
        },
        {
            "name": "fallback_local_selector_linear",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "selector_fallback_local_selector": True,
        },
        {
            "name": "blend_mean_pairs",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "blend_candidates": "mean_pairs",
        },
        {
            "name": "blend_residual_pairs",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "blend_candidates": "residual_pairs",
        },
        {
            "name": "fallback_local_selector_blend_mean_pairs",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "selector_fallback_local_selector": True,
            "blend_candidates": "mean_pairs",
        },
        {
            "name": "independent_kinematic_source_gate",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "source_gate": "independent_train_margin",
            "source_gate_sources": "kinematic",
            "source_gate_router": "speed",
            "source_gate_max_rate": "0.35",
            "source_gate_min_precision": "0.55",
        },
        {
            "name": "conservative_intent_source_gate",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "source_gate": "independent_train_margin",
            "source_gate_sources": "kinematic,temporal",
            "source_gate_router": "intent",
            "source_gate_max_rate": "0.18",
            "source_gate_min_precision": "0.60",
            "source_gate_local_selector": True,
        },
        {
            "name": "conservative_intent_speed_source_gate",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "source_gate": "independent_train_margin",
            "source_gate_sources": "kinematic,temporal",
            "source_gate_router": "intent_speed",
            "source_gate_max_rate": "0.18",
            "source_gate_min_precision": "0.60",
            "source_gate_local_selector": True,
        },
        {
            "name": "conservative_learned_cap_speed_fine",
            "selector_model": "pairwise_logistic",
            "selector_features": "family_reliability_contextual",
            "selector_target": "frame_delta",
            "pairwise_iterations": "900",
            "pairwise_lr": "0.12",
            "pairwise_l2": "0.002",
            "pairwise_max_pairs_per_frame": "160",
            "selector_family_calibration": "speed_source_family",
            "selector_family_calibration_min_count": "6",
            "source_gate": "independent_train_margin",
            "source_gate_sources": "kinematic,temporal",
            "source_gate_router": "speed_fine",
            "source_gate_max_rate": "0.22",
            "source_gate_min_precision": "0.62",
            "source_gate_deny_prefixes": "ridge_residual_pc2,ridge_residual_pc4,ridge_residual_pc5",
        },
        {
            "name": "independent_kinematic_learned_source_gate",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "source_gate": "independent_train_margin",
            "source_gate_sources": "kinematic,learned",
            "source_gate_router": "speed",
            "source_gate_max_rate": "0.35",
            "source_gate_min_precision": "0.55",
        },
        {
            "name": "zero_shot_reflex_kinematic_linear",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "kinematic_profile": "reflex",
        },
        {
            "name": "zero_shot_reflex_guarded_reactive",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "kinematic_profile": "reflex",
            "source_gate": "train_margin",
            "source_gate_sources": "kinematic",
            "source_gate_router": "speed",
            "source_gate_max_rate": "0.20",
            "source_gate_candidate_prefixes": "yield_,lane_offset_,lane_change_,avoid_",
        },
        {
            "name": "zero_shot_reflex_fast_slow_scene_gate",
            "selector_model": "linear",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
            "kinematic_profile": "reflex",
            "aux_feature_set": "temporal_summary",
            "residual_modes": "3",
            "residual_grouping": "off",
            "selector_fallback_source_options": "kinematic;kinematic,temporal",
            "source_gate": "train_margin",
            "source_gate_sources": "kinematic",
            "source_gate_candidate_prefixes": "yield_,lane_offset_,lane_change_,avoid_",
            "source_gate_router": "speed",
            "source_gate_max_rate": "0.20",
            "source_gate_route_allowlist": "speed:fast",
            "scene_gate": "train_margin",
            "scene_gate_router": "speed",
            "scene_gate_max_rate": "0.20",
            "scene_aux_feature_set": "external_embeddings",
        },
        {
            "name": "pairwise_logistic_contextual",
            "selector_model": "pairwise_logistic",
            "selector_features": "contextual",
            "selector_target": "frame_delta",
        },
        {
            "name": "pairwise_logistic_family_reliability",
            "selector_model": "pairwise_logistic",
            "selector_features": "family_reliability_contextual",
            "selector_target": "frame_delta",
        },
        {
            "name": "pairwise_logistic_family_reliability_strong",
            "selector_model": "pairwise_logistic",
            "selector_features": "family_reliability_contextual",
            "selector_target": "frame_delta",
            "pairwise_iterations": "900",
            "pairwise_lr": "0.12",
            "pairwise_l2": "0.002",
            "pairwise_max_pairs_per_frame": "160",
        },
        {
            "name": "memory_pairwise_family_reliability_top1",
            "selector_model": "pairwise_logistic",
            "selector_features": "family_reliability_contextual",
            "selector_target": "frame_delta",
            "memory_candidates": "nearest_train",
            "memory_top_k": "1",
        },
        {
            "name": "memory_pairwise_family_reliability_top3",
            "selector_model": "pairwise_logistic",
            "selector_features": "family_reliability_contextual",
            "selector_target": "frame_delta",
            "memory_candidates": "nearest_train",
            "memory_top_k": "3",
        },
        {
            "name": "memory_pairwise_family_reliability_top5",
            "selector_model": "pairwise_logistic",
            "selector_features": "family_reliability_contextual",
            "selector_target": "frame_delta",
            "memory_candidates": "nearest_train",
            "memory_top_k": "5",
        },
    ]
    if args.external_embedding_cache.is_file():
        base_runs.append(
            {
                "name": "external_scene_pairwise_family_reliability",
                "selector_model": "pairwise_logistic",
                "selector_features": "family_reliability_contextual",
                "selector_target": "frame_delta",
                "scene_aux_feature_set": "external_embeddings",
            }
        )
    if getattr(args, "neural_candidate_model", None) and args.neural_candidate_model.is_file():
        base_runs.extend(
            [
                {
                    "name": "neural_candidates_contextual",
                    "selector_model": "linear",
                    "selector_features": "contextual",
                    "selector_target": "frame_delta",
                    "neural_candidate_model": str(args.neural_candidate_model),
                    "neural_top_k": "8",
                    "neural_residual_modes_per_anchor": "1",
                },
                {
                    "name": "neural_residual_pairs_family_calibration",
                    "selector_model": "linear",
                    "selector_features": "contextual",
                    "selector_target": "frame_delta",
                    "blend_candidates": "residual_pairs",
                    "selector_family_calibration": "speed_source_family",
                    "selector_family_calibration_min_count": "4",
                    "neural_candidate_model": str(args.neural_candidate_model),
                    "neural_top_k": "8",
                    "neural_residual_modes_per_anchor": "1",
                },
            ]
        )
    if getattr(args, "transformer_candidate_model", None) and args.transformer_candidate_model.is_file():
        base_runs.extend(
            [
                {
                    "name": "transformer_candidates_contextual",
                    "selector_model": "linear",
                    "selector_features": "contextual",
                    "selector_target": "frame_delta",
                    "transformer_candidate_model": str(args.transformer_candidate_model),
                    "transformer_top_k": "12",
                },
                {
                    "name": "transformer_residual_pairs_family_calibration",
                    "selector_model": "linear",
                    "selector_features": "contextual",
                    "selector_target": "frame_delta",
                    "blend_candidates": "residual_pairs",
                    "selector_family_calibration": "speed_source_family",
                    "selector_family_calibration_min_count": "4",
                    "transformer_candidate_model": str(args.transformer_candidate_model),
                    "transformer_top_k": "12",
                },
            ]
        )
    return base_runs[:2] if args.smoke else base_runs


def _command_for_run(run: dict[str, Any], output: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_wod_trajectory_model_cv.py"),
        "--output",
        str(output),
        "--frame-cache",
        str(args.frame_cache),
        "--rfs-backend",
        "local" if args.smoke or args.pilot else "official",
        "--selector-model",
        str(run["selector_model"]),
        "--selector-features",
        str(run["selector_features"]),
        "--selector-target",
        str(run["selector_target"]),
        "--selector-ridge",
        "175",
        "--ridge",
        "30",
        "--residual-modes",
        str(run.get("residual_modes", "5")),
        "--residual-grouping",
        str(run.get("residual_grouping", "speed")),
        "--kinematic-profile",
        str(run.get("kinematic_profile", "expanded")),
        "--selector-kinematic-fallback",
        "train_margin",
        "--selector-fallback-router",
        "speed",
        "--selector-fallback-source-options",
        str(run.get("selector_fallback_source_options", "kinematic;kinematic,learned;kinematic,temporal")),
        "--include-frame-diagnostics",
        "--progress-every-fold",
    ]
    if args.smoke:
        command.extend(["--max-preference-frames", "40", "--folds", "2"])
    elif args.pilot:
        command.extend(["--max-preference-frames", "160", "--folds", "4"])
    if run.get("aux_feature_set"):
        command.extend(["--aux-feature-set", str(run["aux_feature_set"])])
    if run.get("pairwise_iterations"):
        command.extend(["--selector-pairwise-iterations", str(run["pairwise_iterations"])])
    if run.get("pairwise_lr"):
        command.extend(["--selector-pairwise-lr", str(run["pairwise_lr"])])
    if run.get("pairwise_l2"):
        command.extend(["--selector-pairwise-l2", str(run["pairwise_l2"])])
    if run.get("pairwise_max_pairs_per_frame"):
        command.extend(["--selector-pairwise-max-pairs-per-frame", str(run["pairwise_max_pairs_per_frame"])])
    if run.get("selector_source_policy"):
        command.extend(["--selector-source-policy", str(run["selector_source_policy"])])
    if run.get("selector_source_calibration"):
        command.extend(["--selector-source-calibration", str(run["selector_source_calibration"])])
    if run.get("selector_family_calibration"):
        command.extend(["--selector-family-calibration", str(run["selector_family_calibration"])])
    if run.get("selector_family_calibration_min_count"):
        command.extend(
            [
                "--selector-family-calibration-min-count",
                str(run["selector_family_calibration_min_count"]),
            ]
        )
    if run.get("selector_fallback_local_selector"):
        command.append("--selector-fallback-local-selector")
    if run.get("blend_candidates"):
        command.extend(["--blend-candidates", str(run["blend_candidates"])])
    if run.get("source_gate"):
        command.extend(["--source-gate", str(run["source_gate"])])
    if run.get("source_gate_sources"):
        command.extend(["--source-gate-sources", str(run["source_gate_sources"])])
    if run.get("source_gate_candidate_prefixes"):
        command.extend(["--source-gate-candidate-prefixes", str(run["source_gate_candidate_prefixes"])])
    if run.get("source_gate_deny_prefixes"):
        command.extend(["--source-gate-deny-prefixes", str(run["source_gate_deny_prefixes"])])
    if run.get("source_gate_router"):
        command.extend(["--source-gate-router", str(run["source_gate_router"])])
    if run.get("source_gate_max_rate"):
        command.extend(["--source-gate-max-rate", str(run["source_gate_max_rate"])])
    if run.get("source_gate_min_precision"):
        command.extend(["--source-gate-min-precision", str(run["source_gate_min_precision"])])
    if run.get("source_gate_route_allowlist"):
        command.extend(["--source-gate-route-allowlist", str(run["source_gate_route_allowlist"])])
    if run.get("source_gate_local_selector"):
        command.append("--source-gate-local-selector")
    if run.get("scene_gate"):
        command.extend(["--scene-gate", str(run["scene_gate"])])
    if run.get("scene_gate_router"):
        command.extend(["--scene-gate-router", str(run["scene_gate_router"])])
    if run.get("scene_gate_max_rate"):
        command.extend(["--scene-gate-max-rate", str(run["scene_gate_max_rate"])])
    if run.get("memory_candidates"):
        command.extend(
            ["--memory-candidates", str(run["memory_candidates"]), "--memory-top-k", str(run["memory_top_k"])]
        )
    if run.get("scene_aux_feature_set"):
        command.extend(
            [
                "--scene-aux-feature-set",
                str(run["scene_aux_feature_set"]),
                "--external-embedding-cache",
                str(args.external_embedding_cache),
            ]
        )
    if run.get("neural_candidate_model"):
        command.extend(["--neural-candidate-model", str(run["neural_candidate_model"])])
        command.extend(["--neural-top-k", str(run.get("neural_top_k", "8"))])
        command.extend(
            [
                "--neural-residual-modes-per-anchor",
                str(run.get("neural_residual_modes_per_anchor", "0")),
            ]
        )
    if run.get("transformer_candidate_model"):
        command.extend(["--transformer-candidate-model", str(run["transformer_candidate_model"])])
        command.extend(["--transformer-top-k", str(run.get("transformer_top_k", "12"))])
    return command


def _run_command(command: list[str], *, continue_on_error: bool) -> int:
    try:
        completed = subprocess.run(command, cwd=ROOT, check=False)
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(json.dumps({"phase": "run_failed_to_start", "error": str(exc), "command": command}), file=sys.stderr)
        return 1
    if completed.returncode != 0 and not continue_on_error:
        payload = {"phase": "run_failed", "returncode": completed.returncode, "command": command}
        print(json.dumps(payload), file=sys.stderr)
    return int(completed.returncode)


def _metrics(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    worst = _worst_slice(payload)
    selected = payload.get("combined_ranker_mean_rfs")
    oracle = payload.get("combined_oracle_mean_rfs")
    return {
        "combined_ranker_mean_rfs": selected,
        "combined_ranker_mean_normalized_rfs": payload.get("combined_ranker_mean_normalized_rfs"),
        "combined_oracle_mean_rfs": oracle,
        "oracle_gap": _oracle_gap(selected, oracle),
        "worst_slice": worst,
    }


def _worst_slice(payload: dict[str, Any]) -> dict[str, Any] | None:
    slices = payload.get("slices", {})
    if not isinstance(slices, dict):
        return None
    rows = [
        {"slice": str(name), "mean_regret": float(data["mean_regret"])}
        for name, data in slices.items()
        if isinstance(data, dict) and "mean_regret" in data
    ]
    return max(rows, key=lambda row: (row["mean_regret"], row["slice"])) if rows else None


def _best_successful_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    successful = [
        row
        for row in rows
        if row.get("returncode") == 0 and isinstance(row.get("combined_ranker_mean_rfs"), (int, float))
    ]
    if not successful:
        return None
    return max(successful, key=lambda row: float(row["combined_ranker_mean_rfs"]))


def _promoted_run(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    promoted = [
        row
        for row in rows
        if row.get("returncode") == 0
        and row.get("pilot_baseline_passed") is True
        and isinstance(row.get("combined_ranker_mean_rfs"), (int, float))
    ]
    if not promoted:
        return None
    return max(promoted, key=lambda row: float(row["combined_ranker_mean_rfs"]))


def _validation_candidate_run(
    rows: list[dict[str, Any]], *, mode: str, min_rfs_gain: float = 0.1
) -> dict[str, Any] | None:
    if mode != "official":
        return None
    candidates = [
        row
        for row in rows
        if row.get("returncode") == 0
        and _passes_validation_baseline(row, min_rfs_gain=min_rfs_gain)
        and isinstance(row.get("combined_ranker_mean_rfs"), (int, float))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda row: float(row["combined_ranker_mean_rfs"]))


def _passes_validation_baseline(row: dict[str, Any], *, min_rfs_gain: float) -> bool:
    selected_delta = _float_or_none(row.get("baseline_selected_rfs_delta"))
    normalized_delta = _float_or_none(row.get("baseline_normalized_rfs_delta"))
    worst_delta = _float_or_none(row.get("baseline_worst_slice_regret_delta"))
    if selected_delta is None or normalized_delta is None or worst_delta is None:
        return False
    required_gain = float(min_rfs_gain)
    selected_passed = selected_delta > 0.0 if required_gain <= 0.0 else selected_delta >= required_gain
    return selected_passed and normalized_delta > 0.0 and worst_delta <= 0.0


def _mode(args: argparse.Namespace) -> str:
    if args.smoke:
        return "smoke"
    if args.pilot:
        return "pilot"
    return "official"


def _oracle_gap(selected: Any, oracle: Any) -> float | None:
    if isinstance(selected, (int, float)) and isinstance(oracle, (int, float)):
        return float(oracle) - float(selected)
    return None


def _add_baseline_comparisons(rows: list[dict[str, Any]], *, baseline_report: Path | None = None) -> None:
    baseline = _baseline_row_from_report(baseline_report) if baseline_report is not None else None
    if baseline is None:
        baseline = next(
            (
                row
                for row in rows
                if row.get("name") == "baseline_recheck"
                and row.get("returncode") == 0
                and isinstance(row.get("combined_ranker_mean_rfs"), (int, float))
            ),
            None,
        )
    if baseline is None:
        return
    baseline_rfs = float(baseline["combined_ranker_mean_rfs"])
    baseline_normalized = _float_or_none(baseline.get("combined_ranker_mean_normalized_rfs"))
    baseline_worst = _worst_regret(baseline)
    for row in rows:
        selected = _float_or_none(row.get("combined_ranker_mean_rfs"))
        normalized = _float_or_none(row.get("combined_ranker_mean_normalized_rfs"))
        worst = _worst_regret(row)
        row["baseline_selected_rfs_delta"] = selected - baseline_rfs if selected is not None else None
        row["baseline_normalized_rfs_delta"] = (
            normalized - baseline_normalized if normalized is not None and baseline_normalized is not None else None
        )
        row["baseline_worst_slice_regret_delta"] = (
            worst - baseline_worst if worst is not None and baseline_worst is not None else None
        )
        row["pilot_baseline_passed"] = _passes_pilot_baseline(row)


def _baseline_row_from_report(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    selected = _float_or_none(payload.get("combined_ranker_mean_rfs"))
    if selected is None:
        return None
    return {
        "name": "promotion_baseline",
        "returncode": 0,
        "combined_ranker_mean_rfs": selected,
        "combined_ranker_mean_normalized_rfs": payload.get("combined_ranker_mean_normalized_rfs"),
        "combined_oracle_mean_rfs": payload.get("combined_oracle_mean_rfs"),
        "worst_slice": _worst_slice(payload),
    }


def _passes_pilot_baseline(row: dict[str, Any]) -> bool:
    selected_delta = _float_or_none(row.get("baseline_selected_rfs_delta"))
    normalized_delta = _float_or_none(row.get("baseline_normalized_rfs_delta"))
    worst_delta = _float_or_none(row.get("baseline_worst_slice_regret_delta"))
    if selected_delta is None or normalized_delta is None or worst_delta is None:
        return False
    return selected_delta > 0.0 and normalized_delta > 0.0 and worst_delta <= 0.0


def _float_or_none(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _worst_regret(row: dict[str, Any]) -> float | None:
    worst = row.get("worst_slice")
    if not isinstance(worst, dict):
        return None
    value = worst.get("mean_regret")
    return float(value) if isinstance(value, (int, float)) else None


if __name__ == "__main__":
    raise SystemExit(main())
