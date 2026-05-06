#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
from itertools import islice
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.kinematic_candidates import kinematic_trajectories
from minimal_shot_av.model.anchor_trajectory_model import fit_anchor_residual_trajectory_model
from minimal_shot_av.model.learned_trajectory_model import (
    FEATURE_SET_BASE,
    FEATURE_SET_EXTERNAL_EMBEDDINGS,
    FEATURE_SET_TEMPORAL,
    _frame_features,
    RESIDUAL_GROUP_INTENT,
    RESIDUAL_GROUP_INTENT_SPEED,
    RESIDUAL_GROUP_OFF,
    RESIDUAL_GROUP_SPEED,
    RidgeTrajectoryModel,
    fit_ridge_trajectory_model,
)
from minimal_shot_av.model.neural_trajectory_model import NeuralAnchorResidualTrajectoryModel
from minimal_shot_av.model.rfs_metric import ManeuverCandidate, RfsReference, score_candidate
from minimal_shot_av.model.rfs_metric import Trajectory
from minimal_shot_av.model.transformer_trajectory_model import TransformerTrajectoryProposalModel
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame, load_preference_frames
from minimal_shot_av.model.wod_ranker import WodPreferenceRanker, candidate_ranker_row, raw_features
from minimal_shot_av.model.wod_ranker import selector_numeric_features, speed_bin
from minimal_shot_av.model.world_model import (
    FEATURE_MODE_EGO_TEMPORAL,
    FEATURE_MODE_EXTERNAL_EMBEDDINGS,
    FEATURE_MODE_SCENE_TOKENS,
    LearnedWorldModel,
    attach_external_embedding_cache,
    attach_scene_token_cache,
    external_embedding_dimension,
    fit_world_model,
    load_external_embedding_cache,
    load_scene_token_cache,
)
from minimal_shot_av.model.zero_shot_eval import CandidateRecord, RfsScorer, load_official_rfs_scorer


WORLD_MODEL_OFF = "off"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Segment-grouped CV for the non-text WOD trajectory proposal model."
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "wod_ridge_trajectory_cv.json")
    parser.add_argument(
        "--selector-audit-output",
        type=Path,
        help="Optional JSONL export of per-frame selector/oracle audit rows.",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--ridge", type=float, default=30.0)
    parser.add_argument(
        "--feature-set",
        choices=(FEATURE_SET_BASE, FEATURE_SET_TEMPORAL, FEATURE_SET_EXTERNAL_EMBEDDINGS),
        default=FEATURE_SET_BASE,
    )
    parser.add_argument(
        "--aux-feature-set",
        choices=(FEATURE_SET_TEMPORAL,),
        help="Train a second proposal model and add its candidates as an auxiliary source.",
    )
    parser.add_argument(
        "--scene-aux-feature-set",
        choices=(FEATURE_SET_EXTERNAL_EMBEDDINGS,),
        help="Train an external-embedding proposal model and add its frame-conditioned candidates.",
    )
    parser.add_argument(
        "--neural-candidate-model",
        type=Path,
        help="Optional trained neural anchor-residual proposal model JSON to add as learned candidates.",
    )
    parser.add_argument(
        "--neural-candidate-models",
        default="",
        help=(
            "Optional comma-separated trained neural anchor-residual proposal model JSONs. "
            "These are added as an ensemble of learned candidates."
        ),
    )
    parser.add_argument("--neural-top-k", type=int, default=8)
    parser.add_argument("--neural-residual-modes-per-anchor", type=int, default=0)
    parser.add_argument(
        "--transformer-candidate-model",
        type=Path,
        help="Optional trained transformer trajectory proposal model .pt to add as learned candidates.",
    )
    parser.add_argument(
        "--transformer-candidate-models",
        default="",
        help="Optional comma-separated trained transformer proposal model .pt files for learned candidates.",
    )
    parser.add_argument("--transformer-top-k", type=int, default=12)
    parser.add_argument("--selector-ridge", type=float, default=1.0)
    parser.add_argument(
        "--selector-target",
        choices=(
            "absolute",
            "absolute_normalized",
            "frame_delta",
            "frame_delta_normalized",
            "frame_delta_oracle_weighted",
            "frame_delta_normalized_oracle_weighted",
            "frame_delta_risk_weighted",
            "frame_delta_normalized_risk_weighted",
            "frame_zscore",
            "frame_zscore_normalized",
            "frame_rank",
            "oracle_binary",
            "pairwise_delta",
        ),
        default="absolute",
    )
    parser.add_argument(
        "--selector-features",
        choices=(
            "linear",
            "squared",
            "contextual",
            "intent_contextual",
            "world_contextual",
            "geometry_contextual",
            "family_reliability_contextual",
            "external_contextual",
            "contextual_external",
            "camera_contextual",
            "image_contextual",
        ),
        default="linear",
    )
    parser.add_argument(
        "--selector-model",
        choices=("linear", "random_fourier", "listwise_softmax", "pairwise_logistic", "boosted_stumps", "knn"),
        default="linear",
        help="Selector model family. random_fourier adds deterministic non-linear features before ridge fitting.",
    )
    parser.add_argument("--selector-rff-dim", type=int, default=256)
    parser.add_argument("--selector-rff-scale", type=float, default=3.0)
    parser.add_argument("--selector-rff-seed", type=int, default=1009)
    parser.add_argument("--selector-listwise-iterations", type=int, default=600)
    parser.add_argument("--selector-listwise-lr", type=float, default=0.2)
    parser.add_argument("--selector-listwise-temperature", type=float, default=0.35)
    parser.add_argument("--selector-pairwise-iterations", type=int, default=400)
    parser.add_argument("--selector-pairwise-lr", type=float, default=0.15)
    parser.add_argument("--selector-pairwise-l2", type=float, default=0.01)
    parser.add_argument("--selector-pairwise-max-pairs-per-frame", type=int, default=96)
    parser.add_argument("--selector-stump-iterations", type=int, default=80)
    parser.add_argument("--selector-stump-lr", type=float, default=0.08)
    parser.add_argument("--selector-stump-thresholds", type=int, default=16)
    parser.add_argument("--selector-knn-k", type=int, default=16)
    parser.add_argument("--selector-knn-temperature", type=float, default=4.0)
    parser.add_argument(
        "--selector-route-targets",
        default="",
        help=(
            "Optional comma-separated alternate selector targets. When set with "
            "--selector-route-router, train one selector per target and choose the "
            "best target per train-fold route."
        ),
    )
    parser.add_argument(
        "--selector-route-router",
        choices=("off", "intent", "speed", "speed_fine", "intent_speed"),
        default="off",
        help="Route selector target choice by train-fold group.",
    )
    parser.add_argument(
        "--selector-source-guard",
        choices=("off", "intent", "speed", "intent_speed"),
        default="off",
        help="Restrict selector choices to sources that were competitive for the same train-fold slice.",
    )
    parser.add_argument(
        "--selector-source-guard-margin",
        type=float,
        default=0.0,
        help="Allowed train-fold mean RFS gap from the best source for --selector-source-guard.",
    )
    parser.add_argument(
        "--selector-source-calibration",
        choices=("off", "intent", "speed", "intent_speed"),
        default="off",
        help="Add train-fold source reliability offsets to selector scores by group.",
    )
    parser.add_argument(
        "--selector-source-calibration-scale",
        type=float,
        default=1.0,
        help="Multiply train-fold source calibration offsets before applying them.",
    )
    parser.add_argument(
        "--selector-source-policy",
        choices=("off", "oracle_source", "source_score_speed", "source_score_intent_speed"),
        default="off",
        help=(
            "Optionally select source first from train-fold source reliability, then rank within source. "
            "source_score_* routes by mean best-candidate RFS instead of oracle win rate."
        ),
    )
    parser.add_argument(
        "--zero-shot-geometry-filter",
        choices=("off", "conservative", "reactive", "affordance"),
        default="off",
        help=(
            "Drop non-kinematic candidates that violate fixed motion-consistency priors before selection. "
            "This uses only candidate geometry, ego speed, and intent; it does not fit on frame outcomes."
        ),
    )
    parser.add_argument(
        "--selector-family-calibration",
        choices=("off", "speed_source_family"),
        default="off",
        help="Apply conservative held-out reliability offsets by speed, source, and candidate family.",
    )
    parser.add_argument("--selector-family-calibration-min-count", type=int, default=8)
    parser.add_argument(
        "--selector-postprocess",
        choices=("off", "safety_filter", "safety_utility"),
        default="off",
        help="Apply a safety postprocess after normal selection.",
    )
    parser.add_argument(
        "--safety-utility-ridge",
        type=float,
        default=10.0,
        help="Ridge penalty for --selector-postprocess safety_utility risk/utility models.",
    )
    parser.add_argument(
        "--safety-utility-floor",
        type=float,
        default=7.0,
        help="Train-fold RFS floor used to define safety risk for --selector-postprocess safety_utility.",
    )
    parser.add_argument(
        "--safety-utility-min-risk-margin",
        type=float,
        default=0.5,
        help="Minimum predicted risk reduction required to replace an already safe selected candidate.",
    )
    parser.add_argument(
        "--safety-utility-max-utility-drop",
        type=float,
        default=0.75,
        help="Maximum predicted utility loss allowed when replacing an already safe selected candidate.",
    )
    parser.add_argument(
        "--selector-kinematic-fallback",
        choices=("off", "train_margin"),
        default="off",
        help="Learn a train-fold selector margin threshold for falling back to the best kinematic candidate.",
    )
    parser.add_argument(
        "--selector-fallback-sources",
        default="kinematic",
        help="Comma-separated sources eligible for --selector-kinematic-fallback train_margin.",
    )
    parser.add_argument(
        "--selector-fallback-router",
        choices=("off", "intent", "speed", "speed_fine", "intent_speed"),
        default="off",
        help="Choose fallback source policy per train-fold group.",
    )
    parser.add_argument(
        "--selector-fallback-source-options",
        default="kinematic;kinematic,temporal",
        help="Semicolon-separated source lists for --selector-fallback-router.",
    )
    parser.add_argument(
        "--selector-fallback-local-selector",
        action="store_true",
        help="Rank fallback candidates with a selector trained only on the fallback source pool.",
    )
    parser.add_argument(
        "--scene-gate",
        choices=("off", "train_margin"),
        default="off",
        help="Learn a train-fold scene override gate that can replace the baseline selection with a scene candidate.",
    )
    parser.add_argument(
        "--scene-gate-margin",
        type=float,
        default=0.0,
        help="Minimum train-fold RFS gain required when choosing the scene-gate threshold.",
    )
    parser.add_argument(
        "--scene-gate-router",
        choices=("off", "intent", "speed", "speed_fine", "intent_speed"),
        default="speed",
        help="Choose scene-gate thresholds globally or per train-fold group.",
    )
    parser.add_argument("--scene-gate-ridge", type=float, default=1.0)
    parser.add_argument(
        "--scene-gate-max-rate",
        type=float,
        default=0.12,
        help="Maximum train-fold scene override rate allowed while fitting scene-gate thresholds.",
    )
    parser.add_argument(
        "--source-gate",
        choices=("off", "train_margin", "independent_train_margin"),
        default="off",
        help=(
            "Learn train-fold source override gates from the baseline selection "
            "to best candidates from named sources."
        ),
    )
    parser.add_argument(
        "--source-gate-sources",
        default="kinematic,learned,temporal,scene",
        help="Comma-separated candidate sources eligible for --source-gate.",
    )
    parser.add_argument(
        "--source-gate-candidate-prefixes",
        default="",
        help="Optional comma-separated candidate-name prefixes eligible for --source-gate train_margin.",
    )
    parser.add_argument(
        "--source-gate-deny-prefixes",
        default="",
        help="Optional comma-separated candidate-name prefixes excluded from --source-gate train_margin.",
    )
    parser.add_argument(
        "--source-gate-route-allowlist",
        default="",
        help=(
            "Optional comma-separated router keys eligible for --source-gate, e.g. speed:fast. "
            "Entries may be source-qualified as source|route, e.g. kinematic|speed:fast."
        ),
    )
    parser.add_argument("--source-gate-margin", type=float, default=0.0)
    parser.add_argument(
        "--source-gate-router",
        choices=("off", "intent", "speed", "speed_fine", "intent_speed"),
        default="speed",
    )
    parser.add_argument("--source-gate-ridge", type=float, default=1.0)
    parser.add_argument("--source-gate-max-rate", type=float, default=0.25)
    parser.add_argument(
        "--source-gate-min-precision",
        type=float,
        default=0.0,
        help="Minimum train-fold positive-gain precision required while fitting source-gate thresholds.",
    )
    parser.add_argument(
        "--source-gate-min-route-observations",
        type=int,
        default=0,
        help="Minimum train-fold observations required before fitting a source-gate route threshold.",
    )
    parser.add_argument(
        "--source-gate-min-route-positives",
        type=int,
        default=0,
        help="Minimum positive train-fold overrides required before a source-gate route may activate.",
    )
    parser.add_argument(
        "--source-gate-local-selector",
        action="store_true",
        help="Rank each gated source with a selector trained only on that source before applying source gates.",
    )
    parser.add_argument(
        "--source-veto-gate",
        choices=("off", "train_margin"),
        default="off",
        help="Learn a train-fold gate that can veto selected sources back to a fallback source pool.",
    )
    parser.add_argument(
        "--source-veto-sources",
        default="temporal",
        help="Comma-separated selected sources eligible for veto.",
    )
    parser.add_argument(
        "--source-veto-fallback-sources",
        default="kinematic",
        help="Comma-separated fallback sources used when a selected source is vetoed.",
    )
    parser.add_argument(
        "--source-veto-router",
        choices=("off", "intent", "speed", "speed_fine", "intent_speed"),
        default="speed",
    )
    parser.add_argument("--source-veto-ridge", type=float, default=1.0)
    parser.add_argument("--source-veto-max-rate", type=float, default=0.20)
    parser.add_argument("--source-veto-min-precision", type=float, default=0.0)
    parser.add_argument("--source-veto-min-route-observations", type=int, default=0)
    parser.add_argument("--source-veto-min-route-positives", type=int, default=0)
    parser.add_argument(
        "--kinematic-profile",
        choices=("base", "expanded", "reflex"),
        default="base",
        help="Kinematic candidate set to include in each frame.",
    )
    parser.add_argument(
        "--blend-candidates",
        choices=("off", "mean_pairs", "residual_pairs"),
        default="off",
        help=(
            "Add learned-family candidates that interpolate proposals from different sources. "
            "residual_pairs also blends learned residual modes with kinematic representatives."
        ),
    )
    parser.add_argument("--residual-modes", type=int, default=3)
    parser.add_argument(
        "--residual-grouping",
        choices=(
            RESIDUAL_GROUP_OFF,
            RESIDUAL_GROUP_INTENT,
            RESIDUAL_GROUP_SPEED,
            RESIDUAL_GROUP_INTENT_SPEED,
        ),
        default=RESIDUAL_GROUP_OFF,
        help="Add train-fold context-conditioned residual candidates.",
    )
    parser.add_argument("--pairwise-residuals", action="store_true")
    parser.add_argument(
        "--include-frame-diagnostics",
        action="store_true",
        help="Store per-frame selected-vs-oracle candidate diagnostics in folds_detail.",
    )
    parser.add_argument("--anchor-count", type=int, default=0)
    parser.add_argument("--anchor-top-k", type=int, default=0)
    parser.add_argument("--anchor-residual-modes-per-anchor", type=int, default=0)
    parser.add_argument("--anchor-iterations", type=int, default=25)
    parser.add_argument(
        "--world-model",
        choices=(
            WORLD_MODEL_OFF,
            FEATURE_MODE_EGO_TEMPORAL,
            FEATURE_MODE_SCENE_TOKENS,
            FEATURE_MODE_EXTERNAL_EMBEDDINGS,
        ),
        default=WORLD_MODEL_OFF,
        help="Add learned world-model candidates to the segment-grouped CV candidate set.",
    )
    parser.add_argument("--world-latent-dim", type=int, default=8)
    parser.add_argument("--world-ridge", type=float, default=10.0)
    parser.add_argument(
        "--world-latent-source",
        choices=("all", "scene"),
        default="all",
        help="Features used to build the world-model latent space.",
    )
    parser.add_argument("--world-memory-top-k", type=int, default=0)
    parser.add_argument(
        "--memory-candidates",
        choices=("off", "nearest_train", "nearest_preferences"),
        default="off",
        help="Add train-fold nearest-neighbor trajectory candidates without using test-fold labels.",
    )
    parser.add_argument("--memory-top-k", type=int, default=0)
    parser.add_argument(
        "--memory-feature-set",
        choices=(FEATURE_SET_TEMPORAL, FEATURE_SET_EXTERNAL_EMBEDDINGS),
        default=FEATURE_SET_TEMPORAL,
    )
    parser.add_argument("--target-rfs", type=float, default=10.0)
    parser.add_argument(
        "--scene-token-cache",
        type=Path,
        help="JSON cache from scripts/build_wod_scene_token_cache.py for fast scene-token world-model sweeps.",
    )
    parser.add_argument(
        "--external-embedding-cache",
        type=Path,
        help="JSON cache with one fixed-length external embedding vector per WOD frame.",
    )
    parser.add_argument(
        "--allow-missing-external-embeddings",
        action="store_true",
        help="Attach available external embeddings and leave missing frames as zero-filled selector context.",
    )
    parser.add_argument(
        "--world-max-neighbor-distance",
        type=float,
        help="Only emit world candidates when the nearest learned experience is within this latent distance.",
    )
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument(
        "--frame-cache",
        type=Path,
        help="Read/write a lightweight parsed WOD preference-frame cache to avoid repeated TFRecord scans.",
    )
    parser.add_argument(
        "--rfs-backend",
        choices=("official", "local"),
        default="official",
        help="Use official Waymo RFS for report artifacts; local is for fast tests.",
    )
    parser.add_argument(
        "--waymo-src",
        type=Path,
        default=ROOT / "waymo-open-dataset" / "src",
        help="Path containing waymo_open_dataset/metrics/python/rater_feedback_utils.py.",
    )
    parser.add_argument("--progress-every-fold", action="store_true", help="Print CV fold progress to stderr.")
    args = parser.parse_args()

    include_camera_images = (
        args.selector_features in {"camera_contextual", "image_contextual"}
        or (args.world_model == FEATURE_MODE_SCENE_TOKENS and args.scene_token_cache is None)
    )
    if args.frame_cache is not None and args.frame_cache.exists():
        frames = _load_frame_cache(args.frame_cache)
        if args.max_preference_frames is not None:
            frames = frames[: args.max_preference_frames]
    else:
        frame_iter = load_preference_frames(
            args.val_dir,
            max_shards=args.max_shards,
            max_records=args.max_records,
            include_camera_images=include_camera_images,
        )
        if args.max_preference_frames is not None:
            frame_iter = islice(frame_iter, args.max_preference_frames)
        frames = list(frame_iter)
        if args.frame_cache is not None:
            _write_frame_cache(frames, args.frame_cache)
    if args.progress_every_fold:
        print(json.dumps({"phase": "loaded_frames", "frames": len(frames)}), file=sys.stderr, flush=True)
    if args.world_model == FEATURE_MODE_EXTERNAL_EMBEDDINGS and args.external_embedding_cache is None:
        raise ValueError("--external-embedding-cache is required when --world-model external_embeddings")
    if args.allow_missing_external_embeddings and args.world_model == FEATURE_MODE_EXTERNAL_EMBEDDINGS:
        raise ValueError("--allow-missing-external-embeddings cannot be used with --world-model external_embeddings")
    if args.allow_missing_external_embeddings and args.scene_aux_feature_set == FEATURE_SET_EXTERNAL_EMBEDDINGS:
        raise ValueError(
            "--allow-missing-external-embeddings cannot be used with --scene-aux-feature-set external_embeddings"
        )
    if args.allow_missing_external_embeddings and args.memory_feature_set == FEATURE_SET_EXTERNAL_EMBEDDINGS:
        raise ValueError(
            "--allow-missing-external-embeddings cannot be used with --memory-feature-set external_embeddings"
        )
    if args.scene_token_cache is not None:
        frames = attach_scene_token_cache(frames, load_scene_token_cache(args.scene_token_cache))
    external_embedding_source = None
    if args.external_embedding_cache is not None:
        external_payload = json.loads(args.external_embedding_cache.read_text(encoding="utf-8"))
        external_embedding_source = str(external_payload.get("source", args.external_embedding_cache))
        frames = attach_external_embedding_cache(
            frames,
            load_external_embedding_cache(args.external_embedding_cache),
            require_all=not args.allow_missing_external_embeddings,
        )
        if args.progress_every_fold:
            print(
                json.dumps({"phase": "attached_external_embeddings", "frames": len(frames)}),
                file=sys.stderr,
                flush=True,
            )
    if not frames:
        raise RuntimeError("no WOD-E2E preference frames loaded")

    scorer = _local_rfs_score if args.rfs_backend == "local" else load_official_rfs_scorer(args.waymo_src)
    report = cross_validate_trajectory_model(
        frames,
        folds=args.folds,
        seed=args.seed,
        ridge=args.ridge,
        feature_set=args.feature_set,
        aux_feature_set=args.aux_feature_set,
        scene_aux_feature_set=args.scene_aux_feature_set,
        neural_candidate_model_path=args.neural_candidate_model,
        neural_candidate_model_paths=tuple(_split_paths(args.neural_candidate_models)),
        neural_top_k=args.neural_top_k,
        neural_residual_modes_per_anchor=args.neural_residual_modes_per_anchor,
        transformer_candidate_model_path=args.transformer_candidate_model,
        transformer_candidate_model_paths=tuple(_split_paths(args.transformer_candidate_models)),
        transformer_top_k=args.transformer_top_k,
        selector_ridge=args.selector_ridge,
        selector_target=args.selector_target,
        selector_features=args.selector_features,
        selector_model=args.selector_model,
        selector_rff_dim=args.selector_rff_dim,
        selector_rff_scale=args.selector_rff_scale,
        selector_rff_seed=args.selector_rff_seed,
        selector_listwise_iterations=args.selector_listwise_iterations,
        selector_listwise_lr=args.selector_listwise_lr,
        selector_listwise_temperature=args.selector_listwise_temperature,
        selector_pairwise_iterations=args.selector_pairwise_iterations,
        selector_pairwise_lr=args.selector_pairwise_lr,
        selector_pairwise_l2=args.selector_pairwise_l2,
        selector_pairwise_max_pairs_per_frame=args.selector_pairwise_max_pairs_per_frame,
        selector_stump_iterations=args.selector_stump_iterations,
        selector_stump_lr=args.selector_stump_lr,
        selector_stump_thresholds=args.selector_stump_thresholds,
        selector_knn_k=args.selector_knn_k,
        selector_knn_temperature=args.selector_knn_temperature,
        selector_route_targets=tuple(_split_optional_csv(args.selector_route_targets)),
        selector_route_router=args.selector_route_router,
        residual_modes=args.residual_modes,
        residual_grouping=args.residual_grouping,
        include_pairwise_residuals=args.pairwise_residuals,
        include_frame_diagnostics=args.include_frame_diagnostics or bool(args.selector_audit_output),
        anchor_count=args.anchor_count,
        anchor_top_k=args.anchor_top_k,
        anchor_residual_modes_per_anchor=args.anchor_residual_modes_per_anchor,
        anchor_iterations=args.anchor_iterations,
        world_model_mode=args.world_model,
        world_latent_dim=args.world_latent_dim,
        world_ridge=args.world_ridge,
        world_latent_source=args.world_latent_source,
        world_memory_top_k=args.world_memory_top_k,
        world_max_neighbor_distance=args.world_max_neighbor_distance,
        memory_candidates=args.memory_candidates,
        memory_top_k=args.memory_top_k,
        memory_feature_set=args.memory_feature_set,
        target_rfs=args.target_rfs,
        external_embedding_source=external_embedding_source,
        selector_source_guard=args.selector_source_guard,
        selector_source_guard_margin=args.selector_source_guard_margin,
        selector_source_calibration=args.selector_source_calibration,
        selector_source_calibration_scale=args.selector_source_calibration_scale,
        selector_source_policy=args.selector_source_policy,
        zero_shot_geometry_filter=args.zero_shot_geometry_filter,
        selector_family_calibration=args.selector_family_calibration,
        selector_family_calibration_min_count=args.selector_family_calibration_min_count,
        selector_postprocess=args.selector_postprocess,
        safety_utility_ridge=args.safety_utility_ridge,
        safety_utility_floor=args.safety_utility_floor,
        safety_utility_min_risk_margin=args.safety_utility_min_risk_margin,
        safety_utility_max_utility_drop=args.safety_utility_max_utility_drop,
        selector_kinematic_fallback=args.selector_kinematic_fallback,
        selector_fallback_sources=tuple(_split_sources(args.selector_fallback_sources)),
        selector_fallback_router=args.selector_fallback_router,
        selector_fallback_source_options=tuple(_split_source_options(args.selector_fallback_source_options)),
        selector_fallback_local_selector=args.selector_fallback_local_selector,
        scene_gate=args.scene_gate,
        scene_gate_margin=args.scene_gate_margin,
        scene_gate_router=args.scene_gate_router,
        scene_gate_ridge=args.scene_gate_ridge,
        scene_gate_max_rate=args.scene_gate_max_rate,
        source_gate=args.source_gate,
        source_gate_sources=tuple(_split_sources(args.source_gate_sources)),
        source_gate_candidate_prefixes=tuple(_split_optional_csv(args.source_gate_candidate_prefixes)),
        source_gate_deny_prefixes=tuple(_split_optional_csv(args.source_gate_deny_prefixes)),
        source_gate_route_allowlist=tuple(_split_optional_csv(args.source_gate_route_allowlist)),
        source_gate_margin=args.source_gate_margin,
        source_gate_router=args.source_gate_router,
        source_gate_ridge=args.source_gate_ridge,
        source_gate_max_rate=args.source_gate_max_rate,
        source_gate_min_precision=args.source_gate_min_precision,
        source_gate_min_route_observations=args.source_gate_min_route_observations,
        source_gate_min_route_positives=args.source_gate_min_route_positives,
        source_gate_local_selector=args.source_gate_local_selector,
        source_veto_gate=args.source_veto_gate,
        source_veto_sources=tuple(_split_sources(args.source_veto_sources)),
        source_veto_fallback_sources=tuple(_split_sources(args.source_veto_fallback_sources)),
        source_veto_router=args.source_veto_router,
        source_veto_ridge=args.source_veto_ridge,
        source_veto_max_rate=args.source_veto_max_rate,
        source_veto_min_precision=args.source_veto_min_precision,
        source_veto_min_route_observations=args.source_veto_min_route_observations,
        source_veto_min_route_positives=args.source_veto_min_route_positives,
        kinematic_profile=args.kinematic_profile,
        blend_candidates=args.blend_candidates,
        progress_every_fold=args.progress_every_fold,
        scorer=scorer,
        score_backend="local_rfs_metric" if args.rfs_backend == "local" else "official_waymo_rfs",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.selector_audit_output:
        _write_selector_audit_rows(report, args.selector_audit_output)
    print(json.dumps({key: value for key, value in report.items() if key != "folds_detail"}, indent=2))
    print(f"wrote {args.output}")
    return 0


def _write_frame_cache(frames: list[WodE2EPreferenceFrame], path: Path) -> None:
    payload = {
        "schema": "wod_preference_frames_v1",
        "frames": [_frame_to_payload(frame) for frame in frames],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_selector_audit_rows(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for fold in report.get("folds_detail", []):
            fold_dict = dict(fold)
            fold_index = int(fold_dict.get("fold_index", -1))
            for diagnostic in fold_dict.get("frame_diagnostics", []):
                row = dict(diagnostic)
                row["fold_index"] = fold_index
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _load_frame_cache(path: Path) -> list[WodE2EPreferenceFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "wod_preference_frames_v1":
        raise ValueError(f"unsupported WOD preference frame cache schema: {payload.get('schema')!r}")
    frames = payload.get("frames", [])
    if not isinstance(frames, list):
        raise ValueError("WOD preference frame cache frames must be a list")
    return [_frame_from_payload(frame) for frame in frames]


def _frame_to_payload(frame: WodE2EPreferenceFrame) -> dict[str, object]:
    return {
        "frame_name": frame.frame_name,
        "past_trajectory": _json_trajectory(frame.past_trajectory),
        "future_trajectory": _json_trajectory(frame.future_trajectory),
        "intent": int(frame.intent),
        "init_speed_mps": float(frame.init_speed_mps),
        "references": [
            {
                "label": reference.label,
                "trajectory": _json_trajectory(reference.trajectory),
                "score": float(reference.score),
            }
            for reference in frame.references
        ],
    }


def _frame_from_payload(payload: dict[str, object]) -> WodE2EPreferenceFrame:
    return WodE2EPreferenceFrame(
        frame_name=str(payload["frame_name"]),
        past_trajectory=_trajectory_from_payload(payload["past_trajectory"]),
        future_trajectory=_trajectory_from_payload(payload["future_trajectory"]),
        intent=int(payload["intent"]),
        init_speed_mps=float(payload["init_speed_mps"]),
        references=[
            RfsReference(
                label=str(reference["label"]),
                trajectory=_trajectory_from_payload(reference["trajectory"]),
                score=float(reference["score"]),
            )
            for reference in payload["references"]
        ],
    )


def _json_trajectory(trajectory: Trajectory) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in trajectory]


def _trajectory_from_payload(payload: object) -> Trajectory:
    if not isinstance(payload, list):
        raise ValueError("trajectory payload must be a list")
    return [(float(point[0]), float(point[1])) for point in payload]


def _split_sources(value: str) -> list[str]:
    sources = [source.strip() for source in value.split(",") if source.strip()]
    if not sources:
        raise ValueError("at least one fallback source is required")
    return sources


def _split_optional_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_paths(value: str) -> list[Path]:
    return [Path(item) for item in _split_optional_csv(value)]


def _split_source_options(value: str) -> list[tuple[str, ...]]:
    options = [tuple(_split_sources(option)) for option in value.split(";") if option.strip()]
    if not options:
        raise ValueError("at least one fallback source option is required")
    return options


def _effective_fallback_sources(
    fallback_sources: tuple[str, ...],
    router: str,
    source_options: tuple[tuple[str, ...], ...],
) -> list[str]:
    if router == "off":
        return list(fallback_sources)
    return sorted({source for option in source_options for source in option})


def cross_validate_trajectory_model(
    frames: list[WodE2EPreferenceFrame],
    *,
    folds: int,
    seed: int,
    ridge: float,
    feature_set: str = FEATURE_SET_BASE,
    aux_feature_set: str | None = None,
    scene_aux_feature_set: str | None = None,
    neural_candidate_model_path: Path | None = None,
    neural_candidate_model_paths: tuple[Path, ...] = (),
    neural_top_k: int = 8,
    neural_residual_modes_per_anchor: int = 0,
    transformer_candidate_model_path: Path | None = None,
    transformer_candidate_model_paths: tuple[Path, ...] = (),
    transformer_top_k: int = 12,
    selector_ridge: float = 1.0,
    selector_target: str = "absolute",
    selector_features: str = "linear",
    selector_model: str = "linear",
    selector_rff_dim: int = 256,
    selector_rff_scale: float = 3.0,
    selector_rff_seed: int = 1009,
    selector_listwise_iterations: int = 600,
    selector_listwise_lr: float = 0.2,
    selector_listwise_temperature: float = 0.35,
    selector_pairwise_iterations: int = 400,
    selector_pairwise_lr: float = 0.15,
    selector_pairwise_l2: float = 0.01,
    selector_pairwise_max_pairs_per_frame: int = 96,
    selector_stump_iterations: int = 80,
    selector_stump_lr: float = 0.08,
    selector_stump_thresholds: int = 16,
    selector_knn_k: int = 16,
    selector_knn_temperature: float = 4.0,
    selector_route_targets: tuple[str, ...] = (),
    selector_route_router: str = "off",
    residual_modes: int,
    residual_grouping: str = RESIDUAL_GROUP_OFF,
    include_pairwise_residuals: bool = False,
    include_frame_diagnostics: bool = False,
    anchor_count: int = 0,
    anchor_top_k: int = 0,
    anchor_residual_modes_per_anchor: int = 0,
    anchor_iterations: int = 25,
    world_model_mode: str = WORLD_MODEL_OFF,
    world_latent_dim: int = 8,
    world_ridge: float = 10.0,
    world_latent_source: str = "all",
    world_memory_top_k: int = 0,
    world_max_neighbor_distance: float | None = None,
    memory_candidates: str = "off",
    memory_top_k: int = 0,
    memory_feature_set: str = FEATURE_SET_TEMPORAL,
    target_rfs: float = 10.0,
    external_embedding_source: str | None = None,
    selector_source_guard: str = "off",
    selector_source_guard_margin: float = 0.0,
    selector_source_calibration: str = "off",
    selector_source_calibration_scale: float = 1.0,
    selector_source_policy: str = "off",
    zero_shot_geometry_filter: str = "off",
    selector_family_calibration: str = "off",
    selector_family_calibration_min_count: int = 8,
    selector_postprocess: str = "off",
    safety_utility_ridge: float = 10.0,
    safety_utility_floor: float = 7.0,
    safety_utility_min_risk_margin: float = 0.5,
    safety_utility_max_utility_drop: float = 0.75,
    selector_kinematic_fallback: str = "off",
    selector_fallback_sources: tuple[str, ...] = ("kinematic",),
    selector_fallback_router: str = "off",
    selector_fallback_source_options: tuple[tuple[str, ...], ...] = (("kinematic",), ("kinematic", "temporal")),
    selector_fallback_local_selector: bool = False,
    scene_gate: str = "off",
    scene_gate_margin: float = 0.0,
    scene_gate_router: str = "speed",
    scene_gate_ridge: float = 1.0,
    scene_gate_max_rate: float = 0.12,
    source_gate: str = "off",
    source_gate_sources: tuple[str, ...] = ("kinematic", "learned", "temporal", "scene"),
    source_gate_candidate_prefixes: tuple[str, ...] = (),
    source_gate_deny_prefixes: tuple[str, ...] = (),
    source_gate_route_allowlist: tuple[str, ...] = (),
    source_gate_margin: float = 0.0,
    source_gate_router: str = "speed",
    source_gate_ridge: float = 1.0,
    source_gate_max_rate: float = 0.25,
    source_gate_min_precision: float = 0.0,
    source_gate_min_route_observations: int = 0,
    source_gate_min_route_positives: int = 0,
    source_gate_local_selector: bool = False,
    source_veto_gate: str = "off",
    source_veto_sources: tuple[str, ...] = ("temporal",),
    source_veto_fallback_sources: tuple[str, ...] = ("kinematic",),
    source_veto_router: str = "speed",
    source_veto_ridge: float = 1.0,
    source_veto_max_rate: float = 0.20,
    source_veto_min_precision: float = 0.0,
    source_veto_min_route_observations: int = 0,
    source_veto_min_route_positives: int = 0,
    kinematic_profile: str = "base",
    blend_candidates: str = "off",
    progress_every_fold: bool = False,
    scorer: RfsScorer | None = None,
    score_backend: str = "local_rfs_metric",
) -> dict[str, object]:
    scorer = scorer or _local_rfs_score
    if world_model_mode not in {
        WORLD_MODEL_OFF,
        FEATURE_MODE_EGO_TEMPORAL,
        FEATURE_MODE_SCENE_TOKENS,
        FEATURE_MODE_EXTERNAL_EMBEDDINGS,
    }:
        raise ValueError(f"unsupported world model mode: {world_model_mode}")
    if world_model_mode == FEATURE_MODE_EXTERNAL_EMBEDDINGS:
        external_embedding_dimension(frames)
    if memory_candidates not in {"off", "nearest_train", "nearest_preferences"}:
        raise ValueError(f"unsupported memory candidate mode: {memory_candidates}")
    if memory_top_k < 0:
        raise ValueError("--memory-top-k must be non-negative")
    if neural_top_k < 0:
        raise ValueError("--neural-top-k must be non-negative")
    if neural_residual_modes_per_anchor < 0:
        raise ValueError("--neural-residual-modes-per-anchor must be non-negative")
    if transformer_top_k < 0:
        raise ValueError("--transformer-top-k must be non-negative")
    if zero_shot_geometry_filter not in {"off", "conservative", "reactive", "affordance"}:
        raise ValueError(f"unsupported zero-shot geometry filter: {zero_shot_geometry_filter}")
    if selector_postprocess not in {"off", "safety_filter", "safety_utility"}:
        raise ValueError(f"unsupported selector postprocess: {selector_postprocess}")
    if safety_utility_ridge <= 0.0:
        raise ValueError("--safety-utility-ridge must be positive")
    if safety_utility_min_risk_margin < 0.0:
        raise ValueError("--safety-utility-min-risk-margin must be non-negative")
    if safety_utility_max_utility_drop < 0.0:
        raise ValueError("--safety-utility-max-utility-drop must be non-negative")
    neural_paths = tuple(
        dict.fromkeys(
            [
                *([] if neural_candidate_model_path is None else [neural_candidate_model_path]),
                *neural_candidate_model_paths,
            ]
        )
    )
    neural_candidate_models = tuple(NeuralAnchorResidualTrajectoryModel.load(path) for path in neural_paths)
    transformer_paths = tuple(
        dict.fromkeys(
            [
                *([] if transformer_candidate_model_path is None else [transformer_candidate_model_path]),
                *transformer_candidate_model_paths,
            ]
        )
    )
    transformer_candidate_models = tuple(
        TransformerTrajectoryProposalModel.load(path) for path in transformer_paths
    )
    fold_frame_sets = _split_frame_names_by_segment(frames, folds=folds, seed=seed)
    fold_reports: list[dict[str, object]] = []
    for fold_index, test_names in enumerate(fold_frame_sets):
        if progress_every_fold:
            print(
                json.dumps({"phase": "fold_start", "fold_index": fold_index, "test_frames": len(test_names)}),
                file=sys.stderr,
                flush=True,
            )
        train_frames = [frame for frame in frames if frame.frame_name not in test_names]
        test_frames = [frame for frame in frames if frame.frame_name in test_names]
        if not train_frames or not test_frames:
            raise ValueError("each fold must have at least one train and one test frame")
        model = fit_ridge_trajectory_model(
            train_frames,
            ridge=ridge,
            residual_modes=residual_modes,
            feature_set=feature_set,
            residual_grouping=residual_grouping,
        )
        aux_model = None
        if aux_feature_set is not None:
            aux_model = fit_ridge_trajectory_model(
                train_frames,
                ridge=ridge,
                residual_modes=residual_modes,
                feature_set=aux_feature_set,
                residual_grouping=residual_grouping,
            )
        scene_aux_model = None
        if scene_aux_feature_set is not None:
            scene_aux_model = fit_ridge_trajectory_model(
                train_frames,
                ridge=ridge,
                residual_modes=residual_modes,
                feature_set=scene_aux_feature_set,
                residual_grouping=residual_grouping,
            )
        anchor_model = None
        if anchor_count > 0:
            anchor_model = fit_anchor_residual_trajectory_model(
                train_frames,
                anchor_count=anchor_count,
                ridge=ridge,
                anchor_iterations=anchor_iterations,
                seed=seed + fold_index,
                residual_modes_per_anchor=anchor_residual_modes_per_anchor,
            )
        world_model = None
        if world_model_mode != WORLD_MODEL_OFF:
            if progress_every_fold:
                print(
                    json.dumps(
                        {
                            "phase": "fit_world_model",
                            "fold_index": fold_index,
                            "train_frames": len(train_frames),
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
            world_model = fit_world_model(
                train_frames,
                latent_dim=world_latent_dim,
                ridge=world_ridge,
                feature_mode=world_model_mode,
                latent_source=world_latent_source,
            )
        memory_model = (
            _fit_memory_candidate_model(
                train_frames,
                mode=memory_candidates,
                top_k=memory_top_k,
                feature_set=memory_feature_set,
            )
            if memory_candidates != "off" and memory_top_k > 0
            else None
        )
        if progress_every_fold:
            print(
                json.dumps({"phase": "score_train_rows", "fold_index": fold_index, "train_frames": len(train_frames)}),
                file=sys.stderr,
                flush=True,
            )
        train_rows = _scored_candidate_rows(
            train_frames,
            model,
            scorer,
            residual_modes=residual_modes,
            include_pairwise_residuals=include_pairwise_residuals,
            kinematic_profile=kinematic_profile,
            aux_model=aux_model,
            scene_aux_model=scene_aux_model,
            anchor_model=anchor_model,
            anchor_top_k=anchor_top_k,
            anchor_residual_modes_per_anchor=anchor_residual_modes_per_anchor,
            world_model=world_model,
            world_memory_top_k=world_memory_top_k,
            world_max_neighbor_distance=world_max_neighbor_distance,
            memory_model=memory_model,
            blend_candidates=blend_candidates,
            neural_candidate_models=neural_candidate_models,
            neural_top_k=neural_top_k,
            neural_residual_modes_per_anchor=neural_residual_modes_per_anchor,
            transformer_candidate_models=transformer_candidate_models,
            transformer_top_k=transformer_top_k,
        )
        if progress_every_fold:
            print(
                json.dumps({"phase": "evaluate_fold", "fold_index": fold_index, "test_frames": len(test_frames)}),
                file=sys.stderr,
                flush=True,
            )
        selector_train_rows = _selector_training_rows_for_gates(
            train_rows,
            scene_gate=scene_gate,
            source_gate=source_gate,
            source_gate_sources=source_gate_sources,
        )
        family_reliability = _fit_family_reliability_features(selector_train_rows)
        if selector_features == "family_reliability_contextual":
            _apply_family_reliability_features(selector_train_rows, family_reliability)
            _apply_family_reliability_features(train_rows, family_reliability)
        selector = _fit_selector(
            selector_train_rows,
            ridge=selector_ridge,
            target_mode=selector_target,
            feature_mode=selector_features,
            model_family=selector_model,
            rff_dim=selector_rff_dim,
            rff_scale=selector_rff_scale,
            rff_seed=selector_rff_seed + fold_index,
            listwise_iterations=selector_listwise_iterations,
            listwise_lr=selector_listwise_lr,
            listwise_temperature=selector_listwise_temperature,
            pairwise_iterations=selector_pairwise_iterations,
            pairwise_lr=selector_pairwise_lr,
            pairwise_l2=selector_pairwise_l2,
            pairwise_max_pairs_per_frame=selector_pairwise_max_pairs_per_frame,
            stump_iterations=selector_stump_iterations,
            stump_lr=selector_stump_lr,
            stump_thresholds=selector_stump_thresholds,
            knn_k=selector_knn_k,
            knn_temperature=selector_knn_temperature,
        )
        routed_selectors, selector_route_policy = _fit_selector_route_policy(
            selector_train_rows,
            primary_selector=selector,
            primary_target=selector_target,
            alternate_targets=selector_route_targets,
            router=selector_route_router,
            ridge=selector_ridge,
            feature_mode=selector_features,
            model_family=selector_model,
            rff_dim=selector_rff_dim,
            rff_scale=selector_rff_scale,
            rff_seed=selector_rff_seed + fold_index,
            listwise_iterations=selector_listwise_iterations,
            listwise_lr=selector_listwise_lr,
            listwise_temperature=selector_listwise_temperature,
            pairwise_iterations=selector_pairwise_iterations,
            pairwise_lr=selector_pairwise_lr,
            pairwise_l2=selector_pairwise_l2,
            pairwise_max_pairs_per_frame=selector_pairwise_max_pairs_per_frame,
            stump_iterations=selector_stump_iterations,
            stump_lr=selector_stump_lr,
            stump_thresholds=selector_stump_thresholds,
            knn_k=selector_knn_k,
            knn_temperature=selector_knn_temperature,
        )
        source_guard = _fit_source_guard(
            selector_train_rows,
            mode=selector_source_guard,
            margin=selector_source_guard_margin,
        )
        source_calibration = _fit_source_calibration(
            selector_train_rows,
            mode=selector_source_calibration,
            scale=selector_source_calibration_scale,
        )
        source_policy = _fit_source_policy(selector_train_rows, mode=selector_source_policy)
        family_calibration = _fit_family_calibration(
            selector_train_rows,
            mode=selector_family_calibration,
            min_count=selector_family_calibration_min_count,
        )
        safety_utility_policy = _fit_safety_utility_policy(
            selector_train_rows,
            selector,
            mode=selector_postprocess,
            ridge=safety_utility_ridge,
            safety_floor=safety_utility_floor,
            min_risk_margin=safety_utility_min_risk_margin,
            max_utility_drop=safety_utility_max_utility_drop,
            source_calibration=source_calibration,
            family_calibration=family_calibration,
        )
        fallback_selectors = (
            _fit_fallback_selectors(
                selector_train_rows,
                fallback_sources=selector_fallback_sources,
                source_options=selector_fallback_source_options,
                ridge=selector_ridge,
                target_mode=selector_target,
                feature_mode=selector_features,
                model_family=selector_model,
                rff_dim=selector_rff_dim,
                rff_scale=selector_rff_scale,
                rff_seed=selector_rff_seed + fold_index,
                listwise_iterations=selector_listwise_iterations,
                listwise_lr=selector_listwise_lr,
                listwise_temperature=selector_listwise_temperature,
                pairwise_iterations=selector_pairwise_iterations,
                pairwise_lr=selector_pairwise_lr,
                pairwise_l2=selector_pairwise_l2,
                pairwise_max_pairs_per_frame=selector_pairwise_max_pairs_per_frame,
                stump_iterations=selector_stump_iterations,
                stump_lr=selector_stump_lr,
                stump_thresholds=selector_stump_thresholds,
                knn_k=selector_knn_k,
                knn_temperature=selector_knn_temperature,
            )
            if selector_fallback_local_selector
            else {}
        )
        source_gate_selectors = (
            _fit_fallback_selectors(
                selector_train_rows,
                fallback_sources=source_gate_sources,
                source_options=tuple((source,) for source in source_gate_sources),
                ridge=selector_ridge,
                target_mode=selector_target,
                feature_mode=selector_features,
                model_family=selector_model,
                rff_dim=selector_rff_dim,
                rff_scale=selector_rff_scale,
                rff_seed=selector_rff_seed + fold_index,
                listwise_iterations=selector_listwise_iterations,
                listwise_lr=selector_listwise_lr,
                listwise_temperature=selector_listwise_temperature,
                pairwise_iterations=selector_pairwise_iterations,
                pairwise_lr=selector_pairwise_lr,
                pairwise_l2=selector_pairwise_l2,
                pairwise_max_pairs_per_frame=selector_pairwise_max_pairs_per_frame,
                stump_iterations=selector_stump_iterations,
                stump_lr=selector_stump_lr,
                stump_thresholds=selector_stump_thresholds,
                knn_k=selector_knn_k,
                knn_temperature=selector_knn_temperature,
            )
            if source_gate_local_selector and source_gate != "off"
            else {}
        )
        fallback_policy = _fit_fallback_policy(
            selector_train_rows,
            selector,
            mode=selector_kinematic_fallback,
            fallback_sources=selector_fallback_sources,
            router=selector_fallback_router,
            source_options=selector_fallback_source_options,
            source_calibration=source_calibration,
            fallback_selectors=fallback_selectors,
        )
        scene_gate_policy = _fit_scene_gate_policy(
            train_rows,
            selector,
            mode=scene_gate,
            margin=scene_gate_margin,
            router=scene_gate_router,
            ridge=scene_gate_ridge,
            max_rate=scene_gate_max_rate,
            source_calibration=source_calibration,
            fallback_policy=fallback_policy,
            fallback_selectors=fallback_selectors,
        )
        source_gate_policy = _fit_source_gate_policy(
            train_rows,
            selector,
            mode=source_gate,
            sources=source_gate_sources,
            candidate_prefixes=source_gate_candidate_prefixes,
            deny_prefixes=source_gate_deny_prefixes,
            route_allowlist=source_gate_route_allowlist,
            margin=source_gate_margin,
            router=source_gate_router,
            ridge=source_gate_ridge,
            max_rate=source_gate_max_rate,
            min_precision=source_gate_min_precision,
            min_route_observations=source_gate_min_route_observations,
            min_route_positives=source_gate_min_route_positives,
            source_calibration=source_calibration,
            fallback_policy=fallback_policy,
            fallback_selectors=fallback_selectors,
            source_selectors=source_gate_selectors,
        )
        source_veto_policy = _fit_source_veto_policy(
            train_rows,
            selector,
            mode=source_veto_gate,
            sources=source_veto_sources,
            fallback_sources=source_veto_fallback_sources,
            router=source_veto_router,
            ridge=source_veto_ridge,
            max_rate=source_veto_max_rate,
            min_precision=source_veto_min_precision,
            min_route_observations=source_veto_min_route_observations,
            min_route_positives=source_veto_min_route_positives,
            source_calibration=source_calibration,
            fallback_policy=fallback_policy,
            fallback_selectors=fallback_selectors,
            scene_gate_policy=scene_gate_policy,
            source_gate_policy=source_gate_policy,
            source_gate_selectors=source_gate_selectors,
        )
        fold_reports.append(
            _evaluate_fold(
                test_frames,
                model,
                scorer,
                selector,
                source_guard=source_guard,
                fallback_policy=fallback_policy,
                source_calibration=source_calibration,
                source_policy=source_policy,
                zero_shot_geometry_filter=zero_shot_geometry_filter,
                family_calibration=family_calibration,
                fallback_selectors=fallback_selectors,
                scene_gate_policy=scene_gate_policy,
                source_gate_policy=source_gate_policy,
                source_gate_selectors=source_gate_selectors,
                source_veto_policy=source_veto_policy,
                safety_utility_policy=safety_utility_policy,
                fold_index=fold_index,
                residual_modes=residual_modes,
                include_pairwise_residuals=include_pairwise_residuals,
                kinematic_profile=kinematic_profile,
                blend_candidates=blend_candidates,
                include_frame_diagnostics=include_frame_diagnostics,
                aux_model=aux_model,
                scene_aux_model=scene_aux_model,
                anchor_model=anchor_model,
                anchor_top_k=anchor_top_k,
                anchor_residual_modes_per_anchor=anchor_residual_modes_per_anchor,
                world_model=world_model,
                world_memory_top_k=world_memory_top_k,
                world_max_neighbor_distance=world_max_neighbor_distance,
                memory_model=memory_model,
                neural_candidate_models=neural_candidate_models,
                neural_top_k=neural_top_k,
                neural_residual_modes_per_anchor=neural_residual_modes_per_anchor,
                transformer_candidate_models=transformer_candidate_models,
                transformer_top_k=transformer_top_k,
                family_reliability=family_reliability if selector_features == "family_reliability_contextual" else None,
                routed_selectors=routed_selectors,
                selector_route_policy=selector_route_policy,
            )
        )
    report = {
        "benchmark_type": "segment_grouped_cross_validation",
        "score_backend": score_backend,
        "frames": sum(int(fold["frames"]) for fold in fold_reports),
        "fold_count": folds,
        "ridge": float(ridge),
        "feature_set": feature_set,
        "aux_feature_set": aux_feature_set,
        "scene_aux_feature_set": scene_aux_feature_set,
        "neural_candidate_model": str(neural_paths[0]) if len(neural_paths) == 1 else None,
        "neural_candidate_models": [str(path) for path in neural_paths],
        "neural_top_k": int(neural_top_k),
        "neural_residual_modes_per_anchor": int(neural_residual_modes_per_anchor),
        "transformer_candidate_model": str(transformer_paths[0]) if len(transformer_paths) == 1 else None,
        "transformer_candidate_models": [str(path) for path in transformer_paths],
        "transformer_top_k": int(transformer_top_k),
        "selector_ridge": float(selector_ridge),
        "selector_target": selector_target,
        "selector_features": selector_features,
        "selector_model": selector_model,
        "selector_rff_dim": int(selector_rff_dim),
        "selector_rff_scale": float(selector_rff_scale),
        "selector_rff_seed": int(selector_rff_seed),
        "selector_listwise_iterations": int(selector_listwise_iterations),
        "selector_listwise_lr": float(selector_listwise_lr),
        "selector_listwise_temperature": float(selector_listwise_temperature),
        "selector_pairwise_iterations": int(selector_pairwise_iterations),
        "selector_pairwise_lr": float(selector_pairwise_lr),
        "selector_pairwise_l2": float(selector_pairwise_l2),
        "selector_pairwise_max_pairs_per_frame": int(selector_pairwise_max_pairs_per_frame),
        "selector_stump_iterations": int(selector_stump_iterations),
        "selector_stump_lr": float(selector_stump_lr),
        "selector_stump_thresholds": int(selector_stump_thresholds),
        "selector_knn_k": int(selector_knn_k),
        "selector_knn_temperature": float(selector_knn_temperature),
        "selector_route_targets": list(selector_route_targets),
        "selector_route_router": selector_route_router,
        "selector_route_enabled": selector_route_policy is not None if fold_reports else False,
        "residual_modes": int(residual_modes),
        "residual_grouping": residual_grouping,
        "pairwise_residuals": bool(include_pairwise_residuals),
        "kinematic_profile": kinematic_profile,
        "blend_candidates": blend_candidates,
        "anchor_count": int(anchor_count),
        "anchor_top_k": int(anchor_top_k),
        "anchor_residual_modes_per_anchor": int(anchor_residual_modes_per_anchor),
        "anchor_iterations": int(anchor_iterations),
        "world_model": world_model_mode,
        "world_latent_dim": int(world_latent_dim),
        "world_latent_source": world_latent_source,
        "world_ridge": float(world_ridge),
        "world_memory_top_k": int(world_memory_top_k),
        "world_max_neighbor_distance": world_max_neighbor_distance,
        "memory_candidates": memory_candidates,
        "memory_top_k": int(memory_top_k),
        "memory_feature_set": memory_feature_set,
        "target_rfs": float(target_rfs),
        "selector_source_guard": selector_source_guard,
        "selector_source_guard_margin": float(selector_source_guard_margin),
        "selector_source_calibration": selector_source_calibration,
        "selector_source_calibration_scale": float(selector_source_calibration_scale),
        "selector_source_policy": selector_source_policy,
        "zero_shot_geometry_filter": zero_shot_geometry_filter,
        "zero_shot_geometry_filtered_rate": _weighted_mean(fold_reports, "zero_shot_geometry_filtered_rate"),
        "zero_shot_geometry_filtered_non_kinematic_rate": _weighted_mean(
            fold_reports,
            "zero_shot_geometry_filtered_non_kinematic_rate",
        ),
        "selector_family_calibration": selector_family_calibration,
        "selector_family_calibration_min_count": int(selector_family_calibration_min_count),
        "selector_postprocess": selector_postprocess,
        "safety_utility_ridge": float(safety_utility_ridge),
        "safety_utility_floor": float(safety_utility_floor),
        "safety_utility_min_risk_margin": float(safety_utility_min_risk_margin),
        "safety_utility_max_utility_drop": float(safety_utility_max_utility_drop),
        "selector_kinematic_fallback": selector_kinematic_fallback,
        "selector_fallback_sources": _effective_fallback_sources(
            selector_fallback_sources,
            selector_fallback_router,
            selector_fallback_source_options,
        ),
        "selector_fallback_local_selector": bool(selector_fallback_local_selector),
        "selector_fallback_source_options": [list(option) for option in selector_fallback_source_options],
        "selector_fallback_router": selector_fallback_router,
        "scene_gate": scene_gate,
        "scene_gate_margin": float(scene_gate_margin),
        "scene_gate_router": scene_gate_router,
        "scene_gate_ridge": float(scene_gate_ridge),
        "scene_gate_max_rate": float(scene_gate_max_rate),
        "scene_gate_enabled": scene_gate != "off",
        "scene_gate_selected_rate": _weighted_mean(fold_reports, "scene_gate_selected_rate"),
        "scene_gate_oracle_positive_rate": _weighted_mean(fold_reports, "scene_gate_oracle_positive_rate"),
        "scene_gate_precision": _weighted_mean(fold_reports, "scene_gate_precision"),
        "scene_gate_mean_gain": _weighted_mean(fold_reports, "scene_gate_mean_gain"),
        "scene_gate_false_positive_loss": _weighted_mean(fold_reports, "scene_gate_false_positive_loss"),
        "source_gate": source_gate,
        "source_gate_sources": list(source_gate_sources),
        "source_gate_candidate_prefixes": list(source_gate_candidate_prefixes),
        "source_gate_deny_prefixes": list(source_gate_deny_prefixes),
        "source_gate_route_allowlist": list(source_gate_route_allowlist),
        "source_gate_margin": float(source_gate_margin),
        "source_gate_router": source_gate_router,
        "source_gate_ridge": float(source_gate_ridge),
        "source_gate_max_rate": float(source_gate_max_rate),
        "source_gate_min_precision": float(source_gate_min_precision),
        "source_gate_min_route_observations": int(source_gate_min_route_observations),
        "source_gate_min_route_positives": int(source_gate_min_route_positives),
        "source_gate_local_selector": bool(source_gate_local_selector),
        "source_gate_enabled": source_gate != "off",
        "source_gate_selected_rate": _weighted_mean(fold_reports, "source_gate_selected_rate"),
        "source_gate_precision": _weighted_mean(fold_reports, "source_gate_precision"),
        "source_gate_mean_gain": _weighted_mean(fold_reports, "source_gate_mean_gain"),
        "source_gate_false_positive_loss": _weighted_mean(fold_reports, "source_gate_false_positive_loss"),
        "source_veto_gate": source_veto_gate,
        "source_veto_sources": list(source_veto_sources),
        "source_veto_fallback_sources": list(source_veto_fallback_sources),
        "source_veto_router": source_veto_router,
        "source_veto_ridge": float(source_veto_ridge),
        "source_veto_max_rate": float(source_veto_max_rate),
        "source_veto_min_precision": float(source_veto_min_precision),
        "source_veto_min_route_observations": int(source_veto_min_route_observations),
        "source_veto_min_route_positives": int(source_veto_min_route_positives),
        "source_veto_enabled": source_veto_gate != "off",
        "source_veto_selected_rate": _weighted_mean(fold_reports, "source_veto_selected_rate"),
        "source_veto_precision": _weighted_mean(fold_reports, "source_veto_precision"),
        "source_veto_mean_gain": _weighted_mean(fold_reports, "source_veto_mean_gain"),
        "source_veto_false_positive_loss": _weighted_mean(fold_reports, "source_veto_false_positive_loss"),
        "safety_utility_selected_rate": _weighted_mean(fold_reports, "safety_utility_selected_rate"),
        "safety_utility_precision": _weighted_mean(fold_reports, "safety_utility_precision"),
        "safety_utility_mean_gain": _weighted_mean(fold_reports, "safety_utility_mean_gain"),
        "safety_utility_false_positive_loss": _weighted_mean(
            fold_reports,
            "safety_utility_false_positive_loss",
        ),
        "selector_kinematic_fallback_threshold_mean": _finite_mean_or_none(
            [
                value
                for fold in fold_reports
                for value in _fallback_policy_thresholds(fold.get("selector_fallback_policy"))
            ]
        )
        if selector_kinematic_fallback != "off"
        else None,
        "notes": [
            "Generated from segment-grouped cross-validation over WOD-E2E validation preference frames.",
            "This is an internal held-out development benchmark, not a leaderboard/test score.",
            "Simulator code and simulator metrics were not used.",
            "Simulator metrics are not used for model selection.",
        ],
        "frame_diagnostics_enabled": bool(include_frame_diagnostics),
        "frame_diagnostics_count": sum(
            len(fold.get("frame_diagnostics", [])) for fold in fold_reports
        )
        if include_frame_diagnostics
        else 0,
        "kinematic_constant_velocity_mean_rfs": _weighted_mean(fold_reports, "kinematic_constant_velocity_mean_rfs"),
        "kinematic_oracle_mean_rfs": _weighted_mean(fold_reports, "kinematic_oracle_mean_rfs"),
        "learned_mean_candidate_mean_rfs": _weighted_mean(fold_reports, "learned_mean_candidate_mean_rfs"),
        "learned_oracle_mean_rfs": _weighted_mean(fold_reports, "learned_oracle_mean_rfs"),
        "combined_oracle_mean_rfs": _weighted_mean(fold_reports, "combined_oracle_mean_rfs"),
        "combined_oracle_mean_normalized_rfs": _weighted_mean(
            fold_reports,
            "combined_oracle_mean_normalized_rfs",
        ),
        "reference_ceiling_mean_rfs": _weighted_mean(fold_reports, "reference_ceiling_mean_rfs"),
        "future_trajectory_mean_rfs": _weighted_mean(fold_reports, "future_trajectory_mean_rfs"),
        "future_trajectory_mean_normalized_rfs": _weighted_mean(
            fold_reports,
            "future_trajectory_mean_normalized_rfs",
        ),
        "target_rfs_reachable_by_references": _weighted_mean(fold_reports, "reference_ceiling_mean_rfs")
        >= float(target_rfs),
        "target_rfs_reachable_by_candidates": _weighted_mean(fold_reports, "combined_oracle_mean_rfs")
        >= float(target_rfs),
        "combined_ranker_mean_rfs": _weighted_mean(fold_reports, "combined_ranker_mean_rfs"),
        "combined_ranker_mean_normalized_rfs": _weighted_mean(
            fold_reports,
            "combined_ranker_mean_normalized_rfs",
        ),
        "learned_mean_gain_vs_constant_velocity": _weighted_delta(
            fold_reports,
            "learned_mean_candidate_mean_rfs",
            "kinematic_constant_velocity_mean_rfs",
        ),
        "learned_oracle_gain_vs_kinematic_oracle": _weighted_delta(
            fold_reports,
            "learned_oracle_mean_rfs",
            "kinematic_oracle_mean_rfs",
        ),
        "combined_oracle_gain_vs_kinematic_oracle": _weighted_delta(
            fold_reports,
            "combined_oracle_mean_rfs",
            "kinematic_oracle_mean_rfs",
        ),
        "combined_ranker_gain_vs_constant_velocity": _weighted_delta(
            fold_reports,
            "combined_ranker_mean_rfs",
            "kinematic_constant_velocity_mean_rfs",
        ),
        "combined_ranker_regret_to_oracle": _weighted_delta(
            fold_reports,
            "combined_oracle_mean_rfs",
            "combined_ranker_mean_rfs",
        ),
        "combined_ranker_top1_oracle_match_rate": _weighted_mean(
            fold_reports,
            "combined_ranker_top1_oracle_match_rate",
        ),
        "selected_anchor_rate": _weighted_mean(fold_reports, "selected_anchor_rate"),
        "selected_learned_rate": _weighted_mean(fold_reports, "selected_learned_rate"),
        "selected_scene_rate": _weighted_mean(fold_reports, "selected_scene_rate"),
        "selected_temporal_rate": _weighted_mean(fold_reports, "selected_temporal_rate"),
        "selected_kinematic_rate": _weighted_mean(fold_reports, "selected_kinematic_rate"),
        "selected_memory_rate": _weighted_mean(fold_reports, "selected_memory_rate"),
        "oracle_anchor_rate": _weighted_mean(fold_reports, "oracle_anchor_rate"),
        "oracle_scene_rate": _weighted_mean(fold_reports, "oracle_scene_rate"),
        "oracle_temporal_rate": _weighted_mean(fold_reports, "oracle_temporal_rate"),
        "oracle_memory_rate": _weighted_mean(fold_reports, "oracle_memory_rate"),
        "slices": _aggregate_fold_slices(fold_reports),
        "selector_regret_buckets": _aggregate_fold_regret_buckets(fold_reports),
        "selector_opportunity_buckets": _aggregate_fold_opportunity_buckets(fold_reports),
        "folds_detail": fold_reports,
    }
    if world_model_mode != WORLD_MODEL_OFF:
        report.update(
            {
                "world_imagined_mean_rfs": _weighted_mean(fold_reports, "world_imagined_mean_rfs"),
                "world_oracle_mean_rfs": _weighted_mean(fold_reports, "world_oracle_mean_rfs"),
                "combined_with_world_oracle_mean_rfs": _weighted_mean(
                    fold_reports,
                    "combined_oracle_mean_rfs",
                ),
                "combined_with_world_ranker_mean_rfs": _weighted_mean(
                    fold_reports,
                    "combined_ranker_mean_rfs",
                ),
                "selected_world_rate": _weighted_mean(fold_reports, "selected_world_rate"),
                "oracle_world_rate": _weighted_mean(fold_reports, "oracle_world_rate"),
            }
        )
    if external_embedding_source is not None:
        report.update(
            {
                "embedding_source": external_embedding_source or "external_embedding_cache",
                "embedding_dimension": external_embedding_dimension(frames),
                "embedding_frame_count": sum(1 for frame in frames if frame.external_embedding is not None),
            }
        )
    return report


def _evaluate_fold(
    frames: list[WodE2EPreferenceFrame],
    model: RidgeTrajectoryModel,
    scorer: RfsScorer,
    selector: WodPreferenceRanker,
    *,
    source_guard: dict[tuple[str, ...], set[str]] | None = None,
    fallback_policy: dict[str, object] | None = None,
    source_calibration: dict[str, dict[str, float]] | None = None,
    source_policy: dict[str, object] | None = None,
    zero_shot_geometry_filter: str = "off",
    family_calibration: dict[str, object] | None = None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
    scene_gate_policy: dict[str, object] | None = None,
    source_gate_policy: dict[str, object] | None = None,
    source_gate_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
    source_veto_policy: dict[str, object] | None = None,
    safety_utility_policy: dict[str, object] | None = None,
    fold_index: int,
    residual_modes: int,
    include_pairwise_residuals: bool,
    kinematic_profile: str,
    blend_candidates: str,
    include_frame_diagnostics: bool,
    aux_model: RidgeTrajectoryModel | None,
    scene_aux_model: RidgeTrajectoryModel | None,
    anchor_model,
    anchor_top_k: int,
    anchor_residual_modes_per_anchor: int,
    world_model: LearnedWorldModel | None,
    world_memory_top_k: int,
    world_max_neighbor_distance: float | None,
    memory_model: dict[str, object] | None,
    neural_candidate_models: tuple[NeuralAnchorResidualTrajectoryModel, ...],
    neural_top_k: int,
    neural_residual_modes_per_anchor: int,
    transformer_candidate_models: tuple[TransformerTrajectoryProposalModel, ...],
    transformer_top_k: int,
    family_reliability: dict[str, object] | None = None,
    routed_selectors: dict[str, WodPreferenceRanker] | None = None,
    selector_route_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    kinematic_first_scores: list[float] = []
    kinematic_oracle_scores: list[float] = []
    learned_mean_scores: list[float] = []
    learned_oracle_scores: list[float] = []
    world_imagined_scores: list[float] = []
    world_oracle_scores: list[float] = []
    reference_ceiling_scores: list[float] = []
    future_trajectory_scores: list[float] = []
    future_trajectory_normalized_scores: list[float] = []
    combined_oracle_scores: list[float] = []
    combined_oracle_normalized_scores: list[float] = []
    combined_ranker_scores: list[float] = []
    combined_ranker_normalized_scores: list[float] = []
    scene_gate_gains: list[float] = []
    scene_gate_false_positive_losses: list[float] = []
    scene_gate_positive_oracle_count = 0
    scene_gate_override_count = 0
    scene_gate_true_positive_count = 0
    source_gate_gains: list[float] = []
    source_gate_false_positive_losses: list[float] = []
    source_gate_override_count = 0
    source_gate_true_positive_count = 0
    source_veto_gains: list[float] = []
    source_veto_false_positive_losses: list[float] = []
    source_veto_override_count = 0
    source_veto_true_positive_count = 0
    safety_utility_gains: list[float] = []
    safety_utility_false_positive_losses: list[float] = []
    safety_utility_override_count = 0
    safety_utility_true_positive_count = 0
    zero_shot_geometry_filtered_count = 0
    zero_shot_geometry_filtered_non_kinematic_count = 0
    zero_shot_geometry_candidate_count = 0
    zero_shot_geometry_non_kinematic_count = 0
    opportunity_accumulators: dict[str, dict[str, object]] = {}
    top1_matches = 0
    selected_source_counts: dict[str, int] = defaultdict(int)
    oracle_source_counts: dict[str, int] = defaultdict(int)
    slice_accumulators: dict[str, dict[str, object]] = {}
    regret_accumulators: dict[str, dict[str, object]] = {}
    frame_diagnostics: list[dict[str, object]] = []
    for frame in frames:
        all_rows = _frame_candidate_rows(
            frame,
            model,
            scorer,
            residual_modes=residual_modes,
            include_pairwise_residuals=include_pairwise_residuals,
            kinematic_profile=kinematic_profile,
            aux_model=aux_model,
            scene_aux_model=scene_aux_model,
            anchor_model=anchor_model,
            anchor_top_k=anchor_top_k,
            anchor_residual_modes_per_anchor=anchor_residual_modes_per_anchor,
            world_model=world_model,
            world_memory_top_k=world_memory_top_k,
            world_max_neighbor_distance=world_max_neighbor_distance,
            memory_model=memory_model,
            blend_candidates=blend_candidates,
            neural_candidate_models=neural_candidate_models,
            neural_top_k=neural_top_k,
            neural_residual_modes_per_anchor=neural_residual_modes_per_anchor,
            transformer_candidate_models=transformer_candidate_models,
            transformer_top_k=transformer_top_k,
        )
        if family_reliability is not None:
            _apply_family_reliability_features(all_rows, family_reliability)
        rows = _zero_shot_geometry_filter_rows(all_rows, mode=zero_shot_geometry_filter)
        zero_shot_geometry_candidate_count += len(all_rows)
        zero_shot_geometry_filtered_count += len(all_rows) - len(rows)
        kept_row_ids = {id(row) for row in rows}
        zero_shot_geometry_non_kinematic_count += sum(
            1 for row in all_rows if str(row["source"]) != "kinematic"
        )
        zero_shot_geometry_filtered_non_kinematic_count += sum(
            1
            for row in all_rows
            if str(row["source"]) != "kinematic" and id(row) not in kept_row_ids
        )
        kinematic_scores = [float(row["rfs_score"]) for row in all_rows if str(row["source"]) == "kinematic"]
        learned_scores = [float(row["rfs_score"]) for row in all_rows if str(row["source"]) == "learned"]
        world_scores = [float(row["rfs_score"]) for row in all_rows if str(row["source"]) == "world"]
        policy_rows = [row for row in rows if str(row["source"]) != "scene"] if scene_gate_policy is not None else rows
        policy_rows = _source_gate_baseline_rows(source_gate_policy, policy_rows) or policy_rows or rows
        active_selector = _selector_for_route(selector, routed_selectors, selector_route_policy, rows[0])
        selected = _select_with_policy(
            active_selector,
            policy_rows,
            source_guard=source_guard,
            fallback_policy=fallback_policy,
            source_calibration=source_calibration,
            source_policy=source_policy,
            family_calibration=family_calibration,
            fallback_selectors=fallback_selectors,
        )
        baseline_selected = selected
        selected = _apply_scene_gate(
            scene_gate_policy,
            active_selector,
            rows,
            baseline_selected,
            source_calibration=source_calibration,
        )
        selected_before_source_gate = selected
        selected = _apply_source_gate(
            source_gate_policy,
            active_selector,
            rows,
            selected,
            source_calibration=source_calibration,
            source_selectors=source_gate_selectors,
        )
        selected_after_source_gate = selected
        selected_before_veto = selected
        selected = _apply_source_veto(
            source_veto_policy,
            active_selector,
            rows,
            selected,
            source_calibration=source_calibration,
            fallback_selectors=fallback_selectors,
        )
        selected_after_veto = selected
        selected_before_safety_utility = selected
        selected = _apply_safety_utility_policy(
            safety_utility_policy,
            active_selector,
            rows,
            selected,
            source_calibration=source_calibration,
            family_calibration=family_calibration,
        )
        oracle = max(all_rows, key=lambda row: float(row["rfs_score"]))
        oracle_score = float(oracle["rfs_score"])
        selected_score = float(selected["rfs_score"])
        reference_ceiling = _reference_ceiling(frame)
        reference_ceiling_scores.append(reference_ceiling)
        future_trajectory_scores.append(
            float(scorer(_record(frame, "ground_truth_future", -1, frame.future_trajectory), frame))
        )
        future_trajectory_normalized_scores.append(
            _normalized_rfs_score(future_trajectory_scores[-1], reference_ceiling)
        )
        scene_rows = [row for row in rows if str(row["source"]) == "scene"]
        if scene_gate_policy is not None and scene_rows:
            best_scene = _select_by_calibrated_score(active_selector, scene_rows, source_calibration)
            scene_gain = float(best_scene["rfs_score"]) - float(baseline_selected["rfs_score"])
            if scene_gain > 0.0:
                scene_gate_positive_oracle_count += 1
            if selected is best_scene:
                scene_gate_override_count += 1
                scene_gate_gains.append(scene_gain)
                if scene_gain > 0.0:
                    scene_gate_true_positive_count += 1
                else:
                    scene_gate_false_positive_losses.append(-scene_gain)
        if selected_after_source_gate is not selected_before_source_gate:
            source_gate_override_count += 1
            source_gain = float(selected_after_source_gate["rfs_score"]) - float(
                selected_before_source_gate["rfs_score"]
            )
            source_gate_gains.append(source_gain)
            if source_gain > 0.0:
                source_gate_true_positive_count += 1
            else:
                source_gate_false_positive_losses.append(-source_gain)
        if selected_after_veto is not selected_before_veto:
            source_veto_override_count += 1
            veto_gain = float(selected_after_veto["rfs_score"]) - float(selected_before_veto["rfs_score"])
            source_veto_gains.append(veto_gain)
            if veto_gain > 0.0:
                source_veto_true_positive_count += 1
            else:
                source_veto_false_positive_losses.append(-veto_gain)
        if selected is not selected_before_safety_utility:
            safety_utility_override_count += 1
            safety_gain = float(selected["rfs_score"]) - float(selected_before_safety_utility["rfs_score"])
            safety_utility_gains.append(safety_gain)
            if safety_gain > 0.0:
                safety_utility_true_positive_count += 1
            else:
                safety_utility_false_positive_losses.append(-safety_gain)
        kinematic_first_scores.append(kinematic_scores[0])
        kinematic_oracle_scores.append(max(kinematic_scores))
        learned_mean_scores.append(learned_scores[0])
        learned_oracle_scores.append(max(learned_scores))
        if world_scores:
            world_imagined_scores.append(world_scores[0])
            world_oracle_scores.append(max(world_scores))
        combined_ranker_scores.append(selected_score)
        combined_ranker_normalized_scores.append(_normalized_rfs_score(selected_score, reference_ceiling))
        combined_oracle_scores.append(oracle_score)
        combined_oracle_normalized_scores.append(_normalized_rfs_score(oracle_score, reference_ceiling))
        selected_source_counts[str(selected["source"])] += 1
        oracle_source_counts[str(oracle["source"])] += 1
        if selected_score == oracle_score:
            top1_matches += 1
        _add_slice_observation(slice_accumulators, f"speed:{_speed_bin(frame.init_speed_mps)}", selected, oracle)
        _add_slice_observation(slice_accumulators, f"intent:{int(frame.intent)}", selected, oracle)
        _add_regret_observations(regret_accumulators, frame, selected, oracle)
        _add_opportunity_observations(opportunity_accumulators, frame, selected, rows)
        if include_frame_diagnostics:
            frame_diagnostics.append(_frame_diagnostic(frame, selected, oracle, rows))
    report = {
        "fold_index": fold_index,
        "frames": len(frames),
        "selector_route_policy": selector_route_policy,
        "selector_fallback_policy": fallback_policy,
        "scene_gate_policy": scene_gate_policy,
        "kinematic_constant_velocity_mean_rfs": _mean(kinematic_first_scores),
        "kinematic_oracle_mean_rfs": _mean(kinematic_oracle_scores),
        "learned_mean_candidate_mean_rfs": _mean(learned_mean_scores),
        "learned_oracle_mean_rfs": _mean(learned_oracle_scores),
        "world_imagined_mean_rfs": _mean(world_imagined_scores) if world_imagined_scores else 0.0,
        "world_oracle_mean_rfs": _mean(world_oracle_scores) if world_oracle_scores else 0.0,
        "reference_ceiling_mean_rfs": _mean(reference_ceiling_scores),
        "future_trajectory_mean_rfs": _mean(future_trajectory_scores),
        "future_trajectory_mean_normalized_rfs": _mean(future_trajectory_normalized_scores),
        "combined_oracle_mean_rfs": _mean(combined_oracle_scores),
        "combined_oracle_mean_normalized_rfs": _mean(combined_oracle_normalized_scores),
        "combined_ranker_mean_rfs": _mean(combined_ranker_scores),
        "combined_ranker_mean_normalized_rfs": _mean(combined_ranker_normalized_scores),
        "combined_ranker_top1_oracle_match_rate": float(top1_matches / len(frames)),
        "scene_gate_selected_rate": float(scene_gate_override_count / len(frames)),
        "scene_gate_oracle_positive_rate": float(scene_gate_positive_oracle_count / len(frames)),
        "scene_gate_precision": float(scene_gate_true_positive_count / scene_gate_override_count)
        if scene_gate_override_count
        else 0.0,
        "scene_gate_mean_gain": _mean(scene_gate_gains) if scene_gate_gains else 0.0,
        "scene_gate_false_positive_loss": _mean(scene_gate_false_positive_losses)
        if scene_gate_false_positive_losses
        else 0.0,
        "source_gate_policy": source_gate_policy,
        "source_gate_selected_rate": float(source_gate_override_count / len(frames)),
        "source_gate_precision": float(source_gate_true_positive_count / source_gate_override_count)
        if source_gate_override_count
        else 0.0,
        "source_gate_mean_gain": _mean(source_gate_gains) if source_gate_gains else 0.0,
        "source_gate_false_positive_loss": _mean(source_gate_false_positive_losses)
        if source_gate_false_positive_losses
        else 0.0,
        "source_veto_policy": source_veto_policy,
        "source_veto_selected_rate": float(source_veto_override_count / len(frames)),
        "source_veto_precision": float(source_veto_true_positive_count / source_veto_override_count)
        if source_veto_override_count
        else 0.0,
        "source_veto_mean_gain": _mean(source_veto_gains) if source_veto_gains else 0.0,
        "source_veto_false_positive_loss": _mean(source_veto_false_positive_losses)
        if source_veto_false_positive_losses
        else 0.0,
        "safety_utility_policy": safety_utility_policy,
        "safety_utility_selected_rate": float(safety_utility_override_count / len(frames)),
        "safety_utility_precision": float(safety_utility_true_positive_count / safety_utility_override_count)
        if safety_utility_override_count
        else 0.0,
        "safety_utility_mean_gain": _mean(safety_utility_gains) if safety_utility_gains else 0.0,
        "safety_utility_false_positive_loss": _mean(safety_utility_false_positive_losses)
        if safety_utility_false_positive_losses
        else 0.0,
        "zero_shot_geometry_filtered_rate": float(
            zero_shot_geometry_filtered_count / zero_shot_geometry_candidate_count
        )
        if zero_shot_geometry_candidate_count
        else 0.0,
        "zero_shot_geometry_filtered_non_kinematic_rate": float(
            zero_shot_geometry_filtered_non_kinematic_count / zero_shot_geometry_non_kinematic_count
        )
        if zero_shot_geometry_non_kinematic_count
        else 0.0,
        "selected_anchor_rate": selected_source_counts["anchor"] / len(frames),
        "selected_learned_rate": selected_source_counts["learned"] / len(frames),
        "selected_scene_rate": selected_source_counts["scene"] / len(frames),
        "selected_temporal_rate": selected_source_counts["temporal"] / len(frames),
        "selected_kinematic_rate": selected_source_counts["kinematic"] / len(frames),
        "selected_world_rate": selected_source_counts["world"] / len(frames),
        "selected_memory_rate": selected_source_counts["memory"] / len(frames),
        "oracle_anchor_rate": oracle_source_counts["anchor"] / len(frames),
        "oracle_scene_rate": oracle_source_counts["scene"] / len(frames),
        "oracle_temporal_rate": oracle_source_counts["temporal"] / len(frames),
        "oracle_world_rate": oracle_source_counts["world"] / len(frames),
        "oracle_memory_rate": oracle_source_counts["memory"] / len(frames),
        "slices": _finalize_slices(slice_accumulators),
        "selector_regret_buckets": _finalize_regret_buckets(regret_accumulators),
        "selector_opportunity_buckets": _finalize_opportunity_buckets(opportunity_accumulators),
    }
    if include_frame_diagnostics:
        report["frame_diagnostics"] = frame_diagnostics
    return report


def _scored_candidate_rows(
    frames: list[WodE2EPreferenceFrame],
    model: RidgeTrajectoryModel,
    scorer: RfsScorer,
    *,
    residual_modes: int,
    include_pairwise_residuals: bool = False,
    kinematic_profile: str = "base",
    aux_model: RidgeTrajectoryModel | None = None,
    scene_aux_model: RidgeTrajectoryModel | None = None,
    anchor_model=None,
    anchor_top_k: int = 0,
    anchor_residual_modes_per_anchor: int = 0,
    world_model: LearnedWorldModel | None = None,
    world_memory_top_k: int = 0,
    world_max_neighbor_distance: float | None = None,
    memory_model: dict[str, object] | None = None,
    blend_candidates: str = "off",
    neural_candidate_models: tuple[NeuralAnchorResidualTrajectoryModel, ...] = (),
    neural_top_k: int = 8,
    neural_residual_modes_per_anchor: int = 0,
    transformer_candidate_models: tuple[TransformerTrajectoryProposalModel, ...] = (),
    transformer_top_k: int = 12,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for frame in frames:
        rows.extend(
            _frame_candidate_rows(
                frame,
                model,
                scorer,
                residual_modes=residual_modes,
                include_pairwise_residuals=include_pairwise_residuals,
                kinematic_profile=kinematic_profile,
                aux_model=aux_model,
                scene_aux_model=scene_aux_model,
                anchor_model=anchor_model,
                anchor_top_k=anchor_top_k,
                anchor_residual_modes_per_anchor=anchor_residual_modes_per_anchor,
                world_model=world_model,
                world_memory_top_k=world_memory_top_k,
                world_max_neighbor_distance=world_max_neighbor_distance,
                memory_model=memory_model,
                blend_candidates=blend_candidates,
                neural_candidate_models=neural_candidate_models,
                neural_top_k=neural_top_k,
                neural_residual_modes_per_anchor=neural_residual_modes_per_anchor,
                transformer_candidate_models=transformer_candidate_models,
                transformer_top_k=transformer_top_k,
            )
        )
    return rows


def _frame_diagnostic(
    frame: WodE2EPreferenceFrame,
    selected: dict[str, object],
    oracle: dict[str, object],
    rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    selected_score = float(selected["rfs_score"])
    oracle_score = float(oracle["rfs_score"])
    best_by_source = _best_rows_by_source(rows or [])
    return {
        "frame_name": frame.frame_name,
        "intent": int(frame.intent),
        "speed_bin": _speed_bin(frame.init_speed_mps),
        "init_speed_mps": float(frame.init_speed_mps),
        "selected_source": str(selected["source"]),
        "selected_candidate": str(selected["candidate_name"]),
        "selected_family": _candidate_family(selected),
        "selected_score": selected_score,
        "oracle_source": str(oracle["source"]),
        "oracle_candidate": str(oracle["candidate_name"]),
        "oracle_family": _candidate_family(oracle),
        "oracle_score": oracle_score,
        "regret": oracle_score - selected_score,
        "oracle_matched": selected_score == oracle_score,
        "top_candidates_by_source": {
            source: {
                "candidate": str(row["candidate_name"]),
                "family": _candidate_family(row),
                "score": float(row["rfs_score"]),
                "gap_vs_selected": float(row["rfs_score"]) - selected_score,
            }
            for source, row in sorted(best_by_source.items())
        },
    }


def _frame_candidate_rows(
    frame: WodE2EPreferenceFrame,
    model: RidgeTrajectoryModel,
    scorer: RfsScorer,
    *,
    residual_modes: int,
    include_pairwise_residuals: bool = False,
    kinematic_profile: str = "base",
    aux_model: RidgeTrajectoryModel | None = None,
    scene_aux_model: RidgeTrajectoryModel | None = None,
    anchor_model=None,
    anchor_top_k: int = 0,
    anchor_residual_modes_per_anchor: int = 0,
    world_model: LearnedWorldModel | None = None,
    world_memory_top_k: int = 0,
    world_max_neighbor_distance: float | None = None,
    memory_model: dict[str, object] | None = None,
    blend_candidates: str = "off",
    neural_candidate_models: tuple[NeuralAnchorResidualTrajectoryModel, ...] = (),
    neural_top_k: int = 8,
    neural_residual_modes_per_anchor: int = 0,
    transformer_candidate_models: tuple[TransformerTrajectoryProposalModel, ...] = (),
    transformer_top_k: int = 12,
) -> list[dict[str, object]]:
    if blend_candidates not in {"off", "mean_pairs", "residual_pairs"}:
        raise ValueError(f"unsupported blend candidate mode: {blend_candidates}")
    candidates = [
        ("kinematic", name, trajectory, {})
        for name, trajectory in kinematic_trajectories(frame.past_trajectory, profile=kinematic_profile)
    ]
    candidates.extend(
            ("learned", name, trajectory, {})
            for name, trajectory in model.candidate_trajectories_for_frame(
                frame,
                max_residual_modes=residual_modes,
                include_pairwise_residuals=include_pairwise_residuals,
            )
    )
    if aux_model is not None:
        candidates.extend(
            ("temporal", f"temporal_{name}", trajectory, {})
            for name, trajectory in aux_model.candidate_trajectories(
                frame.past_trajectory,
                intent=frame.intent,
                init_speed_mps=frame.init_speed_mps,
                max_residual_modes=residual_modes,
                include_pairwise_residuals=include_pairwise_residuals,
            )
        )
    if scene_aux_model is not None:
        candidates.extend(
            ("scene", f"scene_aux_{name}", trajectory, {})
            for name, trajectory in scene_aux_model.candidate_trajectories_for_frame(
                frame,
                max_residual_modes=residual_modes,
                include_pairwise_residuals=include_pairwise_residuals,
            )
        )
    if anchor_model is not None and anchor_top_k > 0:
        candidates.extend(
            ("anchor", name, trajectory, {})
            for name, trajectory, _ in anchor_model.candidate_trajectories(
                frame.past_trajectory,
                intent=frame.intent,
                init_speed_mps=frame.init_speed_mps,
                top_k=anchor_top_k,
                residual_modes_per_anchor=anchor_residual_modes_per_anchor,
            )
        )
    if world_model is not None:
        candidates.extend(
            ("world", name, trajectory, _world_candidate_metadata(prediction))
            for name, trajectory, prediction in world_model.candidate_trajectories(
                frame,
                memory_top_k=world_memory_top_k,
                max_neighbor_distance=world_max_neighbor_distance,
            )
        )
    if memory_model is not None:
        candidates.extend(
            ("memory", name, trajectory, {})
            for name, trajectory in _memory_candidate_trajectories(memory_model, frame)
        )
    if neural_candidate_models and neural_top_k > 0:
        for model_index, neural_candidate_model in enumerate(neural_candidate_models):
            candidates.extend(
                ("learned", _ensemble_candidate_name(name, model_index, len(neural_candidate_models)), trajectory, {})
                for name, trajectory, _confidence in neural_candidate_model.candidate_trajectories_for_frame(
                    frame,
                    top_k=neural_top_k,
                    residual_modes_per_anchor=neural_residual_modes_per_anchor,
                )
            )
    if transformer_candidate_models and transformer_top_k > 0:
        for model_index, transformer_candidate_model in enumerate(transformer_candidate_models):
            candidates.extend(
                (
                    "learned",
                    _ensemble_candidate_name(name, model_index, len(transformer_candidate_models)),
                    trajectory,
                    {},
                )
                for name, trajectory, _confidence in transformer_candidate_model.candidate_trajectories_for_frame(
                    frame,
                    top_k=transformer_top_k,
                )
            )
    if blend_candidates in {"mean_pairs", "residual_pairs"}:
        candidates.extend(
            _blend_mean_pair_candidates(
                candidates,
                include_learned_residuals=blend_candidates == "residual_pairs",
            )
        )
    rows: list[dict[str, object]] = []
    for candidate_index, (source, candidate_name, trajectory, metadata) in enumerate(candidates):
        record = _record(frame, candidate_name, candidate_index, trajectory)
        row = candidate_ranker_row(
            frame=frame,
            trajectory=record.trajectory,
            candidate_name=candidate_name,
            candidate_index=candidate_index,
            source=source,
        )
        _add_world_candidate_features(row, metadata)
        row["rfs_score"] = float(scorer(record, frame))
        row["rfs_score_normalized"] = _normalized_rfs_score(float(row["rfs_score"]), _reference_ceiling(frame))
        rows.append(
            {
                **row,
                "source": source,
            }
        )
    return rows


def _ensemble_candidate_name(name: str, model_index: int, model_count: int) -> str:
    if model_count <= 1:
        return name
    return f"ensemble{model_index}_{name}"


def _reference_ceiling(frame: WodE2EPreferenceFrame) -> float:
    if not frame.references:
        return 0.0
    return max(float(reference.score) for reference in frame.references)


def _zero_shot_geometry_filter_rows(rows: list[dict[str, object]], *, mode: str) -> list[dict[str, object]]:
    if mode == "off":
        return rows
    if mode not in {"conservative", "reactive", "affordance"}:
        raise ValueError(f"unsupported zero-shot geometry filter: {mode}")
    kept = [row for row in rows if not _zero_shot_geometry_reject(row, mode=mode)]
    return kept or rows


def _zero_shot_geometry_reject(row: dict[str, object], *, mode: str) -> bool:
    source = str(row["source"])
    if source == "kinematic":
        return False
    features = row.get("features", {})
    if not isinstance(features, dict):
        return False
    speed = max(0.0, float(features.get("init_speed_mps", 0.0)))
    intent = int(round(float(features.get("intent", 1.0))))
    max_speed = max(0.0, float(features.get("max_speed_mps", 0.0)))
    mean_speed = max(0.0, float(features.get("mean_speed_mps", 0.0)))
    final_speed = max(0.0, float(features.get("final_speed_mps", 0.0)))
    max_accel = max(0.0, float(features.get("max_abs_accel_mps2", 0.0)))
    mean_accel = max(0.0, float(features.get("mean_abs_accel_mps2", 0.0)))
    lateral_abs = max(0.0, float(features.get("max_lateral_abs", 0.0)))
    lateral_range = max(0.0, float(features.get("lateral_range", 0.0)))
    final_lateral = float(features.get("signed_lateral_5s", 0.0))
    heading_change = max(0.0, float(features.get("max_abs_heading_change", 0.0)))
    mean_heading_change = max(0.0, float(features.get("mean_abs_heading_change", 0.0)))
    expected_progress = max(0.0, float(features.get("expected_progress_5s", speed * 5.0)))
    progress = float(features.get("x_5s", 0.0))
    progress_ratio = float(features.get("progress_ratio_5s", progress / max(1.0, expected_progress)))
    stop_distance_error = max(0.0, float(features.get("stop_distance_error", abs(progress))))
    reverse_distance = max(0.0, float(features.get("reverse_distance", 0.0)))
    monotonic_forward_rate = float(features.get("monotonic_forward_rate", 1.0))
    lateral_to_progress_ratio = max(0.0, float(features.get("lateral_to_progress_ratio", 0.0)))
    curvature_per_meter = max(0.0, float(features.get("curvature_per_meter", 0.0)))
    turn_alignment_5s = float(features.get("turn_lateral_alignment_5s", 0.0))
    x_1s = float(features.get("x_1s", 0.0))
    x_3s = float(features.get("x_3s", 0.0))
    x_5s = float(features.get("x_5s", 0.0))

    if mode == "affordance":
        # A label-free world critic: reject only candidates whose motion is
        # internally inconsistent with the ego speed, route intent, and a
        # drivable forward frame. It is deliberately conservative and never
        # removes kinematic fallbacks.
        if reverse_distance > max(1.5, speed * 0.45):
            return True
        if monotonic_forward_rate < (0.82 if speed < 5.0 else 0.9):
            return True
        if speed < 1.4 and stop_distance_error > 9.0 and final_speed > 4.0:
            return True
        if speed >= 8.0 and progress_ratio < 0.12 and max_speed < speed * 0.35:
            return True
        if x_5s > 3.0 and lateral_to_progress_ratio > (0.95 if speed < 5.0 else 0.65):
            return True
        if curvature_per_meter > (0.7 if speed < 5.0 else 0.38):
            return True
        if intent in {2, 3} and abs(final_lateral) > 2.0 and turn_alignment_5s < -1.0:
            return True
        if intent == 1 and abs(final_lateral) > max(5.0, abs(x_5s) * 0.55):
            return True
        return False

    speed_headroom = 6.0 if mode == "conservative" else 4.0
    if max_speed > max(8.0, speed * 2.35 + speed_headroom):
        return True
    if mean_speed > max(6.0, speed * 1.85 + speed_headroom * 0.55):
        return True
    if speed < 1.4 and final_speed > (4.5 if mode == "conservative" else 3.25):
        return True
    if max_accel > (9.0 if mode == "conservative" else 7.0):
        return True
    if mean_accel > (4.5 if mode == "conservative" else 3.5):
        return True
    if x_1s < -0.75 or x_3s < -1.25 or x_5s < -1.75:
        return True

    if speed < 1.4:
        lateral_limit = 4.0 if mode == "conservative" else 2.75
    elif speed < 5.0:
        lateral_limit = 6.5 if mode == "conservative" else 4.75
    else:
        lateral_limit = 10.0 if mode == "conservative" else 8.0
    if lateral_abs > lateral_limit or lateral_range > lateral_limit * 1.35:
        return True

    heading_limit = 1.35 if mode == "conservative" else 1.05
    if speed >= 11.0:
        heading_limit *= 0.72
    elif speed >= 5.0:
        heading_limit *= 0.88
    if heading_change > heading_limit:
        return True
    if mean_heading_change > heading_limit * 0.42:
        return True

    if intent == 1:
        straight_lateral_limit = max(3.0, abs(x_5s) * (0.42 if mode == "conservative" else 0.33))
        if speed < 5.0 and abs(final_lateral) > straight_lateral_limit:
            return True
    elif intent == 2 and final_lateral < (-3.0 if mode == "conservative" else -2.0):
        return True
    elif intent == 3 and final_lateral > (3.0 if mode == "conservative" else 2.0):
        return True
    return False


def _normalized_rfs_score(score: float, reference_ceiling: float) -> float:
    if reference_ceiling <= 1e-8:
        return float(score)
    return float(score) * 10.0 / float(reference_ceiling)


def _blend_mean_pair_candidates(
    candidates: list[tuple[str, str, Trajectory, dict[str, float]]],
    *,
    include_learned_residuals: bool = False,
) -> list[tuple[str, str, Trajectory, dict[str, float]]]:
    representatives = [
        (source, name, trajectory)
        for source, name, trajectory, _metadata in candidates
        if _is_blend_representative(source, name, include_learned_residuals=include_learned_residuals)
    ]
    blends: list[tuple[str, str, Trajectory, dict[str, float]]] = []
    seen_names: set[str] = set()
    for left_index, (left_source, left_name, left_trajectory) in enumerate(representatives):
        for right_source, right_name, right_trajectory in representatives[left_index + 1 :]:
            if left_source == right_source:
                continue
            for left_weight in (0.25, 0.5, 0.75):
                right_weight = 1.0 - left_weight
                name = (
                    f"blend_{_short_candidate_key(left_source, left_name)}_"
                    f"{int(left_weight * 100):02d}_"
                    f"{_short_candidate_key(right_source, right_name)}_{int(right_weight * 100):02d}"
                )
                if name in seen_names:
                    continue
                seen_names.add(name)
                blends.append(
                    (
                        "learned",
                        name,
                        _weighted_average_trajectory(
                            left_trajectory,
                            right_trajectory,
                            left_weight=left_weight,
                        ),
                        {},
                    )
                )
    return blends


def _fit_memory_candidate_model(
    train_frames: list[WodE2EPreferenceFrame],
    *,
    mode: str,
    top_k: int,
    feature_set: str,
) -> dict[str, object]:
    if mode not in {"nearest_train", "nearest_preferences"}:
        raise ValueError(f"unsupported memory candidate mode: {mode}")
    rows = []
    for frame in train_frames:
        raw_features = _memory_raw_features(frame, feature_set)
        trajectories: list[tuple[str, Trajectory]] = [("future", frame.future_trajectory)]
        if mode == "nearest_preferences":
            trajectories.extend(
                (f"pref{index}_score{int(round(float(reference.score) * 100)):04d}", reference.trajectory)
                for index, reference in enumerate(frame.references)
            )
        rows.append(
            {
                "frame_name": frame.frame_name,
                "features": raw_features,
                "trajectories": trajectories,
            }
        )
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    for row, features in zip(rows, x):
        row["features_norm"] = ((features - mean) / scale).tolist()
    return {
        "mode": mode,
        "top_k": int(top_k),
        "feature_set": feature_set,
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "rows": rows,
    }


def _memory_candidate_trajectories(
    memory_model: dict[str, object],
    frame: WodE2EPreferenceFrame,
) -> list[tuple[str, Trajectory]]:
    rows = list(memory_model["rows"])
    if not rows:
        return []
    query = np.asarray(_memory_raw_features(frame, str(memory_model["feature_set"])), dtype=np.float64)
    mean = np.asarray(memory_model["feature_mean"], dtype=np.float64)
    scale = np.asarray(memory_model["feature_scale"], dtype=np.float64)
    query_norm = (query - mean) / scale
    distances = [
        (float(np.mean((np.asarray(row["features_norm"], dtype=np.float64) - query_norm) ** 2)), row)
        for row in rows
    ]
    distances.sort(key=lambda item: (item[0], str(item[1]["frame_name"])))
    candidates: list[tuple[str, Trajectory]] = []
    seen: set[tuple[tuple[float, float], ...]] = set()
    for rank, (_distance, row) in enumerate(distances[: int(memory_model["top_k"])], start=1):
        for suffix, trajectory in row["trajectories"]:
            key = tuple((round(float(x), 3), round(float(y), 3)) for x, y in trajectory)
            if key in seen:
                continue
            seen.add(key)
            candidates.append((f"memory_neighbor_{rank}_{suffix}", trajectory))
    return candidates


def _memory_raw_features(frame: WodE2EPreferenceFrame, feature_set: str) -> list[float]:
    return _frame_features(
        frame.past_trajectory,
        intent=frame.intent,
        init_speed_mps=frame.init_speed_mps,
        feature_set=feature_set,
        external_embedding=frame.external_embedding,
    )


def _is_blend_representative(
    source: str,
    candidate_name: str,
    *,
    include_learned_residuals: bool = False,
) -> bool:
    if source == "kinematic":
        return candidate_name in {"constant_velocity", "constant_acceleration", "hold_position"}
    if source == "learned":
        return candidate_name == "ridge_mean" or (
            include_learned_residuals and candidate_name.startswith("ridge_residual")
        )
    if source == "temporal":
        return candidate_name == "temporal_ridge_mean"
    if source == "scene":
        return candidate_name == "scene_aux_ridge_scene_mean"
    return False


def _short_candidate_key(source: str, candidate_name: str) -> str:
    key = candidate_name
    for prefix in ("scene_aux_", "temporal_", "ridge_", "constant_"):
        key = key.replace(prefix, "")
    return f"{source}_{key}".replace(" ", "_").replace(":", "_")


def _weighted_average_trajectory(
    left: Trajectory,
    right: Trajectory,
    *,
    left_weight: float,
) -> Trajectory:
    if len(left) != len(right):
        raise ValueError("blend candidates require trajectories with the same waypoint count")
    right_weight = 1.0 - float(left_weight)
    return [
        (
            float(left_x) * float(left_weight) + float(right_x) * right_weight,
            float(left_y) * float(left_weight) + float(right_y) * right_weight,
        )
        for (left_x, left_y), (right_x, right_y) in zip(left, right)
    ]


class RandomFourierPreferenceRanker:
    def __init__(
        self,
        *,
        numeric_features: list[str],
        candidate_names: list[str],
        candidate_families: list[str],
        feature_mean: list[float],
        feature_scale: list[float],
        projection: list[list[float]],
        phase: list[float],
        weights: list[float],
        bias: float,
    ) -> None:
        self.numeric_features = numeric_features
        self.candidate_names = candidate_names
        self.candidate_families = candidate_families
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.projection = projection
        self.phase = phase
        self.weights = weights
        self.bias = bias
        self._feature_mean_array = np.asarray(feature_mean, dtype=np.float64)
        self._feature_scale_array = np.asarray(feature_scale, dtype=np.float64)
        self._projection_array = np.asarray(projection, dtype=np.float64)
        self._phase_array = np.asarray(phase, dtype=np.float64)
        self._weights_array = np.asarray(weights, dtype=np.float64)

    def predict_row(self, row: dict[str, object]) -> float:
        raw = np.asarray(
            raw_features(row, self.numeric_features, self.candidate_names, self.candidate_families),
            dtype=np.float64,
        )
        x_norm = (raw - self._feature_mean_array) / self._feature_scale_array
        transformed = _random_fourier_features(
            x_norm.reshape(1, -1),
            self._projection_array,
            self._phase_array,
        )[0]
        return float(self.bias + transformed @ self._weights_array)

    def select_row(self, rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            raise ValueError("at least one candidate row is required")
        return max(
            rows,
            key=lambda row: (
                self.predict_row(row),
                -int(row.get("candidate_index", 0)),
                str(row["candidate_name"]),
            ),
        )


class BoostedStumpPreferenceRanker:
    def __init__(
        self,
        *,
        numeric_features: list[str],
        candidate_names: list[str],
        candidate_families: list[str],
        feature_mean: list[float],
        feature_scale: list[float],
        bias: float,
        learning_rate: float,
        stumps: list[dict[str, float]],
    ) -> None:
        self.numeric_features = numeric_features
        self.candidate_names = candidate_names
        self.candidate_families = candidate_families
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.bias = float(bias)
        self.learning_rate = float(learning_rate)
        self.stumps = stumps
        self._feature_mean_array = np.asarray(feature_mean, dtype=np.float64)
        self._feature_scale_array = np.asarray(feature_scale, dtype=np.float64)

    def predict_row(self, row: dict[str, object]) -> float:
        raw = np.asarray(
            raw_features(row, self.numeric_features, self.candidate_names, self.candidate_families),
            dtype=np.float64,
        )
        x_norm = (raw - self._feature_mean_array) / self._feature_scale_array
        score = self.bias
        for stump in self.stumps:
            feature_index = int(stump["feature_index"])
            value = float(stump["left_value"]) if x_norm[feature_index] <= float(stump["threshold"]) else float(
                stump["right_value"]
            )
            score += self.learning_rate * value
        return float(score)

    def select_row(self, rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            raise ValueError("at least one candidate row is required")
        return max(
            rows,
            key=lambda row: (
                self.predict_row(row),
                -int(row.get("candidate_index", 0)),
                str(row["candidate_name"]),
            ),
        )


class KnnPreferenceRanker:
    def __init__(
        self,
        *,
        numeric_features: list[str],
        candidate_names: list[str],
        candidate_families: list[str],
        feature_mean: list[float],
        feature_scale: list[float],
        train_features: list[list[float]],
        train_targets: list[float],
        k: int,
        temperature: float,
    ) -> None:
        self.numeric_features = numeric_features
        self.candidate_names = candidate_names
        self.candidate_families = candidate_families
        self.feature_mean = feature_mean
        self.feature_scale = feature_scale
        self.train_features = train_features
        self.train_targets = train_targets
        self.k = int(k)
        self.temperature = float(temperature)
        self._feature_mean_array = np.asarray(feature_mean, dtype=np.float64)
        self._feature_scale_array = np.asarray(feature_scale, dtype=np.float64)
        self._train_features_array = np.asarray(train_features, dtype=np.float64)
        self._train_targets_array = np.asarray(train_targets, dtype=np.float64)

    def predict_row(self, row: dict[str, object]) -> float:
        raw = np.asarray(
            raw_features(row, self.numeric_features, self.candidate_names, self.candidate_families),
            dtype=np.float64,
        )
        x_norm = (raw - self._feature_mean_array) / self._feature_scale_array
        distances = np.mean((self._train_features_array - x_norm) ** 2, axis=1)
        k = min(max(1, self.k), len(distances))
        nearest_indices = np.argpartition(distances, k - 1)[:k]
        nearest_distances = distances[nearest_indices]
        scale = max(1e-8, float(self.temperature))
        weights = np.exp(-(nearest_distances - float(np.min(nearest_distances))) / scale)
        total = float(np.sum(weights))
        if total <= 0.0:
            return float(np.mean(self._train_targets_array[nearest_indices]))
        return float(np.sum(weights * self._train_targets_array[nearest_indices]) / total)

    def select_row(self, rows: list[dict[str, object]]) -> dict[str, object]:
        if not rows:
            raise ValueError("at least one candidate row is required")
        return max(
            rows,
            key=lambda row: (
                self.predict_row(row),
                -int(row.get("candidate_index", 0)),
                str(row["candidate_name"]),
            ),
        )


def _fit_selector(
    rows: list[dict[str, object]],
    *,
    ridge: float,
    target_mode: str = "absolute",
    feature_mode: str = "linear",
    model_family: str = "linear",
    rff_dim: int = 256,
    rff_scale: float = 3.0,
    rff_seed: int = 1009,
    listwise_iterations: int = 600,
    listwise_lr: float = 0.2,
    listwise_temperature: float = 0.35,
    pairwise_iterations: int = 400,
    pairwise_lr: float = 0.15,
    pairwise_l2: float = 0.01,
    pairwise_max_pairs_per_frame: int = 96,
    stump_iterations: int = 80,
    stump_lr: float = 0.08,
    stump_thresholds: int = 16,
    knn_k: int = 16,
    knn_temperature: float = 4.0,
):
    numeric_features = _selector_numeric_features(feature_mode)
    candidate_names = sorted({str(row["candidate_name"]) for row in rows})
    candidate_families = sorted({str(row["features"].get("candidate_family", row["candidate_name"])) for row in rows})
    x = np.asarray(
        [raw_features(row, numeric_features, candidate_names, candidate_families) for row in rows],
        dtype=np.float64,
    )
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean) / scale
    if model_family == "linear":
        return _fit_linear_selector(
            rows,
            x_norm,
            mean=mean,
            scale=scale,
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            ridge=ridge,
            target_mode=target_mode,
        )
    if model_family == "listwise_softmax":
        weights, bias = _fit_listwise_softmax_weights(
            rows,
            x_norm,
            ridge=ridge,
            iterations=listwise_iterations,
            learning_rate=listwise_lr,
            temperature=listwise_temperature,
        )
        return WodPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            weights=weights.tolist(),
            bias=float(bias),
        )
    if model_family == "pairwise_logistic":
        weights = _fit_pairwise_logistic_weights(
            rows,
            x_norm,
            iterations=pairwise_iterations,
            learning_rate=pairwise_lr,
            l2=pairwise_l2,
            max_pairs_per_frame=pairwise_max_pairs_per_frame,
        )
        return WodPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            weights=weights.tolist(),
            bias=0.0,
        )
    if model_family == "boosted_stumps":
        if target_mode == "pairwise_delta":
            raise ValueError("boosted_stumps does not support pairwise_delta selector target")
        if stump_iterations <= 0:
            raise ValueError("--selector-stump-iterations must be positive")
        if stump_lr <= 0.0:
            raise ValueError("--selector-stump-lr must be positive")
        if stump_thresholds <= 0:
            raise ValueError("--selector-stump-thresholds must be positive")
        y = _selector_targets(rows, target_mode)
        bias, stumps = _fit_boosted_stumps(
            x_norm,
            y,
            iterations=stump_iterations,
            learning_rate=stump_lr,
            max_thresholds=stump_thresholds,
        )
        return BoostedStumpPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            bias=bias,
            learning_rate=stump_lr,
            stumps=stumps,
        )
    if model_family == "knn":
        if target_mode == "pairwise_delta":
            raise ValueError("knn selector does not support pairwise_delta selector target")
        if knn_k <= 0:
            raise ValueError("--selector-knn-k must be positive")
        if knn_temperature <= 0.0:
            raise ValueError("--selector-knn-temperature must be positive")
        return KnnPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            train_features=x_norm.tolist(),
            train_targets=_selector_targets(rows, target_mode).tolist(),
            k=knn_k,
            temperature=knn_temperature,
        )
    if model_family != "random_fourier":
        raise ValueError(f"unsupported selector model: {model_family!r}")
    if rff_dim <= 0:
        raise ValueError("--selector-rff-dim must be positive")
    if rff_scale <= 0.0:
        raise ValueError("--selector-rff-scale must be positive")
    rng = np.random.default_rng(int(rff_seed))
    projection = rng.normal(loc=0.0, scale=1.0 / float(rff_scale), size=(x_norm.shape[1], int(rff_dim)))
    phase = rng.uniform(0.0, 2.0 * math.pi, size=int(rff_dim))
    x_model = _random_fourier_features(x_norm, projection, phase)
    if target_mode == "pairwise_delta":
        weights = _fit_pairwise_selector_weights(rows, x_model, ridge=ridge)
        return RandomFourierPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            projection=projection.tolist(),
            phase=phase.tolist(),
            weights=weights.tolist(),
            bias=0.0,
        )
    y = _selector_targets(rows, target_mode)
    design = np.concatenate([np.ones((x_model.shape[0], 1)), x_model], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RandomFourierPreferenceRanker(
        numeric_features=numeric_features,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        projection=projection.tolist(),
        phase=phase.tolist(),
        weights=weights[1:].tolist(),
        bias=float(weights[0]),
    )


def _fit_linear_selector(
    rows: list[dict[str, object]],
    x_norm: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    numeric_features: list[str],
    candidate_names: list[str],
    candidate_families: list[str],
    ridge: float,
    target_mode: str,
) -> WodPreferenceRanker:
    if target_mode == "pairwise_delta":
        weights = _fit_pairwise_selector_weights(rows, x_norm, ridge=ridge)
        return WodPreferenceRanker(
            numeric_features=numeric_features,
            candidate_names=candidate_names,
            candidate_families=candidate_families,
            feature_mean=mean.tolist(),
            feature_scale=scale.tolist(),
            weights=weights.tolist(),
            bias=0.0,
        )
    y = _selector_targets(rows, target_mode)
    design = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return WodPreferenceRanker(
        numeric_features=numeric_features,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=mean.tolist(),
        feature_scale=scale.tolist(),
        weights=weights[1:].tolist(),
        bias=float(weights[0]),
    )


def _random_fourier_features(x_norm: np.ndarray, projection: np.ndarray, phase: np.ndarray) -> np.ndarray:
    mapped = (2.0 / float(projection.shape[1])) ** 0.5 * np.cos(x_norm @ projection + phase)
    return np.concatenate([x_norm, mapped], axis=1)


def _fit_listwise_softmax_weights(
    rows: list[dict[str, object]],
    x_norm: np.ndarray,
    *,
    ridge: float,
    iterations: int,
    learning_rate: float,
    temperature: float,
) -> tuple[np.ndarray, float]:
    if iterations <= 0:
        raise ValueError("--selector-listwise-iterations must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--selector-listwise-lr must be positive")
    if temperature <= 0.0:
        raise ValueError("--selector-listwise-temperature must be positive")
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row["frame_name"])].append(index)
    grouped_indices = list(groups.values())
    if not grouped_indices:
        raise ValueError("listwise selector training requires at least one frame group")
    scores = np.asarray([float(row["rfs_score"]) for row in rows], dtype=np.float64)
    targets = [
        _softmax((scores[indices] - np.max(scores[indices])) / float(temperature))
        for indices in grouped_indices
    ]
    weights = np.zeros(x_norm.shape[1], dtype=np.float64)
    bias = 0.0
    frame_count = float(len(grouped_indices))
    penalty = float(ridge) / max(1.0, float(len(rows)))
    for _ in range(int(iterations)):
        grad_w = penalty * weights
        grad_b = 0.0
        for indices, target in zip(grouped_indices, targets):
            frame_x = x_norm[indices]
            logits = frame_x @ weights + bias
            probs = _softmax(logits - np.max(logits))
            diff = probs - target
            grad_w += frame_x.T @ diff / frame_count
            grad_b += float(np.sum(diff)) / frame_count
        weights -= float(learning_rate) * grad_w
        bias -= float(learning_rate) * grad_b
    return weights, bias


def _fit_pairwise_logistic_weights(
    rows: list[dict[str, object]],
    x_norm: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    l2: float,
    max_pairs_per_frame: int,
) -> np.ndarray:
    if iterations <= 0:
        raise ValueError("--selector-pairwise-iterations must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--selector-pairwise-lr must be positive")
    if l2 < 0.0:
        raise ValueError("--selector-pairwise-l2 must be non-negative")
    if max_pairs_per_frame <= 0:
        raise ValueError("--selector-pairwise-max-pairs-per-frame must be positive")
    pair_diffs = _pairwise_training_diffs(rows, x_norm, max_pairs_per_frame=max_pairs_per_frame)
    if pair_diffs.size == 0:
        raise ValueError("pairwise logistic selector training requires at least one ordered candidate pair")
    weights = np.zeros(x_norm.shape[1], dtype=np.float64)
    pair_count = float(pair_diffs.shape[0])
    for _ in range(int(iterations)):
        margins = np.clip(pair_diffs @ weights, -60.0, 60.0)
        # d/dw log(1 + exp(-margin)) = -diff * sigmoid(-margin)
        loss_weights = 1.0 / (1.0 + np.exp(margins))
        gradient = -(pair_diffs.T @ loss_weights) / pair_count
        gradient += float(l2) * weights
        weights -= float(learning_rate) * gradient
    return weights


def _pairwise_training_diffs(
    rows: list[dict[str, object]],
    x_norm: np.ndarray,
    *,
    max_pairs_per_frame: int,
) -> np.ndarray:
    indices_by_frame: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_frame[str(row["frame_name"])].append(index)
    scores = np.asarray([float(row["rfs_score"]) for row in rows], dtype=np.float64)
    diffs: list[np.ndarray] = []
    for indices in indices_by_frame.values():
        ordered_pairs: list[tuple[float, int, int]] = []
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                delta = float(scores[left_index] - scores[right_index])
                if abs(delta) <= 1e-9:
                    continue
                better, worse = (left_index, right_index) if delta > 0.0 else (right_index, left_index)
                ordered_pairs.append((abs(delta), better, worse))
        ordered_pairs.sort(reverse=True)
        if len(ordered_pairs) > max_pairs_per_frame:
            if max_pairs_per_frame == 1:
                ordered_pairs = [ordered_pairs[0]]
            else:
                positions = np.linspace(0, len(ordered_pairs) - 1, num=max_pairs_per_frame)
                ordered_pairs = [ordered_pairs[int(round(position))] for position in positions]
        diffs.extend(x_norm[better] - x_norm[worse] for _delta, better, worse in ordered_pairs)
    return np.asarray(diffs, dtype=np.float64) if diffs else np.zeros((0, x_norm.shape[1]), dtype=np.float64)


def _fit_selector_route_policy(
    rows: list[dict[str, object]],
    *,
    primary_selector: WodPreferenceRanker,
    primary_target: str,
    alternate_targets: tuple[str, ...],
    router: str,
    ridge: float,
    feature_mode: str,
    model_family: str,
    rff_dim: int,
    rff_scale: float,
    rff_seed: int,
    listwise_iterations: int,
    listwise_lr: float,
    listwise_temperature: float,
    pairwise_iterations: int,
    pairwise_lr: float,
    pairwise_l2: float,
    pairwise_max_pairs_per_frame: int,
    stump_iterations: int,
    stump_lr: float,
    stump_thresholds: int,
    knn_k: int,
    knn_temperature: float,
) -> tuple[dict[str, WodPreferenceRanker] | None, dict[str, object] | None]:
    targets = [primary_target, *[target for target in alternate_targets if target != primary_target]]
    if router == "off" or len(targets) <= 1:
        return None, None
    if router not in {"intent", "speed", "speed_fine", "intent_speed"}:
        raise ValueError(f"unsupported selector route router: {router}")
    selectors: dict[str, WodPreferenceRanker] = {primary_target: primary_selector}
    for target_index, target in enumerate(targets[1:], start=1):
        selectors[target] = _fit_selector(
            rows,
            ridge=ridge,
            target_mode=target,
            feature_mode=feature_mode,
            model_family=model_family,
            rff_dim=rff_dim,
            rff_scale=rff_scale,
            rff_seed=rff_seed + 7919 * target_index,
            listwise_iterations=listwise_iterations,
            listwise_lr=listwise_lr,
            listwise_temperature=listwise_temperature,
            pairwise_iterations=pairwise_iterations,
            pairwise_lr=pairwise_lr,
            pairwise_l2=pairwise_l2,
            pairwise_max_pairs_per_frame=pairwise_max_pairs_per_frame,
            stump_iterations=stump_iterations,
            stump_lr=stump_lr,
            stump_thresholds=stump_thresholds,
            knn_k=knn_k,
            knn_temperature=knn_temperature,
        )
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_fallback_router_key(row, router)].append(row)
    routes: dict[str, str] = {}
    route_scores: dict[str, dict[str, float]] = {}
    global_scores = {
        target: _selector_route_train_mean(rows, selector)
        for target, selector in selectors.items()
    }
    global_target = max(global_scores, key=lambda target: (global_scores[target], -targets.index(target)))
    for key, group_rows in groups.items():
        scores = {
            target: _selector_route_train_mean(group_rows, selector)
            for target, selector in selectors.items()
        }
        routes[key] = max(scores, key=lambda target: (scores[target], -targets.index(target)))
        route_scores[key] = {target: float(score) for target, score in scores.items()}
    policy = {
        "router": router,
        "primary_target": primary_target,
        "targets": targets,
        "global_target": global_target,
        "global_scores": {target: float(score) for target, score in global_scores.items()},
        "routes": routes,
        "route_scores": route_scores,
    }
    return selectors, policy


def _selector_route_train_mean(rows: list[dict[str, object]], selector: WodPreferenceRanker) -> float:
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    scores = [
        float(selector.select_row(frame_rows)["rfs_score"])
        for frame_rows in rows_by_frame.values()
        if frame_rows
    ]
    return _mean(scores) if scores else float("-inf")


def _selector_for_route(
    primary_selector: WodPreferenceRanker,
    selectors: dict[str, WodPreferenceRanker] | None,
    policy: dict[str, object] | None,
    row: dict[str, object],
) -> WodPreferenceRanker:
    if not selectors or not policy:
        return primary_selector
    router = str(policy.get("router", "off"))
    if router == "off":
        return primary_selector
    route_key = _fallback_router_key(row, router)
    target = dict(policy.get("routes", {})).get(route_key, policy.get("global_target"))
    if target is None:
        return primary_selector
    return selectors.get(str(target), primary_selector)


def _fit_boosted_stumps(
    x: np.ndarray,
    y: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    max_thresholds: int,
) -> tuple[float, list[dict[str, float]]]:
    prediction = np.full(y.shape, float(np.mean(y)), dtype=np.float64)
    stumps: list[dict[str, float]] = []
    for _ in range(int(iterations)):
        residual = y - prediction
        stump = _fit_regression_stump(x, residual, max_thresholds=max_thresholds)
        if stump is None:
            break
        update = np.where(
            x[:, int(stump["feature_index"])] <= float(stump["threshold"]),
            float(stump["left_value"]),
            float(stump["right_value"]),
        )
        prediction += float(learning_rate) * update
        stumps.append(stump)
    return float(np.mean(y)), stumps


def _fit_regression_stump(
    x: np.ndarray,
    residual: np.ndarray,
    *,
    max_thresholds: int,
) -> dict[str, float] | None:
    best: dict[str, float] | None = None
    best_loss = float("inf")
    for feature_index in range(x.shape[1]):
        values = x[:, feature_index]
        if float(np.max(values) - np.min(values)) < 1e-8:
            continue
        quantiles = np.linspace(0.0, 1.0, num=int(max_thresholds) + 2, dtype=np.float64)[1:-1]
        thresholds = np.unique(np.quantile(values, quantiles))
        for threshold in thresholds:
            left = values <= threshold
            if not np.any(left) or np.all(left):
                continue
            left_value = float(np.mean(residual[left]))
            right_value = float(np.mean(residual[~left]))
            fitted = np.where(left, left_value, right_value)
            loss = float(np.mean((residual - fitted) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = {
                    "feature_index": float(feature_index),
                    "threshold": float(threshold),
                    "left_value": left_value,
                    "right_value": right_value,
                }
    return best


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = float(np.sum(exp_values))
    if total <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.shape[0]), dtype=np.float64)
    return exp_values / total


def _fit_pairwise_selector_weights(rows: list[dict[str, object]], x_norm: np.ndarray, *, ridge: float) -> np.ndarray:
    indices_by_frame: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        indices_by_frame[str(row["frame_name"])].append(index)
    diffs: list[np.ndarray] = []
    targets: list[float] = []
    scores = np.asarray([float(row["rfs_score"]) for row in rows], dtype=np.float64)
    for indices in indices_by_frame.values():
        for left_position, left_index in enumerate(indices):
            for right_index in indices[left_position + 1 :]:
                diffs.append(x_norm[left_index] - x_norm[right_index])
                targets.append(float(scores[left_index] - scores[right_index]))
    if not diffs:
        raise ValueError("pairwise selector training requires at least one same-frame candidate pair")
    design = np.asarray(diffs, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    return np.linalg.solve(design.T @ design + penalty, design.T @ y)


def _fit_source_guard(
    rows: list[dict[str, object]],
    *,
    mode: str,
    margin: float,
) -> dict[tuple[str, ...], set[str]] | None:
    if mode == "off":
        return None
    if mode not in {"intent", "speed", "intent_speed"}:
        raise ValueError(f"unsupported selector source guard: {mode}")
    best_by_frame_source: dict[tuple[tuple[str, ...], str, str], float] = {}
    for row in rows:
        key = (_source_guard_key(row, mode), str(row["frame_name"]), str(row["source"]))
        best_by_frame_source[key] = max(float(row["rfs_score"]), best_by_frame_source.get(key, float("-inf")))
    scores_by_group_source: dict[tuple[tuple[str, ...], str], list[float]] = defaultdict(list)
    for (group, _frame_name, source), score in best_by_frame_source.items():
        scores_by_group_source[(group, source)].append(score)
    means_by_group: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for (group, source), scores in scores_by_group_source.items():
        means_by_group[group][source] = _mean(scores)
    guard: dict[tuple[str, ...], set[str]] = {}
    for group, means in means_by_group.items():
        best = max(means.values())
        guard[group] = {source for source, mean in means.items() if mean >= best - float(margin)}
    return guard


def _fit_source_calibration(
    rows: list[dict[str, object]],
    *,
    mode: str,
    scale: float = 1.0,
) -> dict[str, dict[str, float]] | None:
    if mode == "off":
        return None
    if mode not in {"intent", "speed", "intent_speed"}:
        raise ValueError(f"unsupported selector source calibration: {mode}")
    if scale < 0.0:
        raise ValueError("--selector-source-calibration-scale must be non-negative")
    best_by_frame_source: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (_fallback_router_key(row, mode), str(row["frame_name"]), str(row["source"]))
        best_by_frame_source[key] = max(float(row["rfs_score"]), best_by_frame_source.get(key, float("-inf")))
    scores_by_group_source: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (group, _frame_name, source), score in best_by_frame_source.items():
        scores_by_group_source[(group, source)].append(score)
    means_by_group: dict[str, dict[str, float]] = defaultdict(dict)
    for (group, source), scores in scores_by_group_source.items():
        means_by_group[group][source] = _mean(scores)
    calibration: dict[str, dict[str, float]] = {}
    for group, source_means in means_by_group.items():
        center = _mean(list(source_means.values()))
        calibration[group] = {
            source: float((source_mean - center) * scale)
            for source, source_mean in source_means.items()
        }
    return calibration


def _fit_source_policy(rows: list[dict[str, object]], *, mode: str) -> dict[str, object] | None:
    if mode == "off":
        return None
    if mode not in {"oracle_source", "source_score_speed", "source_score_intent_speed"}:
        raise ValueError(f"unsupported selector source policy: {mode}")
    router = "intent_speed" if mode == "source_score_intent_speed" else "speed"
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    source_wins_by_route: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    source_scores: dict[str, float] = defaultdict(float)
    source_counts: dict[str, int] = defaultdict(int)
    source_scores_by_route: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    source_counts_by_route: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for frame_rows in rows_by_frame.values():
        oracle = max(frame_rows, key=lambda row: float(row["rfs_score"]))
        route = _source_policy_route_key(frame_rows, {"router": router})
        oracle_source = str(oracle["source"])
        source_wins_by_route[route][oracle_source] += 1
        for source, source_row in _best_rows_by_source(frame_rows).items():
            score = float(source_row["rfs_score"])
            source_scores[source] += score
            source_counts[source] += 1
            source_scores_by_route[route][source] += score
            source_counts_by_route[route][source] += 1
    global_source_scores = {
        source: float(source_scores[source] / source_counts[source])
        for source in source_scores
        if source_counts[source]
    }
    routes = {}
    if mode == "oracle_source":
        for route, counts_by_source in source_wins_by_route.items():
            total = sum(counts_by_source.values())
            routes[route] = {
                source: float(count / total)
                for source, count in counts_by_source.items()
                if total
            }
    else:
        for route, scores_by_source in source_scores_by_route.items():
            routes[route] = {
                source: float(score / source_counts_by_route[route][source])
                for source, score in scores_by_source.items()
                if source_counts_by_route[route][source]
            }
    return {
        "mode": mode,
        "router": router,
        "routes": routes,
        "global_source_scores": global_source_scores,
    }


def _source_policy_route_key(rows: list[dict[str, object]], source_policy: dict[str, object]) -> str:
    if not rows:
        return "__all__"
    router = str(source_policy.get("router", "speed"))
    if router == "off":
        return "__all__"
    return _fallback_router_key(rows[0], router)


def _fit_family_calibration(
    rows: list[dict[str, object]],
    *,
    mode: str,
    min_count: int,
) -> dict[str, object] | None:
    if mode == "off":
        return None
    if mode != "speed_source_family":
        raise ValueError(f"unsupported selector family calibration: {mode}")
    if min_count <= 0:
        raise ValueError("--selector-family-calibration-min-count must be positive")
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    gains_by_key: dict[str, list[float]] = defaultdict(list)
    gains_by_source: dict[str, list[float]] = defaultdict(list)
    for frame_rows in rows_by_frame.values():
        frame_mean = _mean([float(row["rfs_score"]) for row in frame_rows])
        for row in frame_rows:
            gain = float(row["rfs_score"]) - frame_mean
            key = _family_calibration_key(row)
            gains_by_key[key].append(gain)
            gains_by_source[str(row["source"])].append(gain)
    source_offsets = {
        source: float(np.clip(_mean(values), -1.0, 1.0))
        for source, values in gains_by_source.items()
        if values
    }
    offsets = {
        key: float(np.clip(_mean(values), -1.0, 1.0))
        for key, values in gains_by_key.items()
        if len(values) >= min_count
    }
    counts = {key: len(values) for key, values in gains_by_key.items()}
    return {
        "mode": mode,
        "min_count": int(min_count),
        "offsets": offsets,
        "source_offsets": source_offsets,
        "counts": counts,
    }


def _family_calibration_key(row: dict[str, object]) -> str:
    return f"{_fallback_router_key(row, 'speed')}|{row['source']}|{_candidate_family(row)}"


def _family_calibration_offset(row: dict[str, object], family_calibration: dict[str, object]) -> float:
    key = _family_calibration_key(row)
    offsets = dict(family_calibration.get("offsets", {}))
    if key in offsets:
        return float(offsets[key])
    source_offsets = dict(family_calibration.get("source_offsets", {}))
    return float(source_offsets.get(str(row["source"]), 0.0))


SAFETY_UTILITY_FEATURES = [
    "selector_score",
    "selector_margin_to_best",
    "selector_margin_to_best_kinematic",
    "hard_geometry_reject",
    "source_is_kinematic",
    "source_is_learned",
    "speed",
    "intent",
    "max_speed",
    "final_speed",
    "max_accel",
    "progress_ratio",
    "stop_distance_error",
    "reverse_distance",
    "monotonic_forward_rate",
    "lateral_to_progress_ratio",
    "curvature_per_meter",
    "final_speed_ratio",
]


def _fit_safety_utility_policy(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    mode: str,
    ridge: float,
    safety_floor: float,
    min_risk_margin: float,
    max_utility_drop: float,
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None,
) -> dict[str, object] | None:
    if mode == "off":
        return None
    if mode == "safety_filter":
        return {"mode": "safety_filter", "hard_filter": "catastrophic_geometry"}
    if mode != "safety_utility":
        raise ValueError(f"unsupported selector postprocess: {mode}")
    if ridge <= 0.0:
        raise ValueError("--safety-utility-ridge must be positive")
    x, risk_y, utility_y = _safety_utility_training_matrix(
        rows,
        selector,
        safety_floor=float(safety_floor),
        source_calibration=source_calibration,
        family_calibration=family_calibration,
    )
    if x.size == 0:
        return None
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean) / scale
    penalty = np.eye(x_norm.shape[1], dtype=np.float64) * float(ridge)
    risk_weights = np.linalg.solve(x_norm.T @ x_norm + penalty, x_norm.T @ (risk_y - risk_y.mean()))
    utility_weights = np.linalg.solve(
        x_norm.T @ x_norm + penalty,
        x_norm.T @ (utility_y - utility_y.mean()),
    )
    return {
        "mode": "safety_utility",
        "feature_names": list(SAFETY_UTILITY_FEATURES),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "risk_bias": float(risk_y.mean()),
        "risk_weights": risk_weights.tolist(),
        "utility_bias": float(utility_y.mean()),
        "utility_weights": utility_weights.tolist(),
        "safety_floor": float(safety_floor),
        "min_risk_margin": float(min_risk_margin),
        "max_utility_drop": float(max_utility_drop),
        "hard_filter": "catastrophic_geometry",
    }


def _safety_utility_training_matrix(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    safety_floor: float,
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    features: list[list[float]] = []
    risk_targets: list[float] = []
    utility_targets: list[float] = []
    for frame_rows in rows_by_frame.values():
        context = _safety_utility_context(
            frame_rows,
            selector,
            source_calibration=source_calibration,
            family_calibration=family_calibration,
        )
        for row in frame_rows:
            features.append(_safety_utility_features(row, context))
            hard_penalty = 2.0 if _safety_hard_reject(row) else 0.0
            score = float(row["rfs_score"])
            risk_targets.append(max(0.0, float(safety_floor) - score) + hard_penalty)
            utility_targets.append(score)
    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(risk_targets, dtype=np.float64),
        np.asarray(utility_targets, dtype=np.float64),
    )


def _apply_safety_utility_policy(
    policy: dict[str, object] | None,
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    selected: dict[str, object],
    *,
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None,
) -> dict[str, object]:
    if not policy:
        return selected
    if policy.get("mode") == "safety_filter":
        return _apply_safety_filter_policy(
            selector,
            rows,
            selected,
            source_calibration=source_calibration,
            family_calibration=family_calibration,
        )
    if policy.get("mode") != "safety_utility":
        raise ValueError(f"unsupported selector postprocess: {policy.get('mode')}")
    safe_rows = [row for row in rows if not _safety_hard_reject(row)]
    eligible_rows = safe_rows or [row for row in rows if str(row["source"]) == "kinematic"] or rows
    context = _safety_utility_context(
        eligible_rows,
        selector,
        source_calibration=source_calibration,
        family_calibration=family_calibration,
    )
    candidate = min(
        eligible_rows,
        key=lambda row: (
            _safety_utility_predict(policy, row, context, target="risk"),
            -_safety_utility_predict(policy, row, context, target="utility"),
            -_calibrated_score(selector, row, source_calibration, family_calibration),
            int(row.get("candidate_index", 0)),
            str(row["candidate_name"]),
        ),
    )
    if _safety_hard_reject(selected):
        return candidate
    if selected not in eligible_rows:
        return candidate
    selected_risk = _safety_utility_predict(policy, selected, context, target="risk")
    candidate_risk = _safety_utility_predict(policy, candidate, context, target="risk")
    selected_utility = _safety_utility_predict(policy, selected, context, target="utility")
    candidate_utility = _safety_utility_predict(policy, candidate, context, target="utility")
    min_risk_margin = float(policy.get("min_risk_margin", 0.0))
    max_utility_drop = float(policy.get("max_utility_drop", float("inf")))
    if candidate_risk <= selected_risk - min_risk_margin and candidate_utility >= selected_utility - max_utility_drop:
        return candidate
    return selected


def _apply_safety_filter_policy(
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    selected: dict[str, object],
    *,
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None,
) -> dict[str, object]:
    if not _safety_hard_reject(selected):
        return selected
    safe_rows = [row for row in rows if not _safety_hard_reject(row)]
    eligible_rows = safe_rows or [row for row in rows if str(row["source"]) == "kinematic"] or rows
    return _select_by_calibrated_score(
        selector,
        eligible_rows,
        source_calibration,
        family_calibration,
    )


def _safety_utility_context(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None,
) -> dict[str, object]:
    scores = [
        _calibrated_score(selector, row, source_calibration, family_calibration)
        for row in rows
    ]
    best_score = max(scores) if scores else 0.0
    kinematic_scores = [
        score
        for row, score in zip(rows, scores)
        if str(row["source"]) == "kinematic"
    ]
    best_kinematic_score = max(kinematic_scores) if kinematic_scores else best_score
    return {
        "best_score": float(best_score),
        "best_kinematic_score": float(best_kinematic_score),
        "scores_by_row_id": {id(row): float(score) for row, score in zip(rows, scores)},
    }


def _safety_utility_features(row: dict[str, object], context: dict[str, object]) -> list[float]:
    features = row.get("features", {})
    if not isinstance(features, dict):
        features = {}
    scores_by_row_id = context.get("scores_by_row_id", {})
    selector_score = float(dict(scores_by_row_id).get(id(row), context.get("best_score", 0.0)))
    speed = max(0.0, float(features.get("init_speed_mps", 0.0)))
    expected_progress = max(1.0, float(features.get("expected_progress_5s", speed * 5.0)))
    progress_ratio = float(features.get("progress_ratio_5s", float(features.get("x_5s", 0.0)) / expected_progress))
    source = str(row["source"])
    return [
        selector_score,
        selector_score - float(context.get("best_score", selector_score)),
        selector_score - float(context.get("best_kinematic_score", selector_score)),
        1.0 if _safety_hard_reject(row) else 0.0,
        1.0 if source == "kinematic" else 0.0,
        1.0 if source == "learned" else 0.0,
        speed,
        float(features.get("intent", 1.0)),
        max(0.0, float(features.get("max_speed_mps", 0.0))),
        max(0.0, float(features.get("final_speed_mps", 0.0))),
        max(0.0, float(features.get("max_abs_accel_mps2", 0.0))),
        progress_ratio,
        max(0.0, float(features.get("stop_distance_error", 0.0))),
        max(0.0, float(features.get("reverse_distance", 0.0))),
        float(features.get("monotonic_forward_rate", 1.0)),
        max(0.0, float(features.get("lateral_to_progress_ratio", 0.0))),
        max(0.0, float(features.get("curvature_per_meter", 0.0))),
        max(0.0, float(features.get("final_speed_ratio", 0.0))),
    ]


def _safety_utility_predict(
    policy: dict[str, object],
    row: dict[str, object],
    context: dict[str, object],
    *,
    target: str,
) -> float:
    values = np.asarray(_safety_utility_features(row, context), dtype=np.float64)
    mean = np.asarray(policy["feature_mean"], dtype=np.float64)
    scale = np.asarray(policy["feature_scale"], dtype=np.float64)
    x_norm = (values - mean) / scale
    if target == "risk":
        weights = np.asarray(policy["risk_weights"], dtype=np.float64)
        return float(policy["risk_bias"]) + float(np.dot(x_norm, weights))
    if target == "utility":
        weights = np.asarray(policy["utility_weights"], dtype=np.float64)
        return float(policy["utility_bias"]) + float(np.dot(x_norm, weights))
    raise ValueError(f"unsupported safety utility target: {target}")


def _safety_hard_reject(row: dict[str, object]) -> bool:
    return _catastrophic_motion_reject(row)


def _catastrophic_motion_reject(row: dict[str, object]) -> bool:
    source = str(row["source"])
    if source == "kinematic":
        return False
    features = row.get("features", {})
    if not isinstance(features, dict):
        return False

    speed = max(0.0, float(features.get("init_speed_mps", 0.0)))
    max_speed = max(0.0, float(features.get("max_speed_mps", 0.0)))
    mean_accel = max(0.0, float(features.get("mean_abs_accel_mps2", 0.0)))
    max_accel = max(0.0, float(features.get("max_abs_accel_mps2", 0.0)))
    progress = float(features.get("x_5s", 0.0))
    x_1s = float(features.get("x_1s", progress))
    x_3s = float(features.get("x_3s", progress))
    reverse_distance = max(0.0, float(features.get("reverse_distance", 0.0)))
    monotonic_forward_rate = float(features.get("monotonic_forward_rate", 1.0))
    lateral_to_progress_ratio = max(0.0, float(features.get("lateral_to_progress_ratio", 0.0)))
    curvature_per_meter = max(0.0, float(features.get("curvature_per_meter", 0.0)))

    if reverse_distance > max(15.0, speed * 3.0):
        return True
    if min(x_1s, x_3s, progress) < -max(8.0, speed * 2.0):
        return True
    if monotonic_forward_rate < 0.2:
        return True
    if max_accel > 40.0 or mean_accel > 20.0:
        return True
    if max_speed > max(45.0, speed * 5.0 + 15.0):
        return True
    if curvature_per_meter > 6.0:
        return True
    if progress > 3.0 and lateral_to_progress_ratio > 8.0:
        return True
    return False


def _selector_training_rows_for_gates(
    rows: list[dict[str, object]],
    *,
    scene_gate: str,
    source_gate: str,
    source_gate_sources: tuple[str, ...],
) -> list[dict[str, object]]:
    held_out_sources = set()
    if scene_gate != "off":
        held_out_sources.add("scene")
    if source_gate == "independent_train_margin":
        held_out_sources.update(str(source) for source in source_gate_sources)
    if not held_out_sources:
        return rows
    train_rows = [
        row
        for row in rows
        if str(row["source"]) not in held_out_sources
    ]
    return train_rows or rows


def _fit_fallback_selectors(
    rows: list[dict[str, object]],
    *,
    fallback_sources: tuple[str, ...],
    source_options: tuple[tuple[str, ...], ...],
    ridge: float,
    target_mode: str,
    feature_mode: str,
    model_family: str,
    rff_dim: int,
    rff_scale: float,
    rff_seed: int,
    listwise_iterations: int,
    listwise_lr: float,
    listwise_temperature: float,
    pairwise_iterations: int,
    pairwise_lr: float,
    pairwise_l2: float,
    pairwise_max_pairs_per_frame: int,
    stump_iterations: int = 80,
    stump_lr: float = 0.08,
    stump_thresholds: int = 16,
    knn_k: int = 16,
    knn_temperature: float = 4.0,
) -> dict[tuple[str, ...], object]:
    selectors: dict[tuple[str, ...], object] = {}
    for sources in {tuple(fallback_sources), *source_options}:
        source_set = set(sources)
        source_rows = [row for row in rows if str(row["source"]) in source_set]
        if source_rows:
            selectors[tuple(sources)] = _fit_selector(
                source_rows,
                ridge=ridge,
                target_mode=target_mode,
                feature_mode=feature_mode,
                model_family=model_family,
                rff_dim=rff_dim,
                rff_scale=rff_scale,
                rff_seed=rff_seed,
                listwise_iterations=listwise_iterations,
                listwise_lr=listwise_lr,
                listwise_temperature=listwise_temperature,
                pairwise_iterations=pairwise_iterations,
                pairwise_lr=pairwise_lr,
                pairwise_l2=pairwise_l2,
                pairwise_max_pairs_per_frame=pairwise_max_pairs_per_frame,
                stump_iterations=stump_iterations,
                stump_lr=stump_lr,
                stump_thresholds=stump_thresholds,
                knn_k=knn_k,
                knn_temperature=knn_temperature,
            )
    return selectors


def _fit_kinematic_fallback_threshold(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    mode: str,
    fallback_sources: tuple[str, ...] = ("kinematic",),
    source_calibration: dict[str, dict[str, float]] | None = None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
) -> float | None:
    if mode == "off":
        return None
    if mode != "train_margin":
        raise ValueError(f"unsupported selector kinematic fallback: {mode}")
    observations: list[tuple[float, float, float]] = []
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    for frame_rows in rows_by_frame.values():
        selected = _select_by_calibrated_score(selector, frame_rows, source_calibration)
        fallback_rows = [row for row in frame_rows if str(row["source"]) in fallback_sources]
        if not fallback_rows:
            continue
        fallback_selector = _fallback_selector_for_sources(fallback_selectors, fallback_sources) or selector
        fallback = _select_by_calibrated_score(fallback_selector, fallback_rows, source_calibration)
        margin = _calibrated_score(selector, selected, source_calibration) - _calibrated_score(
            selector,
            fallback,
            source_calibration,
        )
        observations.append((margin, float(selected["rfs_score"]), float(fallback["rfs_score"])))
    if not observations:
        return None
    thresholds = [float("-inf"), *sorted({margin for margin, _selected, _fallback in observations}), float("inf")]
    best_threshold = float("-inf")
    best_mean = float("-inf")
    for threshold in thresholds:
        scores = [
            fallback_score if margin < threshold else selected_score
            for margin, selected_score, fallback_score in observations
        ]
        mean_score = _mean(scores)
        if mean_score > best_mean:
            best_mean = mean_score
            best_threshold = threshold
    return best_threshold


def _fit_fallback_policy(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    mode: str,
    fallback_sources: tuple[str, ...],
    router: str,
    source_options: tuple[tuple[str, ...], ...],
    source_calibration: dict[str, dict[str, float]] | None = None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
) -> dict[str, object] | None:
    if mode == "off":
        return None
    if router == "off":
        threshold = _fit_kinematic_fallback_threshold(
            rows,
            selector,
            mode=mode,
            fallback_sources=fallback_sources,
            source_calibration=source_calibration,
            fallback_selectors=fallback_selectors,
        )
        return {
            "router": "off",
            **_fallback_route_payload(threshold, fallback_sources),
        }
    if router not in {"intent", "speed", "speed_fine", "intent_speed"}:
        raise ValueError(f"unsupported selector fallback router: {router}")
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[_fallback_router_key(row, router)].append(row)
    routes: dict[str, dict[str, object]] = {}
    for key, group_rows in groups.items():
        best_route: dict[str, object] | None = None
        best_mean = float("-inf")
        for sources in source_options:
            threshold = _fit_kinematic_fallback_threshold(
                group_rows,
                selector,
                mode=mode,
                fallback_sources=sources,
                source_calibration=source_calibration,
                fallback_selectors=fallback_selectors,
            )
            route = _fallback_route_payload(threshold, sources)
            mean_score = _fallback_policy_train_mean(
                group_rows,
                selector,
                route,
                source_calibration,
                fallback_selectors,
            )
            if mean_score > best_mean:
                best_mean = mean_score
                best_route = route
        if best_route is not None:
            routes[key] = best_route
    return {"router": router, "routes": routes}


def _fallback_policy_train_mean(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    route: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
) -> float:
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    return _mean(
        [
            float(
                _select_with_policy(
                    selector,
                    frame_rows,
                    source_guard=None,
                    fallback_policy={"router": "off", **route},
                    source_calibration=source_calibration,
                    fallback_selectors=fallback_selectors,
                )["rfs_score"]
            )
            for frame_rows in rows_by_frame.values()
        ]
    )


def _fit_scene_gate_policy(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    mode: str,
    margin: float,
    router: str,
    ridge: float,
    max_rate: float,
    source_calibration: dict[str, dict[str, float]] | None,
    fallback_policy: dict[str, object] | None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
) -> dict[str, object] | None:
    if mode == "off":
        return None
    if mode != "train_margin":
        raise ValueError(f"unsupported scene gate: {mode}")
    if router not in {"off", "intent", "speed", "speed_fine", "intent_speed"}:
        raise ValueError(f"unsupported scene gate router: {router}")
    if ridge <= 0.0:
        raise ValueError("--scene-gate-ridge must be positive")
    if max_rate <= 0.0 or max_rate > 1.0:
        raise ValueError("--scene-gate-max-rate must be in (0, 1]")

    observations: list[dict[str, object]] = []
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    for frame_rows in rows_by_frame.values():
        scene_rows = [row for row in frame_rows if str(row["source"]) == "scene"]
        non_scene_rows = [row for row in frame_rows if str(row["source"]) != "scene"]
        if not scene_rows or not non_scene_rows:
            continue
        baseline = _select_with_policy(
            selector,
            non_scene_rows,
            source_guard=None,
            fallback_policy=fallback_policy,
            source_calibration=source_calibration,
            fallback_selectors=fallback_selectors,
        )
        scene = _select_by_calibrated_score(selector, scene_rows, source_calibration)
        gain = float(scene["rfs_score"]) - float(baseline["rfs_score"])
        observations.append(
            {
                "features": _scene_gate_features(selector, baseline, scene, source_calibration),
                "gain": gain,
                "baseline_score": float(baseline["rfs_score"]),
                "scene_score": float(scene["rfs_score"]),
                "frame_name": str(baseline["frame_name"]),
                "group": "__all__" if router == "off" else _fallback_router_key(baseline, router),
            }
        )
    if not observations:
        return {
            "mode": mode,
            "router": router,
            "decision": "never",
            "threshold": None,
            "routes": {},
            "train_frames": 0,
        }
    mean, scale, weights, _predicted_gains = _fit_gate_linear_model(observations, ridge=ridge)
    threshold_gains = _crossfit_gate_predictions(observations, ridge=ridge)

    routes: dict[str, dict[str, object]] = {}
    for group in sorted({str(observation["group"]) for observation in observations}):
        indices = [index for index, observation in enumerate(observations) if str(observation["group"]) == group]
        routes[group] = _scene_gate_route_payload(
            _fit_scene_gate_threshold(
                [observations[index] for index in indices],
                threshold_gains[indices],
                margin=margin,
                max_rate=max_rate,
            )
        )
    global_threshold = _fit_scene_gate_threshold(observations, threshold_gains, margin=margin, max_rate=max_rate)
    return {
        "mode": mode,
        "router": router,
        "margin": float(margin),
        "max_rate": float(max_rate),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights[1:].tolist(),
        "bias": float(weights[0]),
        "threshold": None if global_threshold is None else float(global_threshold),
        "decision": _scene_gate_route_payload(global_threshold)["decision"],
        "routes": {} if router == "off" else routes,
        "train_frames": len(observations),
        "train_positive_rate": float(np.mean([float(observation["gain"]) > 0.0 for observation in observations])),
        "threshold_fit": "segment_crossfit",
    }


def _fit_source_gate_policy(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    mode: str,
    sources: tuple[str, ...],
    candidate_prefixes: tuple[str, ...],
    deny_prefixes: tuple[str, ...] = (),
    route_allowlist: tuple[str, ...],
    margin: float,
    router: str,
    ridge: float,
    max_rate: float,
    min_precision: float,
    min_route_observations: int,
    min_route_positives: int,
    source_calibration: dict[str, dict[str, float]] | None,
    fallback_policy: dict[str, object] | None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
    source_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
) -> dict[str, object] | None:
    if mode == "off":
        return None
    if mode not in {"train_margin", "independent_train_margin"}:
        raise ValueError(f"unsupported source gate: {mode}")
    if router not in {"off", "intent", "speed", "speed_fine", "intent_speed"}:
        raise ValueError(f"unsupported source gate router: {router}")
    if ridge <= 0.0:
        raise ValueError("--source-gate-ridge must be positive")
    if max_rate <= 0.0 or max_rate > 1.0:
        raise ValueError("--source-gate-max-rate must be in (0, 1]")
    if min_precision < 0.0 or min_precision > 1.0:
        raise ValueError("--source-gate-min-precision must be in [0, 1]")
    if min_route_observations < 0:
        raise ValueError("--source-gate-min-route-observations must be non-negative")
    if min_route_positives < 0:
        raise ValueError("--source-gate-min-route-positives must be non-negative")
    source_names = tuple(dict.fromkeys(str(source) for source in sources))
    if not source_names:
        raise ValueError("at least one source-gate source is required")
    prefixes = tuple(str(prefix) for prefix in candidate_prefixes)
    denied_prefixes = tuple(str(prefix) for prefix in deny_prefixes)
    allowed_routes = frozenset(str(route) for route in route_allowlist)

    observations: list[dict[str, object]] = []
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    for frame_rows in rows_by_frame.values():
        baseline_rows = [row for row in frame_rows if str(row["source"]) != "scene"]
        if mode == "independent_train_margin":
            held_out_sources = set(source_names)
            independent_baseline_rows = [
                row
                for row in baseline_rows
                if str(row["source"]) not in held_out_sources
            ]
            if independent_baseline_rows:
                baseline_rows = independent_baseline_rows
        if not baseline_rows:
            baseline_rows = frame_rows
        baseline = _select_with_policy(
            selector,
            baseline_rows,
            source_guard=None,
            fallback_policy=fallback_policy,
            source_calibration=source_calibration,
            fallback_selectors=fallback_selectors,
        )
        for source in source_names:
            source_rows = [
                row
                for row in frame_rows
                if str(row["source"]) == source
                and _candidate_prefix_allowed(row, prefixes)
                and _candidate_prefix_allowed(row, denied_prefixes, invert=True)
            ]
            if not source_rows:
                continue
            source_selector = _fallback_selector_for_sources(source_selectors, (source,)) or selector
            candidate = _select_by_calibrated_score(source_selector, source_rows, source_calibration)
            if candidate is baseline:
                continue
            gain = float(candidate["rfs_score"]) - float(baseline["rfs_score"])
            group = "__all__" if router == "off" else _fallback_router_key(baseline, router)
            if allowed_routes and group not in allowed_routes and f"{source}|{group}" not in allowed_routes:
                continue
            observations.append(
                {
                    "features": _source_gate_features(selector, baseline, candidate, source_calibration),
                    "gain": gain,
                    "baseline_score": float(baseline["rfs_score"]),
                    "source_score": float(candidate["rfs_score"]),
                    "frame_name": str(baseline["frame_name"]),
                    "source": source,
                    "group": group,
                }
            )
    if not observations:
        return {
            "mode": mode,
            "router": router,
            "sources": list(source_names),
            "candidate_prefixes": list(prefixes),
            "deny_prefixes": list(denied_prefixes),
            "route_allowlist": sorted(allowed_routes),
            "decision": "never",
            "threshold": None,
            "routes": {},
            "train_observations": 0,
        }
    if mode == "independent_train_margin":
        return _fit_independent_source_gate_policy(
            observations,
            router=router,
            source_names=source_names,
            prefixes=prefixes,
            denied_prefixes=denied_prefixes,
            allowed_routes=allowed_routes,
            margin=margin,
            max_rate=max_rate,
            min_precision=min_precision,
            min_route_observations=min_route_observations,
            min_route_positives=min_route_positives,
            ridge=ridge,
        )
    mean, scale, weights, _predicted_gains = _fit_gate_linear_model(observations, ridge=ridge)
    threshold_gains = _crossfit_gate_predictions(observations, ridge=ridge)

    routes: dict[str, dict[str, object]] = {}
    route_keys = sorted({f"{observation['source']}|{observation['group']}" for observation in observations})
    for route_key in route_keys:
        source, group = route_key.split("|", 1)
        indices = [
            index
            for index, observation in enumerate(observations)
            if str(observation["source"]) == source and str(observation["group"]) == group
        ]
        route_observations = [
            {
                **observations[index],
                "scene_score": observations[index]["source_score"],
            }
            for index in indices
        ]
        routes[route_key] = _scene_gate_route_payload(
            _fit_scene_gate_threshold(
                route_observations,
                threshold_gains[indices],
                margin=margin,
                max_rate=max_rate,
                min_precision=min_precision,
                min_observations=min_route_observations,
                min_positive_overrides=min_route_positives,
            )
        )
    global_observations = [{**observation, "scene_score": observation["source_score"]} for observation in observations]
    global_threshold = _fit_scene_gate_threshold(
        global_observations,
        threshold_gains,
        margin=margin,
        max_rate=max_rate,
        min_precision=min_precision,
        min_observations=min_route_observations,
        min_positive_overrides=min_route_positives,
    )
    return {
        "mode": mode,
        "router": router,
        "sources": list(source_names),
        "candidate_prefixes": list(prefixes),
        "deny_prefixes": list(denied_prefixes),
        "route_allowlist": sorted(allowed_routes),
        "margin": float(margin),
        "max_rate": float(max_rate),
        "min_precision": float(min_precision),
        "min_route_observations": int(min_route_observations),
        "min_route_positives": int(min_route_positives),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights[1:].tolist(),
        "bias": float(weights[0]),
        "threshold": None if global_threshold is None else float(global_threshold),
        "decision": _scene_gate_route_payload(global_threshold)["decision"],
        "routes": {} if router == "off" else routes,
        "train_observations": len(observations),
        "train_positive_rate": float(np.mean([float(observation["gain"]) > 0.0 for observation in observations])),
        "threshold_fit": "segment_crossfit",
    }


def _fit_source_veto_policy(
    rows: list[dict[str, object]],
    selector: WodPreferenceRanker,
    *,
    mode: str,
    sources: tuple[str, ...],
    fallback_sources: tuple[str, ...],
    router: str,
    ridge: float,
    max_rate: float,
    min_precision: float,
    min_route_observations: int,
    min_route_positives: int,
    source_calibration: dict[str, dict[str, float]] | None,
    fallback_policy: dict[str, object] | None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
    scene_gate_policy: dict[str, object] | None,
    source_gate_policy: dict[str, object] | None,
    source_gate_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
) -> dict[str, object] | None:
    if mode == "off":
        return None
    if mode != "train_margin":
        raise ValueError(f"unsupported source veto gate: {mode}")
    if router not in {"off", "intent", "speed", "speed_fine", "intent_speed"}:
        raise ValueError(f"unsupported source veto router: {router}")
    if ridge <= 0.0:
        raise ValueError("--source-veto-ridge must be positive")
    if max_rate <= 0.0 or max_rate > 1.0:
        raise ValueError("--source-veto-max-rate must be in (0, 1]")
    if min_precision < 0.0 or min_precision > 1.0:
        raise ValueError("--source-veto-min-precision must be in [0, 1]")
    if min_route_observations < 0:
        raise ValueError("--source-veto-min-route-observations must be non-negative")
    if min_route_positives < 0:
        raise ValueError("--source-veto-min-route-positives must be non-negative")
    veto_sources = tuple(dict.fromkeys(str(source) for source in sources))
    fallback_source_names = tuple(dict.fromkeys(str(source) for source in fallback_sources))
    if not veto_sources:
        raise ValueError("at least one source-veto source is required")
    if not fallback_source_names:
        raise ValueError("at least one source-veto fallback source is required")

    observations: list[dict[str, object]] = []
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    for frame_rows in rows_by_frame.values():
        selected = _select_final_for_gate_training(
            selector,
            frame_rows,
            source_calibration=source_calibration,
            fallback_policy=fallback_policy,
            fallback_selectors=fallback_selectors,
            scene_gate_policy=scene_gate_policy,
            source_gate_policy=source_gate_policy,
            source_gate_selectors=source_gate_selectors,
        )
        if str(selected["source"]) not in veto_sources:
            continue
        fallback_rows = [row for row in frame_rows if str(row["source"]) in fallback_source_names]
        if not fallback_rows:
            continue
        fallback_selector = _fallback_selector_for_sources(fallback_selectors, fallback_source_names) or selector
        fallback = _select_by_calibrated_score(fallback_selector, fallback_rows, source_calibration)
        if fallback is selected:
            continue
        observations.append(
            {
                "features": _source_gate_features(selector, selected, fallback, source_calibration),
                "gain": float(fallback["rfs_score"]) - float(selected["rfs_score"]),
                "baseline_score": float(selected["rfs_score"]),
                "scene_score": float(fallback["rfs_score"]),
                "frame_name": str(selected["frame_name"]),
                "source": str(selected["source"]),
                "group": "__all__" if router == "off" else _fallback_router_key(selected, router),
            }
        )
    if not observations:
        return {
            "mode": mode,
            "router": router,
            "sources": list(veto_sources),
            "fallback_sources": list(fallback_source_names),
            "decision": "never",
            "threshold": None,
            "routes": {},
            "train_observations": 0,
        }
    mean, scale, weights, _predicted_gains = _fit_gate_linear_model(observations, ridge=ridge)
    threshold_gains = _crossfit_gate_predictions(observations, ridge=ridge)
    routes: dict[str, dict[str, object]] = {}
    route_keys = sorted({f"{observation['source']}|{observation['group']}" for observation in observations})
    for route_key in route_keys:
        source, group = route_key.split("|", 1)
        indices = [
            index
            for index, observation in enumerate(observations)
            if str(observation["source"]) == source and str(observation["group"]) == group
        ]
        routes[route_key] = _scene_gate_route_payload(
            _fit_scene_gate_threshold(
                [observations[index] for index in indices],
                threshold_gains[indices],
                margin=0.0,
                max_rate=max_rate,
                min_precision=min_precision,
                min_observations=min_route_observations,
                min_positive_overrides=min_route_positives,
            )
        )
    global_threshold = _fit_scene_gate_threshold(
        observations,
        threshold_gains,
        margin=0.0,
        max_rate=max_rate,
        min_precision=min_precision,
        min_observations=min_route_observations,
        min_positive_overrides=min_route_positives,
    )
    return {
        "mode": mode,
        "router": router,
        "sources": list(veto_sources),
        "fallback_sources": list(fallback_source_names),
        "max_rate": float(max_rate),
        "min_precision": float(min_precision),
        "min_route_observations": int(min_route_observations),
        "min_route_positives": int(min_route_positives),
        "feature_mean": mean.tolist(),
        "feature_scale": scale.tolist(),
        "weights": weights[1:].tolist(),
        "bias": float(weights[0]),
        "threshold": None if global_threshold is None else float(global_threshold),
        "decision": _scene_gate_route_payload(global_threshold)["decision"],
        "routes": {} if router == "off" else routes,
        "train_observations": len(observations),
        "train_positive_rate": float(np.mean([float(observation["gain"]) > 0.0 for observation in observations])),
        "threshold_fit": "segment_crossfit_source_veto",
    }


def _select_final_for_gate_training(
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    *,
    source_calibration: dict[str, dict[str, float]] | None,
    fallback_policy: dict[str, object] | None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
    scene_gate_policy: dict[str, object] | None,
    source_gate_policy: dict[str, object] | None,
    source_gate_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
) -> dict[str, object]:
    policy_rows = [row for row in rows if str(row["source"]) != "scene"] if scene_gate_policy is not None else rows
    policy_rows = _source_gate_baseline_rows(source_gate_policy, policy_rows) or policy_rows or rows
    selected = _select_with_policy(
        selector,
        policy_rows,
        source_guard=None,
        fallback_policy=fallback_policy,
        source_calibration=source_calibration,
        fallback_selectors=fallback_selectors,
    )
    selected = _apply_scene_gate(scene_gate_policy, selector, rows, selected, source_calibration=source_calibration)
    return _apply_source_gate(
        source_gate_policy,
        selector,
        rows,
        selected,
        source_calibration=source_calibration,
        source_selectors=source_gate_selectors,
    )


def _fit_gate_linear_model(
    observations: list[dict[str, object]],
    *,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray([observation["features"] for observation in observations], dtype=np.float64)
    y = np.asarray([float(observation["gain"]) for observation in observations], dtype=np.float64)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x_norm = (x - mean) / scale
    design = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[0, 0] = 0.0
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, scale, weights, design @ weights


def _fit_independent_source_gate_policy(
    observations: list[dict[str, object]],
    *,
    router: str,
    source_names: tuple[str, ...],
    prefixes: tuple[str, ...],
    denied_prefixes: tuple[str, ...],
    allowed_routes: frozenset[str],
    margin: float,
    max_rate: float,
    min_precision: float,
    min_route_observations: int,
    min_route_positives: int,
    ridge: float,
) -> dict[str, object]:
    source_models: dict[str, dict[str, object]] = {}
    routes: dict[str, dict[str, object]] = {}
    global_decisions: dict[str, dict[str, object]] = {}
    train_positive_rates: dict[str, float] = {}
    train_observations_by_source: dict[str, int] = {}
    for source in source_names:
        source_observations = [
            observation
            for observation in observations
            if str(observation["source"]) == source
        ]
        if not source_observations:
            global_decisions[source] = {"decision": "never", "threshold": None}
            train_positive_rates[source] = 0.0
            train_observations_by_source[source] = 0
            continue
        mean, scale, weights, _predicted_gains = _fit_gate_linear_model(source_observations, ridge=ridge)
        predicted_gains = _crossfit_gate_predictions(source_observations, ridge=ridge)
        source_models[source] = {
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            "weights": weights[1:].tolist(),
            "bias": float(weights[0]),
        }
        train_positive_rates[source] = float(
            np.mean([float(observation["gain"]) > 0.0 for observation in source_observations])
        )
        train_observations_by_source[source] = len(source_observations)
        threshold_observations = [
            {**observation, "scene_score": observation["source_score"]}
            for observation in source_observations
        ]
        global_threshold = _fit_scene_gate_threshold(
            threshold_observations,
            predicted_gains,
            margin=margin,
            max_rate=max_rate,
            min_precision=min_precision,
            min_observations=min_route_observations,
            min_positive_overrides=min_route_positives,
        )
        global_decisions[source] = _scene_gate_route_payload(global_threshold)
        for group in sorted({str(observation["group"]) for observation in source_observations}):
            indices = [
                index
                for index, observation in enumerate(source_observations)
                if str(observation["group"]) == group
            ]
            route_observations = [
                {**source_observations[index], "scene_score": source_observations[index]["source_score"]}
                for index in indices
            ]
            route_key = f"{source}|{group}"
            routes[route_key] = _scene_gate_route_payload(
                _fit_scene_gate_threshold(
                    route_observations,
                    predicted_gains[indices],
                    margin=margin,
                    max_rate=max_rate,
                    min_precision=min_precision,
                    min_observations=min_route_observations,
                    min_positive_overrides=min_route_positives,
                )
            )
    return {
        "mode": "independent_train_margin",
        "router": router,
        "sources": list(source_names),
        "candidate_prefixes": list(prefixes),
        "deny_prefixes": list(denied_prefixes),
        "route_allowlist": sorted(allowed_routes),
        "margin": float(margin),
        "max_rate": float(max_rate),
        "min_precision": float(min_precision),
        "min_route_observations": int(min_route_observations),
        "min_route_positives": int(min_route_positives),
        "source_models": source_models,
        "source_decisions": global_decisions,
        "routes": {} if router == "off" else routes,
        "train_observations": len(observations),
        "train_observations_by_source": train_observations_by_source,
        "train_positive_rate_by_source": train_positive_rates,
        "threshold_fit": "segment_crossfit_independent_sources",
    }


def _predict_gate_linear_model(
    observations: list[dict[str, object]],
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    x = np.asarray([observation["features"] for observation in observations], dtype=np.float64)
    x_norm = (x - mean) / scale
    design = np.concatenate([np.ones((x_norm.shape[0], 1)), x_norm], axis=1)
    return design @ weights


def _crossfit_gate_predictions(
    observations: list[dict[str, object]],
    *,
    ridge: float,
) -> np.ndarray:
    segments = sorted({_segment_id(str(observation["frame_name"])) for observation in observations})
    folds = min(3, len(segments))
    if folds < 3:
        return _fit_gate_linear_model(observations, ridge=ridge)[3]

    predictions = np.zeros(len(observations), dtype=np.float64)
    for fold_index in range(folds):
        valid_segments = {
            segment
            for index, segment in enumerate(segments)
            if index % folds == fold_index
        }
        train_observations = [
            observation
            for observation in observations
            if _segment_id(str(observation["frame_name"])) not in valid_segments
        ]
        valid_indices = [
            index
            for index, observation in enumerate(observations)
            if _segment_id(str(observation["frame_name"])) in valid_segments
        ]
        if not train_observations or not valid_indices:
            return _fit_gate_linear_model(observations, ridge=ridge)[3]
        mean, scale, weights, _predicted = _fit_gate_linear_model(train_observations, ridge=ridge)
        valid_observations = [observations[index] for index in valid_indices]
        predictions[valid_indices] = _predict_gate_linear_model(
            valid_observations,
            mean=mean,
            scale=scale,
            weights=weights,
        )
    return predictions


def _fit_scene_gate_threshold(
    observations: list[dict[str, object]],
    predicted_gains: np.ndarray,
    *,
    margin: float,
    max_rate: float,
    min_precision: float = 0.0,
    min_observations: int = 0,
    min_positive_overrides: int = 0,
) -> float | None:
    if not observations:
        return None
    if len(observations) < int(min_observations):
        return None
    thresholds = [float("-inf"), *sorted({float(value) for value in predicted_gains}), float("inf")]
    best_threshold: float | None = None
    best_mean = _mean([float(observation["baseline_score"]) for observation in observations])
    for threshold in thresholds:
        adjusted_scores = []
        override_count = 0
        positive_override_count = 0
        for observation, predicted_gain in zip(observations, predicted_gains):
            if float(predicted_gain) >= threshold:
                override_count += 1
                if float(observation["scene_score"]) > float(observation["baseline_score"]):
                    positive_override_count += 1
                adjusted_scores.append(float(observation["scene_score"]) - float(margin))
            else:
                adjusted_scores.append(float(observation["baseline_score"]))
        if override_count / len(observations) > float(max_rate):
            continue
        if override_count and positive_override_count / override_count < float(min_precision):
            continue
        if positive_override_count < int(min_positive_overrides):
            continue
        mean_score = _mean(adjusted_scores)
        if mean_score > best_mean:
            best_mean = mean_score
            best_threshold = float(threshold)
    return best_threshold


def _scene_gate_route_payload(threshold: float | None) -> dict[str, object]:
    if threshold is None or threshold == float("inf"):
        return {"decision": "never", "threshold": None}
    if threshold == float("-inf"):
        return {"decision": "always", "threshold": None}
    return {"decision": "margin", "threshold": float(threshold)}


def _apply_scene_gate(
    policy: dict[str, object] | None,
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    baseline: dict[str, object],
    *,
    source_calibration: dict[str, dict[str, float]] | None,
) -> dict[str, object]:
    if policy is None:
        return baseline
    scene_rows = [row for row in rows if str(row["source"]) == "scene"]
    if not scene_rows:
        return baseline
    scene = _select_by_calibrated_score(selector, scene_rows, source_calibration)
    route = _scene_gate_route_for_row(policy, baseline)
    decision = str(route.get("decision", "never"))
    if decision == "never":
        return baseline
    if decision == "always":
        return scene
    threshold = _finite_float_or_none(route.get("threshold"))
    if threshold is None:
        return baseline
    predicted_gain = _scene_gate_predict(policy, selector, baseline, scene, source_calibration)
    return scene if predicted_gain >= threshold else baseline


def _apply_source_gate(
    policy: dict[str, object] | None,
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    baseline: dict[str, object],
    *,
    source_calibration: dict[str, dict[str, float]] | None,
    source_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
) -> dict[str, object]:
    if policy is None:
        return baseline
    prefixes = tuple(str(prefix) for prefix in policy.get("candidate_prefixes", []))
    denied_prefixes = tuple(str(prefix) for prefix in policy.get("deny_prefixes", []))
    allowed_routes = set(str(route) for route in policy.get("route_allowlist", []))
    router = str(policy.get("router", "off"))
    route_group = _fallback_router_key(baseline, router) if router != "off" else "__all__"
    best_candidate = baseline
    best_predicted_gain = float("-inf")
    for source in policy.get("sources", []):
        if allowed_routes and route_group not in allowed_routes and f"{source}|{route_group}" not in allowed_routes:
            continue
        source_rows = [
            row
            for row in rows
            if str(row["source"]) == str(source)
            and _candidate_prefix_allowed(row, prefixes)
            and _candidate_prefix_allowed(row, denied_prefixes, invert=True)
        ]
        if not source_rows:
            continue
        source_selector = _fallback_selector_for_sources(source_selectors, (str(source),)) or selector
        candidate = _select_by_calibrated_score(source_selector, source_rows, source_calibration)
        if candidate is baseline:
            continue
        route = _source_gate_route_for_candidate(policy, baseline, str(source))
        decision = str(route.get("decision", "never"))
        if decision == "never":
            continue
        predicted_gain = _source_gate_predict(policy, selector, baseline, candidate, source_calibration)
        if decision == "always":
            predicted_gain = max(predicted_gain, 0.0)
        else:
            threshold = _finite_float_or_none(route.get("threshold"))
            if threshold is None or predicted_gain < threshold:
                continue
        if predicted_gain > best_predicted_gain:
            best_predicted_gain = predicted_gain
            best_candidate = candidate
    return best_candidate


def _apply_source_veto(
    policy: dict[str, object] | None,
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    selected: dict[str, object],
    *,
    source_calibration: dict[str, dict[str, float]] | None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
) -> dict[str, object]:
    if policy is None:
        return selected
    if str(selected["source"]) not in {str(source) for source in policy.get("sources", [])}:
        return selected
    fallback_sources = tuple(str(source) for source in policy.get("fallback_sources", ("kinematic",)))
    fallback_rows = [row for row in rows if str(row["source"]) in fallback_sources]
    if not fallback_rows:
        return selected
    fallback_selector = _fallback_selector_for_sources(fallback_selectors, fallback_sources) or selector
    fallback = _select_by_calibrated_score(fallback_selector, fallback_rows, source_calibration)
    if fallback is selected:
        return selected
    route = _source_veto_route_for_row(policy, selected)
    decision = str(route.get("decision", "never"))
    if decision == "never":
        return selected
    predicted_gain = _source_veto_predict(policy, selector, selected, fallback, source_calibration)
    if decision == "always":
        return fallback
    threshold = _finite_float_or_none(route.get("threshold"))
    return fallback if threshold is not None and predicted_gain >= threshold else selected


def _source_gate_baseline_rows(
    policy: dict[str, object] | None,
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if policy is None or policy.get("mode") != "independent_train_margin":
        return rows
    held_out_sources = {str(source) for source in policy.get("sources", [])}
    baseline_rows = [
        row
        for row in rows
        if str(row["source"]) not in held_out_sources
    ]
    return baseline_rows or rows


def _candidate_prefix_allowed(row: dict[str, object], prefixes: tuple[str, ...], *, invert: bool = False) -> bool:
    if not prefixes:
        return True
    candidate_name = str(row["candidate_name"])
    matched = any(candidate_name.startswith(prefix) for prefix in prefixes)
    return not matched if invert else matched


def _source_gate_route_for_candidate(
    policy: dict[str, object],
    baseline: dict[str, object],
    source: str,
) -> dict[str, object]:
    router = str(policy.get("router", "off"))
    source_decisions = dict(policy.get("source_decisions", {}))
    if router == "off":
        source_route = source_decisions.get(source)
        if source_route is not None:
            return dict(source_route)
        return {"decision": str(policy.get("decision", "never")), "threshold": policy.get("threshold")}
    route = dict(policy.get("routes", {})).get(f"{source}|{_fallback_router_key(baseline, router)}")
    if route is not None:
        return dict(route)
    source_route = source_decisions.get(source)
    if source_route is not None:
        return dict(source_route)
    return {"decision": str(policy.get("decision", "never")), "threshold": policy.get("threshold")}


def _source_veto_route_for_row(policy: dict[str, object], selected: dict[str, object]) -> dict[str, object]:
    router = str(policy.get("router", "off"))
    if router == "off":
        return {"decision": str(policy.get("decision", "never")), "threshold": policy.get("threshold")}
    route = dict(policy.get("routes", {})).get(f"{selected['source']}|{_fallback_router_key(selected, router)}")
    if route is not None:
        return dict(route)
    return {"decision": str(policy.get("decision", "never")), "threshold": policy.get("threshold")}


def _source_veto_predict(
    policy: dict[str, object],
    selector: WodPreferenceRanker,
    selected: dict[str, object],
    fallback: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
) -> float:
    mean = np.asarray(policy.get("feature_mean", []), dtype=np.float64)
    scale = np.asarray(policy.get("feature_scale", []), dtype=np.float64)
    weights = np.concatenate(
        [
            np.asarray([float(policy.get("bias", 0.0))], dtype=np.float64),
            np.asarray(policy.get("weights", []), dtype=np.float64),
        ]
    )
    features = np.asarray(_source_gate_features(selector, selected, fallback, source_calibration), dtype=np.float64)
    if mean.size != features.size or scale.size != features.size or weights.size != features.size + 1:
        return float("-inf")
    x_norm = (features - mean) / scale
    return float(np.concatenate([np.asarray([1.0]), x_norm]) @ weights)


def _source_gate_predict(
    policy: dict[str, object],
    selector: WodPreferenceRanker,
    baseline: dict[str, object],
    candidate: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
) -> float:
    features = np.asarray(_source_gate_features(selector, baseline, candidate, source_calibration), dtype=np.float64)
    source_model = dict(policy.get("source_models", {})).get(str(candidate["source"]))
    model = dict(source_model) if source_model is not None else policy
    mean = np.asarray(model["feature_mean"], dtype=np.float64)
    scale = np.asarray(model["feature_scale"], dtype=np.float64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    return float(float(model["bias"]) + ((features - mean) / scale) @ weights)


def _scene_gate_route_for_row(policy: dict[str, object], row: dict[str, object]) -> dict[str, object]:
    router = str(policy.get("router", "off"))
    if router == "off":
        return {"decision": str(policy.get("decision", "never")), "threshold": policy.get("threshold")}
    route = dict(policy.get("routes", {})).get(_fallback_router_key(row, router))
    if route is not None:
        return dict(route)
    return {"decision": str(policy.get("decision", "never")), "threshold": policy.get("threshold")}


def _scene_gate_predict(
    policy: dict[str, object],
    selector: WodPreferenceRanker,
    baseline: dict[str, object],
    scene: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
) -> float:
    features = np.asarray(_scene_gate_features(selector, baseline, scene, source_calibration), dtype=np.float64)
    mean = np.asarray(policy["feature_mean"], dtype=np.float64)
    scale = np.asarray(policy["feature_scale"], dtype=np.float64)
    weights = np.asarray(policy["weights"], dtype=np.float64)
    return float(float(policy["bias"]) + ((features - mean) / scale) @ weights)


def _scene_gate_features(
    selector: WodPreferenceRanker,
    baseline: dict[str, object],
    scene: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
) -> list[float]:
    return _source_gate_features(selector, baseline, scene, source_calibration)


def _source_gate_features(
    selector: WodPreferenceRanker,
    baseline: dict[str, object],
    candidate: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
) -> list[float]:
    baseline_features = baseline["features"]
    candidate_features = candidate["features"]
    baseline_score = _calibrated_score(selector, baseline, source_calibration)
    candidate_score = _calibrated_score(selector, candidate, source_calibration)
    values = [
        candidate_score - baseline_score,
        candidate_score,
        baseline_score,
        float(baseline_features["intent"]),
        float(baseline_features["init_speed_mps"]),
    ]
    for source in ("kinematic", "learned", "temporal", "scene", "anchor", "world"):
        values.append(1.0 if str(candidate["source"]) == source else 0.0)
        values.append(1.0 if str(baseline["source"]) == source else 0.0)
    for feature_name in (
        "speed_bin_stopped_or_creep",
        "speed_bin_slow",
        "speed_bin_urban",
        "speed_bin_fast",
        "intent_1",
        "intent_2",
        "intent_3",
    ):
        values.append(float(baseline_features.get(feature_name, 0.0)))
    for feature_name in (
        "endpoint_distance",
        "total_distance",
        "final_lateral_abs",
        "max_lateral_abs",
        "lateral_range",
        "forward_progress",
        "mean_abs_heading_change",
        "signed_lateral_5s",
    ):
        scene_value = float(candidate_features.get(feature_name, 0.0))
        baseline_value = float(baseline_features.get(feature_name, 0.0))
        values.extend([scene_value, scene_value - baseline_value])
    for index in range(8):
        feature_name = f"external_embedding_{index:02d}"
        values.append(float(candidate_features.get(feature_name, baseline_features.get(feature_name, 0.0))))
    return values


def _fallback_policy_thresholds(policy: object) -> list[object]:
    if not policy:
        return []
    policy_dict = dict(policy)
    if policy_dict.get("router") == "off":
        return [policy_dict.get("threshold")]
    return [route.get("threshold") for route in dict(policy_dict.get("routes", {})).values()]


def _fallback_route_payload(threshold: float | None, sources: tuple[str, ...]) -> dict[str, object]:
    if threshold is None or threshold == float("-inf"):
        return {"decision": "never", "threshold": None, "sources": list(sources)}
    if threshold == float("inf"):
        return {"decision": "always", "threshold": None, "sources": list(sources)}
    return {"decision": "margin", "threshold": float(threshold), "sources": list(sources)}


def _fallback_decision(route: dict[str, object]) -> str:
    raw_decision = route.get("decision")
    if raw_decision is not None:
        decision = str(raw_decision)
        if decision not in {"always", "never", "margin"}:
            raise ValueError(f"unsupported fallback decision: {decision}")
        return decision
    return "margin" if route.get("threshold") is not None else "never"


def _select_with_policy(
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    *,
    source_guard: dict[tuple[str, ...], set[str]] | None,
    fallback_policy: dict[str, object] | None,
    source_calibration: dict[str, dict[str, float]] | None = None,
    source_policy: dict[str, object] | None = None,
    family_calibration: dict[str, object] | None = None,
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None = None,
) -> dict[str, object]:
    selected = _select_with_source_policy(selector, rows, source_policy, source_calibration, family_calibration)
    route = _fallback_route_for_rows(fallback_policy, selected)
    if route is not None:
        decision = _fallback_decision(route)
        threshold = _finite_float_or_none(route.get("threshold"))
        fallback_sources = tuple(str(source) for source in route.get("sources", ("kinematic",)))
        fallback_rows = [row for row in rows if str(row["source"]) in fallback_sources]
        if fallback_rows:
            fallback_selector = _fallback_selector_for_sources(fallback_selectors, fallback_sources) or selector
            fallback = _select_by_calibrated_score(
                fallback_selector,
                fallback_rows,
                source_calibration,
                family_calibration,
            )
            margin = _calibrated_score(selector, selected, source_calibration, family_calibration) - _calibrated_score(
                selector,
                fallback,
                source_calibration,
                family_calibration,
            )
            if decision == "always" or (decision == "margin" and threshold is not None and margin < threshold):
                selected = fallback
    if source_guard is not None:
        key = _source_guard_key(selected, _source_guard_mode_from_key(source_guard))
        allowed = source_guard.get(key)
        if allowed and str(selected["source"]) not in allowed:
            guarded_rows = [row for row in rows if str(row["source"]) in allowed]
            if guarded_rows:
                selected = _select_by_calibrated_score(selector, guarded_rows, source_calibration, family_calibration)
    return selected


def _select_with_source_policy(
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    source_policy: dict[str, object] | None,
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None,
) -> dict[str, object]:
    if not source_policy:
        return _select_by_calibrated_score(selector, rows, source_calibration, family_calibration)
    if source_policy.get("mode") not in {"oracle_source", "source_score_speed", "source_score_intent_speed"}:
        raise ValueError(f"unsupported selector source policy: {source_policy.get('mode')}")
    route_key = _source_policy_route_key(rows, source_policy)
    route_scores = dict(dict(source_policy.get("routes", {})).get(route_key, {}))
    source_scores = route_scores or dict(source_policy.get("global_source_scores", {}))
    available_sources = sorted({str(row["source"]) for row in rows})
    ranked_sources = sorted(
        available_sources,
        key=lambda source: (float(source_scores.get(source, 0.0)), source),
        reverse=True,
    )
    for source in ranked_sources:
        source_rows = [row for row in rows if str(row["source"]) == source]
        if source_rows:
            return _select_by_calibrated_score(selector, source_rows, source_calibration, family_calibration)
    return _select_by_calibrated_score(selector, rows, source_calibration, family_calibration)


def _select_by_calibrated_score(
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None = None,
) -> dict[str, object]:
    if not rows:
        raise ValueError("at least one candidate row is required")
    return max(
        rows,
        key=lambda row: (
            _calibrated_score(selector, row, source_calibration, family_calibration),
            -int(row.get("candidate_index", 0)),
            str(row["candidate_name"]),
        ),
    )


def _fallback_selector_for_sources(
    fallback_selectors: dict[tuple[str, ...], WodPreferenceRanker] | None,
    sources: tuple[str, ...],
) -> WodPreferenceRanker | None:
    if not fallback_selectors:
        return None
    return fallback_selectors.get(tuple(sources))


def _calibrated_score(
    selector: WodPreferenceRanker,
    row: dict[str, object],
    source_calibration: dict[str, dict[str, float]] | None,
    family_calibration: dict[str, object] | None = None,
) -> float:
    score = selector.predict_row(row)
    if source_calibration:
        group_offsets = source_calibration.get(_source_calibration_key(row, source_calibration))
        if group_offsets:
            score += group_offsets.get(str(row["source"]), 0.0)
    if family_calibration:
        score += _family_calibration_offset(row, family_calibration)
    return float(score)


def _source_calibration_key(
    row: dict[str, object],
    source_calibration: dict[str, dict[str, float]],
) -> str:
    first_key = next(iter(source_calibration), "")
    if "|" in first_key:
        return _fallback_router_key(row, "intent_speed")
    if first_key.startswith("intent:"):
        return _fallback_router_key(row, "intent")
    return _fallback_router_key(row, "speed")


def _select_with_source_guard(
    selector: WodPreferenceRanker,
    rows: list[dict[str, object]],
    source_guard: dict[tuple[str, ...], set[str]] | None,
) -> dict[str, object]:
    return _select_with_policy(
        selector,
        rows,
        source_guard=source_guard,
        fallback_policy=None,
        source_calibration=None,
    )


def _fallback_route_for_rows(
    fallback_policy: dict[str, object] | None,
    selected: dict[str, object],
) -> dict[str, object] | None:
    if fallback_policy is None:
        return None
    if fallback_policy.get("router") == "off":
        return fallback_policy
    router = str(fallback_policy.get("router"))
    key = _fallback_router_key(selected, router)
    route = dict(fallback_policy.get("routes", {})).get(key)
    return dict(route) if route is not None else None


def _fallback_router_key(row: dict[str, object], router: str) -> str:
    features = row["features"]
    intent_key = f"intent:{int(round(float(features['intent'])))}"
    speed_key = f"speed:{_speed_bin(float(features['init_speed_mps']))}"
    speed_fine_key = f"speed:{_speed_fine_bin(float(features['init_speed_mps']))}"
    if router == "intent":
        return intent_key
    if router == "speed":
        return speed_key
    if router == "speed_fine":
        return speed_fine_key
    if router == "intent_speed":
        return f"{intent_key}|{speed_key}"
    raise ValueError(f"unsupported selector fallback router: {router}")


def _source_guard_mode_from_key(source_guard: dict[tuple[str, ...], set[str]]) -> str:
    first_key = next(iter(source_guard), ())
    if len(first_key) == 2:
        return "intent_speed"
    if first_key and first_key[0].startswith("intent:"):
        return "intent"
    return "speed"


def _source_guard_key(row: dict[str, object], mode: str) -> tuple[str, ...]:
    features = row["features"]
    intent_key = f"intent:{int(round(float(features['intent'])))}"
    speed_key = f"speed:{_speed_bin(float(features['init_speed_mps']))}"
    if mode == "intent":
        return (intent_key,)
    if mode == "speed":
        return (speed_key,)
    if mode == "intent_speed":
        return (intent_key, speed_key)
    raise ValueError(f"unsupported selector source guard: {mode}")


def _world_candidate_metadata(prediction) -> dict[str, float]:
    nearest_distance = prediction.nearest_experiences[0][1] if prediction.nearest_experiences else 0.0
    return {
        "world_nearest_distance": float(nearest_distance),
        "world_neighbor_count": float(len(prediction.nearest_experiences)),
    }


def _add_world_candidate_features(row: dict[str, object], metadata: dict[str, float]) -> None:
    features = row["features"]
    nearest_distance = float(metadata.get("world_nearest_distance", 0.0))
    neighbor_count = float(metadata.get("world_neighbor_count", 0.0))
    features["world_nearest_distance"] = nearest_distance
    features["world_nearest_distance_log"] = float(np.log1p(max(0.0, nearest_distance)))
    features["world_neighbor_count"] = neighbor_count
    features["source_world_x_world_nearest_distance_log"] = (
        float(features["source_world"]) * float(features["world_nearest_distance_log"])
    )


def _fit_family_reliability_features(rows: list[dict[str, object]]) -> dict[str, object]:
    rows_by_frame: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        rows_by_frame[str(row["frame_name"])].append(row)
    family_stats: dict[str, dict[str, float]] = defaultdict(_empty_reliability_stats)
    source_stats: dict[str, dict[str, float]] = defaultdict(_empty_reliability_stats)
    global_score = 0.0
    global_regret = 0.0
    global_count = 0.0
    for frame_rows in rows_by_frame.values():
        oracle = max(frame_rows, key=lambda row: float(row["rfs_score"]))
        oracle_score = float(oracle["rfs_score"])
        oracle_family_key = _family_reliability_key(oracle)
        oracle_source = str(oracle["source"])
        for row in frame_rows:
            score = float(row["rfs_score"])
            regret = max(0.0, oracle_score - score)
            family_key = _family_reliability_key(row)
            source = str(row["source"])
            _update_reliability_stats(
                family_stats[family_key],
                score=score,
                regret=regret,
                oracle=float(family_key == oracle_family_key),
            )
            _update_reliability_stats(
                source_stats[source],
                score=score,
                regret=regret,
                oracle=float(source == oracle_source),
            )
            global_score += score
            global_regret += regret
            global_count += 1.0
    return {
        "family": _finalize_reliability_stats(family_stats),
        "source": _finalize_reliability_stats(source_stats),
        "global": {
            "mean_rfs": global_score / global_count if global_count else 0.0,
            "oracle_rate": 0.0,
            "regret_mean": global_regret / global_count if global_count else 0.0,
            "count_log": float(np.log1p(global_count)),
        },
    }


def _empty_reliability_stats() -> dict[str, float]:
    return {"score": 0.0, "oracle": 0.0, "regret": 0.0, "count": 0.0}


def _update_reliability_stats(stats: dict[str, float], *, score: float, regret: float, oracle: float) -> None:
    stats["score"] += float(score)
    stats["regret"] += float(regret)
    stats["oracle"] += float(oracle)
    stats["count"] += 1.0


def _finalize_reliability_stats(raw_stats: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for key, stats in raw_stats.items():
        count = max(1.0, float(stats["count"]))
        finalized[key] = {
            "mean_rfs": float(stats["score"] / count),
            "oracle_rate": float(stats["oracle"] / count),
            "regret_mean": float(stats["regret"] / count),
            "count_log": float(np.log1p(count)),
        }
    return finalized


def _apply_family_reliability_features(rows: list[dict[str, object]], profile: dict[str, object]) -> None:
    family_stats = dict(profile.get("family", {}))
    source_stats = dict(profile.get("source", {}))
    global_stats = dict(profile.get("global", {}))
    for row in rows:
        family = dict(family_stats.get(_family_reliability_key(row), global_stats))
        source = dict(source_stats.get(str(row["source"]), global_stats))
        features = row["features"]
        features["family_reliability_mean_rfs"] = float(family.get("mean_rfs", 0.0))
        features["family_reliability_oracle_rate"] = float(family.get("oracle_rate", 0.0))
        features["family_reliability_regret_mean"] = float(family.get("regret_mean", 0.0))
        features["family_reliability_count_log"] = float(family.get("count_log", 0.0))
        features["source_reliability_mean_rfs"] = float(source.get("mean_rfs", 0.0))
        features["source_reliability_oracle_rate"] = float(source.get("oracle_rate", 0.0))
        features["source_reliability_regret_mean"] = float(source.get("regret_mean", 0.0))
        features["source_reliability_count_log"] = float(source.get("count_log", 0.0))


def _family_reliability_key(row: dict[str, object]) -> str:
    return f"{row['source']}|{_candidate_family(row)}"


def _selector_numeric_features(feature_mode: str) -> list[str]:
    return selector_numeric_features(feature_mode)


def _selector_targets(rows: list[dict[str, object]], target_mode: str) -> np.ndarray:
    risk_weighted = target_mode.endswith("_risk_weighted")
    target_mode = target_mode.removesuffix("_risk_weighted") + "_oracle_weighted" if risk_weighted else target_mode
    oracle_weighted = target_mode.endswith("_oracle_weighted")
    unweighted_target_mode = target_mode.removesuffix("_oracle_weighted") if oracle_weighted else target_mode
    score_key = "rfs_score_normalized" if unweighted_target_mode.endswith("_normalized") else "rfs_score"
    base_target_mode = unweighted_target_mode.removesuffix("_normalized")
    scores = np.asarray([float(row.get(score_key, row["rfs_score"])) for row in rows], dtype=np.float64)
    if base_target_mode == "absolute":
        return scores
    if base_target_mode == "oracle_binary":
        best_by_frame: dict[str, float] = {}
        for row, score in zip(rows, scores):
            frame_name = str(row["frame_name"])
            best_by_frame[frame_name] = max(float(score), best_by_frame.get(frame_name, float("-inf")))
        return np.asarray(
            [1.0 if float(score) == best_by_frame[str(row["frame_name"])] else 0.0 for row, score in zip(rows, scores)],
            dtype=np.float64,
        )
    if base_target_mode == "frame_rank":
        indices_by_frame: dict[str, list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            indices_by_frame[str(row["frame_name"])].append(index)
        targets = np.zeros(len(rows), dtype=np.float64)
        for indices in indices_by_frame.values():
            ranked_indices = sorted(indices, key=lambda index: (scores[index], -index))
            denominator = max(1, len(ranked_indices) - 1)
            for rank, index in enumerate(ranked_indices):
                targets[index] = rank / denominator
        return targets
    if base_target_mode not in {"frame_delta", "frame_zscore"}:
        raise ValueError(f"unsupported selector target mode: {target_mode}")
    mean_by_frame: dict[str, float] = {}
    scale_by_frame: dict[str, float] = {}
    scores_by_frame: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        scores_by_frame[str(row["frame_name"])].append(float(score))
    for frame_name, frame_scores in scores_by_frame.items():
        mean_by_frame[frame_name] = sum(frame_scores) / len(frame_scores)
        scale = float(np.std(np.asarray(frame_scores, dtype=np.float64)))
        scale_by_frame[frame_name] = scale if scale > 1e-8 else 1.0
    targets = [float(score) - mean_by_frame[str(row["frame_name"])] for row, score in zip(rows, scores)]
    if base_target_mode == "frame_zscore":
        targets = [target / scale_by_frame[str(row["frame_name"])] for row, target in zip(rows, targets)]
    targets_array = np.asarray(targets, dtype=np.float64)
    if oracle_weighted:
        if base_target_mode != "frame_delta":
            raise ValueError(f"oracle-weighted selector target requires frame_delta mode: {target_mode}")
        targets_array = targets_array * _selector_oracle_gap_weights(rows, scores)
    if risk_weighted:
        targets_array = targets_array * _selector_slice_risk_weights(rows)
    return targets_array


def _selector_oracle_gap_weights(rows: list[dict[str, object]], scores: np.ndarray) -> np.ndarray:
    frame_scores: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores):
        frame_scores[str(row["frame_name"])].append(float(score))
    gap_by_frame: dict[str, float] = {}
    for frame_name, values in frame_scores.items():
        best = max(values)
        mean = sum(values) / len(values)
        gap_by_frame[frame_name] = max(0.0, best - mean)
    positive_gaps = [gap for gap in gap_by_frame.values() if gap > 1e-8]
    if not positive_gaps:
        return np.ones(len(rows), dtype=np.float64)
    median_gap = float(np.median(np.asarray(positive_gaps, dtype=np.float64)))
    if median_gap <= 1e-8:
        median_gap = 1.0
    return np.asarray(
        [min(3.0, max(1.0, gap_by_frame[str(row["frame_name"])] / median_gap)) for row in rows],
        dtype=np.float64,
    )


def _selector_slice_risk_weights(rows: list[dict[str, object]]) -> np.ndarray:
    weights: list[float] = []
    for row in rows:
        features = row.get("features", {})
        intent = int(round(float(features.get("intent", 1.0)))) if isinstance(features, dict) else 1
        init_speed = float(features.get("init_speed_mps", 0.0)) if isinstance(features, dict) else 0.0
        weight = 1.0
        if intent == 2:
            weight *= 1.75
        elif intent == 3:
            weight *= 2.5
        if _speed_bin(init_speed) == "fast":
            weight *= 1.35
        weights.append(min(4.0, weight))
    return np.asarray(weights, dtype=np.float64)


def _record(
    frame: WodE2EPreferenceFrame,
    candidate_name: str,
    candidate_index: int,
    trajectory: Trajectory,
) -> CandidateRecord:
    return CandidateRecord(
        frame_name=frame.frame_name,
        trajectory=[(float(x), float(y)) for x, y in trajectory],
        source="wod_cv_candidate",
        candidate_name=candidate_name,
        candidate_index=candidate_index,
    )


def _local_rfs_score(candidate: CandidateRecord, frame: WodE2EPreferenceFrame) -> float:
    return float(
        score_candidate(
            ManeuverCandidate(candidate.candidate_name, candidate.trajectory),
            frame.references,
            frame.init_speed_mps,
        ).combined_score
    )


def _split_frame_names_by_segment(
    frames: list[WodE2EPreferenceFrame],
    *,
    folds: int,
    seed: int,
) -> list[set[str]]:
    if folds < 2:
        raise ValueError("--folds must be at least 2")
    frames_by_segment: dict[str, set[str]] = defaultdict(set)
    for frame in frames:
        frames_by_segment[_segment_id(frame.frame_name)].add(frame.frame_name)
    segments = sorted(frames_by_segment, key=lambda segment: _stable_hash_key(segment, seed))
    if folds > len(segments):
        raise ValueError(f"--folds={folds} exceeds segment count {len(segments)}")
    return [
        {frame for segment in segments[fold_index::folds] for frame in frames_by_segment[segment]}
        for fold_index in range(folds)
    ]


def _segment_id(frame_name: str) -> str:
    head, separator, tail = frame_name.rpartition("-")
    if separator and tail.isdigit():
        return head
    return frame_name


def _stable_hash_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _speed_bin(speed_mps: float) -> str:
    return speed_bin(speed_mps)


def _speed_fine_bin(speed_mps: float) -> str:
    coarse = speed_bin(speed_mps)
    if coarse != "fast":
        return coarse
    if speed_mps < 14.0:
        return "fast_low"
    if speed_mps < 18.0:
        return "fast_mid"
    return "fast_high"


def _add_slice_observation(
    accumulators: dict[str, dict[str, object]],
    key: str,
    selected: dict[str, object],
    oracle: dict[str, object],
) -> None:
    accumulator = accumulators.setdefault(
        key,
        {
            "frames": 0,
            "selected_scores": [],
            "oracle_scores": [],
            "selected_source_counts": defaultdict(int),
            "oracle_source_counts": defaultdict(int),
        },
    )
    selected_score = float(selected["rfs_score"])
    oracle_score = float(oracle["rfs_score"])
    accumulator["frames"] = int(accumulator["frames"]) + 1
    accumulator["selected_scores"].append(selected_score)
    accumulator["oracle_scores"].append(oracle_score)
    accumulator["selected_source_counts"][str(selected["source"])] += 1
    accumulator["oracle_source_counts"][str(oracle["source"])] += 1


def _add_regret_observations(
    accumulators: dict[str, dict[str, object]],
    frame: WodE2EPreferenceFrame,
    selected: dict[str, object],
    oracle: dict[str, object],
) -> None:
    regret = max(0.0, float(oracle["rfs_score"]) - float(selected["rfs_score"]))
    bucket_keys = (
        f"speed:{_speed_bin(frame.init_speed_mps)}",
        f"intent:{int(frame.intent)}",
        f"selected_source:{selected['source']}",
        f"oracle_source:{oracle['source']}",
        f"source_pair:{selected['source']}->{oracle['source']}",
        f"selected_family:{_candidate_family(selected)}",
        f"oracle_family:{_candidate_family(oracle)}",
    )
    for key in bucket_keys:
        accumulator = accumulators.setdefault(
            key,
            {
                "frames": 0,
                "total_regret": 0.0,
                "max_regret": 0.0,
                "oracle_source_counts": defaultdict(int),
                "selected_source_counts": defaultdict(int),
            },
        )
        accumulator["frames"] = int(accumulator["frames"]) + 1
        accumulator["total_regret"] = float(accumulator["total_regret"]) + regret
        accumulator["max_regret"] = max(float(accumulator["max_regret"]), regret)
        accumulator["oracle_source_counts"][str(oracle["source"])] += 1
        accumulator["selected_source_counts"][str(selected["source"])] += 1


def _add_opportunity_observations(
    accumulators: dict[str, dict[str, object]],
    frame: WodE2EPreferenceFrame,
    selected: dict[str, object],
    rows: list[dict[str, object]],
) -> None:
    selected_score = float(selected["rfs_score"])
    best_by_source = _best_rows_by_source(rows)
    for candidate in best_by_source.values():
        gain = float(candidate["rfs_score"]) - selected_score
        if gain <= 0.0:
            continue
        bucket_keys = (
            f"speed:{_speed_bin(frame.init_speed_mps)}",
            f"intent:{int(frame.intent)}",
            f"opportunity_source:{candidate['source']}",
            f"opportunity_family:{_candidate_family(candidate)}",
            f"source_pair:{selected['source']}->{candidate['source']}",
        )
        for key in bucket_keys:
            accumulator = accumulators.setdefault(
                key,
                {
                    "opportunities": 0,
                    "total_potential_gain": 0.0,
                    "max_potential_gain": 0.0,
                    "candidate_source_counts": defaultdict(int),
                    "selected_source_counts": defaultdict(int),
                },
            )
            accumulator["opportunities"] = int(accumulator["opportunities"]) + 1
            accumulator["total_potential_gain"] = float(accumulator["total_potential_gain"]) + gain
            accumulator["max_potential_gain"] = max(float(accumulator["max_potential_gain"]), gain)
            accumulator["candidate_source_counts"][str(candidate["source"])] += 1
            accumulator["selected_source_counts"][str(selected["source"])] += 1


def _best_rows_by_source(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    best_by_source: dict[str, dict[str, object]] = {}
    for row in rows:
        source = str(row["source"])
        previous = best_by_source.get(source)
        if previous is None or float(row["rfs_score"]) > float(previous["rfs_score"]):
            best_by_source[source] = row
    return best_by_source


def _candidate_family(row: dict[str, object]) -> str:
    features = row.get("features", {})
    if isinstance(features, dict) and features.get("candidate_family") is not None:
        return str(features["candidate_family"])
    return str(row.get("candidate_name", "unknown"))


def _finalize_regret_buckets(
    accumulators: dict[str, dict[str, object]],
    *,
    limit: int = 24,
) -> list[dict[str, object]]:
    buckets = []
    for key, accumulator in accumulators.items():
        frames = int(accumulator["frames"])
        total_regret = float(accumulator["total_regret"])
        buckets.append(
            {
                "bucket": key,
                "frames": frames,
                "total_regret": total_regret,
                "mean_regret": total_regret / frames if frames else 0.0,
                "max_regret": float(accumulator["max_regret"]),
                "selected_source_rates": _source_rates(accumulator["selected_source_counts"], frames),
                "oracle_source_rates": _source_rates(accumulator["oracle_source_counts"], frames),
            }
        )
    buckets.sort(key=lambda item: (-float(item["total_regret"]), str(item["bucket"])))
    return buckets[:limit]


def _finalize_opportunity_buckets(
    accumulators: dict[str, dict[str, object]],
    *,
    limit: int = 24,
) -> list[dict[str, object]]:
    buckets = []
    for key, accumulator in accumulators.items():
        opportunities = int(accumulator["opportunities"])
        total_gain = float(accumulator["total_potential_gain"])
        buckets.append(
            {
                "bucket": key,
                "opportunities": opportunities,
                "total_potential_gain": total_gain,
                "mean_potential_gain": total_gain / opportunities if opportunities else 0.0,
                "max_potential_gain": float(accumulator["max_potential_gain"]),
                "selected_source_rates": _source_rates(accumulator["selected_source_counts"], opportunities),
                "candidate_source_rates": _source_rates(accumulator["candidate_source_counts"], opportunities),
            }
        )
    buckets.sort(key=lambda item: (-float(item["total_potential_gain"]), str(item["bucket"])))
    return buckets[:limit]


def _finalize_slices(accumulators: dict[str, dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        key: _finalize_slice(accumulator)
        for key, accumulator in sorted(accumulators.items())
    }


def _finalize_slice(accumulator: dict[str, object]) -> dict[str, object]:
    frames = int(accumulator["frames"])
    selected_scores = [float(value) for value in accumulator["selected_scores"]]
    oracle_scores = [float(value) for value in accumulator["oracle_scores"]]
    selected_source_counts = accumulator["selected_source_counts"]
    oracle_source_counts = accumulator["oracle_source_counts"]
    return {
        "frames": frames,
        "selected_mean_rfs": _mean(selected_scores),
        "oracle_mean_rfs": _mean(oracle_scores),
        "mean_regret": _mean([oracle - selected for selected, oracle in zip(selected_scores, oracle_scores)]),
        "selected_source_rates": _source_rates(selected_source_counts, frames),
        "oracle_source_rates": _source_rates(oracle_source_counts, frames),
    }


def _source_rates(source_counts: dict[str, int], frames: int) -> dict[str, float]:
    sources = ("kinematic", "learned", "scene", "temporal", "anchor", "world", "memory")
    return {source: float(source_counts[source] / frames) for source in sources}


def _aggregate_fold_slices(fold_reports: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    accumulators: dict[str, dict[str, object]] = {}
    for fold in fold_reports:
        for key, slice_report in dict(fold.get("slices", {})).items():
            accumulator = accumulators.setdefault(
                str(key),
                {
                    "frames": 0,
                    "selected_scores": [],
                    "oracle_scores": [],
                    "selected_source_counts": defaultdict(int),
                    "oracle_source_counts": defaultdict(int),
                },
            )
            frames = int(slice_report["frames"])
            selected_mean = float(slice_report["selected_mean_rfs"])
            oracle_mean = float(slice_report["oracle_mean_rfs"])
            accumulator["frames"] = int(accumulator["frames"]) + frames
            accumulator["selected_scores"].extend([selected_mean] * frames)
            accumulator["oracle_scores"].extend([oracle_mean] * frames)
            for source, rate in dict(slice_report["selected_source_rates"]).items():
                accumulator["selected_source_counts"][str(source)] += int(round(float(rate) * frames))
            for source, rate in dict(slice_report["oracle_source_rates"]).items():
                accumulator["oracle_source_counts"][str(source)] += int(round(float(rate) * frames))
    return _finalize_slices(accumulators)


def _aggregate_fold_regret_buckets(fold_reports: list[dict[str, object]]) -> list[dict[str, object]]:
    accumulators: dict[str, dict[str, object]] = {}
    for fold in fold_reports:
        for bucket in list(fold.get("selector_regret_buckets", [])):
            bucket_dict = dict(bucket)
            key = str(bucket_dict["bucket"])
            frames = int(bucket_dict["frames"])
            accumulator = accumulators.setdefault(
                key,
                {
                    "frames": 0,
                    "total_regret": 0.0,
                    "max_regret": 0.0,
                    "selected_source_counts": defaultdict(float),
                    "oracle_source_counts": defaultdict(float),
                },
            )
            accumulator["frames"] = int(accumulator["frames"]) + frames
            accumulator["total_regret"] = float(accumulator["total_regret"]) + float(bucket_dict["total_regret"])
            accumulator["max_regret"] = max(float(accumulator["max_regret"]), float(bucket_dict["max_regret"]))
            for source, rate in dict(bucket_dict["selected_source_rates"]).items():
                accumulator["selected_source_counts"][str(source)] += float(rate) * frames
            for source, rate in dict(bucket_dict["oracle_source_rates"]).items():
                accumulator["oracle_source_counts"][str(source)] += float(rate) * frames
    return _finalize_regret_buckets(accumulators)


def _aggregate_fold_opportunity_buckets(fold_reports: list[dict[str, object]]) -> list[dict[str, object]]:
    accumulators: dict[str, dict[str, object]] = {}
    for fold in fold_reports:
        for bucket in list(fold.get("selector_opportunity_buckets", [])):
            bucket_dict = dict(bucket)
            key = str(bucket_dict["bucket"])
            opportunities = int(bucket_dict["opportunities"])
            accumulator = accumulators.setdefault(
                key,
                {
                    "opportunities": 0,
                    "total_potential_gain": 0.0,
                    "max_potential_gain": 0.0,
                    "selected_source_counts": defaultdict(float),
                    "candidate_source_counts": defaultdict(float),
                },
            )
            accumulator["opportunities"] = int(accumulator["opportunities"]) + opportunities
            accumulator["total_potential_gain"] = float(accumulator["total_potential_gain"]) + float(
                bucket_dict["total_potential_gain"]
            )
            accumulator["max_potential_gain"] = max(
                float(accumulator["max_potential_gain"]),
                float(bucket_dict["max_potential_gain"]),
            )
            for source, rate in dict(bucket_dict["selected_source_rates"]).items():
                accumulator["selected_source_counts"][str(source)] += float(rate) * opportunities
            for source, rate in dict(bucket_dict["candidate_source_rates"]).items():
                accumulator["candidate_source_counts"][str(source)] += float(rate) * opportunities
    return _finalize_opportunity_buckets(accumulators)


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty list")
    return float(sum(values) / len(values))


def _finite_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _finite_mean_or_none(values: list[object]) -> float | None:
    finite_values = [
        finite_value
        for value in values
        if (finite_value := _finite_float_or_none(value)) is not None
    ]
    return _mean(finite_values) if finite_values else None


def _weighted_mean(fold_reports: list[dict[str, object]], key: str) -> float:
    total_frames = sum(int(fold["frames"]) for fold in fold_reports)
    return float(sum(float(fold[key]) * int(fold["frames"]) for fold in fold_reports) / total_frames)


def _weighted_delta(fold_reports: list[dict[str, object]], left_key: str, right_key: str) -> float:
    total_frames = sum(int(fold["frames"]) for fold in fold_reports)
    total = sum((float(fold[left_key]) - float(fold[right_key])) * int(fold["frames"]) for fold in fold_reports)
    return float(total / total_frames)


if __name__ == "__main__":
    raise SystemExit(main())
