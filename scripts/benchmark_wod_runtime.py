#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.model.learned_trajectory_model import (  # noqa: E402
    FEATURE_SET_BASE,
    FEATURE_SET_TEMPORAL,
    RidgeTrajectoryModel,
    fit_ridge_trajectory_model,
)
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame  # noqa: E402
from minimal_shot_av.model.wod_ranker import WodPreferenceRanker, candidate_ranker_row, raw_features  # noqa: E402
from minimal_shot_av.model.wod_ranker import selector_numeric_features  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark online WOD-E2E non-text trajectory runtime.")
    parser.add_argument("--model", type=Path, help="Optional trained base ridge model JSON.")
    parser.add_argument("--aux-model", type=Path, help="Optional trained temporal auxiliary model JSON.")
    parser.add_argument(
        "--architecture",
        choices=("monolithic", "fast-slow"),
        default="monolithic",
        help="Runtime architecture to benchmark.",
    )
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--residual-modes", type=int, default=3)
    parser.add_argument("--fast-residual-modes", type=int, default=1)
    parser.add_argument("--slow-refresh-frames", type=int, default=5)
    parser.add_argument("--target-ms", type=float, default=14.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model = RidgeTrajectoryModel.load(args.model) if args.model else None
    aux_model = RidgeTrajectoryModel.load(args.aux_model) if args.aux_model else None
    if args.architecture == "fast-slow":
        report = benchmark_fast_slow_runtime(
            model=model,
            aux_model=aux_model,
            frame_count=args.frames,
            warmup=args.warmup,
            slow_residual_modes=args.residual_modes,
            fast_residual_modes=args.fast_residual_modes,
            slow_refresh_frames=args.slow_refresh_frames,
            target_ms=args.target_ms,
        )
    else:
        report = benchmark_runtime(
            model=model,
            aux_model=aux_model,
            frame_count=args.frames,
            warmup=args.warmup,
            residual_modes=args.residual_modes,
            target_ms=args.target_ms,
        )
    encoded = json.dumps(report, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["meets_target"] else 1


def benchmark_runtime(
    *,
    model: RidgeTrajectoryModel | None = None,
    aux_model: RidgeTrajectoryModel | None = None,
    frame_count: int = 5000,
    warmup: int = 200,
    residual_modes: int = 3,
    target_ms: float = 14.0,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    frames = _synthetic_frames(max(32, min(256, frame_count)))
    model = model or fit_ridge_trajectory_model(frames, ridge=30.0, residual_modes=residual_modes)
    aux_model = aux_model or fit_ridge_trajectory_model(
        frames,
        ridge=30.0,
        residual_modes=residual_modes,
        feature_set=FEATURE_SET_TEMPORAL,
    )
    sample_rows = _candidate_rows(
        frames[0],
        model,
        aux_model,
        residual_modes=residual_modes,
        measure_sections=False,
    )[0]
    ranker = _zero_ranker(sample_rows)

    for index in range(warmup):
        frame = frames[index % len(frames)]
        _run_online_frame(frame, model, aux_model, ranker, residual_modes=residual_modes)

    total_ms: list[float] = []
    generation_ms: list[float] = []
    feature_ms: list[float] = []
    selection_ms: list[float] = []
    candidate_counts: list[int] = []
    benchmark_start = time.perf_counter()
    for index in range(frame_count):
        frame = frames[index % len(frames)]
        result = _run_online_frame(frame, model, aux_model, ranker, residual_modes=residual_modes)
        total_ms.append(result["total_ms"])
        generation_ms.append(result["generation_ms"])
        feature_ms.append(result["feature_ms"])
        selection_ms.append(result["selection_ms"])
        candidate_counts.append(result["candidate_count"])
    elapsed_s = time.perf_counter() - benchmark_start
    p95_total_ms = _percentile(total_ms, 95.0)
    return {
        "benchmark_type": "online_wod_non_text_runtime",
        "frames": frame_count,
        "warmup": warmup,
        "target_ms": float(target_ms),
        "meets_target": bool(p95_total_ms <= target_ms),
        "feature_sets": {
            "base": model.feature_set,
            "auxiliary": aux_model.feature_set,
        },
        "selector_features": "contextual",
        "residual_modes": residual_modes,
        "candidate_count_mean": statistics.mean(candidate_counts),
        "total_ms": _latency_summary(total_ms),
        "generation_ms": _latency_summary(generation_ms),
        "feature_ms": _latency_summary(feature_ms),
        "selection_ms": _latency_summary(selection_ms),
        "throughput_fps": frame_count / elapsed_s if elapsed_s > 0.0 else None,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "notes": [
            "Measures online trajectory generation, feature extraction, and ranker selection only.",
            "Excludes TFRecord parsing, JPEG decode, official RFS scoring, and offline model training.",
            "Uses only local numeric trajectory models and ranker arithmetic.",
        ],
    }


def benchmark_fast_slow_runtime(
    *,
    model: RidgeTrajectoryModel | None = None,
    aux_model: RidgeTrajectoryModel | None = None,
    frame_count: int = 5000,
    warmup: int = 200,
    slow_residual_modes: int = 3,
    fast_residual_modes: int = 1,
    slow_refresh_frames: int = 5,
    target_ms: float = 14.0,
) -> dict[str, Any]:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if slow_refresh_frames <= 0:
        raise ValueError("slow_refresh_frames must be positive")
    if fast_residual_modes <= 0:
        raise ValueError("fast_residual_modes must be positive")
    if slow_residual_modes < fast_residual_modes:
        raise ValueError("slow_residual_modes must be >= fast_residual_modes")

    frames = _synthetic_frames(max(32, min(256, frame_count)))
    model = model or fit_ridge_trajectory_model(frames, ridge=30.0, residual_modes=slow_residual_modes)
    aux_model = aux_model or fit_ridge_trajectory_model(
        frames,
        ridge=30.0,
        residual_modes=slow_residual_modes,
        feature_set=FEATURE_SET_TEMPORAL,
    )
    sample_rows = _candidate_rows(
        frames[0],
        model,
        aux_model,
        residual_modes=slow_residual_modes,
        measure_sections=False,
    )[0]
    ranker = _zero_ranker(sample_rows)
    cache = _SlowCandidateCache()

    for index in range(warmup):
        frame = frames[index % len(frames)]
        _run_fast_slow_frame(
            frame,
            model,
            aux_model,
            ranker,
            cache,
            frame_index=index,
            fast_residual_modes=fast_residual_modes,
            slow_residual_modes=slow_residual_modes,
            slow_refresh_frames=slow_refresh_frames,
        )

    total_ms: list[float] = []
    fast_generation_ms: list[float] = []
    feature_ms: list[float] = []
    selection_ms: list[float] = []
    slow_refresh_ms: list[float] = []
    candidate_counts: list[int] = []
    stale_cache_frames = 0
    benchmark_start = time.perf_counter()
    for index in range(frame_count):
        frame = frames[index % len(frames)]
        result = _run_fast_slow_frame(
            frame,
            model,
            aux_model,
            ranker,
            cache,
            frame_index=index,
            fast_residual_modes=fast_residual_modes,
            slow_residual_modes=slow_residual_modes,
            slow_refresh_frames=slow_refresh_frames,
        )
        total_ms.append(result["total_ms"])
        fast_generation_ms.append(result["fast_generation_ms"])
        feature_ms.append(result["feature_ms"])
        selection_ms.append(result["selection_ms"])
        if result["slow_refresh_ms"] > 0.0:
            slow_refresh_ms.append(result["slow_refresh_ms"])
        candidate_counts.append(result["candidate_count"])
        stale_cache_frames += int(result["cache_age_frames"] >= slow_refresh_frames)
    elapsed_s = time.perf_counter() - benchmark_start
    p95_total_ms = _percentile(total_ms, 95.0)
    return {
        "benchmark_type": "online_wod_fast_slow_runtime",
        "architecture": "fast_reflex_slow_deliberation",
        "frames": frame_count,
        "warmup": warmup,
        "target_ms": float(target_ms),
        "meets_target": bool(p95_total_ms <= target_ms),
        "control_loop_contract": "synchronous fast loop excludes asynchronous slow refresh latency",
        "slow_loop_contract": "refreshes richer candidate context every N frames; fast loop remains safe if stale",
        "feature_sets": {
            "fast": model.feature_set,
            "slow_auxiliary": aux_model.feature_set,
        },
        "selector_features": "contextual",
        "fast_residual_modes": fast_residual_modes,
        "slow_residual_modes": slow_residual_modes,
        "slow_refresh_frames": slow_refresh_frames,
        "candidate_count_mean": statistics.mean(candidate_counts),
        "stale_cache_frame_rate": stale_cache_frames / frame_count,
        "total_ms": _latency_summary(total_ms),
        "fast_generation_ms": _latency_summary(fast_generation_ms),
        "feature_ms": _latency_summary(feature_ms),
        "selection_ms": _latency_summary(selection_ms),
        "async_slow_refresh_ms": _latency_summary(slow_refresh_ms) if slow_refresh_ms else None,
        "throughput_fps": frame_count / elapsed_s if elapsed_s > 0.0 else None,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": platform.python_version(),
        },
        "notes": [
            "Fast loop measures compact candidate generation, feature extraction, and selection.",
            "Slow refresh latency is reported separately because it can run asynchronously and be preempted.",
            "If the slow cache is stale, the fast loop still has local learned candidates and ranker safety features.",
            "Excludes TFRecord parsing, JPEG decode, official RFS scoring, and offline model training.",
        ],
    }


def _run_online_frame(
    frame: WodE2EPreferenceFrame,
    model: RidgeTrajectoryModel,
    aux_model: RidgeTrajectoryModel,
    ranker: WodPreferenceRanker,
    *,
    residual_modes: int,
) -> dict[str, float | int]:
    start_ns = time.perf_counter_ns()
    rows, generation_ns, feature_ns = _candidate_rows(
        frame,
        model,
        aux_model,
        residual_modes=residual_modes,
        measure_sections=True,
    )
    selection_start_ns = time.perf_counter_ns()
    ranker.select_row(rows)
    end_ns = time.perf_counter_ns()
    return {
        "candidate_count": len(rows),
        "generation_ms": (generation_ns or 0) / 1_000_000.0,
        "feature_ms": (feature_ns or 0) / 1_000_000.0,
        "selection_ms": (end_ns - selection_start_ns) / 1_000_000.0,
        "total_ms": (end_ns - start_ns) / 1_000_000.0,
    }


class _SlowCandidateCache:
    def __init__(self) -> None:
        self.trajectory: tuple[str, str, list[tuple[float, float]]] | None = None
        self.updated_at_frame: int | None = None

    def age(self, frame_index: int) -> int:
        if self.updated_at_frame is None:
            return 10**9
        return max(0, frame_index - self.updated_at_frame)


def _run_fast_slow_frame(
    frame: WodE2EPreferenceFrame,
    model: RidgeTrajectoryModel,
    aux_model: RidgeTrajectoryModel,
    ranker: WodPreferenceRanker,
    cache: _SlowCandidateCache,
    *,
    frame_index: int,
    fast_residual_modes: int,
    slow_residual_modes: int,
    slow_refresh_frames: int,
) -> dict[str, float | int]:
    slow_refresh_ms = 0.0
    if cache.updated_at_frame is None or frame_index - cache.updated_at_frame >= slow_refresh_frames:
        slow_start_ns = time.perf_counter_ns()
        cache.trajectory = _select_slow_candidate_trajectory(
            frame,
            model,
            aux_model,
            ranker,
            residual_modes=slow_residual_modes,
            skip_first_model_modes=fast_residual_modes,
        )
        cache.updated_at_frame = frame_index
        slow_refresh_ms = (time.perf_counter_ns() - slow_start_ns) / 1_000_000.0

    start_ns = time.perf_counter_ns()
    fast_generation_start_ns = start_ns
    candidates = [
        ("fast", name, trajectory)
        for name, trajectory in model.candidate_trajectories(
            frame.past_trajectory,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
            max_residual_modes=fast_residual_modes,
        )
    ]
    fast_generation_end_ns = time.perf_counter_ns()
    if cache.trajectory is not None:
        candidates.append(cache.trajectory)
    feature_start_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    for candidate_index, (source, candidate_name, trajectory) in enumerate(candidates):
        rows.append(
            candidate_ranker_row(
                frame=frame,
                trajectory=trajectory,
                source=source,
                candidate_name=candidate_name,
                candidate_index=candidate_index,
            )
        )
    selection_start_ns = time.perf_counter_ns()
    ranker.select_row(rows)
    end_ns = time.perf_counter_ns()
    return {
        "candidate_count": len(rows),
        "cache_age_frames": cache.age(frame_index),
        "slow_refresh_ms": slow_refresh_ms,
        "fast_generation_ms": (fast_generation_end_ns - fast_generation_start_ns) / 1_000_000.0,
        "feature_ms": (selection_start_ns - feature_start_ns) / 1_000_000.0,
        "selection_ms": (end_ns - selection_start_ns) / 1_000_000.0,
        "total_ms": (end_ns - start_ns) / 1_000_000.0,
    }


def _select_slow_candidate_trajectory(
    frame: WodE2EPreferenceFrame,
    model: RidgeTrajectoryModel,
    aux_model: RidgeTrajectoryModel,
    ranker: WodPreferenceRanker,
    *,
    residual_modes: int,
    skip_first_model_modes: int,
) -> tuple[str, str, list[tuple[float, float]]] | None:
    slow: list[tuple[str, str, list[tuple[float, float]]]] = []
    for mode_index, (name, trajectory) in enumerate(
        model.candidate_trajectories(
            frame.past_trajectory,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
            max_residual_modes=residual_modes,
        )
    ):
        if mode_index >= skip_first_model_modes:
            slow.append(("slow", f"slow_{name}", trajectory))
    slow.extend(
        ("slow_temporal", f"slow_temporal_{name}", trajectory)
        for name, trajectory in aux_model.candidate_trajectories(
            frame.past_trajectory,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
            max_residual_modes=residual_modes,
        )
    )
    if not slow:
        return None
    rows = [
        candidate_ranker_row(
            frame=frame,
            trajectory=trajectory,
            source=source,
            candidate_name=candidate_name,
            candidate_index=candidate_index,
        )
        for candidate_index, (source, candidate_name, trajectory) in enumerate(slow)
    ]
    selected = ranker.select_row(rows)
    return slow[int(selected["candidate_index"])]


def _candidate_rows(
    frame: WodE2EPreferenceFrame,
    model: RidgeTrajectoryModel,
    aux_model: RidgeTrajectoryModel,
    *,
    residual_modes: int,
    measure_sections: bool,
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    generation_start_ns = time.perf_counter_ns()
    candidates = [
        ("learned", name, trajectory)
        for name, trajectory in model.candidate_trajectories(
            frame.past_trajectory,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
            max_residual_modes=residual_modes,
        )
    ]
    candidates.extend(
        ("temporal", f"temporal_{name}", trajectory)
        for name, trajectory in aux_model.candidate_trajectories(
            frame.past_trajectory,
            intent=frame.intent,
            init_speed_mps=frame.init_speed_mps,
            max_residual_modes=residual_modes,
        )
    )
    feature_start_ns = time.perf_counter_ns()
    rows: list[dict[str, Any]] = []
    for candidate_index, (source, candidate_name, trajectory) in enumerate(candidates):
        rows.append(
            candidate_ranker_row(
                frame=frame,
                trajectory=trajectory,
                source=source,
                candidate_name=candidate_name,
                candidate_index=candidate_index,
            )
        )
    end_ns = time.perf_counter_ns()
    if not measure_sections:
        return rows, None, None
    return rows, feature_start_ns - generation_start_ns, end_ns - feature_start_ns


def _zero_ranker(rows: list[dict[str, Any]]) -> WodPreferenceRanker:
    numeric_features = selector_numeric_features("contextual")
    candidate_names = sorted({str(row["candidate_name"]) for row in rows})
    candidate_families = sorted({str(row["features"].get("candidate_family", row["candidate_name"])) for row in rows})
    feature_len = len(raw_features(rows[0], numeric_features, candidate_names, candidate_families))
    return WodPreferenceRanker(
        numeric_features=numeric_features,
        candidate_names=candidate_names,
        candidate_families=candidate_families,
        feature_mean=[0.0] * feature_len,
        feature_scale=[1.0] * feature_len,
        weights=[0.0] * feature_len,
        bias=0.0,
    )


def _synthetic_frames(count: int) -> list[WodE2EPreferenceFrame]:
    frames: list[WodE2EPreferenceFrame] = []
    for index in range(count):
        step = 0.25 + (index % 17) * 0.04
        lateral = ((index % 9) - 4) * 0.015
        past = [
            (
                float((point - 15) * step),
                float(lateral * (point - 15) * (point - 15) / 16.0),
            )
            for point in range(16)
        ]
        future = [
            (
                float(point * step),
                float(lateral * point * point / 8.0),
            )
            for point in range(1, 21)
        ]
        frames.append(
            WodE2EPreferenceFrame(
                frame_name=f"synthetic-{index}",
                past_trajectory=past,
                future_trajectory=future,
                intent=index % 4,
                init_speed_mps=step * 4.0,
                references=[],
            )
        )
    return frames


def _latency_summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": float(statistics.mean(values)),
        "median": float(statistics.median(values)),
        "p95": _percentile(values, 95.0),
        "max": float(max(values)),
        "unit": "ms",
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (percentile / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return float(ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction)


if __name__ == "__main__":
    raise SystemExit(main())
