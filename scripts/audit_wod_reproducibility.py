#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STORED = ROOT / "artifacts" / "wod_fastkin_gate_ridge175_scene020_cv_official.json"
DEFAULT_RERUN = Path("/tmp/wod_repro_a934_cosmos_scene020.json")
DEFAULT_OUTPUT = ROOT / "artifacts" / "wod_fastkin_gate_ridge175_scene020_repro_audit.json"
DEFAULT_CACHE_PATHS = (
    ROOT / "artifacts" / "cosmos_predict25_wan21_tokenizer_val479.json",
    ROOT / "artifacts" / "wod_preference_frames_val479.json",
)
DEFAULT_PROVENANCE_COMMIT = "a934d2740cdf831c5f7ee58f5ddae3c7de8692e8"
DEFAULT_PROVENANCE_COMMAND = (
    "evaluate_wod_trajectory_model_cv.py --rfs-backend official --selector-model linear "
    "--selector-features contextual --selector-target frame_delta --selector-ridge 175 "
    "--ridge 30 --residual-modes 3 --residual-grouping off --kinematic-profile base "
    "--selector-kinematic-fallback train_margin --selector-fallback-router speed "
    "--selector-fallback-source-options 'kinematic;kinematic,temporal' "
    "--aux-feature-set temporal_summary --source-gate train_margin "
    "--source-gate-sources kinematic --source-gate-router speed "
    "--source-gate-max-rate 0.20 --source-gate-route-allowlist speed:fast "
    "--scene-gate train_margin --scene-gate-router speed --scene-gate-max-rate 0.20 "
    "--scene-aux-feature-set external_embeddings "
    "--external-embedding-cache artifacts/cosmos_predict25_wan21_tokenizer_val479.json"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit reproducibility drift between WOD validation-CV reports.")
    parser.add_argument("--stored", type=Path, default=DEFAULT_STORED)
    parser.add_argument("--rerun", type=Path, default=DEFAULT_RERUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metric-tolerance", type=float, default=1e-9)
    parser.add_argument("--cache", type=Path, action="append", default=list(DEFAULT_CACHE_PATHS))
    args = parser.parse_args()

    report = audit_reproducibility(
        stored_path=args.stored,
        rerun_path=args.rerun,
        metric_tolerance=args.metric_tolerance,
        cache_paths=tuple(args.cache),
    )
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["passed"] else 1


def audit_reproducibility(
    *,
    stored_path: Path,
    rerun_path: Path,
    metric_tolerance: float = 1e-9,
    cache_paths: tuple[Path, ...] = (),
) -> dict[str, Any]:
    stored = _load_json(stored_path)
    rerun = _load_json(rerun_path)
    metric_deltas = _metric_deltas(stored, rerun)
    source_rate_deltas = _source_rate_deltas(stored, rerun)
    slice_deltas = _slice_deltas(stored, rerun)
    failures: list[dict[str, Any]] = []
    for key, delta in metric_deltas.items():
        if abs(float(delta["delta"])) > metric_tolerance:
            failures.append({"id": "metric_mismatch", "field": key, **delta})
    protocol_failures = _protocol_failures(stored, rerun)
    failures.extend(protocol_failures)
    return {
        "schema": "wod_reproducibility_audit_v1",
        "stored_report": str(stored_path),
        "rerun_report": str(rerun_path),
        "provenance_commit": DEFAULT_PROVENANCE_COMMIT,
        "provenance_command": DEFAULT_PROVENANCE_COMMAND,
        "passed": not failures,
        "failures": failures,
        "metric_tolerance": float(metric_tolerance),
        "metrics": metric_deltas,
        "source_rate_deltas": source_rate_deltas,
        "largest_slice_selected_rfs_deltas": slice_deltas[:12],
        "cache_fingerprints": [_file_fingerprint(path) for path in cache_paths],
        "interpretation": (
            "This audit checks the promoted WOD validation-CV report against the recovered Git history "
            "provenance path. A mismatch means the stored report should be treated as historical "
            "development evidence until the exact code, command, and input-cache path are restored."
        ),
    }


def _metric_deltas(stored: dict[str, Any], rerun: dict[str, Any]) -> dict[str, dict[str, float]]:
    keys = (
        "combined_ranker_mean_rfs",
        "combined_ranker_mean_normalized_rfs",
        "combined_ranker_regret_to_oracle",
        "combined_oracle_mean_rfs",
        "scene_gate_selected_rate",
        "scene_gate_precision",
        "scene_gate_mean_gain",
        "source_gate_selected_rate",
        "source_gate_precision",
        "source_gate_mean_gain",
    )
    return {
        key: {
            "stored": _float(stored.get(key)),
            "rerun": _float(rerun.get(key)),
            "delta": _float(rerun.get(key)) - _float(stored.get(key)),
        }
        for key in keys
    }


def _source_rate_deltas(stored: dict[str, Any], rerun: dict[str, Any]) -> dict[str, dict[str, float]]:
    keys = (
        "selected_kinematic_rate",
        "selected_temporal_rate",
        "selected_learned_rate",
        "selected_scene_rate",
        "selected_memory_rate",
    )
    return {
        key: {
            "stored": _float(stored.get(key)),
            "rerun": _float(rerun.get(key)),
            "delta": _float(rerun.get(key)) - _float(stored.get(key)),
        }
        for key in keys
    }


def _slice_deltas(stored: dict[str, Any], rerun: dict[str, Any]) -> list[dict[str, Any]]:
    stored_slices = stored.get("slices", {})
    rerun_slices = rerun.get("slices", {})
    if not isinstance(stored_slices, dict) or not isinstance(rerun_slices, dict):
        return []
    rows: list[dict[str, Any]] = []
    for name in sorted(set(stored_slices) & set(rerun_slices)):
        left = stored_slices.get(name, {})
        right = rerun_slices.get(name, {})
        if not isinstance(left, dict) or not isinstance(right, dict):
            continue
        stored_rfs = _float(left.get("selected_mean_rfs"))
        rerun_rfs = _float(right.get("selected_mean_rfs"))
        rows.append(
            {
                "slice": name,
                "frames": int(right.get("frames", left.get("frames", 0)) or 0),
                "stored_selected_rfs": stored_rfs,
                "rerun_selected_rfs": rerun_rfs,
                "delta_selected_rfs": rerun_rfs - stored_rfs,
                "stored_source_rates": left.get("selected_source_rates", {}),
                "rerun_source_rates": right.get("selected_source_rates", {}),
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["delta_selected_rfs"])), reverse=True)


def _protocol_failures(stored: dict[str, Any], rerun: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for key in ("benchmark_type", "score_backend", "frames", "fold_count"):
        if stored.get(key) != rerun.get(key):
            failures.append(
                {
                    "id": "protocol_mismatch",
                    "field": key,
                    "stored": stored.get(key),
                    "rerun": rerun.get(key),
                }
            )
    return failures


def _file_fingerprint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "present": False, "size_bytes": None, "sha256": None}
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(path),
        "present": True,
        "size_bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _float(value: object) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
