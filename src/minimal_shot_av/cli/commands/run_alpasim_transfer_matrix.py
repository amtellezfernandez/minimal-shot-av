#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RUNS_ROOT = ROOT / "runs"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"

DEFAULT_MODEL_MATRIX = (
    "token_dagger_iter2,"
    "token_dagger_iter2_clamped,"
    "token_dagger_iter2_axis_constrained_clamped,"
    "token_dagger_iter2_hybrid_clamped,"
    "token_dagger_srcdecay"
)
DEFAULT_PRESET_MATRIX = "front_camera_10scene_smoke,front_camera_30scene_merged"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plan or execute the learned-policy AlpaSim transfer matrix by calling the "
            "scene-level batch runner once per (model, scene preset) pair."
        )
    )
    parser.add_argument("--mode", choices=("print", "both"), default="print")
    parser.add_argument("--models", default=DEFAULT_MODEL_MATRIX)
    parser.add_argument("--scene-presets", default=DEFAULT_PRESET_MATRIX)
    parser.add_argument("--matrix-dir", type=Path, default=None)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--python", default=str(DEFAULT_PYTHON if DEFAULT_PYTHON.is_file() else sys.executable))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--topology", default="1gpu")
    parser.add_argument("--driver-warmup-seconds", type=float, default=10.0)
    parser.add_argument("--wizard-arg", action="append", default=[])
    parser.add_argument("--oracle-actor-proxy", type=Path, default=None)
    parser.add_argument("--alpasim-root", type=Path, default=None)
    parser.add_argument("--allow-existing-matrix-dir", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--scene-offset", type=int, default=0)
    parser.add_argument("--scene-limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    models = _csv_items(args.models)
    presets = _csv_items(args.scene_presets)
    matrix_dir = _resolve_matrix_dir(args)
    _prepare_dir(matrix_dir, allow_existing=args.allow_existing_matrix_dir)

    jobs = []
    for preset in presets:
        for model in models:
            batch_dir = matrix_dir / f"{model}__{preset}"
            command = _job_command(args, model=model, preset=preset, batch_dir=batch_dir)
            jobs.append(
                {
                    "model": model,
                    "scene_preset": preset,
                    "batch_dir": str(batch_dir),
                    "command": command,
                    "status": _batch_status(batch_dir),
                }
            )

    manifest = {
        "schema": "alpasim_transfer_matrix_v1",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "models": models,
        "scene_presets": presets,
        "jobs": jobs,
    }
    manifest_path = matrix_dir / "matrix-manifest.json"
    _write_json(manifest_path, manifest)

    results = []
    failed = False
    for index, job in enumerate(jobs):
        if job["status"] == "completed" and not args.rerun_existing:
            results.append({**job, "result": "skipped_completed", "returncode": 0})
            _update_manifest_status(manifest_path, jobs, active_index=None, completed_indexes={i for i, row in enumerate(results)} if results else set())
            continue
        if args.mode == "print":
            results.append({**job, "result": "planned", "returncode": 0})
            _update_manifest_status(manifest_path, jobs, active_index=None, completed_indexes={i for i, row in enumerate(results)} if results else set())
            continue
        _update_manifest_status(manifest_path, jobs, active_index=index, completed_indexes={i for i, row in enumerate(results)})
        returncode = subprocess.run(job["command"], cwd=ROOT, check=False).returncode
        result = "completed" if returncode == 0 else "failed"
        results.append({**job, "result": result, "returncode": int(returncode)})
        _update_manifest_status(manifest_path, jobs, active_index=None, completed_indexes={i for i, row in enumerate(results)})
        if returncode != 0:
            failed = True
            if not args.continue_on_error:
                break

    summary = {
        "schema": "alpasim_transfer_matrix_summary_v1",
        "matrix_dir": str(matrix_dir),
        "mode": args.mode,
        "models": models,
        "scene_presets": presets,
        "runs": results,
    }
    _write_json(matrix_dir / "matrix-status.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 1 if failed else 0


def _csv_items(value: str) -> list[str]:
    items = [item.strip() for item in str(value).split(",") if item.strip()]
    if not items:
        raise ValueError("expected at least one CSV item")
    return items


def _resolve_matrix_dir(args: argparse.Namespace) -> Path:
    if args.matrix_dir is not None:
        return args.matrix_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.runs_root.resolve() / f"alpasim_transfer_matrix_{stamp}")


def _prepare_dir(path: Path, *, allow_existing: bool) -> None:
    if path.exists():
        if not allow_existing:
            raise SystemExit(f"Matrix dir already exists: {path}")
    else:
        path.mkdir(parents=True)


def _job_command(
    args: argparse.Namespace,
    *,
    model: str,
    preset: str,
    batch_dir: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(ROOT / "scripts" / "run_alpasim_scene_batch.py"),
        "--mode",
        args.mode,
        "--model",
        model,
        "--scene-preset",
        preset,
        "--batch-dir",
        str(batch_dir),
        "--allow-existing-batch-dir",
        "--timeout",
        str(args.timeout),
        "--topology",
        str(args.topology),
        "--driver-warmup-seconds",
        str(args.driver_warmup_seconds),
    ]
    if args.scene_offset:
        command.extend(["--scene-offset", str(args.scene_offset)])
    if args.scene_limit is not None:
        command.extend(["--scene-limit", str(args.scene_limit)])
    if args.rerun_existing:
        command.append("--rerun-existing")
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.alpasim_root is not None:
        command.extend(["--alpasim-root", str(args.alpasim_root)])
    if args.oracle_actor_proxy is not None:
        command.extend(["--oracle-actor-proxy", str(args.oracle_actor_proxy)])
    for override in args.wizard_arg:
        command.extend(["--wizard-arg", override])
    return command


def _batch_status(batch_dir: Path) -> str:
    expected_count = _expected_batch_run_count(batch_dir)
    status_path = batch_dir / "batch-status.json"
    if status_path.is_file():
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return "partial"
        failed = [row for row in payload.get("runs", []) if row.get("result") == "failed"]
        completed = [row for row in payload.get("runs", []) if row.get("result") == "completed"]
        if failed:
            return "failed"
        if completed and len(completed) == len(payload.get("runs", [])) and (
            expected_count is None or len(completed) >= expected_count
        ):
            return "completed"
        return "partial"
    return "missing" if not batch_dir.exists() else "partial"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _expected_batch_run_count(batch_dir: Path) -> int | None:
    manifest_path = batch_dir / "batch-manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    scene_ids = payload.get("scene_ids")
    if isinstance(scene_ids, list):
        return len(scene_ids)
    return None


def _update_manifest_status(
    manifest_path: Path,
    jobs: list[dict[str, Any]],
    *,
    active_index: int | None,
    completed_indexes: set[int],
) -> None:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for index, job in enumerate(jobs):
        if index == active_index:
            status = "running"
        elif index in completed_indexes:
            status = _batch_status(Path(job["batch_dir"]))
        elif _batch_status(Path(job["batch_dir"])) == "completed":
            status = "completed"
        else:
            status = "pending" if active_index is not None and index > active_index else _batch_status(Path(job["batch_dir"]))
        job["status"] = status
        payload["jobs"][index]["status"] = status
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(manifest_path, payload)


if __name__ == "__main__":
    raise SystemExit(main())
