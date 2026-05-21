#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_ROOT = ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0"
DEFAULT_BUCKET = "gs://waymo_open_dataset_end_to_end_camera_v_1_0_0"


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local WOD-E2E train/test download commands.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "val", "test"])
    parser.add_argument("--account-name", default="YOUR_WAYMO_ACCOUNT")
    parser.add_argument("--authors", default="Alba Maria Tellez Fernandez")
    parser.add_argument("--method-prefix", default="alba_maria_tellez_fernandez_blind")
    parser.add_argument(
        "--matrix-output-dir",
        type=Path,
        default=ROOT / "artifacts" / "wod_e2e_submission_matrix",
    )
    parser.add_argument(
        "--frame-list",
        type=Path,
        default=ROOT / "data" / "waymo" / "e2e" / "submission_frames" / "test_frames.json",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON path to write the generated data-restore plan.")
    parser.add_argument("--execute", action="store_true", help="Run gsutil commands instead of only printing them.")
    args = parser.parse_args()

    plan = download_plan(
        data_root=args.data_root,
        bucket=args.bucket,
        splits=args.splits,
        account_name=args.account_name,
        authors=args.authors,
        method_prefix=args.method_prefix,
        matrix_output_dir=args.matrix_output_dir,
        frame_list=args.frame_list,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.execute:
        return 0
    if shutil.which("gsutil") is None:
        raise FileNotFoundError("gsutil is not installed or not on PATH")
    for command in plan["commands"]:
        Path(command[-1]).mkdir(parents=True, exist_ok=True)
        subprocess.run(command, check=True, cwd=ROOT)
    return 0


def download_plan(
    *,
    data_root: Path,
    bucket: str,
    splits: list[str],
    account_name: str = "YOUR_WAYMO_ACCOUNT",
    authors: str = "Alba Maria Tellez Fernandez",
    method_prefix: str = "alba_maria_tellez_fernandez_blind",
    matrix_output_dir: Path = ROOT / "artifacts" / "wod_e2e_submission_matrix",
    frame_list: Path = ROOT / "data" / "waymo" / "e2e" / "submission_frames" / "test_frames.json",
) -> dict[str, Any]:
    commands: list[list[str]] = []
    split_reports: dict[str, dict[str, Any]] = {}
    for split in splits:
        report = split_status(data_root, split)
        target = data_root / split
        source = f"{bucket.rstrip('/')}/{_split_object_prefix(split)}*"
        command = ["gsutil", "-m", "cp", "-n", source, str(target)]
        split_reports[split] = {
            **report,
            "target": str(target),
            "source": source,
            "command": command,
        }
        if not report["complete"]:
            commands.append(command)
    matrix_command = blind_matrix_command(
        data_root=data_root,
        output_dir=matrix_output_dir,
        account_name=account_name,
        authors=authors,
        method_prefix=method_prefix,
        frame_list=frame_list,
    )
    return {
        "data_root": str(data_root),
        "bucket": bucket,
        "commands": commands,
        "splits": split_reports,
        "ready_for_blind_matrix": split_status(data_root, "test")["complete"],
        "two_gate_attack_command": matrix_command,
        "frame_list_note": (
            "Download the official challenge frame-list JSON from the Waymo E2E challenge page "
            "and place it at data/waymo/e2e/submission_frames/test_frames.json."
        ),
    }


def split_status(data_root: Path, split: str) -> dict[str, Any]:
    target = data_root / split
    prefix = _split_object_prefix(split)
    target_shards = sorted(target.glob(f"{prefix}*.tfrecord-*"))
    root_shards = sorted(data_root.glob(f"{prefix}*.tfrecord-*"))
    existing_shards = target_shards or root_shards
    expected, missing_indices = _shard_index_report(existing_shards)
    complete = bool(existing_shards) and (expected is None or not missing_indices)
    return {
        "present": bool(existing_shards),
        "complete": complete,
        "existing_shards": len(existing_shards),
        "expected_shards": expected,
        "missing_shards": len(missing_indices) if expected is not None else None,
        "missing_shard_indices_sample": missing_indices[:20] if expected is not None else None,
        "existing_layout": "split_dir" if target_shards else "root_flat" if root_shards else None,
        "first_shard": str(existing_shards[0]) if existing_shards else None,
        "last_shard": str(existing_shards[-1]) if existing_shards else None,
    }


def _shard_index_report(shards: list[Path]) -> tuple[int | None, list[int]]:
    totals: set[int] = set()
    indices: set[int] = set()
    for shard in shards:
        if "-of-" not in shard.name:
            continue
        prefix, suffix = shard.name.rsplit("-of-", maxsplit=1)
        if suffix.isdigit():
            totals.add(int(suffix))
        index_text = prefix.rsplit("-", maxsplit=1)[-1]
        if index_text.isdigit():
            indices.add(int(index_text))
    if not totals:
        return None, []
    expected = max(totals)
    missing_indices = [index for index in range(expected) if index not in indices]
    return expected, missing_indices


def _split_object_prefix(split: str) -> str:
    if split == "train":
        return "training"
    return split


def blind_matrix_command(
    *,
    data_root: Path,
    output_dir: Path,
    account_name: str,
    authors: str,
    method_prefix: str,
    frame_list: Path,
) -> list[str]:
    return [
        "PYTHONPATH=src:.wod-protos",
        ".venv-wod/bin/python",
        "scripts/run_wod_leaderboard_attack.py",
        "--data-root",
        str(data_root),
        "--frame-list",
        str(frame_list),
        "--output-dir",
        str(output_dir),
        "--account-name",
        account_name,
        "--authors",
        authors,
        "--method-prefix",
        method_prefix,
    ]


if __name__ == "__main__":
    raise SystemExit(main())
