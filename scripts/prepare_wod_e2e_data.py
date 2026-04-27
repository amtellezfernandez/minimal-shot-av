#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0"
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
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
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
) -> dict[str, Any]:
    commands: list[list[str]] = []
    split_reports: dict[str, dict[str, Any]] = {}
    for split in splits:
        report = split_status(data_root, split)
        target = data_root / split
        source = f"{bucket.rstrip('/')}/{split}*"
        command = ["gsutil", "-m", "cp", "-n", source, str(target)]
        split_reports[split] = {
            **report,
            "target": str(target),
            "source": source,
            "command": command,
        }
        if not report["present"]:
            commands.append(command)
    matrix_command = blind_matrix_command(
        data_root=data_root,
        output_dir=matrix_output_dir,
        account_name=account_name,
        authors=authors,
        method_prefix=method_prefix,
    )
    return {
        "data_root": str(data_root),
        "bucket": bucket,
        "commands": commands,
        "splits": split_reports,
        "ready_for_blind_matrix": split_status(data_root, "test")["present"],
        "blind_matrix_command": matrix_command,
        "frame_list_note": (
            "Download the official challenge frame-list JSON from the Waymo E2E challenge page "
            "and place it at data/waymo/e2e/submission_frames/test_frames.json."
        ),
    }


def split_status(data_root: Path, split: str) -> dict[str, Any]:
    target = data_root / split
    target_shards = sorted(target.glob(f"{split}*.tfrecord-*"))
    root_shards = sorted(data_root.glob(f"{split}*.tfrecord-*"))
    existing_shards = target_shards or root_shards
    return {
        "present": bool(existing_shards),
        "existing_shards": len(existing_shards),
        "existing_layout": "split_dir" if target_shards else "root_flat" if root_shards else None,
        "first_shard": str(existing_shards[0]) if existing_shards else None,
        "last_shard": str(existing_shards[-1]) if existing_shards else None,
    }


def blind_matrix_command(
    *,
    data_root: Path,
    output_dir: Path,
    account_name: str,
    authors: str,
    method_prefix: str,
) -> list[str]:
    return [
        "PYTHONPATH=src:.wod-protos",
        ".venv-wod/bin/python",
        "scripts/prepare_wod_e2e_submission_matrix.py",
        "--test-dir",
        str(data_root / "test"),
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
