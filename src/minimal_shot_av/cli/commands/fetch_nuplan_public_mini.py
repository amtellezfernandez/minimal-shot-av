#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "workspace" / "nuplan" / "public_mini"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"
PUBLIC_MINI_URL = "https://motional-nuplan.s3-ap-northeast-1.amazonaws.com/public/nuplan-v1.1/nuplan-v1.1_mini.zip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a small public nuPlan mini DB subset directly from Motional's public mini archive "
            "without downloading the full 8.6 GB zip."
        )
    )
    parser.add_argument("--url", default=PUBLIC_MINI_URL, help="Remote nuPlan mini zip URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive the extracted DB files.",
    )
    parser.add_argument(
        "--db-count",
        type=int,
        default=5,
        help="How many DB members to extract. The smallest files are chosen by default.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Manifest JSON written after extraction.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    payload = fetch_public_mini_bundle(
        url=str(args.url),
        output_dir=args.output_dir,
        db_count=max(1, int(args.db_count)),
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def fetch_public_mini_bundle(*, url: str, output_dir: Path, db_count: int) -> dict[str, Any]:
    remotezip = _remotezip_import()
    output_dir.mkdir(parents=True, exist_ok=True)
    with remotezip.RemoteZip(url) as archive:
        selected = select_smallest_db_members(archive.infolist(), db_count=db_count)
        extracted: list[dict[str, Any]] = []
        for member in selected:
            archive.extract(member["member"], path=output_dir)
            extracted_path = output_dir / member["member"]
            extracted.append(
                {
                    **member,
                    "extracted_path": str(extracted_path),
                    "exists": extracted_path.is_file(),
                }
            )
    return {
        "schema": "nuplan_public_mini_bundle_v1",
        "url": url,
        "output_dir": str(output_dir),
        "db_count": len(extracted),
        "members": extracted,
    }


def select_smallest_db_members(file_infos: list[Any], *, db_count: int) -> list[dict[str, Any]]:
    members = [
        {
            "member": str(info.filename),
            "file_size": int(info.file_size),
            "compress_size": int(info.compress_size),
        }
        for info in file_infos
        if str(info.filename).endswith(".db")
    ]
    members.sort(key=lambda item: (item["file_size"], item["compress_size"], item["member"]))
    return members[:db_count]


def _remotezip_import():
    try:
        import remotezip
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised only outside nuPlan installs.
        raise ImportError(
            "remotezip is not installed. Run `./scripts/bootstrap_nuplan_env.sh` "
            "or install `minimal-shot-av[nuplan]` into the active environment."
        ) from exc
    return remotezip


if __name__ == "__main__":
    raise SystemExit(main())
