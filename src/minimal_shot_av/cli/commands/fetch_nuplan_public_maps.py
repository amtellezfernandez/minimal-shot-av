#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = ROOT / "workspace" / "nuplan" / "maps"
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "manifest.json"
PUBLIC_MAPS_URL = "https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1/nuplan-maps-v1.0.zip"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the public nuPlan maps bundle needed for closed-loop simulation."
    )
    parser.add_argument("--url", default=PUBLIC_MAPS_URL, help="Remote nuPlan maps zip URL.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory that will receive the extracted maps tree.",
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
    payload = fetch_public_maps_bundle(url=str(args.url), output_dir=args.output_dir)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def fetch_public_maps_bundle(*, url: str, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nuplan-maps-") as tmp:
        archive_path = Path(tmp) / "nuplan-maps-v1.0.zip"
        _download_file(url=url, output_path=archive_path)
        extracted_members = extract_public_maps_archive(archive_path=archive_path, output_dir=output_dir)
    return {
        "schema": "nuplan_public_maps_bundle_v1",
        "url": url,
        "output_dir": str(output_dir),
        "map_version": "nuplan-maps-v1.0",
        "members": extracted_members,
    }


def extract_public_maps_archive(*, archive_path: Path, output_dir: Path) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in sorted(archive.infolist(), key=lambda info: info.filename):
            if member.is_dir():
                continue
            archive.extract(member, path=output_dir)
            extracted_path = output_dir / member.filename
            extracted.append(
                {
                    "member": member.filename,
                    "file_size": int(member.file_size),
                    "compressed_size": int(member.compress_size),
                    "extracted_path": str(extracted_path),
                    "exists": extracted_path.is_file(),
                }
            )
    return extracted


def _download_file(*, url: str, output_path: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "minimal-shot-av/nuplan-maps-fetch"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request) as response, output_path.open("wb") as output:
        shutil.copyfileobj(response, output)


if __name__ == "__main__":
    raise SystemExit(main())
