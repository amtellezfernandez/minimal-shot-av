from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "artifacts" / "models_manifest.json"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download published model checkpoints from Hugging Face.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help="Path to the model manifest JSON.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=[],
        help="Model name(s) from the manifest to fetch. Defaults to all models.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload even if a matching local file is already present.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    requested = set(args.name)
    models = manifest.get("models", [])
    if requested:
        models = [model for model in models if model["name"] in requested]
        missing = requested - {model["name"] for model in models}
        if missing:
            raise SystemExit(f"Unknown manifest model(s): {sorted(missing)}")

    repo_id = str(manifest["repo_id"])
    revision = str(manifest.get("revision", "main"))
    if not models:
        print("No models selected.")
        return

    for model in models:
        local_path = ROOT / str(model["local_path"])
        sha256 = str(model["sha256"])
        url = _hf_resolve_url(repo_id, revision, str(model["hf_path"]))
        if not args.force and local_path.is_file() and _sha256(local_path) == sha256:
            print(f"OK  {model['name']} -> {local_path} (already present)")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _download(url, local_path)
        actual = _sha256(local_path)
        if actual != sha256:
            local_path.unlink(missing_ok=True)
            raise SystemExit(
                f"Checksum mismatch for {model['name']}: expected {sha256}, got {actual}"
            )
        print(f"GET {model['name']} -> {local_path}")


def _hf_resolve_url(repo_id: str, revision: str, hf_path: str) -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{hf_path}"


def _download(url: str, target: Path) -> None:
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        with urllib.request.urlopen(url) as response, tmp.open("wb") as out:
            shutil.copyfileobj(response, out)
    except urllib.error.HTTPError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Download failed for {url}: HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        tmp.unlink(missing_ok=True)
        raise SystemExit(f"Download failed for {url}: {exc.reason}") from exc
    tmp.replace(target)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
