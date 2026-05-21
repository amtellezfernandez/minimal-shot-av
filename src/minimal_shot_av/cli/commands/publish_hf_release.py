#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
MODELS_MANIFEST = ROOT / "artifacts" / "models_manifest.json"
RESULTS_MANIFEST = ROOT / "artifacts" / "hf_release" / "results_manifest.json"
RELEASE_DIR = ROOT / "artifacts" / "hf_release"


@dataclass(frozen=True)
class UploadItem:
    local_path: Path
    path_in_repo: str
    repo_id: str
    repo_type: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload tracked checkpoints, results, and release metadata to Hugging Face."
    )
    parser.add_argument("--models-manifest", type=Path, default=MODELS_MANIFEST)
    parser.add_argument("--results-manifest", type=Path, default=RESULTS_MANIFEST)
    parser.add_argument(
        "--results-repo-id",
        default="",
        help="Repo id for results and release metadata. Defaults to the models repo id.",
    )
    parser.add_argument(
        "--results-repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Repo type for the results repo.",
    )
    parser.add_argument("--revision", default="main", help="Target branch or revision.")
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.models_only and args.results_only:
        raise SystemExit("Pass at most one of --models-only and --results-only.")

    _require_hf_auth()

    models_manifest = _load_json(args.models_manifest)
    models_repo_id = str(models_manifest["repo_id"])
    results_repo_id = args.results_repo_id or models_repo_id

    items: list[UploadItem] = []
    if not args.results_only:
        items.extend(_model_items(models_manifest))
    if not args.models_only:
        items.extend(_result_items(_load_json(args.results_manifest), results_repo_id, args.results_repo_type))
        items.extend(_release_items(results_repo_id, args.results_repo_type))

    missing = [item for item in items if not item.local_path.is_file()]
    if missing:
        paths = "\n".join(f"- {item.local_path}" for item in missing)
        raise SystemExit(f"Refusing to upload because these files are missing:\n{paths}")

    if args.dry_run:
        for item in items:
            print(f"DRY {item.repo_type}:{item.repo_id}:{item.path_in_repo} <- {item.local_path}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    _ensure_repo(api, models_repo_id, "model")
    if not args.models_only:
        _ensure_repo(api, results_repo_id, args.results_repo_type)

    for item in items:
        api.upload_file(
            path_or_fileobj=str(item.local_path),
            path_in_repo=item.path_in_repo,
            repo_id=item.repo_id,
            repo_type=item.repo_type,
            revision=args.revision,
            commit_message=f"Publish {item.path_in_repo}",
        )
        print(f"PUT {item.repo_type}:{item.repo_id}:{item.path_in_repo}")
    return 0


def _require_hf_auth() -> None:
    token = os.getenv("HF_TOKEN", "").strip() or os.getenv("HUGGINGFACE_HUB_TOKEN", "").strip()
    if token:
        return
    try:
        from huggingface_hub import get_token
    except ImportError as exc:
        raise SystemExit(
            "huggingface_hub is required. Install it in the current environment before publishing."
        ) from exc
    if get_token():
        return
    raise SystemExit("No Hugging Face token found in env or local HF cache.")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_items(manifest: dict[str, Any]) -> list[UploadItem]:
    repo_id = str(manifest["repo_id"])
    items: list[UploadItem] = []
    for model in manifest.get("models", []):
        items.append(
            UploadItem(
                local_path=ROOT / str(model["local_path"]),
                path_in_repo=str(model["hf_path"]),
                repo_id=repo_id,
                repo_type="model",
            )
        )
    items.append(
        UploadItem(
            local_path=MODELS_MANIFEST,
            path_in_repo="models_manifest.json",
            repo_id=repo_id,
            repo_type="model",
        )
    )
    return items


def _result_items(manifest: dict[str, Any], repo_id: str, repo_type: str) -> list[UploadItem]:
    items: list[UploadItem] = []
    for result in manifest.get("results", []):
        items.append(
            UploadItem(
                local_path=ROOT / str(result["local_path"]),
                path_in_repo=str(result["hf_path"]),
                repo_id=repo_id,
                repo_type=repo_type,
            )
        )
    return items


def _release_items(repo_id: str, repo_type: str) -> list[UploadItem]:
    items: list[UploadItem] = []
    for path in sorted(RELEASE_DIR.glob("*")):
        if not path.is_file():
            continue
        items.append(
            UploadItem(
                local_path=path,
                path_in_repo=f"release/{path.name}",
                repo_id=repo_id,
                repo_type=repo_type,
            )
        )
    return items


def _ensure_repo(api: Any, repo_id: str, repo_type: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
