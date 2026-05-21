#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import json
from itertools import islice
from pathlib import Path
import sys
import types

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.wod_e2e import WodCameraImage, load_preference_frames
from minimal_shot_av.model.world_model import load_external_embedding_cache, write_external_embedding_cache


DEFAULT_REPO = "nvidia/Cosmos-Predict2.5-2B"
DEFAULT_SOURCE_DIR = Path("/tmp/cosmos-predict2.5")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build WOD frame embeddings with the Cosmos-Predict2.5 Wan2.1 tokenizer."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "cosmos_predict25_wan21_tokenizer_val479.json",
    )
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        help="Local tokenizer.pth path. If omitted, download from --repo.",
    )
    parser.add_argument(
        "--cosmos-source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Path to the official nvidia-cosmos/cosmos-predict2.5 source checkout.",
    )
    parser.add_argument("--camera", default="FRONT")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Device for the Cosmos tokenizer. auto uses CUDA when available.",
    )
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-resume", action="store_true", help="Ignore an existing output cache and rebuild it.")
    args = parser.parse_args()

    import torch

    device = _resolve_device(args.device)
    tokenizer_path = args.tokenizer or _download_tokenizer(args.repo)
    vae = _load_wan21_tokenizer(args.cosmos_source_dir, tokenizer_path, device=device)
    frames = load_preference_frames(
        args.data_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=True,
    )
    if args.max_preference_frames is not None:
        frames = islice(frames, args.max_preference_frames)

    cache: dict[str, list[float]] = {}
    if args.output.exists() and not args.no_resume:
        cache = load_external_embedding_cache(args.output)
        print(json.dumps({"resumed_frames": len(cache), "path": str(args.output)}), flush=True)
    with torch.no_grad():
        for seen_count, frame in enumerate(frames, start=1):
            if frame.frame_name in cache:
                if args.progress_every and seen_count % args.progress_every == 0:
                    print(
                        json.dumps(
                            {
                                "seen_frames": seen_count,
                                "cached_frames": len(cache),
                                "path": str(args.output),
                            }
                        ),
                        flush=True,
                    )
                continue
            image = _select_camera(frame.camera_images, args.camera)
            tensor = _image_tensor(image.jpeg, image_size=args.image_size, device=device)
            latent = vae.encode(tensor)
            cache[frame.frame_name] = _latent_summary(latent)
            if args.progress_every and len(cache) % args.progress_every == 0:
                print(
                    json.dumps({"encoded_frames": len(cache), "seen_frames": seen_count, "path": str(args.output)}),
                    flush=True,
                )
            if args.checkpoint_every and len(cache) % args.checkpoint_every == 0:
                write_external_embedding_cache(
                    cache,
                    args.output,
                    source=_source(args.repo, args.camera, args.image_size),
                )

    count = write_external_embedding_cache(cache, args.output, source=_source(args.repo, args.camera, args.image_size))
    dimension = len(next(iter(cache.values()))) if cache else 0
    print(json.dumps({"encoded_frames": count, "dimension": dimension, "path": str(args.output)}, indent=2))
    return 0


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return requested


def _download_tokenizer(repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename="tokenizer.pth"))


def _load_wan21_tokenizer(cosmos_source_dir: Path, tokenizer_path: Path, *, device: str = "cpu"):
    import torch

    if not (cosmos_source_dir / "cosmos_predict2").is_dir():
        raise FileNotFoundError(f"Cosmos-Predict2.5 source checkout not found: {cosmos_source_dir}")
    cosmos_oss = cosmos_source_dir / "packages" / "cosmos-oss"
    for path in (cosmos_source_dir, cosmos_oss):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # The official package checks for a CUDA-extra sentinel during import. The
    # tokenizer encoder itself can run on CPU, so provide the sentinel module.
    sentinel = types.ModuleType("cosmos_cuda")
    sentinel.__version__ = _read_cosmos_version(cosmos_source_dir)
    sys.modules.setdefault("cosmos_cuda", sentinel)

    from cosmos_predict2._src.predict2.tokenizers.wan2pt1 import WanVAE

    return WanVAE(
        vae_pth=str(tokenizer_path),
        device=device,
        dtype=torch.float32,
        is_amp=device == "cuda",
        load_mean_std=False,
    )


def _read_cosmos_version(cosmos_source_dir: Path) -> str:
    about = cosmos_source_dir / "cosmos_predict2" / "__about__.py"
    namespace: dict[str, str] = {}
    exec(about.read_text(encoding="utf-8"), namespace)
    return namespace["__version__"]


def _select_camera(images: list[WodCameraImage], camera: str) -> WodCameraImage:
    for image in images:
        if image.name == camera:
            return image
    if images:
        return images[0]
    raise ValueError("frame has no camera images")


def _image_tensor(jpeg: bytes, *, image_size: int, device: str = "cpu"):
    import torch
    from PIL import Image

    image = Image.open(BytesIO(jpeg)).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).float() / 127.5 - 1.0
    return tensor.to(device)


def _latent_summary(latent) -> list[float]:
    latent = latent.detach().float().cpu()
    channels = latent.shape[1]
    flat = latent.reshape(channels, -1)
    means = flat.mean(dim=1)
    stds = flat.std(dim=1)
    mins = flat.min(dim=1).values
    maxs = flat.max(dim=1).values
    return [float(value) for value in [*means, *stds, *mins, *maxs]]


def _source(repo: str, camera: str, image_size: int) -> str:
    return f"{repo}:tokenizer.pth:wan2pt1:{camera}:{image_size}px"


if __name__ == "__main__":
    raise SystemExit(main())
