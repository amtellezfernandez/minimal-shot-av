#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import json
from itertools import islice
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]

from minimal_shot_av.model.wod_e2e import WodCameraImage, load_preference_frames
from minimal_shot_av.model.world_model import write_external_embedding_cache


DEFAULT_REPO = "nvidia/Cosmos-0.1-Tokenizer-CI8x8"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build WOD frame embeddings with the Cosmos image tokenizer encoder.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "workspace" / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "cosmos_tokenizer_ci8x8_val479.json")
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--encoder", type=Path, help="Local encoder.jit path. If omitted, download from --repo.")
    parser.add_argument("--camera", default="FRONT")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-preference-frames", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    import torch

    encoder_path = args.encoder or _download_encoder(args.repo)
    encoder = torch.jit.load(str(encoder_path), map_location="cpu").eval()
    frames = load_preference_frames(
        args.data_dir,
        max_shards=args.max_shards,
        max_records=args.max_records,
        include_camera_images=True,
    )
    if args.max_preference_frames is not None:
        frames = islice(frames, args.max_preference_frames)

    cache: dict[str, list[float]] = {}
    with torch.no_grad():
        for count, frame in enumerate(frames, start=1):
            image = _select_camera(frame.camera_images, args.camera)
            tensor = _image_tensor(image.jpeg, image_size=args.image_size)
            output = encoder(tensor)
            latent = output[0] if isinstance(output, tuple) else output
            cache[frame.frame_name] = _latent_summary(latent)
            if args.progress_every and count % args.progress_every == 0:
                print(json.dumps({"encoded_frames": count, "path": str(args.output)}), flush=True)
            if args.checkpoint_every and count % args.checkpoint_every == 0:
                write_external_embedding_cache(
                    cache,
                    args.output,
                    source=_source(args.repo, args.camera, args.image_size),
                )

    count = write_external_embedding_cache(cache, args.output, source=_source(args.repo, args.camera, args.image_size))
    dimension = len(next(iter(cache.values()))) if cache else 0
    print(json.dumps({"encoded_frames": count, "dimension": dimension, "path": str(args.output)}, indent=2))
    return 0


def _download_encoder(repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo, filename="encoder.jit"))


def _select_camera(images: list[WodCameraImage], camera: str) -> WodCameraImage:
    for image in images:
        if image.name == camera:
            return image
    if images:
        return images[0]
    raise ValueError("frame has no camera images")


def _image_tensor(jpeg: bytes, *, image_size: int):
    import torch
    from PIL import Image

    image = Image.open(BytesIO(jpeg)).convert("RGB").resize((image_size, image_size))
    array = np.asarray(image).copy()
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0


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
    return f"{repo}:encoder.jit:{camera}:{image_size}px"


if __name__ == "__main__":
    raise SystemExit(main())
