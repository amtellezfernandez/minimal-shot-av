#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from io import BytesIO
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
WOD_PROTOS = ROOT / ".wod-protos"
EXTERNAL_INTERNNAV = ROOT / "artifact" / "external" / "InternNav"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if WOD_PROTOS.exists() and str(WOD_PROTOS) not in sys.path:
    sys.path.insert(0, str(WOD_PROTOS))

from minimal_shot_av.model.internvla_av_bridge import (  # noqa: E402
    internvla_navigation_payload_from_text,
    wod_intent_navigation_instruction,
)
from minimal_shot_av.model.wod_e2e import WodCameraImage, WodE2EPreferenceFrame, load_preference_frames  # noqa: E402


DEFAULT_MODEL_PATH = ROOT / "artifacts" / "models" / "InternVLA-N1-DualVLN"
DEFAULT_FRAME_CACHE = ROOT / "artifacts" / "wod_preference_frames_val479.json"
DEFAULT_VAL_DIR = ROOT / "waymo_open_dataset_end_to_end_camera_v_1_0_0" / "val"
DEFAULT_OUTPUT = ROOT / "artifacts" / "internvla_wod_val.jsonl"
DEFAULT_IMAGE_DIR = ROOT / "artifacts" / "internvla_wod_val_images"
_COORDINATE_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class InternVlaInferenceFrame:
    frame_name: str
    intent: int
    image_jpeg: bytes
    camera_name: str


class InternVlaDualVlnTextRunner:
    def __init__(
        self,
        *,
        model_path: Path,
        device: str,
        torch_dtype: str,
        attention_implementation: str,
        max_new_tokens: int,
        system_mode: str,
        embedding_width: int,
        resize_width: int,
        resize_height: int,
        external_internnav_dir: Path = EXTERNAL_INTERNNAV,
    ) -> None:
        if str(external_internnav_dir) not in sys.path:
            sys.path.insert(0, str(external_internnav_dir))
        try:
            import torch
            from PIL import Image
            from transformers import AutoProcessor, AutoTokenizer
            from internnav.model.basemodel.internvla_n1.internvla_n1 import (
                InternVLAN1ForCausalLM,
                InternVLAN1ModelConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "InternVLA inference requires torch, transformers, diffusers, pillow, and the cloned InternNav "
                "package. The downloader venv only contains Hugging Face download utilities."
            ) from exc

        self.torch = torch
        self.image_cls = Image
        self.device = _resolve_device(torch, device)
        dtype = _resolve_torch_dtype(torch, torch_dtype, device=self.device)
        config = InternVLAN1ModelConfig.from_pretrained(model_path)
        if system_mode == "s2_text":
            _disable_robotics_controller(config)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.processor.tokenizer = self.tokenizer
        self.processor.tokenizer.padding_side = "left"
        self.model = InternVLAN1ForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            attn_implementation=attention_implementation,
            device_map={"": self.device},
        )
        if system_mode == "s2_text":
            _install_s2_only_latent_queries(self.model, config, torch)
        self.model.eval()
        self.max_new_tokens = int(max_new_tokens)
        self.embedding_width = int(embedding_width)
        self.resize_width = int(resize_width)
        self.resize_height = int(resize_height)

    def predict(self, frame: InternVlaInferenceFrame) -> tuple[str, tuple[int, int], float, list[float]]:
        image = self.image_cls.open(BytesIO(frame.image_jpeg)).convert("RGB")
        if self.resize_width > 0 and self.resize_height > 0:
            image = image.resize((self.resize_width, self.resize_height))
        instruction = wod_intent_navigation_instruction(frame.intent)
        content = [
            {"type": "text", "text": _internvla_prompt(instruction) + " you can see "},
            {"type": "image", "image": image},
            {"type": "text", "text": "."},
        ]
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)

        start = time.perf_counter()
        with self.torch.no_grad():
            embedding = self._prompt_embedding(inputs)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        latency_ms = (time.perf_counter() - start) * 1000.0
        model_text = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return model_text.strip(), image.size, latency_ms, embedding

    def _prompt_embedding(self, inputs: object) -> list[float]:
        if self.embedding_width <= 0:
            return []
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        attention_mask = getattr(inputs, "attention_mask", None)
        pooled = _masked_mean(hidden, attention_mask, self.torch)
        return _compress_vector(pooled.detach().float().cpu().numpy().reshape(-1).tolist(), width=self.embedding_width)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local InternVLA-N1-DualVLN System-2 inference on WOD camera frames."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--frame-cache", type=Path, default=DEFAULT_FRAME_CACHE)
    parser.add_argument("--image-manifest", type=Path, help="JSONL manifest from --export-image-manifest.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", default="FRONT")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, or cpu.")
    parser.add_argument("--torch-dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    parser.add_argument("--attention-implementation", default="sdpa")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--resize-width", type=int, default=384)
    parser.add_argument("--resize-height", type=int, default=384)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--embedding-width",
        type=int,
        default=64,
        help="Emit a compressed System-2 prompt/image embedding. Use 0 to disable.",
    )
    parser.add_argument(
        "--system-mode",
        choices=("s2_text", "dual"),
        default="s2_text",
        help="s2_text loads only the System-2 cue generator; dual preserves the full robotics controller.",
    )
    parser.add_argument("--external-internnav-dir", type=Path, default=EXTERNAL_INTERNNAV)
    parser.add_argument("--export-image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument(
        "--export-image-manifest",
        type=Path,
        help="Export selected WOD camera JPEGs and exit before model inference.",
    )
    args = parser.parse_args()

    if args.image_manifest is not None:
        frames = load_image_manifest(args.image_manifest, max_frames=args.max_frames)
    else:
        frames = list(
            load_wod_inference_frames(
                args.val_dir,
                frame_cache=args.frame_cache,
                camera=args.camera,
                max_frames=args.max_frames,
            )
        )

    if args.export_image_manifest is not None:
        count = export_image_manifest(
            frames,
            image_dir=args.export_image_dir,
            manifest_path=args.export_image_manifest,
        )
        _print_summary(
            {
                "exported_images": count,
                "image_dir": str(args.export_image_dir),
                "image_manifest": str(args.export_image_manifest),
            }
        )
        return 0

    runner = InternVlaDualVlnTextRunner(
        model_path=args.model_path,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attention_implementation=args.attention_implementation,
        max_new_tokens=args.max_new_tokens,
        system_mode=args.system_mode,
        embedding_width=args.embedding_width,
        resize_width=args.resize_width,
        resize_height=args.resize_height,
        external_internnav_dir=args.external_internnav_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as stream:
        for index, frame in enumerate(frames, start=1):
            model_text, image_size, latency_ms, embedding = runner.predict(frame)
            payload = internvla_navigation_payload_from_text(
                frame.frame_name,
                model_text,
                image_size=image_size,
                latency_ms=latency_ms,
            )
            payload["intent"] = int(frame.intent)
            payload["camera_name"] = frame.camera_name
            if embedding:
                payload["embedding"] = embedding
            tags = _reasoning_tags(frame.intent, model_text, has_embedding=bool(embedding))
            if tags:
                payload["tags"] = tags
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0):
                print(
                    json.dumps(
                        {"phase": "internvla_inference", "frames_done": index, "frames_total": len(frames)},
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )
    _print_summary({"frames": len(frames), "output": str(args.output)})
    return 0


def load_wod_inference_frames(
    val_dir: Path,
    *,
    frame_cache: Path | None,
    camera: str,
    max_frames: int | None,
) -> Iterable[InternVlaInferenceFrame]:
    target_names = _target_frame_names(frame_cache, max_frames=max_frames)
    targets = set(target_names)
    selected: dict[str, InternVlaInferenceFrame] = {}
    for frame in load_preference_frames(val_dir, include_camera_images=True):
        if targets and frame.frame_name not in targets:
            continue
        selected_frame = _inference_frame_from_wod(frame, camera=camera)
        if targets:
            selected[selected_frame.frame_name] = selected_frame
            if len(selected) >= len(targets):
                break
        else:
            yield selected_frame
            if max_frames is not None and max_frames <= 1:
                break
            if max_frames is not None:
                max_frames -= 1

    if targets:
        missing = [name for name in target_names if name not in selected]
        if missing:
            raise ValueError(f"could not find {len(missing)} target frame(s), first missing: {missing[0]}")
        for name in target_names:
            yield selected[name]


def load_image_manifest(path: Path, *, max_frames: int | None = None) -> list[InternVlaInferenceFrame]:
    frames: list[InternVlaInferenceFrame] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            payload = json.loads(line)
            image_path = _resolve_manifest_path(path.parent, str(payload["image_path"]))
            frames.append(
                InternVlaInferenceFrame(
                    frame_name=str(payload["frame_name"]),
                    intent=int(payload.get("intent", 0)),
                    image_jpeg=image_path.read_bytes(),
                    camera_name=str(payload.get("camera_name", "UNKNOWN")),
                )
            )
            if max_frames is not None and len(frames) >= max_frames:
                break
    return frames


def export_image_manifest(
    frames: Iterable[InternVlaInferenceFrame],
    *,
    image_dir: Path,
    manifest_path: Path,
) -> int:
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with manifest_path.open("w", encoding="utf-8") as stream:
        for frame in frames:
            image_path = image_dir / f"{_safe_filename(frame.frame_name)}_{frame.camera_name.lower()}.jpg"
            image_path.write_bytes(frame.image_jpeg)
            row = {
                "frame_name": frame.frame_name,
                "intent": int(frame.intent),
                "camera_name": frame.camera_name,
                "image_path": _manifest_image_path(image_path, manifest_path.parent),
                "instruction": wod_intent_navigation_instruction(frame.intent),
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    return count


def _inference_frame_from_wod(frame: WodE2EPreferenceFrame, *, camera: str) -> InternVlaInferenceFrame:
    image = _select_camera_image(frame.camera_images, camera=camera)
    return InternVlaInferenceFrame(
        frame_name=frame.frame_name,
        intent=frame.intent,
        image_jpeg=image.jpeg,
        camera_name=image.name,
    )


def _select_camera_image(images: list[WodCameraImage], *, camera: str) -> WodCameraImage:
    for image in images:
        if image.name == camera:
            return image
    if camera == "FRONT":
        for image in images:
            if image.name.startswith("FRONT"):
                return image
    available = ", ".join(image.name for image in images) or "none"
    raise ValueError(f"missing camera {camera}; available cameras: {available}")


def _target_frame_names(frame_cache: Path | None, *, max_frames: int | None) -> list[str]:
    if frame_cache is None:
        return []
    payload = json.loads(frame_cache.read_text(encoding="utf-8"))
    rows = payload.get("frames", [])
    if not isinstance(rows, list):
        raise ValueError("frame cache must contain a frames list")
    names = [str(row["frame_name"]) for row in rows]
    return names[:max_frames] if max_frames is not None else names


def _resolve_device(torch, device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_torch_dtype(torch, dtype: str, *, device: str):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    if dtype == "float32":
        return torch.float32
    return torch.bfloat16 if device.startswith("cuda") else torch.float32


def _masked_mean(hidden: object, attention_mask: object, torch) -> object:
    if attention_mask is None:
        return hidden.mean(dim=1)[0]
    mask = attention_mask.to(hidden.device).unsqueeze(-1).to(hidden.dtype)
    denominator = mask.sum(dim=1).clamp(min=1.0)
    return ((hidden * mask).sum(dim=1) / denominator)[0]


def _compress_vector(values: list[float], *, width: int) -> list[float]:
    if width <= 0:
        return []
    buckets = [0.0 for _ in range(width)]
    counts = [0 for _ in range(width)]
    for index, raw_value in enumerate(values):
        value = float(raw_value)
        if not math.isfinite(value):
            continue
        bucket = index % width
        sign = 1.0 if ((index // width) % 2 == 0) else -1.0
        buckets[bucket] += sign * value
        counts[bucket] += 1
    compressed = [buckets[index] / max(1, counts[index]) for index in range(width)]
    norm = math.sqrt(sum(value * value for value in compressed))
    if norm <= 1e-12:
        return compressed
    return [value / norm for value in compressed]


def _disable_robotics_controller(config: object) -> None:
    if hasattr(config, "system1"):
        delattr(config, "system1")


def _install_s2_only_latent_queries(model: object, config: object, torch) -> None:
    backbone = getattr(model, "model")
    if hasattr(backbone, "latent_queries"):
        return
    parameter = next(model.parameters())
    queries = torch.zeros(
        1,
        int(getattr(config, "n_query")),
        int(getattr(config, "hidden_size")),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    backbone.latent_queries = torch.nn.Parameter(queries, requires_grad=False)


def _internvla_prompt(instruction: str) -> str:
    return (
        "You are an autonomous navigation assistant. Your task is to "
        f"{instruction}. Where should you go next to stay on track? "
        "Please output the next waypoint's coordinates in the image. "
        "Please output STOP when you have successfully completed the task."
    )


def _resolve_manifest_path(manifest_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    manifest_relative = manifest_dir / path
    if manifest_relative.exists():
        return manifest_relative
    root_relative = ROOT / path
    if root_relative.exists():
        return root_relative
    if path.exists():
        return path
    return manifest_relative


def _manifest_image_path(image_path: Path, manifest_dir: Path) -> str:
    try:
        return str(image_path.relative_to(manifest_dir))
    except ValueError:
        return str(image_path)


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _reasoning_tags(intent: int, model_text: str, *, has_embedding: bool) -> list[str]:
    tags = [f"intent_{int(intent)}"]
    normalized = model_text.lower()
    if _COORDINATE_RE.search(model_text):
        tags.append("internvla_pixel_goal")
    if "stop" in normalized:
        tags.append("internvla_stop")
    if "left" in normalized or "←" in normalized:
        tags.append("internvla_left")
    if "right" in normalized or "→" in normalized:
        tags.append("internvla_right")
    if has_embedding:
        tags.append("internvla_s2_embedding")
    return tags


def _print_summary(payload: dict[str, object]) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
