from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Iterator
from types import SimpleNamespace
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_internvla_dualvln_inference.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_internvla_dualvln_inference", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunWodInternVlaDualVlnInferenceTests(unittest.TestCase):
    def test_export_image_manifest_writes_jpegs_and_route_instruction(self) -> None:
        module = _load_module()
        frame = module.InternVlaInferenceFrame(
            frame_name="segment-a-100",
            intent=2,
            image_jpeg=b"jpeg-bytes",
            camera_name="FRONT",
        )
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.jsonl"
            count = module.export_image_manifest([frame], image_dir=root / "images", manifest_path=manifest)
            row = json.loads(manifest.read_text(encoding="utf-8"))
            image_exists = (manifest.parent / row["image_path"]).exists()

        self.assertEqual(1, count)
        self.assertEqual("segment-a-100", row["frame_name"])
        self.assertIn("left", row["instruction"].lower())
        self.assertTrue(image_exists)

    def test_load_image_manifest_reads_relative_image_paths(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image = root / "front.jpg"
            image.write_bytes(b"jpeg-bytes")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "frame_name": "segment-a-100",
                        "intent": 3,
                        "camera_name": "FRONT",
                        "image_path": "front.jpg",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            frames = module.load_image_manifest(manifest)

        self.assertEqual(1, len(frames))
        self.assertEqual("segment-a-100", frames[0].frame_name)
        self.assertEqual(3, frames[0].intent)
        self.assertEqual(b"jpeg-bytes", frames[0].image_jpeg)

    def test_load_image_manifest_keeps_existing_cwd_relative_paths(self) -> None:
        module = _load_module()
        with TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            image = root / "front.jpg"
            image.write_bytes(b"jpeg-bytes")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "frame_name": "segment-a-100",
                        "intent": 1,
                        "camera_name": "FRONT",
                        "image_path": str(image.relative_to(ROOT)),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            frames = module.load_image_manifest(manifest)

        self.assertEqual(b"jpeg-bytes", frames[0].image_jpeg)

    def test_load_image_manifest_prefers_manifest_relative_path(self) -> None:
        module = _load_module()
        with TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            image_name = "__internvla_manifest_priority_test__.jpg"
            cwd_image = ROOT / image_name
            if cwd_image.exists():
                self.skipTest(f"test fixture path already exists: {cwd_image}")
            manifest_image = root / image_name
            cwd_image.write_bytes(b"wrong")
            manifest_image.write_bytes(b"right")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "frame_name": "segment-a-100",
                        "intent": 1,
                        "camera_name": "FRONT",
                        "image_path": image_name,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            try:
                frames = module.load_image_manifest(manifest)
            finally:
                cwd_image.unlink()

        self.assertEqual(b"right", frames[0].image_jpeg)

    def test_select_camera_prefers_exact_camera_name(self) -> None:
        module = _load_module()
        images = [
            module.WodCameraImage(name="FRONT_LEFT", jpeg=b"left"),
            module.WodCameraImage(name="FRONT", jpeg=b"front"),
        ]

        selected = module._select_camera_image(images, camera="FRONT")

        self.assertEqual(b"front", selected.jpeg)

    def test_s2_text_mode_disables_robotics_controller_config(self) -> None:
        module = _load_module()
        config = SimpleNamespace(system1="nextdit_async")

        module._disable_robotics_controller(config)

        self.assertFalse(hasattr(config, "system1"))

    def test_s2_text_mode_installs_dummy_latent_queries_for_internvla_forward(self) -> None:
        module = _load_module()
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        class FakeModel:
            def __init__(self) -> None:
                self.model = SimpleNamespace()
                self.weight = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))

            def parameters(self) -> Iterator[torch.nn.Parameter]:
                return iter([self.weight])

        model = FakeModel()
        config = SimpleNamespace(n_query=4, hidden_size=8)

        module._install_s2_only_latent_queries(model, config, torch)

        self.assertEqual((1, 4, 8), tuple(model.model.latent_queries.shape))
        self.assertFalse(model.model.latent_queries.requires_grad)

    def test_compress_vector_preserves_width_and_normalizes(self) -> None:
        module = _load_module()

        compressed = module._compress_vector([1.0, 2.0, 4.0, 8.0], width=2)

        self.assertEqual(2, len(compressed))
        self.assertAlmostEqual(1.0, sum(value * value for value in compressed))
        self.assertNotEqual(compressed[0], compressed[1])

    def test_reasoning_tags_include_symbolic_internvla_cues(self) -> None:
        module = _load_module()

        tags = module._reasoning_tags(3, "Next waypoint [541, 261], then →.", has_embedding=True)

        self.assertIn("intent_3", tags)
        self.assertIn("internvla_pixel_goal", tags)
        self.assertIn("internvla_right", tags)
        self.assertIn("internvla_s2_embedding", tags)


if __name__ == "__main__":
    unittest.main()
