from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_cosmos_predict25_tokenizer_embedding_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_cosmos_predict25_tokenizer_embedding_cache", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildCosmosPredict25TokenizerEmbeddingCacheTests(unittest.TestCase):
    def test_source_records_repo_tokenizer_camera_and_size(self) -> None:
        module = _load_module()

        source = module._source("nvidia/Cosmos-Predict2.5-2B", "FRONT", 256)

        self.assertEqual("nvidia/Cosmos-Predict2.5-2B:tokenizer.pth:wan2pt1:FRONT:256px", source)

    def test_default_resume_flag_allows_existing_cache(self) -> None:
        module = _load_module()

        parser_defaults = module.DEFAULT_REPO

        self.assertEqual("nvidia/Cosmos-Predict2.5-2B", parser_defaults)

    def test_image_tensor_adds_video_time_dimension(self) -> None:
        module = _load_module()
        try:
            import torch  # noqa: F401
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("PIL or torch is not installed")
        buffer = module.BytesIO()
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(buffer, format="JPEG")

        tensor = module._image_tensor(buffer.getvalue(), image_size=8, device="cpu")

        self.assertEqual((1, 3, 1, 8, 8), tuple(tensor.shape))
        self.assertEqual("cpu", tensor.device.type)

    def test_resolve_device_accepts_cpu(self) -> None:
        module = _load_module()
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")

        self.assertEqual("cpu", module._resolve_device("cpu"))

    def test_latent_summary_returns_channel_statistics(self) -> None:
        module = _load_module()
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        latent = torch.from_numpy(np.arange(16, dtype=np.float32).reshape(1, 2, 1, 2, 4))

        summary = module._latent_summary(latent)

        self.assertEqual(8, len(summary))
        self.assertEqual([3.5, 11.5], summary[:2])
        self.assertEqual([0.0, 8.0], summary[4:6])
        self.assertEqual([7.0, 15.0], summary[6:8])


if __name__ == "__main__":
    unittest.main()
