from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_cosmos_tokenizer_embedding_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_cosmos_tokenizer_embedding_cache", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildCosmosTokenizerEmbeddingCacheTests(unittest.TestCase):
    def test_source_records_repo_camera_and_size(self) -> None:
        module = _load_module()

        source = module._source("nvidia/Cosmos-0.1-Tokenizer-CI8x8", "FRONT", 256)

        self.assertEqual("nvidia/Cosmos-0.1-Tokenizer-CI8x8:encoder.jit:FRONT:256px", source)

    def test_latent_summary_returns_channel_statistics(self) -> None:
        module = _load_module()
        try:
            import torch
        except ModuleNotFoundError:
            self.skipTest("torch is not installed")
        latent = torch.from_numpy(np.arange(16, dtype=np.float32).reshape(1, 2, 2, 4))

        summary = module._latent_summary(latent)

        self.assertEqual(8, len(summary))
        self.assertEqual([3.5, 11.5], summary[:2])
        self.assertEqual([0.0, 8.0], summary[4:6])
        self.assertEqual([7.0, 15.0], summary[6:8])


if __name__ == "__main__":
    unittest.main()
