from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_wod_internvla_embedding_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_wod_internvla_embedding_cache", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildWodInternVlaEmbeddingCacheTests(unittest.TestCase):
    def test_concat_cosmos_symbolic_preserves_cosmos_prefix_and_appends_symbols(self) -> None:
        module = _load_module()
        cosmos = {"frame-a": [0.1, 0.2], "frame-b": [0.3, 0.4]}
        rows = {
            "frame-a": {
                "frame_name": "frame-a",
                "action": "LOOK_DOWN",
                "model_text": "↓",
                "intent": 1,
                "embedding": [3.0, 4.0],
            },
            "frame-b": {
                "frame_name": "frame-b",
                "action": "TURN_LEFT",
                "model_text": "←←←←",
                "intent": 2,
                "embedding": [0.0, 2.0],
            },
        }

        cache = module.build_internvla_embedding_cache(cosmos, rows, variant="concat_cosmos_symbolic")

        self.assertEqual([0.1, 0.2], cache["frame-a"][:2])
        self.assertEqual(13, len(cache["frame-a"]))
        self.assertEqual(1.0, cache["frame-a"][2])
        self.assertEqual(1.0, cache["frame-a"][-1])
        self.assertEqual(1.0, cache["frame-b"][3])
        self.assertEqual(1.0, cache["frame-b"][-3])

    def test_concat_cosmos_vla_uses_unit_internvla_embedding(self) -> None:
        module = _load_module()
        cache = module.build_internvla_embedding_cache(
            {"frame-a": [1.0]},
            {"frame-a": {"frame_name": "frame-a", "embedding": [3.0, 4.0]}},
            variant="concat_cosmos_vla",
        )

        self.assertEqual([1.0, 0.6, 0.8], cache["frame-a"])

    def test_overlay_symbolic_adds_symbols_into_existing_dimensions(self) -> None:
        module = _load_module()
        cache = module.build_internvla_embedding_cache(
            {"frame-a": [0.0, 0.0, 0.0]},
            {
                "frame-a": {
                    "frame_name": "frame-a",
                    "action": "TURN_RIGHT",
                    "intent": 3,
                    "embedding": [1.0, 0.0, 0.0],
                }
            },
            variant="overlay_symbolic",
            symbolic_weight=0.5,
        )

        self.assertEqual([0.0, 0.0, 0.5], cache["frame-a"])

    def test_rejects_frame_set_mismatch(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "frame sets differ"):
            module.build_internvla_embedding_cache(
                {"frame-a": [1.0]},
                {"frame-b": {"frame_name": "frame-b", "embedding": [1.0]}},
                variant="vla_unit",
            )

    def test_cli_writes_external_embedding_cache(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cosmos = root / "cosmos.json"
            cosmos.write_text(
                json.dumps(
                    {
                        "schema": "external_frame_embeddings_v1",
                        "source": "cosmos-test",
                        "dimension": 2,
                        "frames": {"frame-a": [0.1, 0.2]},
                    }
                ),
                encoding="utf-8",
            )
            internvla = root / "internvla.jsonl"
            internvla.write_text(
                json.dumps(
                    {
                        "frame_name": "frame-a",
                        "action": "LOOK_DOWN",
                        "intent": 1,
                        "embedding": [1.0, 0.0],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = root / "cache.json"
            old_argv = sys.argv
            sys.argv = [
                "build_wod_internvla_embedding_cache.py",
                "--cosmos-cache",
                str(cosmos),
                "--internvla-jsonl",
                str(internvla),
                "--variant",
                "concat_cosmos_symbolic",
                "--output",
                str(output),
            ]
            try:
                rc = module.main()
            finally:
                sys.argv = old_argv
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(0, rc)
        self.assertEqual("external_frame_embeddings_v1", payload["schema"])
        self.assertEqual(13, payload["dimension"])
        self.assertEqual("cosmos-predict25+internvla-concat_cosmos_symbolic13", payload["source"])


if __name__ == "__main__":
    unittest.main()
