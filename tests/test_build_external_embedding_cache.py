from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_external_embedding_cache.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_external_embedding_cache", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildExternalEmbeddingCacheTests(unittest.TestCase):
    def test_loads_jsonl_rows_with_custom_keys(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cosmos.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps({"name": "frame-a", "vector": [1.0, 2.0]}),
                        json.dumps({"name": "frame-b", "vector": [3.0, 4.0]}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            cache = module.load_embedding_export(path, frame_key="name", embedding_key="vector")

        self.assertEqual({"frame-a": [1.0, 2.0], "frame-b": [3.0, 4.0]}, cache)

    def test_loads_json_mapping(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cosmos.json"
            path.write_text(json.dumps({"frame-a": [1.0, 2.0], "frame-b": [3.0, 4.0]}), encoding="utf-8")

            cache = module.load_embedding_export(path)

        self.assertEqual({"frame-a": [1.0, 2.0], "frame-b": [3.0, 4.0]}, cache)

    def test_loads_embeddings_container(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cosmos.json"
            path.write_text(
                json.dumps({"model": "cosmos", "embeddings": {"frame-a": [1.0, 2.0]}}),
                encoding="utf-8",
            )

            cache = module.load_embedding_export(path)

        self.assertEqual({"frame-a": [1.0, 2.0]}, cache)

    def test_rejects_ambiguous_json_object(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cosmos.json"
            path.write_text(json.dumps({"model": "cosmos", "dimension": 2}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "frame-to-vector"):
                module.load_embedding_export(path)

    def test_rejects_mapping_duplicate_after_key_coercion(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "duplicate"):
            module._cache_from_mapping({1: [1.0, 2.0], "1": [3.0, 4.0]})

    def test_rejects_duplicate_rows(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "cosmos.json"
            path.write_text(
                json.dumps(
                    [
                        {"frame_name": "frame-a", "embedding": [1.0, 2.0]},
                        {"frame_name": "frame-a", "embedding": [3.0, 4.0]},
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate"):
                module.load_embedding_export(path)

    def test_main_writes_valid_external_embedding_cache(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cosmos.json"
            output_path = root / "cache.json"
            input_path.write_text(json.dumps({"frame-a": [1.0, 2.0]}), encoding="utf-8")

            from unittest.mock import patch

            with patch(
                "sys.argv",
                [
                    "build_external_embedding_cache.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--source",
                    "cosmos-test",
                ],
            ):
                exit_code = module.main()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual("external_frame_embeddings_v1", payload["schema"])
        self.assertEqual("cosmos-test", payload["source"])
        self.assertEqual(2, payload["dimension"])
        self.assertEqual({"frame-a": [1.0, 2.0]}, payload["frames"])

    def test_main_rejects_restamping_existing_cache_without_flag(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cache.json"
            output_path = root / "cache_out.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema": "external_frame_embeddings_v1",
                        "source": "cosmos-a",
                        "dimension": 2,
                        "frames": {"frame-a": [1.0, 2.0]},
                    }
                ),
                encoding="utf-8",
            )

            from unittest.mock import patch

            with patch(
                "sys.argv",
                [
                    "build_external_embedding_cache.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--source",
                    "cosmos-b",
                ],
            ):
                with self.assertRaisesRegex(ValueError, "allow-restamp"):
                    module.main()

    def test_main_can_restamp_existing_cache_with_flag(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "cache.json"
            output_path = root / "cache_out.json"
            input_path.write_text(
                json.dumps(
                    {
                        "schema": "external_frame_embeddings_v1",
                        "source": "cosmos-a",
                        "dimension": 2,
                        "frames": {"frame-a": [1.0, 2.0]},
                    }
                ),
                encoding="utf-8",
            )

            from unittest.mock import patch

            with patch(
                "sys.argv",
                [
                    "build_external_embedding_cache.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--source",
                    "cosmos-b",
                    "--allow-restamp",
                ],
            ):
                exit_code = module.main()
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual("cosmos-b", payload["source"])


if __name__ == "__main__":
    unittest.main()
