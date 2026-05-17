from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "publish_hf_release.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("publish_hf_release", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PublishHfReleaseTests(unittest.TestCase):
    def test_model_items_include_manifest(self) -> None:
        module = _load_module()
        manifest = {
            "repo_id": "user/repo",
            "models": [
                {
                    "local_path": "artifacts/example/model.pt",
                    "hf_path": "models/model.pt",
                }
            ],
        }

        items = module._model_items(manifest)

        self.assertEqual("models/model.pt", items[0].path_in_repo)
        self.assertEqual("models_manifest.json", items[1].path_in_repo)
        self.assertEqual("user/repo", items[1].repo_id)

    def test_result_items_use_requested_repo(self) -> None:
        module = _load_module()
        manifest = {
            "results": [
                {
                    "local_path": "artifacts/example/result.json",
                    "hf_path": "results/result.json",
                }
            ]
        }

        items = module._result_items(manifest, "user/results", "dataset")

        self.assertEqual(1, len(items))
        self.assertEqual("user/results", items[0].repo_id)
        self.assertEqual("dataset", items[0].repo_type)


if __name__ == "__main__":
    unittest.main()
