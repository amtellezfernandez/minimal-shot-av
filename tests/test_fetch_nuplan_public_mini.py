from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_nuplan_public_mini.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_nuplan_public_mini", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _FakeInfo:
    def __init__(self, filename: str, file_size: int, compress_size: int) -> None:
        self.filename = filename
        self.file_size = file_size
        self.compress_size = compress_size


class FetchNuPlanPublicMiniTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_select_smallest_db_members_filters_and_sorts(self) -> None:
        infos = [
            _FakeInfo("data/cache/mini/b.db", 50, 20),
            _FakeInfo("data/cache/mini/a.db", 10, 9),
            _FakeInfo("data/cache/mini/c.db", 10, 8),
            _FakeInfo("docs/readme.txt", 1, 1),
        ]

        selected = self.module.select_smallest_db_members(infos, db_count=2)

        self.assertEqual(
            [
                {
                    "member": "data/cache/mini/c.db",
                    "file_size": 10,
                    "compress_size": 8,
                },
                {
                    "member": "data/cache/mini/a.db",
                    "file_size": 10,
                    "compress_size": 9,
                },
            ],
            selected,
        )


if __name__ == "__main__":
    unittest.main()
