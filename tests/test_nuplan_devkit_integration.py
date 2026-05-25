from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

class NuPlanDevkitIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = ROOT / "src" / "minimal_shot_av" / "model" / "nuplan_devkit_integration.py"
        spec = importlib.util.spec_from_file_location("nuplan_devkit_integration", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_discover_db_files_recurses_nested_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "nested" / "deeper"
            nested.mkdir(parents=True)
            first = nested / "a.db"
            second = root / "flat.db"
            first.write_text("", encoding="utf-8")
            second.write_text("", encoding="utf-8")

            discovered = self.module._discover_db_files(data_root=root, db_files=(str(root),))

            self.assertEqual([str(second), str(first)], discovered)


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    unittest.main()
