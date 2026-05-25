from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "fetch_nuplan_public_maps.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fetch_nuplan_public_maps", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FetchNuPlanPublicMapsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_extract_public_maps_archive_emits_manifest_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            archive_path = tmp_path / "maps.zip"
            output_dir = tmp_path / "maps"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("nuplan-maps-v1.0.json", "{}")
                archive.writestr("sg-one-north/map.gpkg", "map-bytes")

            rows = self.module.extract_public_maps_archive(archive_path=archive_path, output_dir=output_dir)

            self.assertEqual(2, len(rows))
            self.assertTrue((output_dir / "nuplan-maps-v1.0.json").is_file())
            self.assertTrue((output_dir / "sg-one-north" / "map.gpkg").is_file())
            self.assertTrue(all(row["exists"] for row in rows))


if __name__ == "__main__":
    unittest.main()
