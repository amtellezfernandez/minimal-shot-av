from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "liberate_space.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("liberate_space", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class LiberateSpaceTests(unittest.TestCase):
    def test_format_size_uses_binary_units(self) -> None:
        module = _load_module()
        self.assertEqual("1.0K", module._format_size(1024))
        self.assertEqual("1.0M", module._format_size(1024 * 1024))

    def test_path_size_counts_nested_files(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "a" / "b.bin"
            nested.parent.mkdir(parents=True)
            nested.write_bytes(b"x" * 17)

            self.assertEqual(17, module._path_size(root))


if __name__ == "__main__":
    unittest.main()
