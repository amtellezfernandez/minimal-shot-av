from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_wod_e2e_frame_list.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_wod_e2e_frame_list", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildWodE2EFrameListTests(unittest.TestCase):
    def test_frame_list_payload_sorts_names_and_marks_local_source(self) -> None:
        module = _load_module()

        payload = module.frame_list_payload(["b", "a"], source="test")

        self.assertEqual(["a", "b"], payload["frame_names"])
        self.assertEqual(2, payload["frame_count"])
        self.assertFalse(payload["official_challenge_frame_list"])

    def test_frame_list_payload_rejects_duplicate_names(self) -> None:
        module = _load_module()

        with self.assertRaisesRegex(ValueError, "duplicate frame names"):
            module.frame_list_payload(["a", "a"], source="test")


if __name__ == "__main__":
    unittest.main()
