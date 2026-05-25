from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_navsim_intervention_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_navsim_intervention_matrix", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeNavsimInterventionMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_detects_non_co_monotone_public_metric_tradeoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            raw = tmp_path / "raw.json"
            clamped = tmp_path / "clamped.json"
            raw.write_text(
                json.dumps(
                    [
                        _row("a", nc=1.0, dac=0.0, progress=0.2, score=0.2),
                        _row("b", nc=1.0, dac=0.0, progress=0.2, score=0.2),
                    ]
                ),
                encoding="utf-8",
            )
            clamped.write_text(
                json.dumps(
                    [
                        _row("a", nc=0.0, dac=1.0, progress=0.8, score=0.3),
                        _row("b", nc=1.0, dac=1.0, progress=0.7, score=0.4),
                    ]
                ),
                encoding="utf-8",
            )

            report = self.module.analyze_matrix({"raw": raw, "clamped": clamped}, baseline="raw")

        comparison = report["comparisons_vs_baseline"]["clamped"]
        self.assertEqual("tradeoff", comparison["axis_delta"]["class"])
        self.assertIn("progress", comparison["axis_delta"]["improved_axes"])
        self.assertIn("collision", comparison["axis_delta"]["worsened_axes"])
        self.assertEqual(1, comparison["axis_conflict_counts"]["progress_up_collision_up"])


def _row(scene_id: str, *, nc: float, dac: float, progress: float, score: float) -> dict:
    return {
        "scene_id": scene_id,
        "no_at_fault_collisions": nc,
        "drivable_area_compliance": dac,
        "ego_progress": progress,
        "pdm_score": score,
    }


if __name__ == "__main__":
    unittest.main()
