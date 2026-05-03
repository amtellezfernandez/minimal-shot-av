from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_wod_reproducibility.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_wod_reproducibility", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodReproducibilityAuditTests(unittest.TestCase):
    def test_audit_passes_matching_reports(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stored = _write_report(root / "stored.json", rfs=7.6, scene_rate=0.1)
            rerun = _write_report(root / "rerun.json", rfs=7.6, scene_rate=0.1)

            report = module.audit_reproducibility(stored_path=stored, rerun_path=rerun)

        self.assertTrue(report["passed"])
        self.assertEqual([], report["failures"])

    def test_audit_flags_metric_and_reports_slice_delta(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            stored = _write_report(root / "stored.json", rfs=7.659, scene_rate=0.07)
            rerun = _write_report(root / "rerun.json", rfs=7.626, scene_rate=0.04)

            report = module.audit_reproducibility(stored_path=stored, rerun_path=rerun)

        self.assertFalse(report["passed"])
        self.assertIn("metric_mismatch", {failure["id"] for failure in report["failures"]})
        self.assertEqual("speed:fast", report["largest_slice_selected_rfs_deltas"][0]["slice"])


def _write_report(path: Path, *, rfs: float, scene_rate: float) -> Path:
    payload = {
        "benchmark_type": "segment_grouped_cross_validation",
        "score_backend": "official_waymo_rfs",
        "frames": 479,
        "fold_count": 5,
        "combined_ranker_mean_rfs": rfs,
        "combined_ranker_mean_normalized_rfs": rfs + 0.3,
        "combined_ranker_regret_to_oracle": 1.4,
        "combined_oracle_mean_rfs": 9.0,
        "scene_gate_selected_rate": scene_rate,
        "scene_gate_precision": 0.5,
        "scene_gate_mean_gain": 1.0,
        "source_gate_selected_rate": 0.01,
        "source_gate_precision": 0.2,
        "source_gate_mean_gain": 0.5,
        "selected_kinematic_rate": 0.3,
        "selected_temporal_rate": 0.6,
        "selected_learned_rate": 0.02,
        "selected_scene_rate": scene_rate,
        "selected_memory_rate": 0.0,
        "slices": {
            "speed:fast": {
                "frames": 10,
                "selected_mean_rfs": rfs,
                "selected_source_rates": {"scene": scene_rate},
            },
            "speed:slow": {
                "frames": 10,
                "selected_mean_rfs": 7.0,
                "selected_source_rates": {"scene": scene_rate},
            },
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
