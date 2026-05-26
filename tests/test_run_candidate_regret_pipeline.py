from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_candidate_regret_pipeline.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunCandidateRegretPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "run_candidate_regret_pipeline")

    def test_pipeline_attaches_metrics_and_builds_curriculum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            g0_candidates = tmp_path / "g0_candidates.json"
            g0_metrics = tmp_path / "g0_metrics.jsonl"
            out = tmp_path / "out"
            g0_candidates.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
            g0_metrics.write_text(
                "".join(json.dumps(row) + "\n" for row in _metric_rows()),
                encoding="utf-8",
            )

            report = self.module.run_candidate_regret_pipeline(
                g0_candidates=g0_candidates,
                g0_metrics=g0_metrics,
                g1_candidates=None,
                g1_metrics=None,
                output_dir=out,
                score_config=_score_config(),
                utility_regret_threshold=0.001,
            )

            self.assertTrue(report["lifecycle_available"])
            self.assertTrue(report["curriculum_available"])
            self.assertEqual("analysis_ready", report["stages"][0]["readiness"])
            self.assertTrue((out / "g0_analysis.json").is_file())
            self.assertTrue((out / "curriculum.json").is_file())

    def test_generation_ready_only_pipeline_does_not_overclaim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            g0_candidates = tmp_path / "g0_candidates.json"
            out = tmp_path / "out"
            g0_candidates.write_text(json.dumps(_candidate_payload()), encoding="utf-8")

            report = self.module.run_candidate_regret_pipeline(
                g0_candidates=g0_candidates,
                g0_metrics=None,
                g1_candidates=None,
                g1_metrics=None,
                output_dir=out,
                score_config=_score_config(),
                utility_regret_threshold=0.001,
            )

            self.assertFalse(report["lifecycle_available"])
            self.assertFalse(report["curriculum_available"])
            self.assertEqual("generation_ready", report["stages"][0]["readiness"])
            self.assertFalse((out / "g0_analysis.json").exists())


def _candidate_payload() -> dict:
    return {
        "schema": "candidate_set_v1",
        "candidate_sets": [
            {
                "scene_id": "scene",
                "generator": "onevl",
                "top1_candidate_id": "0",
                "candidates": [
                    {"candidate_id": "0", "trajectory": [[1.0, 0.0, 0.0]], "metrics": {}},
                    {"candidate_id": "1", "trajectory": [[0.0, 1.0, 0.1]], "metrics": {}},
                ],
            }
        ],
    }


def _metric_rows() -> list[dict]:
    return [
        {
            "scene_id": "scene",
            "candidate_id": "0",
            "clearance_m": 0.5,
            "progress_m": 3.0,
            "ade_3s_m": 0.2,
        },
        {
            "scene_id": "scene",
            "candidate_id": "1",
            "clearance_m": 2.0,
            "progress_m": 1.0,
            "ade_3s_m": 0.4,
        },
    ]


def _score_config() -> dict:
    return {
        "near_miss_threshold_m": 1.0,
        "failure_penalty": 250.0,
        "progress_weight": 1.0,
        "ade_weight": 2.0,
        "fde_weight": 0.0,
        "clearance_weight": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
