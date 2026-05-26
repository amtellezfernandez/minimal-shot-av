from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "attach_candidate_metrics.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AttachCandidateMetricsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "attach_candidate_metrics")

    def test_attach_metrics_makes_generation_dump_analysis_ready(self) -> None:
        enriched, report = self.module.attach_candidate_metrics(
            _candidate_payload(),
            _metrics_payload(),
            candidate_json="candidates.json",
            metrics_json="metrics.json",
            near_miss_threshold_m=1.0,
        )

        candidate = enriched["candidate_sets"][0]["candidates"][0]
        self.assertEqual(False, candidate["metrics"]["replay_fail"])
        self.assertEqual(3.0, candidate["metrics"]["progress_m"])
        self.assertEqual("analysis_ready", report["validation"]["readiness"])
        self.assertEqual(2, report["matched_candidate_count"])
        self.assertEqual(0, report["missing_metric_candidate_count"])

    def test_cli_writes_enriched_candidates_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            candidates_path = tmp_path / "candidates.json"
            metrics_path = tmp_path / "metrics.jsonl"
            output_path = tmp_path / "enriched.json"
            report_json = tmp_path / "report.json"
            report_md = tmp_path / "report.md"
            candidates_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
            metrics_path.write_text(
                "".join(json.dumps(row) + "\n" for row in _metrics_payload()["metrics"]),
                encoding="utf-8",
            )
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--candidate-json",
                str(candidates_path),
                "--metrics-json",
                str(metrics_path),
                "--output-json",
                str(output_path),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_md),
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            enriched = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual("candidate_set_v1", enriched["schema"])
            self.assertEqual("analysis_ready", report["validation"]["readiness"])
            self.assertIn("Candidate Metric Join", report_md.read_text(encoding="utf-8"))


def _candidate_payload() -> dict:
    return {
        "schema": "candidate_set_v1",
        "candidate_sets": [
            {
                "scene_id": "scene",
                "generator": "onevl",
                "top1_candidate_id": "0",
                "candidates": [
                    {"candidate_id": "0", "trajectory": [[1.0, 0.0, 0.0]], "metrics": {"raw_score": 0.9}},
                    {"candidate_id": "1", "trajectory": [[0.0, 1.0, 0.1]], "metrics": {"raw_score": 0.5}},
                ],
            }
        ],
    }


def _metrics_payload() -> dict:
    return {
        "metrics": [
            {
                "scene_id": "scene",
                "candidate_id": "0",
                "clearance_m": 2.0,
                "progress_m": 3.0,
                "ade_3s_m": 0.2,
            },
            {
                "scene_id": "scene",
                "candidate_id": "1",
                "clearance_m": 0.5,
                "progress_m": 1.0,
                "ade_3s_m": 0.4,
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
