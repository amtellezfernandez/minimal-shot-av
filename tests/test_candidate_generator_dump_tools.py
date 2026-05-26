from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CONVERT_SCRIPT = ROOT / "scripts" / "convert_onevl_inference_to_candidate_sets.py"
VALIDATE_SCRIPT = ROOT / "scripts" / "validate_candidate_generator_dump.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CandidateGeneratorDumpToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.convert = _load_module(CONVERT_SCRIPT, "convert_onevl_inference_to_candidate_sets")
        cls.validate = _load_module(VALIDATE_SCRIPT, "validate_candidate_generator_dump")

    def test_onevl_conversion_produces_generation_ready_candidate_sets(self) -> None:
        candidate_sets = self.convert.convert_onevl_rows_to_candidate_sets(
            [_onevl_row()],
            generator_name="onevl",
            generator_iteration=0,
        )
        report = self.validate.validate_candidate_generator_dump(
            candidate_sets,
            input_json="onevl.json",
        )

        self.assertEqual(1, len(candidate_sets))
        self.assertEqual("onevl_sample_000000", candidate_sets[0]["scene_id"])
        self.assertEqual("generation_ready", report["readiness"])
        self.assertEqual(2, report["replay_metric_missing_candidate_count"])

    def test_analysis_ready_dump_requires_replay_and_utility_metrics(self) -> None:
        candidate_sets = [
            {
                "scene_id": "scene",
                "generator": "planner",
                "top1_candidate_id": "a",
                "candidates": [
                    {
                        "candidate_id": "a",
                        "trajectory": [[0.0, 0.0, 0.0]],
                        "metrics": {
                            "replay_fail": False,
                            "clearance_m": 2.0,
                            "progress_m": 5.0,
                        },
                    }
                ],
            }
        ]

        report = self.validate.validate_candidate_generator_dump(candidate_sets, input_json="ready.json")

        self.assertEqual("analysis_ready", report["readiness"])
        self.assertEqual(1, report["valid_analysis_scene_count"])

    def test_cli_roundtrip_writes_converted_and_validation_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            onevl_path = tmp_path / "onevl.json"
            candidate_path = tmp_path / "candidate_sets.json"
            validation_json = tmp_path / "validation.json"
            validation_md = tmp_path / "validation.md"
            onevl_path.write_text(json.dumps([_onevl_row()]), encoding="utf-8")

            original_argv = sys.argv
            sys.argv = [
                str(CONVERT_SCRIPT),
                "--input-json",
                str(onevl_path),
                "--output-json",
                str(candidate_path),
            ]
            try:
                self.convert.main()
            finally:
                sys.argv = original_argv

            sys.argv = [
                str(VALIDATE_SCRIPT),
                "--input-json",
                str(candidate_path),
                "--output-json",
                str(validation_json),
                "--output-markdown",
                str(validation_md),
            ]
            try:
                self.validate.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
            validation = json.loads(validation_json.read_text(encoding="utf-8"))
            self.assertEqual("candidate_set_v1", payload["schema"])
            self.assertEqual("generation_ready", validation["readiness"])
            self.assertIn("Candidate Generator Dump Validation", validation_md.read_text(encoding="utf-8"))


def _onevl_row() -> dict:
    return {
        "candidate_mode": "sample",
        "num_candidates": 2,
        "candidates": [
            {
                "candidate_id": 0,
                "trajectory": [[1.0, 0.0, 0.0]],
                "raw_text": "[1, 0, 0]",
                "avg_entropy": 0.2,
                "avg_log_prob": -0.1,
                "seq_confidence": 0.9,
            },
            {
                "candidate_id": 1,
                "trajectory": [[0.5, 1.0, 0.1]],
                "raw_text": "[0.5, 1, 0.1]",
                "avg_entropy": 0.3,
                "avg_log_prob": -0.2,
                "seq_confidence": 0.8,
            },
        ],
    }


if __name__ == "__main__":
    unittest.main()
