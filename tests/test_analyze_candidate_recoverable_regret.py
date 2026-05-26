from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_candidate_recoverable_regret.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeCandidateRecoverableRegretTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "analyze_candidate_recoverable_regret")

    def test_gap_decomposition_separates_safety_utility_and_generation_regret(self) -> None:
        candidate_sets = self.module.normalize_candidate_sets(
            _candidate_payload(),
            near_miss_threshold_m=1.0,
        )

        report = self.module.analyze_candidate_recoverable_regret(
            candidate_sets,
            input_json="toy.json",
            score_config=_score_config(),
            utility_regret_threshold=5.0,
            target_limit=None,
        )

        gap = report["gap_decomposition"]
        self.assertEqual(1, gap["recoverable_safety_regret_count"])
        self.assertEqual(1, gap["recoverable_utility_regret_count"])
        self.assertEqual(1, gap["generation_gap_count"])
        self.assertEqual(1, gap["no_regret_count"])
        self.assertEqual(2, report["bootstrap_target_count"])
        self.assertEqual(
            {"recoverable_safety_regret": 1, "recoverable_utility_regret": 1},
            report["bootstrap_target_regret_type_histogram"],
        )

    def test_cli_writes_report_and_bootstrap_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "candidates.json"
            output_json = tmp_path / "report.json"
            output_md = tmp_path / "report.md"
            targets = tmp_path / "targets.jsonl"
            input_path.write_text(json.dumps(_candidate_payload()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--targets-jsonl",
                str(targets),
                "--target-limit",
                "1",
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("candidate_recoverable_regret_v1", report["schema"])
            self.assertEqual(1, len(targets.read_text(encoding="utf-8").splitlines()))
            self.assertIn("Candidate Recoverable-Regret", output_md.read_text(encoding="utf-8"))


def _candidate_payload() -> dict:
    return {
        "schema": "candidate_set_v1",
        "candidate_sets": [
            _candidate_set("safety", "c0", [candidate("c0", fail=True, progress=20), candidate("c1", progress=10)]),
            _candidate_set("utility", "c0", [candidate("c0", progress=10), candidate("c1", progress=20)]),
            _candidate_set(
                "generation",
                "c0",
                [candidate("c0", fail=True, progress=20), candidate("c1", fail=True, progress=15)],
            ),
            _candidate_set("none", "c0", [candidate("c0", progress=15), candidate("c1", progress=13)]),
        ],
    }


def _candidate_set(scene_id: str, top1: str, candidates: list[dict]) -> dict:
    return {
        "scene_id": scene_id,
        "source": "toy",
        "generator": "toy_vla",
        "top1_candidate_id": top1,
        "candidates": candidates,
    }


def candidate(candidate_id: str, *, progress: float, fail: bool = False) -> dict:
    return {
        "candidate_id": candidate_id,
        "metrics": {
            "replay_fail": fail,
            "progress_m": progress,
            "ade_3s_m": 1.0,
            "clearance_m": 0.25 if fail else 2.0,
        },
    }


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
