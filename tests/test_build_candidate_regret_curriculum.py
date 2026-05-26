from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_candidate_regret_curriculum.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildCandidateRegretCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "build_candidate_regret_curriculum")

    def test_curriculum_uses_g0_safety_g1_utility_and_generation_requests(self) -> None:
        curriculum = self.module.build_candidate_regret_curriculum(
            _report("g0", safety=2, utility=0, generation=3),
            g1_report=_report("g1", safety=1, utility=4, generation=3),
            g0_report_path="g0.json",
            g1_report_path="g1.json",
            safety_base_weight=2.0,
            utility_base_weight=1.0,
            max_targets_per_phase=None,
        )

        self.assertEqual("candidate_regret_curriculum_v1", curriculum["schema"])
        self.assertEqual(2, curriculum["phase_summary"]["safety_recovery"]["target_count"])
        self.assertEqual(4, curriculum["phase_summary"]["utility_recovery"]["target_count"])
        self.assertEqual(3, curriculum["candidate_expansion_request_count"])
        self.assertEqual(
            {"recoverable_safety_regret", "recoverable_utility_regret"},
            {row["regret_type"] for row in curriculum["training_targets"]},
        )

    def test_cli_writes_curriculum_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            g0 = tmp_path / "g0.json"
            g1 = tmp_path / "g1.json"
            output_json = tmp_path / "curriculum.json"
            output_md = tmp_path / "curriculum.md"
            targets = tmp_path / "targets.jsonl"
            g0.write_text(json.dumps(_report("g0", safety=2, utility=0, generation=3)), encoding="utf-8")
            g1.write_text(json.dumps(_report("g1", safety=1, utility=4, generation=3)), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--g0-report",
                str(g0),
                "--g1-report",
                str(g1),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--targets-jsonl",
                str(targets),
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(6, payload["training_target_count"])
            self.assertEqual(6, len(targets.read_text(encoding="utf-8").splitlines()))
            self.assertIn("Candidate Regret Curriculum", output_md.read_text(encoding="utf-8"))


def _report(stage: str, *, safety: int, utility: int, generation: int) -> dict:
    targets = [
        _target(stage, index, "recoverable_safety_regret")
        for index in range(safety)
    ] + [
        _target(stage, index, "recoverable_utility_regret")
        for index in range(utility)
    ]
    records = [
        _record(stage, index, "generation_gap")
        for index in range(generation)
    ]
    return {
        "bootstrap_targets": targets,
        "regret_records": records,
    }


def _target(stage: str, index: int, regret_type: str) -> dict:
    return {
        "scene_id": f"{stage}_{regret_type}_{index}",
        "source": "toy",
        "scenario_type": "toy",
        "generator": "toy_generator",
        "top1_candidate_id": "bad",
        "teacher_candidate_id": "good",
        "regret_type": regret_type,
        "recoverable_regret": 10.0 + index,
        "top1_metrics": {"replay_fail": regret_type == "recoverable_safety_regret"},
        "teacher_metrics": {"replay_fail": False},
        "top1_trajectory": [[0.0, 0.0, 0.0]],
        "teacher_trajectory": [[1.0, 0.0, 0.0]],
    }


def _record(stage: str, index: int, regret_type: str) -> dict:
    return {
        "valid": True,
        "scene_id": f"{stage}_{regret_type}_{index}",
        "source": "toy",
        "scenario_type": "toy",
        "generator": "toy_generator",
        "candidate_count": 4,
        "top1_candidate_id": "bad",
        "oracle_candidate_id": "least_bad",
        "regret_type": regret_type,
        "top1_metrics": {"replay_fail": True},
        "oracle_metrics": {"replay_fail": True},
    }


if __name__ == "__main__":
    unittest.main()
