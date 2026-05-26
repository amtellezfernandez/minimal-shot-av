from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_candidate_regret_lifecycle.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SummarizeCandidateRegretLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "summarize_candidate_regret_lifecycle")

    def test_lifecycle_detects_safety_to_utility_shift(self) -> None:
        report = self.module.summarize_lifecycle(
            [
                ("g0", _report(generation=3, safety=5, utility=0, regret=80.0, top1_fail=0.7)),
                ("g1", _report(generation=3, safety=1, utility=4, regret=20.0, top1_fail=0.4)),
            ]
        )

        self.assertEqual("candidate_regret_lifecycle_v1", report["schema"])
        self.assertEqual(-4, report["deltas"][0]["safety_regret_delta"])
        self.assertEqual(4, report["deltas"][0]["utility_regret_delta"])
        self.assertIn("utility regret", report["interpretation"])

    def test_cli_writes_lifecycle_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            g0 = tmp_path / "g0.json"
            g1 = tmp_path / "g1.json"
            output_json = tmp_path / "life.json"
            output_md = tmp_path / "life.md"
            g0.write_text(json.dumps(_report(generation=3, safety=5, utility=0)), encoding="utf-8")
            g1.write_text(json.dumps(_report(generation=3, safety=1, utility=4)), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--report",
                f"g0={g0}",
                "--report",
                f"g1={g1}",
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(2, payload["stage_count"])
            self.assertIn("Candidate Regret Lifecycle", output_md.read_text(encoding="utf-8"))


def _report(
    *,
    generation: int,
    safety: int,
    utility: int,
    regret: float = 10.0,
    top1_fail: float = 0.5,
) -> dict:
    return {
        "valid_scene_count": 10,
        "gap_decomposition": {
            "generation_gap_count": generation,
            "generation_gap_rate": generation / 10.0,
            "recoverable_safety_regret_count": safety,
            "recoverable_safety_regret_rate": safety / 10.0,
            "recoverable_utility_regret_count": utility,
            "recoverable_utility_regret_rate": utility / 10.0,
        },
        "score_summary": {
            "mean_recoverable_regret": regret,
            "top1_replay_fail_rate": top1_fail,
            "oracle_at_k_replay_fail_rate": generation / 10.0,
        },
    }


if __name__ == "__main__":
    unittest.main()
