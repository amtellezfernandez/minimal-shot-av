from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_nuplan_bootstrap_candidate_calibration.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AnalyzeNuPlanBootstrapCandidateCalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "analyze_nuplan_bootstrap_candidate_calibration")

    def test_gap_report_separates_generation_and_selection_gap(self) -> None:
        report = self.module.analyze_bootstrap_candidate_calibration(
            _replay_report(),
            input_replay_json="toy.json",
            selector_fns={},
            teacher_name="replay_oracle",
            near_miss_threshold_m=1.0,
            target_limit=None,
        )

        self.assertEqual(0, report["gap_report"]["generation_gap_count"])
        self.assertEqual(4, report["gap_report"]["proxy_selection_gap_count"])
        self.assertEqual(4, report["bootstrap_target_count"])
        self.assertEqual({"slow_yield": 4}, report["bootstrap_target_token_histogram"])

    def test_cli_writes_report_and_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_json = tmp_path / "report.json"
            output_md = tmp_path / "report.md"
            targets = tmp_path / "targets.jsonl"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--targets-jsonl",
                str(targets),
                "--target-limit",
                "2",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            report = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_bootstrap_candidate_calibration_v1", report["schema"])
            self.assertEqual(2, len(targets.read_text(encoding="utf-8").splitlines()))
            self.assertIn("Bootstrapped Candidate-Calibration", output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
