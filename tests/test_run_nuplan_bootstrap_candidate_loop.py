from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nuplan_bootstrap_candidate_loop.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunNuPlanBootstrapCandidateLoopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "run_nuplan_bootstrap_candidate_loop")

    def test_teacher_promotion_closes_selection_gap_but_not_generation_gap(self) -> None:
        replay = _replay_report()
        targets = _teacher_targets(replay)
        g1_report, application = self.module.build_teacher_promoted_g1_replay_report(replay, targets)
        report = self.module.analyze_bootstrap_candidate_loop(
            replay,
            g1_report,
            targets,
            application=application,
            input_replay_json="toy.json",
            teacher_targets_jsonl="targets.jsonl",
            near_miss_threshold_m=1.0,
        )

        self.assertEqual(4, report["gap_report"]["g0"]["proxy_selection_gap_count"])
        self.assertEqual(0, report["gap_report"]["g1"]["proxy_selection_gap_count"])
        self.assertEqual(0, report["gap_report"]["g0"]["generation_gap_count"])
        self.assertEqual(0, report["gap_report"]["g1"]["generation_gap_count"])
        self.assertEqual(4, report["gap_report"]["selection_gap_closed_count"])
        self.assertEqual(0, report["gap_report"]["generation_gap_delta_count"])
        self.assertEqual(4, report["g1_application"]["changed_token_count"])
        self.assertEqual(0, report["selector_metrics"]["g1_teacher_promoted_top1"]["selected_replay_infeasible_count"])

    def test_cli_writes_loop_report_and_g1_replay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            targets_path = tmp_path / "targets.jsonl"
            output_json = tmp_path / "loop.json"
            output_md = tmp_path / "loop.md"
            output_g1 = tmp_path / "g1.json"
            replay = _replay_report()
            input_path.write_text(json.dumps(replay), encoding="utf-8")
            targets_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in _teacher_targets(replay)),
                encoding="utf-8",
            )
            argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--teacher-targets-jsonl",
                str(targets_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--output-g1-replay-json",
                str(output_g1),
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            report = json.loads(output_json.read_text(encoding="utf-8"))
            g1_report = json.loads(output_g1.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_bootstrap_candidate_loop_v1", report["schema"])
            self.assertEqual("slow_yield", g1_report["scenes"][0]["selected_token"])
            self.assertEqual("bootstrap_teacher_promoted", g1_report["scenes"][0]["selected_token_source"])
            self.assertIn("Bootstrap Candidate Loop", output_md.read_text(encoding="utf-8"))


def _teacher_targets(replay: dict) -> list[dict]:
    return [
        {
            "scene_id": scene["scene_id"],
            "source_db_file": scene["source_db_file"],
            "teacher": "replay_oracle",
            "teacher_token": "slow_yield",
            "proxy_token": scene["selected_token"],
        }
        for scene in replay["scenes"]
    ]


if __name__ == "__main__":
    unittest.main()
