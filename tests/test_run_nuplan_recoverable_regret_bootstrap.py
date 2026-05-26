from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nuplan_recoverable_regret_bootstrap.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunNuPlanRecoverableRegretBootstrapTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "run_nuplan_recoverable_regret_bootstrap")

    def test_oracle_at_k_has_lower_regret_than_proxy(self) -> None:
        replay = _replay_report()
        score_config = _score_config()

        proxy_gap = self.module.regret_gap_summary(
            replay,
            selected_token_fn=lambda scene: str(scene["selected_token"]),
            score_config=score_config,
        )
        oracle_gap = self.module.regret_gap_summary(
            replay,
            selected_token_fn=lambda scene: self.module.oracle_at_k_token_and_score(scene, **score_config)[0],
            score_config=score_config,
        )

        self.assertGreater(proxy_gap["mean_recoverable_regret"], 0.0)
        self.assertEqual(0.0, oracle_gap["mean_recoverable_regret"])
        self.assertEqual(4, proxy_gap["positive_regret_count"])
        self.assertEqual(0, oracle_gap["positive_regret_count"])

    def test_recoverable_regret_targets_capture_teacher_changes(self) -> None:
        targets = self.module.recoverable_regret_targets(
            _replay_report(),
            teacher_name="oracle_at_k",
            **_score_config(),
        )

        self.assertEqual(4, len(targets))
        self.assertTrue(all(row["teacher_changes_proxy"] for row in targets))
        self.assertEqual({"slow_yield"}, {row["teacher_token"] for row in targets})
        self.assertTrue(all(row["recoverable_regret"] > 0.0 for row in targets))

    def test_oracle_at_k_uses_scene_candidate_order_for_expanded_tokens(self) -> None:
        replay = _replay_report()
        scene = replay["scenes"][0]
        scene["candidates"].append(
            {
                "token": "wide_right",
                "speed_scale": 0.5,
                "lateral_offset_m": 0.0,
                "proxy_safe": True,
                "min_proxy_clearance_m": 1.5,
                "final_progress_m": 8.0,
                "score": 5.0,
                "poses": [[1.0 * step, 0.0, 0.0] for step in range(1, 9)],
            }
        )
        scene["candidate_replay_evaluations"].append(
            {
                "token": "wide_right",
                "realized_min_clearance_m": 2.5,
                "realized_near_miss": False,
            }
        )

        token, _score = self.module.oracle_at_k_token_and_score(scene, **_score_config())

        self.assertEqual("slow_yield", token)

    def test_cli_writes_report_targets_and_student(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_json = tmp_path / "report.json"
            output_md = tmp_path / "report.md"
            targets = tmp_path / "targets.jsonl"
            student = tmp_path / "student.json"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--targets-jsonl",
                str(targets),
                "--student-output",
                str(student),
                "--hidden-dim",
                "8",
                "--epochs",
                "180",
                "--seed",
                "11",
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            report = json.loads(output_json.read_text(encoding="utf-8"))
            payload = json.loads(student.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_recoverable_regret_bootstrap_v1", report["schema"])
            self.assertEqual("nuplan_bootstrap_scene_token_student_mlp_v1", payload["model_type"])
            self.assertEqual(4, len(targets.read_text(encoding="utf-8").splitlines()))
            self.assertIn("Recoverable-Regret Bootstrap", output_md.read_text(encoding="utf-8"))

    def test_cli_supports_replay_value_student_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_json = tmp_path / "report.json"
            output_md = tmp_path / "report.md"
            targets = tmp_path / "targets.jsonl"
            student = tmp_path / "student.json"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
                "--targets-jsonl",
                str(targets),
                "--student-output",
                str(student),
                "--student-kind",
                "replay_value_mlp",
                "--hidden-dim",
                "8",
                "--epochs",
                "180",
                "--seed",
                "11",
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            report = json.loads(output_json.read_text(encoding="utf-8"))
            payload = json.loads(student.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_recoverable_regret_bootstrap_v1", report["schema"])
            self.assertEqual("replay_value_mlp", report["student_training"]["student_kind"])
            self.assertEqual("nuplan_replay_value_mlp_selector_v1", payload["model_type"])


def _score_config() -> dict:
    return {
        "near_miss_threshold_m": 1.0,
        "fail_penalty": 250.0,
        "progress_weight": 1.0,
        "ade_weight": 2.0,
        "clearance_weight": 0.0,
    }


if __name__ == "__main__":
    unittest.main()
