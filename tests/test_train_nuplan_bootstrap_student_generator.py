from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nuplan_bootstrap_student_generator.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanBootstrapStudentGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "train_nuplan_bootstrap_student_generator")

    def test_student_examples_and_evaluation_recover_teacher_targets(self) -> None:
        replay = _replay_report()
        targets = _teacher_targets(replay)
        examples = self.module.build_bootstrap_student_examples(replay, targets)
        selector, metrics = self.module.fit_selector_mlp(
            examples,
            hidden_dim=8,
            epochs=250,
            learning_rate=0.05,
            seed=11,
        )
        evaluation = self.module.evaluate_student_family(
            replay,
            student_token_fn=lambda scene: self.module._select_with_score_model(scene, selector),
            targets=targets,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual(8, len(examples))
        self.assertGreater(metrics["scene_accuracy"], 0.7)
        self.assertEqual(4, evaluation["g0_proxy_top1"]["gap_summary"]["selection_gap_count"])
        self.assertEqual(0, evaluation["g1_student"]["selected_replay_infeasible_count"])
        self.assertEqual(0, evaluation["g1_student"]["gap_summary"]["selection_gap_count"])
        self.assertEqual(1.0, evaluation["g1_student"]["teacher_agreement"]["teacher_agreement_rate"])

    def test_scene_token_student_learns_teacher_token_policy(self) -> None:
        replay = _replay_report()
        targets = _teacher_targets(replay)
        teacher_fn = self.module.teacher_target_selector(targets)
        selector, metrics = self.module.fit_scene_token_student(
            replay,
            teacher_token_fn=teacher_fn,
            hidden_dim=8,
            epochs=250,
            learning_rate=0.05,
            seed=13,
        )
        evaluation = self.module.evaluate_student_family(
            replay,
            student_token_fn=lambda scene: self.module.predict_scene_token_student(selector, scene),
            targets=targets,
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("nuplan_bootstrap_scene_token_student_mlp_v1", selector["model_type"])
        self.assertGreater(metrics["scene_accuracy"], 0.7)
        self.assertEqual(0, evaluation["g1_student"]["selected_replay_infeasible_count"])
        self.assertEqual(1.0, evaluation["g1_student"]["teacher_agreement"]["teacher_agreement_rate"])

    def test_scene_token_student_supports_expanded_candidate_vocab(self) -> None:
        replay = _expanded_replay_report()
        targets = _teacher_targets(replay, teacher_token="wide_right")
        teacher_fn = self.module.teacher_target_selector(targets)
        selector, metrics = self.module.fit_scene_token_student(
            replay,
            teacher_token_fn=teacher_fn,
            hidden_dim=8,
            epochs=250,
            learning_rate=0.05,
            seed=17,
        )

        predicted = self.module.predict_scene_token_student(selector, replay["scenes"][0])

        self.assertIn("wide_right", selector["token_order"])
        self.assertGreater(metrics["scene_accuracy"], 0.7)
        self.assertEqual("wide_right", predicted)

    def test_cli_writes_student_model_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            targets_path = tmp_path / "targets.jsonl"
            output_model = tmp_path / "student.json"
            report_json = tmp_path / "eval.json"
            report_md = tmp_path / "eval.md"
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
                "--output",
                str(output_model),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_md),
                "--epochs",
                "250",
                "--hidden-dim",
                "8",
                "--seed",
                "12",
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_model.read_text(encoding="utf-8"))
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_maneuvertoken_selector_mlp_v1", payload["model_type"])
            self.assertEqual("nuplan_bootstrap_student_generator_eval_v1", report["schema"])
            self.assertIn("Bootstrap Student Generator", report_md.read_text(encoding="utf-8"))


def _teacher_targets(replay: dict, *, teacher_token: str = "slow_yield") -> list[dict]:
    return [
        {
            "scene_id": scene["scene_id"],
            "source_db_file": scene["source_db_file"],
            "teacher": "replay_oracle",
            "teacher_token": teacher_token,
            "proxy_token": scene["selected_token"],
        }
        for scene in replay["scenes"]
    ]


def _expanded_replay_report() -> dict:
    replay = _replay_report()
    for scene in replay["scenes"]:
        scene["selected_token"] = "maintain"
        scene["candidates"].append(
            {
                "token": "wide_right",
                "speed_scale": 0.85,
                "lateral_offset_m": -2.0,
                "proxy_safe": True,
                "min_proxy_clearance_m": 3.5,
                "final_progress_m": 11.0,
                "score": 7.5,
                "poses": [[1.6 * step, -0.25 * step, 0.0] for step in range(1, 9)],
            }
        )
        scene["candidate_replay_evaluations"].append(
            {
                "token": "wide_right",
                "realized_min_clearance_m": 3.5,
                "realized_near_miss": False,
            }
        )
        scene["oracle_log_replay_safe_token"] = "wide_right"
    return replay


if __name__ == "__main__":
    unittest.main()
