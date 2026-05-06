from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_rlvr_curriculum.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_rlvr_curriculum", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildRlvrCurriculumTests(unittest.TestCase):
    def test_curriculum_declares_gym_contract_and_graders(self) -> None:
        module = _load_module()

        curriculum = module.build_curriculum(seed_start=1, seed_end=1)

        self.assertEqual("minimal_shot_av_rlvr_curriculum_v1", curriculum["schema"])
        self.assertIn("observation_modes", curriculum["gym"])
        self.assertIn("action_space", curriculum["gym"])
        self.assertIn("benchmark_pass", curriculum["grader_library"])
        self.assertGreater(curriculum["coverage"]["task_count"], 0)
        self.assertGreaterEqual(curriculum["coverage"]["grader_count"], 5)

    def test_tasks_have_splits_axes_observation_and_grader_contracts(self) -> None:
        module = _load_module()

        curriculum = module.build_curriculum(seed_start=1, seed_end=3)
        tasks = curriculum["tasks"]

        self.assertTrue(tasks)
        self.assertTrue({task["split"] for task in tasks} & {"train_curriculum", "blind_holdout"})
        for task in tasks:
            self.assertIn("composition_key", task)
            self.assertIn("primary_hazard_type", task["axes"])
            self.assertFalse(task["observation_contract"]["privileged_state_allowed"])
            self.assertIn("benchmark_pass", task["grader_ids"])
            self.assertEqual("bounded_waypoint_velocity_control", task["action_space_id"])

    def test_cli_writes_curriculum_json(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "curriculum.json"
            payload = module.build_curriculum(seed_start=1, seed_end=1)
            output.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual("minimal_shot_av_rlvr_curriculum_v1", loaded["schema"])


if __name__ == "__main__":
    unittest.main()
