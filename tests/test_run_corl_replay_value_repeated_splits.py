from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_corl_replay_value_repeated_splits.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunCorlReplayValueRepeatedSplitsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "run_corl_replay_value_repeated_splits")

    def test_repeated_split_report_contains_expected_rows(self) -> None:
        report = self.module.run_repeated_splits(
            _replay_report(),
            input_replay_json="synthetic.json",
            split_count=2,
            seed_start=0,
            holdout_fraction=0.5,
            near_miss_threshold_m=1.0,
            scene_token_iterations=1,
            scene_token_hidden_dim=8,
            scene_token_epochs=120,
            scene_token_learning_rate=0.05,
            scene_token_regret_sample_weight=2.0,
            scene_token_changed_target_weight=1.0,
            replay_value_hidden_dim=8,
            replay_value_epochs=120,
            replay_value_learning_rate=0.05,
            fail_penalty=250.0,
            safe_progress_weight=1.0,
            balanced_progress_weight=2.0,
            ade_weight=2.0,
            clearance_weight=0.0,
        )

        self.assertEqual("corl_replay_value_repeated_splits_v1", report["schema"])
        self.assertEqual(2, len(report["splits"]))
        self.assertEqual(5, len(report["rows"]))
        self.assertEqual("Proxy top-1", report["rows"][0]["label"])
        self.assertEqual("Replay-value, balanced", report["rows"][3]["label"])
        self.assertIn("Scene-token student", self.module.report_markdown(report))
        csv_text = self.module.report_csv(report)
        self.assertIn("replay_fail_rate_mean", csv_text)
        self.assertIn('"Replay-value, balanced"', csv_text)

    def test_cli_main_writes_json_csv_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_json = tmp_path / "summary.json"
            output_csv = tmp_path / "summary.csv"
            output_md = tmp_path / "summary.md"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-markdown",
                str(output_md),
                "--split-count",
                "2",
                "--scene-token-iterations",
                "1",
                "--scene-token-hidden-dim",
                "8",
                "--scene-token-epochs",
                "80",
                "--replay-value-hidden-dim",
                "8",
                "--replay-value-epochs",
                "80",
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("corl_replay_value_repeated_splits_v1", payload["schema"])
            self.assertIn("replay_fail_rate_mean", output_csv.read_text(encoding="utf-8"))
            self.assertIn("CoRL Replay-Value Repeated-Split Summary", output_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
