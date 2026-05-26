from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from tests.test_train_nuplan_replay_calibrated_selector import _candidate
from tests.test_train_nuplan_replay_calibrated_selector import _replay
from tests.test_train_nuplan_replay_calibrated_selector import _replay_report


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train_nuplan_regime_value_selector.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TrainNuPlanRegimeValueSelectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "train_nuplan_regime_value_selector")

    def test_regime_type_splits_marginal_and_intervention_recovery(self) -> None:
        marginal = _scene_with_tokens(nudge_safe=True, stop_safe=True)
        intervention = _scene_with_tokens(nudge_safe=False, stop_safe=True)

        self.assertEqual(
            "marginal_recovery_available",
            self.module.scene_regime_type(
                marginal,
                near_miss_threshold_m=1.0,
                marginal_progress_retention=0.5,
            ),
        )
        self.assertEqual(
            "intervention_recovery_required",
            self.module.scene_regime_type(
                intervention,
                near_miss_threshold_m=1.0,
                marginal_progress_retention=0.5,
            ),
        )

    def test_oracle_prefers_marginal_over_low_progress_intervention_when_available(self) -> None:
        scene = _scene_with_tokens(nudge_safe=True, stop_safe=True)

        selected = self.module.select_regime_objective_oracle(
            scene,
            near_miss_threshold_m=1.0,
            marginal_progress_retention=0.5,
            fail_penalty=250.0,
            progress_weight=1.0,
            ade_weight=0.0,
            marginal_bonus=60.0,
            intervention_bonus=40.0,
            low_retention_penalty=80.0,
            safe_proxy_keep_bonus=60.0,
            safe_proxy_switch_penalty=25.0,
        )

        self.assertEqual("nudge_right", selected)

    def test_oracle_allows_intervention_when_no_marginal_safe_token_exists(self) -> None:
        scene = _scene_with_tokens(nudge_safe=False, stop_safe=True)

        selected = self.module.select_regime_objective_oracle(
            scene,
            near_miss_threshold_m=1.0,
            marginal_progress_retention=0.5,
            fail_penalty=250.0,
            progress_weight=1.0,
            ade_weight=0.0,
            marginal_bonus=60.0,
            intervention_bonus=40.0,
            low_retention_penalty=80.0,
            safe_proxy_keep_bonus=60.0,
            safe_proxy_switch_penalty=25.0,
        )

        self.assertEqual("stop", selected)

    def test_cli_main_writes_model_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "replay.json"
            output_path = tmp_path / "selector.json"
            report_json = tmp_path / "eval.json"
            report_md = tmp_path / "eval.md"
            input_path.write_text(json.dumps(_replay_report()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-replay-json",
                str(input_path),
                "--output",
                str(output_path),
                "--report-json",
                str(report_json),
                "--report-markdown",
                str(report_md),
                "--hidden-dim",
                "8",
                "--epochs",
                "180",
                "--seed",
                "7",
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_replay_value_mlp_selector_v1", payload["model_type"])
            self.assertEqual("nuplan_regime_value_selector_eval_v1", report["schema"])
            self.assertIn("Regime-Value Selector", report_md.read_text(encoding="utf-8"))


def _scene_with_tokens(*, nudge_safe: bool, stop_safe: bool) -> dict:
    scene = json.loads(json.dumps(_replay_report()["scenes"][0]))
    scene["candidates"] = [
        _candidate("maintain", proxy_safe=True, clearance=2.0, progress=16.0, score=8.0),
        _candidate("nudge_right", proxy_safe=True, clearance=1.5, progress=12.0, score=6.0),
        _candidate("stop", proxy_safe=True, clearance=2.0, progress=2.0, score=3.0),
    ]
    scene["candidate_replay_evaluations"] = [
        _replay("maintain", clearance=0.2),
        _replay("nudge_right", clearance=1.5 if nudge_safe else 0.2),
        _replay("stop", clearance=1.5 if stop_safe else 0.2),
    ]
    scene["oracle_log_replay_safe_token"] = "nudge_right" if nudge_safe else ("stop" if stop_safe else None)
    return scene


if __name__ == "__main__":
    unittest.main()
