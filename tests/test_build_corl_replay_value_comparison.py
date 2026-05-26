from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_corl_replay_value_comparison.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildCorlReplayValueComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "build_corl_replay_value_comparison")

    def test_build_report_extracts_expected_rows(self) -> None:
        report = self.module.build_report(
            scene_token=_scene_token_report(),
            replay_value_safe=_value_report(progress=12.2, ade=9.1, entropy=3.08),
            replay_value_balanced=_value_report(progress=14.2, ade=8.4, entropy=3.11),
            scene_token_json="scene.json",
            replay_value_safe_json="safe.json",
            replay_value_balanced_json="balanced.json",
        )

        self.assertEqual("corl_replay_value_comparison_v1", report["schema"])
        self.assertEqual(5, len(report["rows"]))
        self.assertEqual("Scene-token student", report["rows"][1]["label"])
        self.assertAlmostEqual(0.097, report["rows"][3]["replay_fail_rate"], places=3)
        self.assertIn("Replay-value, balanced", self.module.report_markdown(report))
        self.assertIn("id,label,replay_fail_count", self.module.report_csv(report))
        self.assertIn("<svg", self.module.report_svg(report))

    def test_cli_writes_json_csv_svg_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            scene_path = tmp_path / "scene.json"
            safe_path = tmp_path / "safe.json"
            balanced_path = tmp_path / "balanced.json"
            output_json = tmp_path / "summary.json"
            output_csv = tmp_path / "summary.csv"
            output_svg = tmp_path / "summary.svg"
            output_md = tmp_path / "summary.md"
            scene_path.write_text(json.dumps(_scene_token_report()), encoding="utf-8")
            safe_path.write_text(json.dumps(_value_report(progress=12.2, ade=9.1, entropy=3.08)), encoding="utf-8")
            balanced_path.write_text(json.dumps(_value_report(progress=14.2, ade=8.4, entropy=3.11)), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--scene-token-json",
                str(scene_path),
                "--replay-value-safe-json",
                str(safe_path),
                "--replay-value-balanced-json",
                str(balanced_path),
                "--output-json",
                str(output_json),
                "--output-csv",
                str(output_csv),
                "--output-svg",
                str(output_svg),
                "--output-markdown",
                str(output_md),
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("corl_replay_value_comparison_v1", payload["schema"])
            self.assertIn("scene-token student", output_csv.read_text(encoding="utf-8").lower())
            self.assertIn("<svg", output_svg.read_text(encoding="utf-8"))
            self.assertIn("Expanded-Bank Replay-Value Comparison", output_md.read_text(encoding="utf-8"))


def _scene_token_report() -> dict:
    return {
        "holdout": {
            "g0_proxy_top1": _row(222, 0.740, 31.181, 2.714, 4.136, 2.773),
            "g1_student_top1": _row(138, 0.460, 23.062, 4.737, 7.555, 2.471),
        }
    }


def _value_report(*, progress: float, ade: float, entropy: float) -> dict:
    return {
        "holdout": {
            "replay_value": _row(29, 0.097, progress, ade, ade + 5.0, entropy),
            "replay_oracle": _row(29, 0.097, 11.258, 9.601, 15.488, 3.053),
        }
    }


def _row(fail_count: int, fail_rate: float, progress: float, ade: float, fde: float, entropy: float) -> dict:
    return {
        "selected_replay_infeasible_count": fail_count,
        "selected_replay_infeasible_rate": fail_rate,
        "mean_selected_progress_m": progress,
        "mean_ade_3s_m": ade,
        "mean_fde_3s_m": fde,
        "selected_token_entropy": entropy,
    }


if __name__ == "__main__":
    unittest.main()
