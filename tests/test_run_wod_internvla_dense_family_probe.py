from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_internvla_dense_family_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_internvla_dense_family_probe", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunWodInternVlaDenseFamilyProbeTests(unittest.TestCase):
    def test_build_command_uses_dense_internvla_candidates_and_family_calibration(self) -> None:
        module = _load_module()
        args = MagicMock()
        args.python = sys.executable
        args.frame_cache = ROOT / "artifacts" / "wod_preference_frames_val479.json"
        args.external_embedding_cache = ROOT / "artifacts" / "cosmos_predict25_wan21_tokenizer_val479.json"
        args.external_candidate_jsonl = ROOT / "artifacts" / "wod_internvla_av_candidates_dense.jsonl"
        args.neural_candidate_models = "a.json,b.json,c.json"
        args.waymo_src = ROOT / "waymo_open-dataset" / "src"
        command = module._build_command(
            args,
            output=ROOT / "artifacts" / "x.json",
            variant=module._variants()[1],
            frames=20,
            folds=2,
        )

        self.assertIn("--external-candidate-jsonl", command)
        self.assertIn("wod_internvla_av_candidates_dense.jsonl", " ".join(command))
        self.assertIn("--selector-family-calibration", command)
        self.assertIn("speed_source_family", command)

    def test_best_report_prefers_highest_rfs_then_overrides(self) -> None:
        module = _load_module()
        reports = [
            {"report": {"combined_ranker_mean_rfs": 7.7, "source_gate_override_count": 1, "scene_gate_precision": 0.2}},
            {"report": {"combined_ranker_mean_rfs": 7.8, "source_gate_override_count": 0, "scene_gate_precision": 0.1}},
            {"report": {"combined_ranker_mean_rfs": 7.8, "source_gate_override_count": 2, "scene_gate_precision": 0.0}},
        ]

        best = module._best_report(reports)

        self.assertEqual(2, best["report"]["source_gate_override_count"])

    def test_main_writes_summary(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            out.mkdir()
            summary = out / "summary.json"
            report = {"combined_ranker_mean_rfs": 7.9, "source_gate_override_count": 1, "scene_gate_precision": 0.0}
            payload = json.dumps(report)

            def fake_run(command):
                label = Path(command[command.index("--output") + 1]).name
                Path(command[command.index("--output") + 1]).write_text(payload, encoding="utf-8")
                return 0

            with patch.object(module, "_run", side_effect=fake_run):
                old_argv = sys.argv
                sys.argv = [
                    "run_wod_internvla_dense_family_probe.py",
                    "--waymo-src",
                    str(ROOT / "waymo_open-dataset" / "src"),
                    "--output-dir",
                    str(out),
                    "--skip-full-official",
                ]
                try:
                    rc = module.main()
                finally:
                    sys.argv = old_argv

            self.assertEqual(0, rc)
            self.assertTrue(summary.is_file())


if __name__ == "__main__":
    unittest.main()
