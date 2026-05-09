from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_source_gate_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_source_gate_probe", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodSourceGateProbeTests(unittest.TestCase):
    def test_base_command_uses_champion_source_gate_recipe(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            output_dir=ROOT / "artifacts" / "probe",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            local_frames=80,
            local_folds=2,
            official_frames=20,
            official_folds=2,
            waymo_src=ROOT / "waymo",
        )

        command = module._base_command(
            args,
            output=ROOT / "out.json",
            backend="local",
            source_gate_ridge=0.12,
            source_gate_max_rate=0.30,
            scene_gate_ridge=2.0,
            scene_gate_max_rate=0.16,
        )

        self.assertEqual("off", command[command.index("--selector-postprocess") + 1])
        self.assertEqual("independent_train_margin", command[command.index("--source-gate") + 1])
        self.assertEqual("0.12", command[command.index("--source-gate-ridge") + 1])
        self.assertEqual("0.3", command[command.index("--source-gate-max-rate") + 1])
        self.assertEqual("2.0", command[command.index("--scene-gate-ridge") + 1])
        self.assertEqual("0.16", command[command.index("--scene-gate-max-rate") + 1])

    def test_local_variants_include_tight_and_loose_candidates(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            output_dir=ROOT / "artifacts" / "probe",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            local_frames=80,
            local_folds=2,
            official_frames=20,
            official_folds=2,
            waymo_src=ROOT / "waymo",
        )

        variants = module._local_variants(args)

        labels = [label for label, _ in variants]
        self.assertEqual(["base", "source_tight", "source_loose", "scene_tight", "scene_loose", "balanced"], labels)

    def test_best_report_prefers_rfs_then_precision(self) -> None:
        module = _load_module()
        reports = [
            {"label": "a", "report": {"combined_ranker_mean_rfs": 7.1, "source_gate_precision": 0.4}},
            {"label": "b", "report": {"combined_ranker_mean_rfs": 7.1, "source_gate_precision": 0.5}},
            {"label": "c", "report": {"combined_ranker_mean_rfs": 7.0, "source_gate_precision": 1.0}},
        ]

        best = module._best_report(reports)

        self.assertEqual("b", best["label"])


if __name__ == "__main__":
    unittest.main()
