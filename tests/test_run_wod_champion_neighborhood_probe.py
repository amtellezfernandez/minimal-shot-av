from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_wod_champion_neighborhood_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_wod_champion_neighborhood_probe", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodChampionNeighborhoodProbeTests(unittest.TestCase):
    def test_local_command_uses_champion_kinematic_recipe(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            local_frames=80,
            local_folds=2,
            official_frames=20,
            official_folds=2,
            waymo_src=ROOT / "workspace" / "waymo-open-dataset/src",
        )

        command = module._build_command(
            args,
            backend="local",
            output=ROOT / "out.json",
            source_gate_ridge=0.12,
            source_gate_max_rate=0.30,
            scene_gate_ridge=2.0,
            scene_gate_max_rate=0.16,
            selector_postprocess="off",
        )

        self.assertEqual("internnav", command[command.index("--kinematic-profile") + 1])
        self.assertEqual("contextual", command[command.index("--selector-features") + 1])
        self.assertEqual("independent_train_margin", command[command.index("--source-gate") + 1])
        self.assertEqual("0.12", command[command.index("--source-gate-ridge") + 1])
        self.assertEqual("0.3", command[command.index("--source-gate-max-rate") + 1])
        self.assertEqual("2.0", command[command.index("--scene-gate-ridge") + 1])
        self.assertEqual("0.16", command[command.index("--scene-gate-max-rate") + 1])

    def test_variants_include_tighter_and_looser_gate_settings(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            frame_cache=ROOT / "frames.json",
            external_embedding_cache=ROOT / "cache.json",
            neural_candidate_models="models.json",
            local_frames=80,
            local_folds=2,
            official_frames=20,
            official_folds=2,
            waymo_src=ROOT / "workspace" / "waymo-open-dataset/src",
        )

        local_variants = module._local_variants(args)
        official_variants = module._official_variants(args)

        self.assertEqual(
            ["base", "source_tighter", "source_looser", "scene_tighter", "scene_looser", "balanced"],
            [v["label"] for v in local_variants],
        )
        self.assertEqual(
            ["base", "source_tighter", "scene_tighter", "balanced"],
            [v["label"] for v in official_variants],
        )

    def test_best_report_prefers_rfs_then_source_gate_overrides(self) -> None:
        module = _load_module()
        reports = [
            {
                "label": "a",
                "report": {
                    "combined_ranker_mean_rfs": 7.7,
                    "source_gate_override_count": 2,
                    "scene_gate_precision": 0.4,
                },
            },
            {
                "label": "b",
                "report": {
                    "combined_ranker_mean_rfs": 7.7,
                    "source_gate_override_count": 3,
                    "scene_gate_precision": 0.1,
                },
            },
            {
                "label": "c",
                "report": {
                    "combined_ranker_mean_rfs": 7.6,
                    "source_gate_override_count": 10,
                    "scene_gate_precision": 1.0,
                },
            },
        ]

        best = module._best_report(reports)

        self.assertEqual("b", best["label"])


if __name__ == "__main__":
    unittest.main()
