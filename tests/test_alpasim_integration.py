from __future__ import annotations

from tempfile import TemporaryDirectory
from pathlib import Path
import sys
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.alpasim_metrics import build_alpasim_evidence, load_alpasim_metrics


class AlpaSimIntegrationTests(unittest.TestCase):
    def test_pyproject_registers_alpasim_plugin_entrypoints(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(
            pyproject["project"]["entry-points"]["alpasim.models"]["spotlight_reflex"],
            "minimal_shot_av.alpasim_spotlight:SpotlightReflexAlpaSimModel",
        )
        self.assertEqual(
            pyproject["project"]["entry-points"]["alpasim.configs"]["spotlight_reflex"],
            "minimal_shot_av.alpasim_configs",
        )

    def test_alpasim_driver_config_exists(self) -> None:
        config_path = Path("src/minimal_shot_av/alpasim_configs/driver/spotlight_reflex.yaml")
        config = config_path.read_text()
        self.assertIn("model_type: spotlight_reflex", config)
        self.assertIn("output_frequency_hz: 4", config)

    def test_imports_alpasim_aggregate_text_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            aggregate_dir = run_dir / "aggregate"
            aggregate_dir.mkdir()
            metrics_file = aggregate_dir / "metrics_results.txt"
            metrics_file.write_text(
                "\n".join(
                    [
                        "Run: spotlight_reflex",
                        "n_clips: 4, n_rollouts/clip: 2",
                        "collision_at_fault        0.00 ± 0.00      MAX",
                        "offroad                   0.01 ± 0.00      MAX",
                        "dist_to_gt_trajectory     2.40 ± 0.50      MAX",
                        "safety_monitor_triggered  0.00 ± 0.00      MAX",
                    ]
                ),
                encoding="utf-8",
            )

            metrics_path, metrics, run_count = load_alpasim_metrics(run_dir)
            evidence = build_alpasim_evidence(run_dir).to_dict()

        self.assertEqual(metrics_path.name, "metrics_results.txt")
        self.assertEqual(run_count, 8)
        self.assertEqual(metrics["collision_at_fault"], 0.0)
        self.assertEqual(evidence["run_count"], 8)
        self.assertTrue(evidence["sensor_realistic"])
        self.assertFalse(evidence["official_compass_score"])
        self.assertTrue(evidence["gates"]["collision_rate"])
        self.assertTrue(evidence["gates"]["route_deviation_m"])

    def test_imports_alpasim_json_metrics(self) -> None:
        with TemporaryDirectory() as tmp:
            metrics_file = Path(tmp) / "metrics_results.json"
            metrics_file.write_text(
                """{
                  "run_count": 3,
                  "collision_at_fault": {"mean": 0.02, "std": 0.01},
                  "offroad": 0.0,
                  "dist_to_gt_trajectory": 4.0,
                  "safety_monitor_triggered": 0.0
                }""",
                encoding="utf-8",
            )

            evidence = build_alpasim_evidence(metrics_file).to_dict()

        self.assertEqual(evidence["run_count"], 3)
        self.assertFalse(evidence["gates"]["collision_rate"])
        self.assertFalse(evidence["gates"]["route_deviation_m"])


if __name__ == "__main__":
    unittest.main()
