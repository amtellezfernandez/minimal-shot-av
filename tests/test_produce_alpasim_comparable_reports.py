from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ProduceAlpaSimComparableReportsTests(unittest.TestCase):
    def test_produces_published_and_front_camera_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            published_ours = tmp_path / "published_ours.json"
            front_ours = tmp_path / "front_ours.json"
            front_alpamayo = tmp_path / "front_alpamayo.json"
            baseline = tmp_path / "baseline.json"
            output_dir = tmp_path / "out"

            published_ours.write_text('{"alpasim_score": 0.83}\n', encoding="utf-8")
            front_ours.write_text('{"alpasim_score": 0.70}\n', encoding="utf-8")
            front_alpamayo.write_text('{"alpasim_score": 0.69}\n', encoding="utf-8")
            baseline.write_text(
                json.dumps(
                    {
                        "system": "alpamayo_1_5",
                        "suite": "physicalai_nurec_alpasim",
                        "source": "unit-test",
                        "metadata": {
                            "evaluation_contract": "physicalai_nurec_alpasim_closed_loop",
                            "scenario_set": "physicalai_av_nurec_910",
                            "score_backend": "alpasim",
                            "sensor_contract": "published_alpamayo_1_5_model_card",
                            "camera_ids": "published_multi_camera_rgb",
                            "context_length": "published_alpamayo_1_5",
                            "ego_history_hz": 10,
                            "output_horizon": "6.4s_64_waypoints_10hz",
                            "route_command_source": "navigation_guidance",
                            "alpasim_version": "published_model_card_unspecified",
                        },
                        "metrics": {
                            "alpasim_score": {
                                "value": 0.81,
                                "higher_is_better": True,
                                "unit": "score",
                                "uncertainty": 0.01,
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "produce_alpasim_comparable_reports.py"),
                    "--published-ours-run",
                    str(published_ours),
                    "--front-ours-run",
                    str(front_ours),
                    "--front-alpamayo-run",
                    str(front_alpamayo),
                    "--published-baseline",
                    str(baseline),
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            published_comparison = json.loads(
                (output_dir / "comparisons" / "spotlight_reflex_vs_alpamayo_1_5_published_contract.json").read_text(
                    encoding="utf-8"
                )
            )
            front_comparison = json.loads(
                (output_dir / "comparisons" / "spotlight_reflex_vs_alpamayo_1_5_front_camera.json").read_text(
                    encoding="utf-8"
                )
            )
            front_report = json.loads(
                (output_dir / "spotlight_reflex_alpasim_front_camera.json").read_text(encoding="utf-8")
            )

        self.assertTrue(published_comparison["beats_all_shared_metrics"])
        self.assertTrue(front_comparison["beats_all_shared_metrics"])
        self.assertEqual("front_wide_120fov_only", front_report["metadata"]["sensor_contract"])


if __name__ == "__main__":
    unittest.main()
