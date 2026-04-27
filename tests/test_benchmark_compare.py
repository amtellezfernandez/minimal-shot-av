from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from minimal_shot_av.neutral.benchmark_compare import (
    ALPASIM_COMPARISON_METADATA_KEYS,
    FAIR_COMPARISON_METADATA_KEYS,
    compare_reports,
    load_metric_report,
    parse_metric_report,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCompareTests(unittest.TestCase):
    def wod_metadata(self, *, selection_mode: str = "first", frame_count: int = 479) -> dict[str, object]:
        return {
            "evaluation_contract": "wod_e2e_rfs",
            "split": "validation_479_preference_frames",
            "frame_count": frame_count,
            "score_backend": "official_waymo_rfs",
            "selection_mode": selection_mode,
        }

    def alpasim_metadata(self, *, camera_ids: object = ("camera_front_wide_120fov",)) -> dict[str, object]:
        return {
            "evaluation_contract": "physicalai_nurec_alpasim_closed_loop",
            "scenario_set": "physicalai_av_nurec_910",
            "score_backend": "alpasim",
            "sensor_contract": "front_wide_120fov_only",
            "camera_ids": list(camera_ids) if isinstance(camera_ids, tuple) else camera_ids,
            "context_length": 1,
            "ego_history_hz": 10,
            "output_horizon": "5.0s_20_waypoints_4hz",
            "route_command_source": "waypoint_commands",
            "alpasim_version": "test-version",
        }

    def test_compares_shared_metrics_with_direction(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "wod_e2e_val",
                "source": "unit-test",
                "metadata": self.wod_metadata(),
                "metrics": {
                    "rfs_overall": {"value": 7.2, "unit": "RFS", "higher_is_better": True},
                    "ade_5s": {"value": 3.0, "unit": "m", "higher_is_better": False},
                },
            }
        )
        baseline = parse_metric_report(
            {
                "system": "baseline",
                "suite": "wod_e2e_val",
                "source": "unit-test",
                "metadata": self.wod_metadata(),
                "metrics": {
                    "rfs_overall": {"value": 6.2, "unit": "RFS", "higher_is_better": True},
                    "ade_5s": {"value": 2.8, "unit": "m", "higher_is_better": False},
                },
            }
        )

        comparison = compare_reports(subject, baseline)

        by_metric = {item.metric: item for item in comparison.comparisons}
        self.assertTrue(by_metric["rfs_overall"].beats_baseline)
        self.assertFalse(by_metric["ade_5s"].beats_baseline)
        self.assertFalse(comparison.beats_all_shared_metrics)

    def test_rejects_wod_comparison_with_different_selection_mode(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "wod_e2e_val479",
                "source": "unit-test",
                "metadata": self.wod_metadata(selection_mode="ranker"),
                "metrics": {"rfs": {"value": 7.5, "unit": "RFS"}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "baseline",
                "suite": "wod_e2e_val479",
                "source": "unit-test",
                "metadata": self.wod_metadata(selection_mode="first"),
                "metrics": {"rfs": {"value": 7.0, "unit": "RFS"}},
            }
        )

        with self.assertRaisesRegex(ValueError, "different selection_mode"):
            compare_reports(subject, baseline)

    def test_rejects_wod_comparison_with_missing_metadata(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "wod_e2e_val479",
                "source": "unit-test",
                "metadata": self.wod_metadata(),
                "metrics": {"rfs": {"value": 7.5, "unit": "RFS"}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "baseline",
                "suite": "wod_e2e_val479",
                "source": "unit-test",
                "metrics": {"rfs": {"value": 7.0, "unit": "RFS"}},
            }
        )

        with self.assertRaisesRegex(ValueError, "incomplete metadata"):
            compare_reports(subject, baseline)

    def test_rejects_cross_suite_comparison(self) -> None:
        subject = parse_metric_report(
            {"system": "ours", "suite": "wod_e2e_val", "source": "unit-test", "metrics": {"rfs": 5.0}}
        )
        baseline = parse_metric_report(
            {"system": "alpamayo", "suite": "physicalai_nurec_alpasim", "source": "unit-test", "metrics": {"rfs": 8.0}}
        )

        with self.assertRaisesRegex(ValueError, "different suites"):
            compare_reports(subject, baseline)

    def test_rejects_runtime_comparison_with_different_contract(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "wod_e2e_online_runtime",
                "source": "unit-test",
                "metadata": {
                    "runtime_contract": "online_wod_non_text_runtime",
                    "excludes_io_and_training": True,
                },
                "metrics": {"p95_total_latency_ms": {"value": 1.0, "unit": "ms", "higher_is_better": False}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "baseline",
                "suite": "wod_e2e_online_runtime",
                "source": "unit-test",
                "metadata": {
                    "runtime_contract": "end_to_end_with_decode",
                    "excludes_io_and_training": True,
                },
                "metrics": {"p95_total_latency_ms": {"value": 14.0, "unit": "ms", "higher_is_better": False}},
            }
        )

        with self.assertRaisesRegex(ValueError, "different runtime_contract"):
            compare_reports(subject, baseline)

    def test_rejects_alpasim_comparison_with_different_camera_contract(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(camera_ids=("camera_front_wide_120fov",)),
                "metrics": {"alpasim_score": {"value": 0.82, "unit": "score"}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "alpamayo_1_5",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(camera_ids="published_multi_camera_rgb"),
                "metrics": {"alpasim_score": {"value": 0.81, "unit": "score"}},
            }
        )

        with self.assertRaisesRegex(ValueError, "different camera_ids"):
            compare_reports(subject, baseline)

    def test_rejects_alpasim_comparison_with_unknown_metadata(self) -> None:
        subject_metadata = self.alpasim_metadata()
        subject_metadata["scenario_set"] = "unknown"
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": subject_metadata,
                "metrics": {"alpasim_score": {"value": 0.82, "unit": "score"}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "baseline",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(),
                "metrics": {"alpasim_score": {"value": 0.81, "unit": "score"}},
            }
        )

        with self.assertRaisesRegex(ValueError, "unknown metadata"):
            compare_reports(subject, baseline)

    def test_rejects_missing_numeric_metric_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "numeric value"):
            parse_metric_report(
                {
                    "system": "broken",
                    "suite": "wod_e2e_val",
                    "source": "unit-test",
                    "metrics": {"rfs_overall": {"value": "unknown"}},
                }
            )

    def test_checked_in_baselines_have_numeric_metrics(self) -> None:
        for path in sorted((ROOT / "benchmarks" / "baselines").glob("*.json")):
            report = load_metric_report(path)
            self.assertTrue(report.metrics)
            for metric in report.metrics.values():
                self.assertIsInstance(metric.value, float)

    def test_checked_in_wod_reports_have_fairness_metadata(self) -> None:
        report_paths = sorted((ROOT / "benchmarks").glob("*/*.json"))
        wod_reports = [
            load_metric_report(path)
            for path in report_paths
            if '"suite": "wod_e2e' in path.read_text() and '"metrics"' in path.read_text()
        ]
        self.assertTrue(wod_reports)
        for report in wod_reports:
            if report.suite.endswith("_online_runtime"):
                continue
            for key in FAIR_COMPARISON_METADATA_KEYS:
                self.assertIn(key, report.metadata, report.source)
                self.assertNotEqual("unknown", report.metadata[key], report.source)

    def test_checked_in_alpasim_baselines_have_fairness_metadata(self) -> None:
        report_paths = sorted((ROOT / "benchmarks" / "baselines").glob("*alpasim*.json"))
        self.assertTrue(report_paths)
        for path in report_paths:
            report = load_metric_report(path)
            for key in ALPASIM_COMPARISON_METADATA_KEYS:
                self.assertIn(key, report.metadata, report.source)
                self.assertNotEqual("unknown", report.metadata[key], report.source)

    def test_cli_output_shape_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            subject_path = Path(tmp) / "subject.json"
            baseline_path = Path(tmp) / "baseline.json"
            subject_path.write_text(
                json.dumps(
                    {
                        "system": "ours",
                        "suite": "physicalai_nurec_alpasim",
                        "source": "unit-test",
                        "metadata": self.alpasim_metadata(),
                        "metrics": {"alpasim_score": {"value": 0.82, "unit": "score"}},
                    }
                ),
                encoding="utf-8",
            )
            baseline_path.write_text(
                json.dumps(
                    {
                        "system": "alpamayo_1_5",
                        "suite": "physicalai_nurec_alpasim",
                        "source": "unit-test",
                        "metadata": self.alpasim_metadata(),
                        "metrics": {"alpasim_score": {"value": 0.81, "unit": "score"}},
                    }
                ),
                encoding="utf-8",
            )

            comparison = compare_reports(load_metric_report(subject_path), load_metric_report(baseline_path))

        self.assertTrue(comparison.beats_all_shared_metrics)
        self.assertEqual("alpamayo_1_5", comparison.baseline_system)

    def test_uncertainty_margin_prevents_overclaiming_tie(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(),
                "metrics": {"alpasim_score": {"value": 0.82, "unit": "score"}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "alpamayo_1_5",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(),
                "metrics": {"alpasim_score": {"value": 0.81, "unit": "score", "uncertainty": 0.01}},
            }
        )

        comparison = compare_reports(subject, baseline)

        self.assertFalse(comparison.beats_all_shared_metrics)
        self.assertEqual(0.01, comparison.comparisons[0].required_margin)

    def test_uncertainty_margin_allows_clear_win(self) -> None:
        subject = parse_metric_report(
            {
                "system": "ours",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(),
                "metrics": {"alpasim_score": {"value": 0.83, "unit": "score"}},
            }
        )
        baseline = parse_metric_report(
            {
                "system": "alpamayo_1_5",
                "suite": "physicalai_nurec_alpasim",
                "source": "unit-test",
                "metadata": self.alpasim_metadata(),
                "metrics": {"alpasim_score": {"value": 0.81, "unit": "score", "uncertainty": 0.01}},
            }
        )

        comparison = compare_reports(subject, baseline)

        self.assertTrue(comparison.beats_all_shared_metrics)


if __name__ == "__main__":
    unittest.main()
