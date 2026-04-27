from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from minimal_shot_av.neutral.benchmark_compare import compare_reports, load_metric_report, parse_metric_report
from minimal_shot_av.neutral.benchmark_reports import (
    alpasim_metrics_report_to_metric_report,
    runtime_report_to_metric_report,
    scenario_eval_report_to_metric_report,
    wod_cv_report_to_metric_report,
    wod_eval_report_to_metric_report,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkReportTests(unittest.TestCase):
    def alpasim_published_metadata(self) -> dict[str, object]:
        return {
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
        }

    def test_builds_metric_report_from_wod_eval_artifact(self) -> None:
        report = wod_eval_report_to_metric_report(
            ROOT / "artifacts" / "model_diagnostic_logged_future_official.json",
            system="spotlight_reflex_logged_future_smoke",
            suite="wod_e2e_smoke_frame",
        )

        parsed = parse_metric_report(report)

        self.assertEqual("wod_e2e_smoke_frame", parsed.suite)
        self.assertAlmostEqual(4.793513563812969, parsed.metrics["rfs"].value)
        self.assertEqual("RFS", parsed.metrics["rfs"].unit)
        self.assertEqual("wod_e2e_rfs", parsed.metadata["evaluation_contract"])
        self.assertEqual(1, parsed.metadata["frame_count"])
        self.assertEqual("official_waymo_rfs", parsed.metadata["score_backend"])

    def test_wod_report_preserves_candidate_headroom_metrics(self) -> None:
        report = wod_eval_report_to_metric_report(
            ROOT / "artifacts" / "wod_kinematic_val479_first_official.json",
            system="wod_kinematic_non_text_val479_first",
            suite="wod_e2e_val479",
        )

        parsed = parse_metric_report(report)

        self.assertAlmostEqual(7.852351864082588, parsed.metrics["best_candidate_rfs"].value)
        self.assertAlmostEqual(0.8299941665897286, parsed.metrics["selection_regret"].value)
        self.assertFalse(parsed.metrics["selection_regret"].higher_is_better)
        self.assertEqual(479, parsed.metadata["frame_count"])

    def test_builds_metric_report_from_wod_cv_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cv_path = Path(tmp) / "cv.json"
            cv_path.write_text(
                """
{
  "benchmark_type": "segment_grouped_cross_validation",
  "score_backend": "local_rfs_metric",
  "frames": 12,
  "fold_count": 3,
  "ridge": 30.0,
  "selector_ridge": 175.0,
  "selector_target": "frame_delta",
  "selector_features": "contextual",
  "selector_fallback_router": "speed",
  "combined_ranker_mean_rfs": 7.5,
  "combined_oracle_mean_rfs": 9.0,
  "combined_ranker_regret_to_oracle": 1.5,
  "combined_ranker_top1_oracle_match_rate": 0.4,
  "selected_temporal_rate": 0.6,
  "selected_kinematic_rate": 0.4,
  "notes": ["Simulator metrics are not used for model selection."]
}
""",
                encoding="utf-8",
            )

            report = wod_cv_report_to_metric_report(cv_path, system="ours", suite="wod_e2e_val479_local_cv")

        parsed = parse_metric_report(report)
        self.assertEqual("wod_e2e_val479_local_cv", parsed.suite)
        self.assertAlmostEqual(7.5, parsed.metrics["combined_ranker_mean_rfs"].value)
        self.assertFalse(parsed.metrics["combined_ranker_regret_to_oracle"].higher_is_better)
        self.assertEqual("local_rfs_metric", parsed.metadata["score_backend"])
        self.assertEqual("speed", parsed.metadata["selector_fallback_router"])
        self.assertIn("not a leaderboard/test score", " ".join(report["notes"]))

    def test_builds_metric_report_from_scenario_eval_artifact(self) -> None:
        report = scenario_eval_report_to_metric_report(
            ROOT / "artifacts" / "agnostic_sim_eval_wod_1_10" / "scenario_eval.json",
            system="spotlight_reflex_procedural_wod",
            suite="procedural_wod_simulator",
        )

        parsed = parse_metric_report(report)

        self.assertEqual(110.0, parsed.metrics["run_count"].value)
        self.assertEqual(1.0, parsed.metrics["success_rate"].value)
        self.assertEqual(0.0, parsed.metrics["collision_rate"].value)
        self.assertEqual(1.0, parsed.metrics["benchmark_pass_rate"].value)

    def test_builds_same_suite_alpasim_report_from_metrics_results(self) -> None:
        report = alpasim_metrics_report_to_metric_report(
            ROOT / "tests" / "fixtures" / "alpasim_metrics_results.json",
            system="ours",
            suite="physicalai_nurec_alpasim",
            required_metrics=("alpasim_score",),
            metadata=self.alpasim_published_metadata(),
        )
        baseline = load_metric_report(ROOT / "benchmarks" / "baselines" / "alpamayo_1_5_alpasim.json")

        parsed = parse_metric_report(report)
        self.assertEqual("physicalai_av_nurec_910", parsed.metadata["scenario_set"])
        comparison = compare_reports(parsed, baseline)
        self.assertEqual(0.82, parsed.metrics["alpasim_score"].value)
        self.assertFalse(parsed.metrics["collision_at_fault"].higher_is_better)
        self.assertFalse(comparison.beats_all_shared_metrics)
        self.assertEqual(0.01, comparison.comparisons[0].required_margin)

    def test_alpasim_report_can_require_published_comparison_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics_results.json"
            metrics_path.write_text('{"collision_at_fault": 0.0}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "missing required AlpaSim metric"):
                alpasim_metrics_report_to_metric_report(
                    metrics_path,
                    system="ours",
                    suite="physicalai_nurec_alpasim",
                    required_metrics=("alpasim_score",),
                )

    def test_builds_metric_report_from_runtime_artifact(self) -> None:
        report = runtime_report_to_metric_report(
            ROOT / "benchmarks" / "current" / "wod_online_runtime.json",
            system="ours",
            suite="wod_e2e_online_runtime",
        )

        parsed = parse_metric_report(report)

        self.assertEqual("wod_e2e_online_runtime", parsed.suite)
        self.assertLess(parsed.metrics["p95_total_latency_ms"].value, 14.0)
        self.assertFalse(parsed.metrics["p95_total_latency_ms"].higher_is_better)
        self.assertEqual("online_wod_non_text_runtime", parsed.metadata["runtime_contract"])

    def test_alpasim_report_rejects_unknown_metric_direction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            metrics_path = Path(tmp) / "metrics_results.json"
            metrics_path.write_text('{"alpasim_score": 0.82, "mystery_error": 1.0}\n', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Unknown AlpaSim metric direction"):
                alpasim_metrics_report_to_metric_report(
                    metrics_path,
                    system="ours",
                    suite="physicalai_nurec_alpasim",
                    required_metrics=("alpasim_score",),
                )

    def test_scenario_report_rejects_missing_required_boolean_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            scenario_path = Path(tmp) / "scenario_eval.json"
            scenario_path.write_text(
                '{"runs": [{"success": true, "benchmark_pass": true}]}\n',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Missing or non-boolean"):
                scenario_eval_report_to_metric_report(
                    scenario_path,
                    system="ours",
                    suite="procedural_wod_simulator",
                )


if __name__ == "__main__":
    unittest.main()
