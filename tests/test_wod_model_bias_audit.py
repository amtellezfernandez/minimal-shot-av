from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_wod_model_bias.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_wod_model_bias", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodModelBiasAuditTests(unittest.TestCase):
    def test_solution_reset_declares_world_model_result_as_insufficient(self) -> None:
        declaration = (ROOT / "models" / "DECLARATION.md").read_text(encoding="utf-8")
        reset = (ROOT / "docs" / "solution-reset.md").read_text(encoding="utf-8")

        self.assertIn("7.606198495114426", declaration)
        self.assertIn("small to claim a meaningful solution", declaration)
        self.assertIn("+0.1", reset)
        self.assertIn("not a finished autonomy architecture", reset)

    def test_declaration_matches_local_best_worst_slice(self) -> None:
        declaration = (ROOT / "models" / "DECLARATION.md").read_text(encoding="utf-8")
        audit = json.loads(
            (ROOT / "benchmarks" / "current" / "wod_model_bias_audit_local_best.json").read_text(encoding="utf-8")
        )
        worst = audit["slice_summary"]["worst_regret_slice"]

        self.assertIn(f"`{worst['slice']}`", declaration)
        self.assertIn(str(worst["selected_mean_rfs"]), declaration)
        self.assertIn(str(worst["oracle_mean_rfs"]), declaration)
        self.assertIn(str(worst["mean_regret"]), declaration)

    def test_current_benchmark_reports_known_bias_risks(self) -> None:
        module = _load_module()

        report = module.audit_bias(
            ROOT / "benchmarks" / "current" / "wod_ridge_trajectory_cv_official.json",
            ROOT / "models" / "DECLARATION.md",
        )

        risk_ids = {risk["id"] for risk in report["risks"]}
        mitigation_ids = {mitigation["id"] for mitigation in report["mitigations"]}
        self.assertEqual("warn", report["status"])
        self.assertIn("validation_preference_bias", risk_ids)
        self.assertIn("visual_blindness_bias", risk_ids)
        self.assertIn("selector_oracle_gap", risk_ids)
        self.assertIn("slice_regret_bias", risk_ids)
        self.assertIn("segment_grouped_cv", mitigation_ids)
        self.assertTrue(report["slice_summary"]["available"])
        self.assertEqual("intent:2", report["slice_summary"]["worst_regret_slice"]["slice"])
        self.assertFalse(report["summary"]["strict_zero_shot_claim_allowed"])

    def test_audit_flags_missing_cross_validation_metadata(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = root / "benchmark.json"
            declaration = root / "DECLARATION.md"
            benchmark.write_text(
                json.dumps(
                    {
                        "metadata": {"split": "validation"},
                        "metrics": {
                            "combined_ranker_mean_rfs": {"value": 5.0},
                            "combined_oracle_mean_rfs": {"value": 6.5},
                        },
                    }
                ),
                encoding="utf-8",
            )
            declaration.write_text("Input cameras: not used in current active path\n", encoding="utf-8")

            report = module.audit_bias(benchmark, declaration)

        self.assertIn("ungrouped_or_missing_cv", {risk["id"] for risk in report["risks"]})

    def test_audit_accepts_raw_cross_validation_report(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = root / "raw_cv.json"
            declaration = root / "DECLARATION.md"
            benchmark.write_text(
                json.dumps(
                    {
                        "benchmark_type": "segment_grouped_cross_validation",
                        "score_backend": "local_rfs_metric",
                        "frames": 4,
                        "fold_count": 2,
                        "combined_ranker_mean_rfs": 7.0,
                        "combined_oracle_mean_rfs": 8.6,
                        "selected_temporal_rate": 0.75,
                        "selected_kinematic_rate": 0.25,
                        "slices": {
                            "intent:3": {
                                "frames": 4,
                                "selected_mean_rfs": 6.0,
                                "oracle_mean_rfs": 8.0,
                                "mean_regret": 2.0,
                                "selected_source_rates": {"temporal": 1.0},
                                "oracle_source_rates": {"kinematic": 1.0},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            declaration.write_text(
                "Input cameras: not used in current active path\n"
                "not a leaderboard/test score\n"
                "simulator metrics were not used\n"
                "simulator metrics are not used\n",
                encoding="utf-8",
            )

            report = module.audit_bias(benchmark, declaration)

        self.assertEqual("warn", report["status"])
        self.assertIn("segment_grouped_cv", {mitigation["id"] for mitigation in report["mitigations"]})
        self.assertEqual("intent:3", report["slice_summary"]["worst_regret_slice"]["slice"])


if __name__ == "__main__":
    unittest.main()
