from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_wod_validation_breakthrough.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_wod_validation_breakthrough", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WodValidationBreakthroughAuditTests(unittest.TestCase):
    def test_audit_accepts_validation_breakthrough(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.86, normalized=8.1, regret=1.4)

            report = module.audit_breakthrough(candidate, baseline)

        self.assertTrue(report["passed"])
        self.assertEqual([], report["failures"])

    def test_audit_rejects_small_gain_and_worse_worst_slice(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.80, normalized=8.05, regret=1.8)

            report = module.audit_breakthrough(candidate, baseline)

        failure_ids = {failure["id"] for failure in report["failures"]}
        self.assertFalse(report["passed"])
        self.assertIn("selected_rfs_gain_too_small", failure_ids)
        self.assertIn("worst_slice_regret_worse", failure_ids)

    def test_audit_rejects_equal_rfs_when_no_min_gain_is_required(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.75, normalized=8.1, regret=1.4)

            report = module.audit_breakthrough(candidate, baseline, min_rfs_gain=0.0)

        self.assertFalse(report["passed"])
        self.assertIn("selected_rfs_gain_too_small", {failure["id"] for failure in report["failures"]})

    def test_audit_rejects_hidden_test_claim(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.90, normalized=8.2, regret=1.2)
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload["hidden_test_confirmed"] = True
            candidate.write_text(json.dumps(payload), encoding="utf-8")

            report = module.audit_breakthrough(candidate, baseline)

        self.assertFalse(report["passed"])
        self.assertIn("hidden_test_claim", {failure["id"] for failure in report["failures"]})

    def test_audit_rejects_incomparable_cv_protocols(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.90, normalized=8.2, regret=1.2)
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            payload["fold_count"] = 4
            candidate.write_text(json.dumps(payload), encoding="utf-8")

            report = module.audit_breakthrough(candidate, baseline)

        self.assertFalse(report["passed"])
        failures = [failure for failure in report["failures"] if failure["id"] == "not_comparable_cv_protocol"]
        self.assertEqual("fold_count", failures[0]["field"])

    def test_audit_rejects_best_observed_wrapper_that_was_not_promoted(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.90, normalized=8.2, regret=1.2)
            wrapper = root / "best_report.json"
            wrapper.write_text(
                json.dumps(
                    {
                        "schema": "wod_breakthrough_best_observed_report_v1",
                        "report_role": "best_observed_not_necessarily_promoted",
                        "promoted": False,
                        "best_report_path": str(candidate),
                    }
                ),
                encoding="utf-8",
            )

            report = module.audit_breakthrough(wrapper, baseline)

        self.assertFalse(report["passed"])
        self.assertIn(
            "candidate_wrapper_not_validation_cv_candidate",
            {failure["id"] for failure in report["failures"]},
        )

    def test_audit_rejects_legacy_pilot_promotion_wrapper(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.90, normalized=8.2, regret=1.2)
            wrapper = root / "promoted_report.json"
            wrapper.write_text(
                json.dumps(
                    {
                        "schema": "wod_breakthrough_promoted_report_v1",
                        "report_role": "promoted_candidate",
                        "promoted": True,
                        "report_path": str(candidate),
                    }
                ),
                encoding="utf-8",
            )

            report = module.audit_breakthrough(wrapper, baseline)

        self.assertFalse(report["passed"])
        self.assertIn(
            "candidate_wrapper_not_validation_cv_candidate",
            {failure["id"] for failure in report["failures"]},
        )

    def test_audit_accepts_validation_candidate_wrapper(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)
            candidate = _write_report(root / "candidate.json", rfs=7.90, normalized=8.2, regret=1.2)
            wrapper = root / "promoted_report.json"
            wrapper.write_text(
                json.dumps(
                    {
                        "schema": "wod_validation_breakthrough_candidate_report_v1",
                        "report_role": "validation_cv_candidate",
                        "promoted": True,
                        "report_path": str(candidate),
                    }
                ),
                encoding="utf-8",
            )

            report = module.audit_breakthrough(wrapper, baseline)

        self.assertTrue(report["passed"])

    def test_audit_reports_missing_promoted_candidate_without_traceback(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            baseline = _write_report(root / "baseline.json", rfs=7.75, normalized=8.0, regret=1.5)

            report = module.audit_breakthrough(root / "missing_promoted.json", baseline)

        self.assertFalse(report["passed"])
        self.assertIn("candidate_report_missing", {failure["id"] for failure in report["failures"]})


def _write_report(path: Path, *, rfs: float, normalized: float, regret: float) -> Path:
    path.write_text(
        json.dumps(
            {
                "benchmark_type": "segment_grouped_cross_validation",
                "fold_count": 5,
                "frames": 479,
                "score_backend": "local_rfs_metric",
                "combined_ranker_mean_rfs": rfs,
                "combined_ranker_mean_normalized_rfs": normalized,
                "slices": {"intent:3": {"frames": 10, "mean_regret": regret}},
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
