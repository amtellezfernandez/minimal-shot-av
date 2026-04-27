from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_wod_improvement_target.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_wod_improvement_target", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _report(*, selected: float, oracle: float, worst_regret: float) -> dict[str, object]:
    return {
        "combined_ranker_mean_rfs": selected,
        "combined_oracle_mean_rfs": oracle,
        "slices": {
            "intent:1": {"frames": 10, "mean_regret": 0.5},
            "intent:3": {"frames": 3, "mean_regret": worst_regret},
        },
    }


class AuditWodImprovementTargetTests(unittest.TestCase):
    def test_flags_target_above_candidate_oracle(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_report(selected=8.0, oracle=9.0, worst_regret=1.0)), encoding="utf-8")
            candidate.write_text(json.dumps(_report(selected=8.2, oracle=9.0, worst_regret=1.0)), encoding="utf-8")

            result = module.audit_improvement_target(
                candidate_path=candidate,
                baseline_path=baseline,
                target_relative_gain=0.25,
            )

        self.assertEqual("fail", result["status"])
        self.assertIn("target_above_candidate_oracle", {failure["id"] for failure in result["failures"]})

    def test_passes_when_target_and_worst_slice_clear(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_report(selected=8.0, oracle=9.0, worst_regret=1.0)), encoding="utf-8")
            candidate.write_text(json.dumps(_report(selected=10.0, oracle=10.0, worst_regret=0.9)), encoding="utf-8")

            result = module.audit_improvement_target(
                candidate_path=candidate,
                baseline_path=baseline,
                target_relative_gain=0.25,
            )

        self.assertEqual("pass", result["status"])

    def test_fails_when_slice_diagnostics_are_missing(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(
                json.dumps({"combined_ranker_mean_rfs": 8.0, "combined_oracle_mean_rfs": 8.0}),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps({"combined_ranker_mean_rfs": 10.0, "combined_oracle_mean_rfs": 10.0}),
                encoding="utf-8",
            )

            result = module.audit_improvement_target(
                candidate_path=candidate,
                baseline_path=baseline,
                target_relative_gain=0.25,
            )

        self.assertEqual("fail", result["status"])
        self.assertIn("missing_slice_diagnostics", {failure["id"] for failure in result["failures"]})

    def test_fails_when_baseline_score_is_not_positive(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(_report(selected=0.0, oracle=1.0, worst_regret=1.0)), encoding="utf-8")
            candidate.write_text(json.dumps(_report(selected=10.0, oracle=10.0, worst_regret=0.9)), encoding="utf-8")

            result = module.audit_improvement_target(
                candidate_path=candidate,
                baseline_path=baseline,
                target_relative_gain=0.25,
            )

        self.assertEqual("fail", result["status"])
        self.assertIn("invalid_baseline_score", {failure["id"] for failure in result["failures"]})


if __name__ == "__main__":
    unittest.main()
