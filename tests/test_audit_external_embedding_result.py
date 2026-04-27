from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_external_embedding_result.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_external_embedding_result", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report(
    *,
    rfs: float,
    world_model: str = "external_embeddings",
    embedding_source: str | None = "cosmos-test",
    worst_regret: float = 1.0,
    selected_world_rate: float = 0.0,
    oracle_world_rate: float = 0.0,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "world_model": world_model,
        "combined_ranker_mean_rfs": rfs,
        "selected_world_rate": selected_world_rate,
        "oracle_world_rate": oracle_world_rate,
        "slices": {
            "intent:1": {
                "frames": 5,
                "mean_regret": 0.2,
                "selected_source_rates": {"world": selected_world_rate},
                "oracle_source_rates": {"world": oracle_world_rate},
            },
            "intent:2": {
                "frames": 3,
                "mean_regret": worst_regret,
                "selected_source_rates": {"world": 0.0},
                "oracle_source_rates": {"world": 0.5},
            },
        },
    }
    if embedding_source is not None:
        payload["embedding_source"] = embedding_source
    return payload


class AuditExternalEmbeddingResultTests(unittest.TestCase):
    def test_passes_when_gain_and_slices_clear_acceptance_bar(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(report(rfs=7.6, world_model="off", embedding_source=None)), encoding="utf-8")
            candidate.write_text(json.dumps(report(rfs=7.75, worst_regret=0.9)), encoding="utf-8")

            result = module.audit_external_embedding_result(candidate_path=candidate, baseline_path=baseline)

        self.assertEqual("pass", result["status"])
        self.assertEqual([], result["failures"])

    def test_reports_world_candidate_slice_diagnostics(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(report(rfs=7.6, world_model="off", embedding_source=None)), encoding="utf-8")
            candidate.write_text(
                json.dumps(report(rfs=7.75, selected_world_rate=0.1, oracle_world_rate=0.4)),
                encoding="utf-8",
            )

            result = module.audit_external_embedding_result(candidate_path=candidate, baseline_path=baseline)

        diagnostics = next(check for check in result["checks"] if check["id"] == "world_candidate_diagnostics")
        self.assertEqual(0.1, diagnostics["selected_world_rate"])
        self.assertEqual(0.4, diagnostics["oracle_world_rate"])
        self.assertEqual("intent:2", diagnostics["top_world_underselection_slices"][0]["slice"])

    def test_fails_when_gain_is_too_small(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(json.dumps(report(rfs=7.6, world_model="off", embedding_source=None)), encoding="utf-8")
            candidate.write_text(json.dumps(report(rfs=7.65)), encoding="utf-8")

            result = module.audit_external_embedding_result(candidate_path=candidate, baseline_path=baseline)

        self.assertEqual("fail", result["status"])
        self.assertIn("insufficient_rfs_gain", {failure["id"] for failure in result["failures"]})

    def test_fails_when_worst_slice_regret_worsens(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(
                json.dumps(report(rfs=7.6, world_model="off", embedding_source=None, worst_regret=1.0)),
                encoding="utf-8",
            )
            candidate.write_text(json.dumps(report(rfs=7.75, worst_regret=1.1)), encoding="utf-8")

            result = module.audit_external_embedding_result(candidate_path=candidate, baseline_path=baseline)

        self.assertEqual("fail", result["status"])
        self.assertIn("worst_slice_regret_worse", {failure["id"] for failure in result["failures"]})

    def test_accepts_wrapped_benchmark_metric_values(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            baseline.write_text(
                json.dumps(
                    {
                        "metrics": {"combined_ranker_mean_rfs": {"value": 7.6}},
                        "slices": {"intent:1": {"frames": 5, "mean_regret": 1.0}},
                    }
                ),
                encoding="utf-8",
            )
            candidate.write_text(
                json.dumps(
                    {
                        "world_model": "external_embeddings",
                        "embedding_source": "cosmos-test",
                        "metrics": {"combined_with_world_ranker_mean_rfs": {"value": 7.75}},
                        "slices": {"intent:1": {"frames": 5, "mean_regret": 0.9}},
                    }
                ),
                encoding="utf-8",
            )

            result = module.audit_external_embedding_result(candidate_path=candidate, baseline_path=baseline)

        self.assertEqual("pass", result["status"])


if __name__ == "__main__":
    unittest.main()
