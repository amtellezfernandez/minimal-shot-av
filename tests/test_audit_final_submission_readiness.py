from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import tarfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_final_submission_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_final_submission_readiness", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditFinalSubmissionReadinessTests(unittest.TestCase):
    def test_final_readiness_accepts_complete_evidence_package(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundles"
            bundle_root.mkdir()
            grand = _write_archive(bundle_root / "grand_commission.tar.gz", module.REQUIRED_GRAND_MEMBERS)
            minor = _write_archive(bundle_root / "minor_commission.tar.gz", module.REQUIRED_MINOR_MEMBERS)
            _write_json(
                bundle_root / "submission_bundles_manifest.json",
                {
                    "grand_commission": {"sha256": module._sha256(grand)},
                    "minor_commission": {"sha256": module._sha256(minor)},
                },
            )
            minimal = _write_minimal(root / "minimal.json")
            judging = _write_judging(root / "judging.json")
            production = _write_production(root / "production.json", decision="no_go")

            report = module.final_readiness_report(
                bundle_root=bundle_root,
                minimal_shot_audit=minimal,
                judging_audit=judging,
                production_audit=production,
            )

            self.assertTrue(report["valid"])
            self.assertEqual([], report["blockers"])

    def test_final_readiness_rejects_missing_production_no_go(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            bundle_root = root / "bundles"
            bundle_root.mkdir()
            grand = _write_archive(bundle_root / "grand_commission.tar.gz", module.REQUIRED_GRAND_MEMBERS)
            minor = _write_archive(bundle_root / "minor_commission.tar.gz", module.REQUIRED_MINOR_MEMBERS)
            _write_json(
                bundle_root / "submission_bundles_manifest.json",
                {
                    "grand_commission": {"sha256": module._sha256(grand)},
                    "minor_commission": {"sha256": module._sha256(minor)},
                },
            )
            minimal = _write_minimal(root / "minimal.json")
            judging = _write_judging(root / "judging.json")
            production = _write_production(root / "production.json", decision="go")

            report = module.final_readiness_report(
                bundle_root=bundle_root,
                minimal_shot_audit=minimal,
                judging_audit=judging,
                production_audit=production,
            )

            self.assertFalse(report["valid"])
            self.assertIn("gate failed: production_no_go_boundary", report["blockers"])
            self.assertIn("production boundary is not an explicit no-go", report["blockers"])


def _write_archive(path: Path, members: set[str]) -> Path:
    with tarfile.open(path, "w:gz") as stream:
        for member in members:
            source = path.parent / member.replace("/", "_")
            source.write_text("{}", encoding="utf-8")
            stream.add(source, arcname=member)
    return path


def _write_minimal(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema": "minimal_shot_claim_audit_v1",
            "valid": True,
            "checks": {
                "active_policy_has_no_episode_lookup": True,
                "closed_loop_no_near_miss": True,
                "closed_loop_zero_collision": True,
                "no_strict_zero_shot_wod_claim": True,
                "validation_preferences_not_primary_claim": True,
                "wod_evidence_declared_auxiliary": True,
            },
        },
    )
    return path


def _write_judging(path: Path) -> Path:
    _write_json(
        path,
        {
            "schema": "sota_judging_criteria_audit_v1",
            "valid": True,
            "criteria": {
                "technical_excellence": {"status": "pass"},
                "novelty": {"status": "pass"},
                "feasibility": {"status": "pass"},
                "adherence_to_brief": {"status": "pass"},
            },
        },
    )
    return path


def _write_production(path: Path, *, decision: str) -> Path:
    authorized = decision == "go"
    _write_json(
        path,
        {
            "schema": "production_av_readiness_audit_v1",
            "production_ready": authorized,
            "deployment_authorization": {
                "decision": decision,
                "real_vehicle_control_authorized": authorized,
                "public_road_deployment_authorized": authorized,
            },
        },
    )
    return path


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
