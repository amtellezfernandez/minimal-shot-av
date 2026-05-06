from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_top_lab_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_top_lab_readiness", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AuditTopLabReadinessTests(unittest.TestCase):
    def test_current_style_package_fails_without_external_hidden_evidence(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            judging = _write_json(root / "judging.json", _judging(valid=True))
            leaderboard = _write_json(
                root / "leaderboard.json",
                {"schema": "wod_e2e_leaderboard_results_v1", "results": []},
            )
            external = _write_json(
                root / "external.json",
                {"audit_type": "external_embedding_result_gate", "status": "fail"},
            )
            sim = _write_json(root / "sim.json", _scenario_eval(suites=("wod",), min_benchmark=0.96))
            ood = _write_json(
                root / "ood.json",
                _scenario_eval(suites=("wod", "adversarial", "gauntlet"), min_benchmark=0.333),
            )
            safety = _write_json(root / "safety.json", {"schema": "draft"})
            curriculum = _write_json(root / "curriculum.json", _rlvr_curriculum())

            report = module.top_lab_readiness_report(
                judging_audit=judging,
                leaderboard_results=leaderboard,
                external_embedding_audit=external,
                sim_eval=sim,
                ood_eval=ood,
                safety_case=safety,
                rlvr_curriculum=curriculum,
                alpasignal_bridge=root / "missing_bridge.json",
            )

        self.assertFalse(report["top_lab_ready"])
        failed = {item["gate"] for item in report["blockers"]}
        self.assertIn("hidden_waymo_test_confirmed", failed)
        self.assertIn("external_embedding_improves_wod", failed)
        self.assertIn("independent_or_external_simulator_evidence", failed)
        self.assertIn("formalish_safety_case_present", failed)

    def test_passes_when_hidden_external_independent_and_safety_evidence_exist(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            judging = _write_json(root / "judging.json", _judging(valid=True))
            leaderboard = _write_json(
                root / "leaderboard.json",
                {
                    "schema": "wod_e2e_leaderboard_results_v1",
                    "results": [
                        {
                            "confirmed_hidden_test": True,
                            "hidden_test_rfs": 8.2,
                            "submission_sha256": "abc",
                        }
                    ],
                },
            )
            external = _write_json(
                root / "external.json",
                {
                    "audit_type": "external_embedding_result_gate",
                    "status": "pass",
                    "failures": [],
                    "checks": [
                        {
                            "id": "world_candidate_diagnostics",
                            "selected_world_rate": 0.3,
                            "oracle_world_rate": 0.4,
                        }
                    ],
                },
            )
            sim = _write_json(root / "sim.json", _scenario_eval(suites=("wod",), min_benchmark=0.96))
            ood = _write_json(
                root / "ood.json",
                _scenario_eval(suites=("wod", "adversarial", "gauntlet", "alpasim"), min_benchmark=0.5),
            )
            safety = _write_json(root / "safety.json", _safety_case())
            curriculum = _write_json(root / "curriculum.json", _rlvr_curriculum())

            report = module.top_lab_readiness_report(
                judging_audit=judging,
                leaderboard_results=leaderboard,
                external_embedding_audit=external,
                sim_eval=sim,
                ood_eval=ood,
                safety_case=safety,
                rlvr_curriculum=curriculum,
                alpasignal_bridge=root / "missing_bridge.json",
            )

        self.assertTrue(report["top_lab_ready"])
        self.assertEqual([], report["blockers"])

    def test_valid_alpasignal_bridge_counts_as_external_simulator_evidence(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            judging = _write_json(root / "judging.json", _judging(valid=True))
            leaderboard = _write_json(
                root / "leaderboard.json",
                {
                    "schema": "wod_e2e_leaderboard_results_v1",
                    "results": [
                        {
                            "confirmed_hidden_test": True,
                            "hidden_test_rfs": 8.2,
                            "submission_sha256": "abc",
                        }
                    ],
                },
            )
            external = _write_json(
                root / "external.json",
                {
                    "audit_type": "external_embedding_result_gate",
                    "status": "pass",
                    "failures": [],
                    "checks": [{"id": "world_candidate_diagnostics"}],
                },
            )
            sim = _write_json(root / "sim.json", _scenario_eval(suites=("wod",), min_benchmark=0.96))
            ood = _write_json(
                root / "ood.json",
                _scenario_eval(suites=("wod", "adversarial", "gauntlet"), min_benchmark=0.5),
            )
            safety = _write_json(root / "safety.json", _safety_case())
            curriculum = _write_json(root / "curriculum.json", _rlvr_curriculum())
            bridge = _write_json(root / "bridge.json", _alpasignal_bridge())

            report = module.top_lab_readiness_report(
                judging_audit=judging,
                leaderboard_results=leaderboard,
                external_embedding_audit=external,
                sim_eval=sim,
                ood_eval=ood,
                safety_case=safety,
                rlvr_curriculum=curriculum,
                alpasignal_bridge=bridge,
            )

        self.assertTrue(report["gates"]["independent_or_external_simulator_evidence"])
        self.assertTrue(report["evidence"]["alpasignal_bridge"]["valid"])
        self.assertTrue(report["top_lab_ready"])

    def test_accepts_existing_list_style_safety_case_sections(self) -> None:
        module = _load_module()
        payload = {
            "schema": "av_safety_case_v1",
            "approved_for_public_road_deployment": False,
            "sections": [
                "operational_design_domain",
                "hazard_analysis",
                "verification_validation",
                "fallback_and_minimal_risk_condition",
                "residual_risk_acceptance",
            ],
            "operational_design_domain": {},
            "hazard_analysis": {},
            "verification_validation": {},
            "fallback_and_minimal_risk_condition": {},
            "residual_risk_acceptance": {},
        }
        with TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "safety.json", payload)

            evidence = module._safety_case_evidence(path)

        self.assertTrue(evidence["valid"])
        self.assertEqual([], evidence["missing_sections"])

    def test_rejects_curriculum_without_holdout_splits(self) -> None:
        module = _load_module()
        payload = _rlvr_curriculum()
        payload["coverage"]["split_counts"] = {"train_curriculum": 500}
        with TemporaryDirectory() as tmp:
            path = _write_json(Path(tmp) / "curriculum.json", payload)

            evidence = module._rlvr_curriculum_evidence(path)

        self.assertFalse(evidence["valid"])
        self.assertIn("blind_holdout", evidence["missing_splits"])


def _write_json(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _judging(*, valid: bool) -> dict:
    return {
        "schema": "sota_judging_criteria_audit_v1",
        "valid": valid,
        "criteria": {
            "technical_excellence": {"status": "pass" if valid else "fail"},
            "novelty": {"status": "pass"},
        },
    }


def _scenario_eval(*, suites: tuple[str, ...], min_benchmark: float) -> dict:
    runs = [{"suite": suite, "trajectory_safety_pass": True} for suite in suites]
    summary = [
        {
            "suite": suite,
            "benchmark_pass_rate": min_benchmark,
            "trajectory_safety_pass_rate": 0.96,
        }
        for suite in suites
    ]
    return {
        "runs": runs,
        "summary": summary,
        "statistics": {"unit": "closed-loop rollout"},
        "curriculum": {"generator": "closed_loop_procedural_curriculum"},
    }


def _safety_case() -> dict:
    return {
        "schema": "av_safety_case_v1",
        "sections": {
            "operational_design_domain": {},
            "hazard_analysis": {},
            "verification_validation": {},
            "fallback_and_minimal_risk_condition": {},
            "residual_risk_acceptance": {},
        },
    }


def _rlvr_curriculum() -> dict:
    task = {
        "composition_key": "wod|construction|debris|rain",
        "axes": {"suite": "wod", "primary_hazard_type": "debris"},
        "grader_ids": ["reach_goal", "no_collision", "trajectory_safety", "benchmark_pass"],
        "observation_contract": {"privileged_state_allowed": False},
    }
    return {
        "schema": "minimal_shot_av_rlvr_curriculum_v1",
        "tasks": [task for _ in range(500)],
        "coverage": {
            "task_count": 500,
            "composition_count": 120,
            "grader_count": 6,
            "observation_mode_count": 4,
            "split_counts": {
                "train_curriculum": 200,
                "dev_holdout": 100,
                "blind_holdout": 100,
                "stress_holdout": 100,
            },
        },
    }


def _alpasignal_bridge() -> dict:
    return {
        "schema": "alpasignal_bridge_audit_v1",
        "valid": True,
        "cases": [{}, {}, {}],
        "gates": {
            "all_cases_emit_finite_20_point_trajectories": True,
            "all_cases_emit_reasoning_text": True,
            "structured_hazard_consumed": True,
            "moving_hazard_preserved_as_actor": True,
            "low_visibility_adds_caution_zone": True,
        },
        "boundary": "Structured simulator bridge, not full photorealism.",
    }


if __name__ == "__main__":
    unittest.main()
