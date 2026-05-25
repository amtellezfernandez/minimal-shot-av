from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_nuplan_selected_token_realized_clearance.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_nuplan_selected_token_realized_clearance", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class AuditNuPlanSelectedTokenRealizedClearanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_enrich_report_fills_realized_clearance_fields(self) -> None:
        report = _base_report()

        def provider(scene):
            return {
                "selected_token_realized_min_clearance_m": 1.5,
                "selected_token_realized_min_clearance_t": 2.0,
                "selected_token_realized_collision": False,
                "selected_token_realized_near_miss": False,
                "realized_clearance_source": "log_replay",
            }

        enriched = self.module.enrich_report_with_realized_clearance(
            report,
            clearance_provider=provider,
            near_miss_threshold_m=1.0,
        )

        scene = enriched["scenes"][0]
        self.assertEqual(1.5, scene["selected_token_realized_min_clearance_m"])
        self.assertEqual(2.0, scene["selected_token_realized_min_clearance_t"])
        self.assertFalse(scene["selected_token_realized_near_miss"])
        self.assertEqual("log_replay", scene["realized_clearance_source"])
        self.assertIn("candidate_replay_evaluations", scene)
        self.assertIn("oracle_log_replay_safe_token", scene)
        self.assertIn("horizon_sensitivity_table", enriched)
        self.assertIn("replay_oracle_diagnostic", scene)
        self.assertIn("selected_failed_but_replay_safe_alternative_exists", scene)
        self.assertIn("proxy_safe_replay_failure_onset_bucket", scene)
        self.assertIn("replay_oracle_diagnostic_table", enriched)
        self.assertIn("failure_rung_by_replay_oracle_diagnostic_table", enriched)
        self.assertIn("replay_oracle_miss_table", enriched)
        self.assertIn("proxy_safe_replay_failure_onset_table", enriched)

    def test_proxy_safe_cases_have_some_realized_clearance(self) -> None:
        report = _base_report()

        enriched = self.module.enrich_report_with_realized_clearance(
            report,
            clearance_provider=lambda scene: {
                "selected_token_realized_min_clearance_m": 1.5,
                "selected_token_realized_min_clearance_t": 1.0,
                "selected_token_realized_collision": False,
                "selected_token_realized_near_miss": False,
                "realized_clearance_source": "log_replay",
            },
            near_miss_threshold_m=1.0,
        )

        proxy_safe = [scene for scene in enriched["scenes"] if scene["proxy_safe_selected"]]
        with_realized = [
            scene for scene in proxy_safe if scene["selected_token_realized_min_clearance_m"] is not None
        ]

        self.assertEqual(1, len(proxy_safe))
        self.assertGreater(len(with_realized), 0)

    def test_realized_near_miss_matches_threshold(self) -> None:
        report = _base_report()

        enriched = self.module.enrich_report_with_realized_clearance(
            report,
            clearance_provider=lambda scene: {
                "selected_token_realized_min_clearance_m": 0.4,
                "selected_token_realized_min_clearance_t": 1.0,
                "selected_token_realized_collision": False,
                "selected_token_realized_near_miss": True,
                "realized_clearance_source": "log_replay",
            },
            near_miss_threshold_m=1.0,
        )

        for scene in enriched["scenes"]:
            clearance = scene["selected_token_realized_min_clearance_m"]
            if clearance is None:
                continue
            self.assertEqual(scene["selected_token_realized_near_miss"], clearance < 1.0)

    def test_cli_main_writes_realized_clearance_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "rollout.json"
            output_json = tmp_path / "realized.json"
            output_md = tmp_path / "realized.md"
            input_path.write_text(json.dumps(_base_report()), encoding="utf-8")
            argv = [
                str(SCRIPT),
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
            ]
            original_argv = sys.argv
            sys.argv = argv
            try:
                original_provider = self.module.compute_scene_log_replay_realized_fields
                self.module.compute_scene_log_replay_realized_fields = lambda scene: None
                self.module.main()
            finally:
                self.module.compute_scene_log_replay_realized_fields = original_provider
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("nuplan_maneuvertoken_realized_clearance_v1", payload["schema"])
            self.assertIn("threshold_sensitivity_table", payload)
            self.assertIn("horizon_sensitivity_table", payload)
            self.assertIn("Failure rung", output_md.read_text(encoding="utf-8"))

    def test_log_replay_failure_rung_uses_replay_specific_label(self) -> None:
        report = _base_report()

        enriched = self.module.enrich_report_with_realized_clearance(
            report,
            clearance_provider=lambda scene: {
                "selected_token_realized_min_clearance_m": 0.4,
                "selected_token_realized_min_clearance_t": 1.0,
                "selected_token_realized_collision": False,
                "selected_token_realized_near_miss": True,
                "selected_token_realized_clearance_trace": [{"time_s": 1.0, "realized_clearance_m": 0.4}],
                "realized_clearance_source": "log_replay",
            },
            near_miss_threshold_m=1.0,
        )

        self.assertEqual("proxy-safe but replay-infeasible", enriched["scenes"][0]["failure_rung"])

    def test_replay_oracle_case_table_distinguishes_alternative_safe_token(self) -> None:
        report = _base_report()
        report["scenes"][0]["candidate_replay_evaluations"] = [
            {"token": "stop", "realized_min_clearance_m": 0.2},
            {"token": "evasive_right", "realized_min_clearance_m": 1.5},
        ]
        report["scenes"][0]["selected_token_realized_near_miss"] = True
        report["scenes"][0]["oracle_log_replay_safe_token"] = "evasive_right"

        table = self.module._replay_oracle_case_table(report["scenes"])
        counts = {row["case"]: row["count"] for row in table}

        self.assertEqual(1, counts["selected token fails but another replay-safe token exists"])

    def test_failure_rung_by_replay_oracle_diagnostic_keeps_ladder_primary(self) -> None:
        scenes = [
            {
                "failure_rung": "safe token existed but selector missed it",
                "replay_oracle_diagnostic": "selected_failed_with_replay_safe_alternative",
            },
            {
                "failure_rung": "proxy-safe but replay-infeasible",
                "replay_oracle_diagnostic": "selected_failed_with_replay_safe_alternative",
            },
        ]

        table = self.module._failure_rung_by_replay_oracle_diagnostic_table(scenes)
        counts = {
            (row["failure_rung"], row["replay_oracle_diagnostic"]): row["count"]
            for row in table
        }

        self.assertEqual(
            1,
            counts[("safe token existed but selector missed it", "selected_failed_with_replay_safe_alternative")],
        )
        self.assertEqual(
            1,
            counts[("proxy-safe but replay-infeasible", "selected_failed_with_replay_safe_alternative")],
        )

    def test_replay_oracle_diagnostic_marks_replay_safe_alternative(self) -> None:
        report = _base_report()
        report["scenes"][0]["failure_rung"] = "proxy-safe but replay-infeasible"
        report["scenes"][0]["selected_token_realized_min_clearance_m"] = 0.4
        report["scenes"][0]["selected_token_realized_near_miss"] = True
        report["scenes"][0]["oracle_log_replay_safe_token"] = "evasive_right"

        diagnostic = self.module._replay_oracle_diagnostic(report["scenes"][0])

        self.assertEqual("selected_failed_with_replay_safe_alternative", diagnostic)

    def test_replay_oracle_miss_table_reports_common_oracle(self) -> None:
        scenes = [
            {
                "replay_oracle_diagnostic": "selected_failed_with_replay_safe_alternative",
                "selected_token": "maintain",
                "oracle_log_replay_safe_token": "slow_yield",
            },
            {
                "replay_oracle_diagnostic": "selected_failed_with_replay_safe_alternative",
                "selected_token": "maintain",
                "oracle_log_replay_safe_token": "slow_yield",
            },
            {
                "replay_oracle_diagnostic": "selected_failed_with_replay_safe_alternative",
                "selected_token": "maintain",
                "oracle_log_replay_safe_token": "lane_recover",
            },
        ]

        table = self.module._replay_oracle_miss_table(scenes)

        self.assertEqual(
            [
                {
                    "selected_token": "maintain",
                    "count": 3,
                    "most_common_oracle_token": "slow_yield",
                    "most_common_oracle_count": 2,
                }
            ],
            table,
        )

    def test_horizon_sensitivity_summary_pins_immediate_failures(self) -> None:
        trace = [
            {"time_s": 1.0, "realized_clearance_m": 0.4},
            {"time_s": 2.0, "realized_clearance_m": 0.3},
            {"time_s": 3.0, "realized_clearance_m": 0.2},
            {"time_s": 4.0, "realized_clearance_m": 0.1},
        ]
        evaluations = [{"realized_clearance_trace": trace}]
        scenes = [
            {
                "proxy_safe_selected": True,
                "selected_token": "stop",
                "selected_token_realized_clearance_trace": trace,
                "candidate_replay_evaluations": evaluations,
            }
        ]

        table = self.module._horizon_sensitivity_table(scenes=scenes, horizons_s=(1.0, 2.0, 3.0, 4.0, 5.0))
        by_horizon = {row["horizon_s"]: row for row in table}

        self.assertEqual(1, by_horizon[1.0]["proxy_safe_selected_failures"])
        self.assertEqual(1, by_horizon[1.0]["proxy_safe_stop_failures"])
        self.assertFalse(by_horizon[5.0]["available"])

    def test_proxy_safe_replay_failure_onset_table_buckets_failures(self) -> None:
        scenes = [
            {
                "failure_rung": "proxy-safe but replay-infeasible",
                "selected_token_realized_clearance_trace": [{"time_s": 1.0, "realized_clearance_m": 0.4}],
            },
            {
                "failure_rung": "proxy-safe but replay-infeasible",
                "selected_token_realized_clearance_trace": [{"time_s": 2.0, "realized_clearance_m": 0.4}],
            },
            {
                "failure_rung": "proxy-safe but replay-infeasible",
                "selected_token_realized_clearance_trace": [{"time_s": 3.0, "realized_clearance_m": 0.4}],
            },
            {
                "failure_rung": "proxy-safe but replay-infeasible",
                "selected_token_realized_clearance_trace": [{"time_s": 4.0, "realized_clearance_m": 0.4}],
            },
        ]
        for scene in scenes:
            scene["proxy_safe_replay_failure_onset_bucket"] = (
                self.module._proxy_safe_replay_failure_onset_bucket(scene)
            )

        table = self.module._proxy_safe_replay_failure_onset_table(scenes)
        by_onset = {row["onset"]: row["count"] for row in table}

        self.assertEqual(1, by_onset["immediate_<=1s"])
        self.assertEqual(1, by_onset["mid_1to2s"])
        self.assertEqual(1, by_onset["mid_2to3s"])
        self.assertEqual(1, by_onset["late_3to4s"])


def _base_report() -> dict:
    return {
        "scene_count": 1,
        "scenes": [
            {
                "scene_id": "scene_a",
                "actor_summary": {"actor_count": 2, "visible_actor_count": 2},
                "safe_token_existed": True,
                "selector_chose_safe_token": True,
                "proxy_safe_selected": True,
                "selected_token_realized_min_clearance_m": None,
                "selected_token": "stop",
            }
        ],
    }


if __name__ == "__main__":
    unittest.main()
