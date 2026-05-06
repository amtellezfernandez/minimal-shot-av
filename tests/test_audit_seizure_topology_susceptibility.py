from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minimal_shot_av.simulator.compositional_scenarios import COMPOSITIONAL_TOPOLOGIES


class SeizureTopologySusceptibilityAuditTests(unittest.TestCase):
    def test_audit_writes_topology_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_seizure_topology_susceptibility.py"),
                    "--seeds-per-topology",
                    "1",
                    "--seed-start",
                    "1",
                    "--suite",
                    "hidden",
                    "--budgets",
                    "0.0",
                    "0.35",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads((Path(temp_dir) / "topology_susceptibility.json").read_text())
            summary_csv = (Path(temp_dir) / "topology_susceptibility_summary.csv").read_text()
            strata_csv = (Path(temp_dir) / "topology_susceptibility_strata.csv").read_text()
            sweep_csv = (Path(temp_dir) / "topology_susceptibility_sweep.csv").read_text()

        self.assertEqual({row["topology"] for row in payload["topology_summary"]}, set(COMPOSITIONAL_TOPOLOGIES))
        self.assertEqual(
            {row["sweep_family"] for row in payload["topology_summary"]},
            {"coupled_stealth", "radius_only", "count_only", "placement_only"},
        )
        self.assertEqual(len(payload["ranking"]), len(COMPOSITIONAL_TOPOLOGIES) * 4)
        self.assertEqual(len(payload["sweep_runs"]), len(COMPOSITIONAL_TOPOLOGIES) * 2 * 4)
        self.assertEqual(
            "phase_locked_phantom_guard_obstacles",
            payload["attack_model"]["name"],
        )
        self.assertTrue(all(float(row["phantom_radius_m"]) <= 1.45 for row in payload["sweep_runs"]))
        self.assertTrue(all(row["stealth_bounded"] for row in payload["sweep_runs"]))
        self.assertIn("confound_audit", payload)
        self.assertIn("claim_boundary", payload["confound_audit"])
        self.assertGreater(len(payload["stratified_summary"]), 0)
        self.assertIn("critical_attack_budget", summary_csv)
        self.assertIn("first_any_collapse_budget", summary_csv)
        self.assertIn("collapse_rate_at_max_budget", summary_csv)
        self.assertIn("primary_hazard_type", strata_csv)
        self.assertIn("dominant_collapse_reason", strata_csv)
        self.assertIn("collapse_reason", sweep_csv)
        self.assertIn("collapse_reason_family", sweep_csv)
        self.assertIn("sweep_family", sweep_csv)
        self.assertIn("curvature_per_100m", sweep_csv)

    def test_matched_hazard_audit_selects_same_hazard_across_topologies(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_seizure_topology_susceptibility.py"),
                    "--seeds-per-topology",
                    "1",
                    "--seed-start",
                    "1",
                    "--suite",
                    "hidden",
                    "--match-primary-hazard",
                    "wrong_way_vehicle",
                    "--sweep-family",
                    "coupled_stealth",
                    "--budgets",
                    "0.0",
                    "0.7",
                    "--output-dir",
                    temp_dir,
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads((Path(temp_dir) / "topology_susceptibility.json").read_text())

        self.assertTrue(payload["attack_model"]["matched_design"])
        self.assertEqual(payload["attack_model"]["match_primary_hazard"], "wrong_way_vehicle")
        self.assertTrue(payload["confound_audit"]["matched_primary_hazard"])
        self.assertEqual(payload["confound_audit"]["selected_hazards"], ["wrong_way_vehicle"])
        self.assertEqual({row["topology"] for row in payload["topology_summary"]}, set(COMPOSITIONAL_TOPOLOGIES))
        self.assertEqual(len(payload["sweep_runs"]), len(COMPOSITIONAL_TOPOLOGIES) * 2)


if __name__ == "__main__":
    unittest.main()
