from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_novel_object_stress import build_report
from scripts.audit_novel_object_stress import GEOMETRY_ONLY_POLICY


SCRIPT = ROOT / "scripts" / "audit_novel_object_stress.py"


class NovelObjectStressAuditTests(unittest.TestCase):
    def test_report_is_valid_for_default_smoke_window(self) -> None:
        report = build_report(seed_start=1, seed_end=30, min_runs=8)

        self.assertTrue(report["valid"], report["failures"])
        self.assertEqual(report["schema"], "novel_object_stress_audit_v1")
        self.assertGreaterEqual(report["summary"]["runs"], 8)
        self.assertGreaterEqual(report["summary"]["label_count"], 2)
        self.assertEqual(report["summary"]["collision_rate"], 0.0)
        self.assertEqual(report["summary"]["near_miss_rate"], 0.0)
        self.assertIn("does not prove camera-based recognition", report["claim_boundary"])

    def test_script_writes_json_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "audit.json"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--seed-start",
                    "1",
                    "--seed-end",
                    "30",
                    "--min-runs",
                    "8",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertTrue(payload["valid"])
        self.assertEqual(payload["summary"]["policy_label_dependency"], GEOMETRY_ONLY_POLICY)


if __name__ == "__main__":
    unittest.main()
