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

from minimal_shot_av.audit import export_internal_audit_log, load_audit_log


class AuditRerunBridgeTests(unittest.TestCase):
    def test_export_internal_audit_log_writes_manifest_and_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            audit_dir = Path(temp_dir) / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--scenario-cluster",
                    "spotlight",
                    "--rollout-preset",
                    "showcase-spotlight",
                    "--artifacts-dir",
                    str(rollout_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            manifest = export_internal_audit_log(rollout_dir / "latest_rollout.json", audit_dir)
            self.assertEqual("internal", manifest["source"])
            loaded_manifest, frames = load_audit_log(audit_dir)

        self.assertEqual(manifest["frame_count"], len(frames))
        self.assertEqual("spotlight", loaded_manifest["scenario_cluster"])
        self.assertIn("step", frames[0])
        self.assertIn("active_obstacles", frames[0])
        self.assertIn("trigger_state", frames[0])

    def test_summary_only_cli_reads_audit_log_without_rerun_dependency(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rollout_dir = Path(temp_dir) / "rollout"
            audit_dir = Path(temp_dir) / "audit"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_demo.py"),
                    "--seed",
                    "3",
                    "--policy",
                    "spotlight-reflex",
                    "--scenario-cluster",
                    "intersection",
                    "--rollout-preset",
                    "showcase-intersection",
                    "--artifacts-dir",
                    str(rollout_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "export_internal_audit.py"),
                    "--rollout-json",
                    str(rollout_dir / "latest_rollout.json"),
                    "--output-dir",
                    str(audit_dir),
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "view_audit_rerun.py"),
                    str(audit_dir),
                    "--summary-only",
                ],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            payload = json.loads(result.stdout)

        self.assertEqual("internal", payload["manifest"]["source"])
        self.assertGreater(payload["frame_count"], 0)
