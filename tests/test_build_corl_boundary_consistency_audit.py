from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_corl_boundary_consistency_audit.py"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildCorlBoundaryConsistencyAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module(SCRIPT, "build_corl_boundary_consistency_audit")

    def test_build_audit_counts_boundary_matches(self) -> None:
        audit = self.module.build_audit(_report(), tolerance=0.05, input_json="demo.json")

        self.assertEqual("corl_boundary_consistency_audit_v1", audit["schema"])
        self.assertEqual(2, audit["split_count"])
        self.assertEqual(2, audit["summary"]["safe_within_tolerance_count"])
        self.assertEqual(1, audit["summary"]["balanced_within_tolerance_count"])
        self.assertEqual(2, audit["summary"]["balanced_progress_gain_positive_count"])
        self.assertIn("Boundary-Consistency Audit", self.module.markdown_audit(audit))

    def test_cli_writes_json_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "summary.json"
            output_json = tmp_path / "audit.json"
            output_md = tmp_path / "audit.md"
            input_path.write_text(json.dumps(_report()), encoding="utf-8")
            original_argv = sys.argv
            sys.argv = [
                str(SCRIPT),
                "--input-json",
                str(input_path),
                "--output-json",
                str(output_json),
                "--output-markdown",
                str(output_md),
            ]
            try:
                self.module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual("corl_boundary_consistency_audit_v1", payload["schema"])
            self.assertIn("Summary", output_md.read_text(encoding="utf-8"))


def _report() -> dict:
    return {
        "splits": [
            {
                "seed": 0,
                "selectors": {
                    "proxy_top1": {"replay_fail_rate": 0.50, "progress_m": 20.0},
                    "scene_token_student": {"replay_fail_rate": 0.45, "progress_m": 19.0},
                    "replay_value_safe": {"replay_fail_rate": 0.10, "progress_m": 12.0},
                    "replay_value_balanced": {"replay_fail_rate": 0.10, "progress_m": 14.0},
                    "replay_value_oracle": {"replay_fail_rate": 0.10, "progress_m": 11.0},
                },
            },
            {
                "seed": 1,
                "selectors": {
                    "proxy_top1": {"replay_fail_rate": 0.60, "progress_m": 21.0},
                    "scene_token_student": {"replay_fail_rate": 0.61, "progress_m": 20.0},
                    "replay_value_safe": {"replay_fail_rate": 0.13, "progress_m": 13.0},
                    "replay_value_balanced": {"replay_fail_rate": 0.20, "progress_m": 15.0},
                    "replay_value_oracle": {"replay_fail_rate": 0.10, "progress_m": 11.0},
                },
            },
        ]
    }


if __name__ == "__main__":
    unittest.main()
