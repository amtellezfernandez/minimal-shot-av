from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_minimal_shot_claim import build_report


class MinimalShotClaimAuditTest(unittest.TestCase):
    def test_passes_when_policy_has_no_episode_dependency_and_wod_is_auxiliary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim = _write_sim(root / "sim.json")
            policy = root / "policy.py"
            policy.write_text("def act(scene):\n    return scene.goal\n", encoding="utf-8")
            model = _write_text(root / "model.md", "validation preference labels; not strict zero-shot")
            grand = _write_text(root / "grand.md", "WOD validation-CV evidence is declared as auxiliary")
            form = _write_text(root / "form.md", "strict zero-shot WOD-E2E should not be described")

            report = build_report(
                sim_eval=sim,
                model_declaration=model,
                grand_submission=grand,
                submission_form=form,
                policy_sources=(policy,),
            )

        self.assertTrue(report["valid"])
        self.assertEqual([], report["failures"])

    def test_fails_when_active_policy_uses_preference_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sim = _write_sim(root / "sim.json")
            policy = root / "policy.py"
            policy.write_text("from x import load_preference_frames\nmode = 'nearest_train'\n", encoding="utf-8")
            model = _write_text(root / "model.md", "validation preference labels; not strict zero-shot")
            grand = _write_text(root / "grand.md", "not the centerpiece of the minimal-shot claim")
            form = _write_text(root / "form.md", "strict zero-shot WOD-E2E should not be described")

            report = build_report(
                sim_eval=sim,
                model_declaration=model,
                grand_submission=grand,
                submission_form=form,
                policy_sources=(policy,),
            )

        self.assertFalse(report["valid"])
        self.assertIn("active_policy_has_no_episode_lookup", report["failures"])
        self.assertGreaterEqual(len(report["metrics"]["policy_static_scan"]["matches"]), 2)


def _write_sim(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "summary": [
                    {
                        "runs": 50,
                        "success_rate": 1.0,
                        "benchmark_pass_rate": 1.0,
                        "collision_rate": 0.0,
                        "near_miss_rate": 0.0,
                    }
                    for _ in range(11)
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_text(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    unittest.main()
