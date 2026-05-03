from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_wod_e2e_submission_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("prepare_wod_e2e_submission_matrix", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class PrepareWodE2ESubmissionMatrixTests(unittest.TestCase):
    def test_matrix_variants_separate_strict_and_calibrated_claims(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            kinematic = root / "kinematic.jsonl"
            strict = root / "strict.jsonl"
            calibrated = root / "calibrated.jsonl"
            for path in (kinematic, strict, calibrated):
                path.write_text("", encoding="utf-8")

            variants = module._matrix_variants(
                argparse.Namespace(
                    strict_spotlight_candidates=strict,
                    calibrated_verifier_candidates=calibrated,
                ),
                kinematic,
            )

        by_name = {variant.name: variant for variant in variants}
        self.assertTrue(by_name["strict_spotlight_reflex"].minimal_shot_claim_allowed)
        self.assertFalse(by_name["strict_spotlight_reflex"].validation_tuned)
        self.assertFalse(by_name["small_verifier_calibrated"].minimal_shot_claim_allowed)
        self.assertTrue(by_name["small_verifier_calibrated"].validation_tuned)
        self.assertEqual("preference_calibrated_leaderboard", by_name["small_verifier_calibrated"].branch)


if __name__ == "__main__":
    unittest.main()
