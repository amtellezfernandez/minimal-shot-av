from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "record_wod_leaderboard_result.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("record_wod_leaderboard_result", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordWodLeaderboardResultTests(unittest.TestCase):
    def test_leaderboard_record_requires_manifest_sha_match(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            manifest = _manifest(Path(tmpdir))

            record = module.leaderboard_record(
                manifest_path=manifest,
                variant="kinematic_constant_velocity",
                submission_sha256="abc123",
                rfs=7.25,
                rank=12,
                notes="first blind upload",
                confirmed_hidden_test=True,
            )

            self.assertEqual("kinematic_constant_velocity", record["variant"])
            self.assertEqual("abc123", record["submission_sha256"])
            self.assertEqual(7.25, record["hidden_test_rfs"])
            self.assertFalse(record["validation_tuned"])
            self.assertTrue(record["minimal_shot_claim_allowed"])
            self.assertFalse(record["beats_sota_snapshot"])

    def test_leaderboard_record_rejects_wrong_sha(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            manifest = _manifest(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                module.leaderboard_record(
                    manifest_path=manifest,
                    variant="kinematic_constant_velocity",
                    submission_sha256="wrong",
                    rfs=7.25,
                    confirmed_hidden_test=True,
                )

    def test_leaderboard_record_rejects_unconfirmed_scores(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            manifest = _manifest(Path(tmpdir))

            with self.assertRaisesRegex(ValueError, "confirmed_hidden_test"):
                module.leaderboard_record(
                    manifest_path=manifest,
                    variant="kinematic_constant_velocity",
                    submission_sha256="abc123",
                    rfs=7.25,
                )

    def test_append_record_rejects_duplicate_variant_hash(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "results.json"
            record = {
                "recorded_at_utc": "2026-04-26T00:00:00Z",
                "variant": "a",
                "submission_sha256": "abc",
            }

            payload = module.append_record(output, record)
            output.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "already recorded"):
                module.append_record(output, record)


def _manifest(root: Path) -> Path:
    path = root / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "pre_registered": True,
                "submissions": [
                    {
                        "variant": "kinematic_constant_velocity",
                        "submission": "kinematic_constant_velocity.tar.gz",
                        "submission_sha256": "abc123",
                        "candidate_name": "constant_velocity",
                        "branch": "strict_minimal_shot",
                        "validation_tuned": False,
                        "minimal_shot_claim_allowed": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
