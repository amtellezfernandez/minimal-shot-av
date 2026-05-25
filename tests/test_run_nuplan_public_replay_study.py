from __future__ import annotations

import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_nuplan_public_replay_study.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_nuplan_public_replay_study", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class RunNuPlanPublicReplayStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_build_public_replay_study_composes_rollout_and_replay_reports(self) -> None:
        globals_dict = self.module.build_public_replay_study.__globals__
        original_load_scenes = globals_dict["_load_scenes"]
        original_run_rollout = globals_dict["run_rollout"]
        original_enrich = globals_dict["enrich_report_with_realized_clearance"]
        try:
            globals_dict["_load_scenes"] = lambda args: [{"scene_id": "scene_a"}]
            globals_dict["run_rollout"] = lambda scenes, selector=None: {"schema": "rollout", "scenes": scenes}
            globals_dict["enrich_report_with_realized_clearance"] = (
                lambda report, clearance_provider, near_miss_threshold_m: {
                    "schema": "replay",
                    "scene_count": len(report["scenes"]),
                    "near_miss_threshold_m": near_miss_threshold_m,
                }
            )

            study = self.module.build_public_replay_study(
                bundle_dir=Path("/tmp/public-mini"),
                scene_limit=25,
                selector_model=None,
                near_miss_threshold_m=1.0,
                sampling_mode="interaction",
                max_scenes_per_db=2,
            )
        finally:
            globals_dict["_load_scenes"] = original_load_scenes
            globals_dict["run_rollout"] = original_run_rollout
            globals_dict["enrich_report_with_realized_clearance"] = original_enrich

        self.assertEqual("rollout", study["rollout_report"]["schema"])
        self.assertEqual("replay", study["replay_report"]["schema"])
        self.assertEqual(1, study["replay_report"]["scene_count"])
        self.assertEqual("nuplan_public_replay_sampling_diagnostics_v1", study["sampling_diagnostics"]["schema"])

    def test_select_interaction_db_members_excludes_tiny_files(self) -> None:
        infos = [
            _FakeInfo("mini/a.db", 5 * 1024 * 1024, 1),
            _FakeInfo("mini/b.db", 20 * 1024 * 1024, 1),
            _FakeInfo("mini/c.db", 30 * 1024 * 1024, 1),
        ]

        selected = self.module.select_interaction_db_members(
            infos,
            db_count=2,
            minimum_file_size_mb=10.0,
        )

        self.assertEqual(["mini/c.db", "mini/b.db"], [row["member"] for row in selected])

    def test_existing_bundle_manifest_reuses_local_db_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_dir = root / "data" / "cache" / "mini"
            db_dir.mkdir(parents=True)
            db_file = db_dir / "sample.db"
            db_file.write_text("placeholder", encoding="utf-8")

            manifest = self.module.existing_public_replay_bundle_manifest(root)

            self.assertEqual("existing_bundle", manifest["selection_mode"])
            self.assertEqual(1, manifest["db_count"])
            self.assertEqual("data/cache/mini/sample.db", manifest["members"][0]["member"])

    def test_sample_public_replay_scenes_caps_scenes_per_db(self) -> None:
        scenes = [
            _scene("a", "/tmp/one.db", actor_count=8, ego_speed_mps=10.0),
            _scene("b", "/tmp/one.db", actor_count=7, ego_speed_mps=9.0),
            _scene("c", "/tmp/one.db", actor_count=6, ego_speed_mps=8.0),
            _scene("d", "/tmp/two.db", actor_count=9, ego_speed_mps=11.0),
        ]

        selected, rows = self.module.sample_public_replay_scenes(
            scenes,
            scene_limit=3,
            sampling_mode="interaction",
            max_scenes_per_db=1,
        )

        self.assertEqual(2, len(selected))
        self.assertEqual(2, len(rows))
        self.assertEqual(1, sum(1 for row in rows if row["source_db_file"] == "/tmp/one.db"))
        self.assertEqual(1, sum(1 for row in rows if row["source_db_file"] == "/tmp/two.db"))

    def test_build_sampling_diagnostics_reports_entropy_and_db_distribution(self) -> None:
        candidate_scenes = [
            _scene("a", "/tmp/one.db", actor_count=4, ego_speed_mps=6.0),
            _scene("b", "/tmp/two.db", actor_count=5, ego_speed_mps=7.0),
        ]
        sampled_scenes = [candidate_scenes[0]]
        rollout_report = {"selected_token_histogram": {"stop": 1}}
        rows = [
            {
                "scene_id": "a",
                "source_db_file": "/tmp/one.db",
                "score": 1.5,
                "sampling_reason": "close_actor",
                "features": {
                    "minimum_actor_distance_m": 4.0,
                    "logged_ttc_s": 2.0,
                },
            }
        ]

        diagnostics = self.module.build_sampling_diagnostics(
            candidate_scenes=candidate_scenes,
            sampled_scenes=sampled_scenes,
            rollout_report=rollout_report,
            sampling_rows=rows,
            sampling_mode="interaction",
            max_scenes_per_db=1,
            invalid_db_files=[],
        )

        self.assertEqual(0.0, diagnostics["selected_token_entropy"])
        self.assertEqual(2, len(diagnostics["db_distribution"]))
        self.assertEqual(1, diagnostics["selected_scene_count"])

    def test_validate_bundle_db_files_skips_malformed_sqlite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            valid = root / "valid.db"
            invalid = root / "invalid.db"
            with sqlite3.connect(str(valid)) as connection:
                connection.execute("CREATE TABLE demo (id INTEGER)")
                connection.commit()
            invalid.write_text("not a sqlite db", encoding="utf-8")

            valid_rows, invalid_rows = self.module.validate_bundle_db_files(root)

            self.assertEqual([valid], valid_rows)
            self.assertEqual(1, len(invalid_rows))
            self.assertIn("invalid.db", invalid_rows[0]["db_file"])


class _FakeInfo:
    def __init__(self, filename: str, file_size: int, compress_size: int) -> None:
        self.filename = filename
        self.file_size = file_size
        self.compress_size = compress_size


def _scene(scene_id: str, source_db_file: str, *, actor_count: int, ego_speed_mps: float) -> dict[str, object]:
    actors = []
    for index in range(actor_count):
        actors.append(
            {
                "visible": True,
                "x_m": float(index + 1),
                "y_m": 0.5 * float(index),
                "vx_mps": -1.0,
                "vy_mps": 0.2,
            }
        )
    return {
        "scene_id": scene_id,
        "source_db_file": source_db_file,
        "ego_state": {"speed_mps": ego_speed_mps},
        "route": {"command": "left", "heading_error_rad": 0.2},
        "actors": actors,
        "expert_trajectory": [
            {"x_m": 1.0, "y_m": 0.0, "heading_rad": 0.1},
            {"x_m": 1.5, "y_m": 0.2, "heading_rad": 0.2},
        ],
    }


if __name__ == "__main__":
    unittest.main()
