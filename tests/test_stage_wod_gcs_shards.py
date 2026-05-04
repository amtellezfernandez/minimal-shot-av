from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "stage_wod_gcs_shards.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage_wod_gcs_shards", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StageWodGcsShardsTests(unittest.TestCase):
    def test_split_object_prefix_matches_waymo_bucket_names(self) -> None:
        module = _load_module()

        self.assertEqual("training", module.split_object_prefix("train"))
        self.assertEqual("val", module.split_object_prefix("val"))
        self.assertEqual("test", module.split_object_prefix("test"))

    def test_select_shard_window_returns_bounded_slice(self) -> None:
        module = _load_module()
        uris = [f"gs://bucket/training.tfrecord-{index:05d}-of-00263" for index in range(10)]

        selected = module.select_shard_window(uris, shard_start=3, max_shards=4)

        self.assertEqual(uris[3:7], selected)

    def test_stage_plan_for_test_builds_copy_commands(self) -> None:
        module = _load_module()
        uris = [f"gs://bucket/test.tfrecord-{index:05d}-of-00266" for index in range(4)]

        plan = module.stage_plan_for_test(
            bucket="gs://bucket",
            split="test",
            uris=uris,
            shard_start=1,
            max_shards=2,
            stage_root=Path("stage"),
        )

        self.assertEqual(2, plan["selected_shards"])
        self.assertEqual(uris[1:3], plan["uris"])
        self.assertEqual(
            [["gsutil", "-m", "cp", "-n", uris[1], "stage/test"], ["gsutil", "-m", "cp", "-n", uris[2], "stage/test"]],
            plan["commands"],
        )


if __name__ == "__main__":
    unittest.main()
