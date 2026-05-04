from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_wod_gcs_shard_samples.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("export_wod_gcs_shard_samples", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ExportWodGcsShardSamplesTests(unittest.TestCase):
    def test_manifest_summarizes_incremental_shard_export(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "frames.jsonl"
            output.write_text("{}\n{}\n", encoding="utf-8")
            args = argparse.Namespace(
                shard_glob="gs://bucket/training*.tfrecord-*",
                shard_start=3,
                shard_count=2,
                records_per_shard=5,
                output=output,
                manifest_output=Path(tmpdir) / "manifest.json",
            )
            rows = [
                {"status": "ok", "shard_index": 3},
                {"status": "error", "shard_index": 4},
            ]

            manifest = module._manifest(args, rows)

        self.assertEqual("wod_gcs_shard_sample_export_manifest_v1", manifest["schema"])
        self.assertEqual(2, manifest["total_rows"])
        self.assertEqual(1, manifest["successful_shards"])
        self.assertEqual(1, manifest["failed_shards"])


if __name__ == "__main__":
    unittest.main()
