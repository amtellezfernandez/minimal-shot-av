from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_alpasim_transfer_matrix.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_alpasim_transfer_matrix", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunAlpaSimTransferMatrixTests(unittest.TestCase):
    def test_job_command_wraps_scene_batch_runner(self) -> None:
        module = _load_module()
        args = argparse.Namespace(
            python="python",
            mode="print",
            timeout=900,
            topology="1gpu",
            driver_warmup_seconds=10.0,
            scene_offset=0,
            scene_limit=10,
            rerun_existing=False,
            continue_on_error=True,
            alpasim_root=Path("/tmp/alpasim"),
            wizard_arg=["wizard.timeout=1200"],
        )

        command = module._job_command(
            args,
            model="token_dagger_iter2",
            preset="front_camera_10scene_smoke",
            batch_dir=Path("/tmp/matrix/job"),
        )

        self.assertIn("run_alpasim_scene_batch.py", command[1])
        self.assertEqual("token_dagger_iter2", command[command.index("--model") + 1])
        self.assertEqual("front_camera_10scene_smoke", command[command.index("--scene-preset") + 1])
        self.assertEqual("10", command[command.index("--scene-limit") + 1])
        self.assertIn("--continue-on-error", command)
        self.assertEqual("wizard.timeout=1200", command[-1])

    def test_batch_status_detects_completed_and_failed(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            complete = root / "complete"
            complete.mkdir()
            (complete / "batch-manifest.json").write_text(
                json.dumps({"scene_ids": ["a", "b"]}),
                encoding="utf-8",
            )
            (complete / "batch-status.json").write_text(
                json.dumps({"runs": [{"result": "completed"}, {"result": "completed"}]}),
                encoding="utf-8",
            )
            failed = root / "failed"
            failed.mkdir()
            (failed / "batch-manifest.json").write_text(
                json.dumps({"scene_ids": ["a", "b"]}),
                encoding="utf-8",
            )
            (failed / "batch-status.json").write_text(
                json.dumps({"runs": [{"result": "completed"}, {"result": "failed"}]}),
                encoding="utf-8",
            )
            partial = root / "partial"
            partial.mkdir()
            (partial / "batch-manifest.json").write_text(
                json.dumps({"scene_ids": ["a", "b", "c"]}),
                encoding="utf-8",
            )
            (partial / "batch-status.json").write_text(
                json.dumps({"runs": [{"result": "completed"}, {"result": "completed"}]}),
                encoding="utf-8",
            )

            self.assertEqual("completed", module._batch_status(complete))
            self.assertEqual("failed", module._batch_status(failed))
            self.assertEqual("partial", module._batch_status(partial))

    def test_update_manifest_status_marks_running_pending_and_completed(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = []
            for name in ("job_a", "job_b", "job_c"):
                batch_dir = root / name
                batch_dir.mkdir()
                jobs.append(
                    {
                        "model": name,
                        "scene_preset": "preset",
                        "batch_dir": str(batch_dir),
                        "command": ["python", "fake.py"],
                        "status": "missing",
                    }
                )
            (root / "job_a" / "batch-status.json").write_text(
                json.dumps({"runs": [{"result": "completed"}]}),
                encoding="utf-8",
            )
            manifest_path = root / "matrix-manifest.json"
            manifest_path.write_text(
                json.dumps({"jobs": jobs}, indent=2),
                encoding="utf-8",
            )

            module._update_manifest_status(
                manifest_path,
                jobs,
                active_index=1,
                completed_indexes={0},
            )

            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            statuses = [job["status"] for job in payload["jobs"]]
            self.assertEqual(["completed", "running", "pending"], statuses)
            self.assertIn("updated_at", payload)


if __name__ == "__main__":
    unittest.main()
