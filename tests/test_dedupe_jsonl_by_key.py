from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "dedupe_jsonl_by_key.py"


class DedupeJsonlByKeyTests(unittest.TestCase):
    def test_deduplicates_jsonl_by_key(self) -> None:
        with TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "rows.jsonl"
            output = Path(tmpdir) / "deduped.jsonl"
            source.write_text(
                "\n".join(
                    [
                        json.dumps({"frame_name": "a", "value": 1}),
                        json.dumps({"frame_name": "b", "value": 2}),
                        json.dumps({"frame_name": "a", "value": 3}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [sys.executable, str(SCRIPT), str(source), "--output", str(output)],
                cwd=ROOT,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            report = json.loads(completed.stdout)
            rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(3, report["total"])
        self.assertEqual(2, report["kept"])
        self.assertEqual(1, report["duplicates"])
        self.assertEqual(["a", "b"], [row["frame_name"] for row in rows])


if __name__ == "__main__":
    unittest.main()
