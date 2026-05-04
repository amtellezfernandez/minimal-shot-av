from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_minor_visual_gallery.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_minor_visual_gallery", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BuildMinorVisualGalleryTests(unittest.TestCase):
    def test_gallery_links_rollouts_and_evidence(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name, cluster in (
                ("minor_construction_seed1", "construction"),
                ("minor_fod_seed2", "foreign object debris"),
                ("minor_spotlight_seed3", "spotlight"),
            ):
                demo = root / name
                demo.mkdir()
                (demo / "latest_rollout.svg").write_text("<svg></svg>", encoding="utf-8")
                (demo / "latest_rollout.json").write_text(
                    json.dumps(
                        {
                            "scenario": {"cluster": cluster, "seed": 1},
                            "rollout": {
                                "success": True,
                                "collision": False,
                                "steps": [{"x": 1.0, "y": 0.0, "min_obstacle_distance": 2.0}],
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            _write_json(root / "minor_eval" / "scenario_eval.json", {"runs": [_row("wod", "construction")]})
            _write_json(root / "minor_ood_eval" / "scenario_eval.json", {"runs": [_row("gauntlet", "case")]})
            _write_json(
                root / "minor_alpasignal_bridge" / "alpasignal_bridge_audit.json",
                {"valid": True, "adapter": "adapter", "cases": [{}, {}, {}]},
            )
            _write_json(
                root / "minor_runtime" / "minor_runtime_constraints.json",
                {
                    "valid": True,
                    "run_count": 3,
                    "total_steps": 300,
                    "p95_step_latency_ms": 0.5,
                    "target_step_ms": 50,
                    "throughput_steps_per_second": 2000,
                },
            )

            output = module.build_gallery(root, root / "minor_visual_gallery")
            text = output.read_text(encoding="utf-8")

        self.assertIn("SoTA Minor Visual Gallery", text)
        self.assertIn("../minor_construction_seed1/latest_rollout.svg", text)
        self.assertIn("Runtime Constraints", text)
        self.assertIn("AlpaSignal Bridge", text)


def _row(suite: str, cluster: str) -> dict[str, object]:
    return {
        "suite": suite,
        "cluster": cluster,
        "success": True,
        "benchmark_pass": True,
        "collision": False,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
