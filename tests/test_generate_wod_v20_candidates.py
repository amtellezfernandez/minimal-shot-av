from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from minimal_shot_av.model.rfs_metric import RfsReference
from minimal_shot_av.model.wod_e2e import WodE2EPreferenceFrame


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_wod_v20_candidates.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_wod_v20_candidates", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sample_frame() -> WodE2EPreferenceFrame:
    trajectory = [(float(index), 0.0) for index in range(1, 21)]
    return WodE2EPreferenceFrame(
        frame_name="segment-a-100",
        past_trajectory=[(-1.0, 0.0), (0.0, 0.0)],
        future_trajectory=trajectory,
        intent=1,
        init_speed_mps=4.0,
        references=[RfsReference("human", trajectory, 9.0)],
    )


class FakeV20Planner:
    def candidate_trajectories_for_frame(self, frame, **kwargs):
        del frame
        return [
            ("v20_mode_0", [(float(index), 0.0) for index in range(1, 21)], 8.5),
            ("v20_mode_1", [(float(index), 0.2) for index in range(1, 21)], 7.5),
        ][: kwargs["top_k"]]


class GenerateWodV20CandidatesTests(unittest.TestCase):
    def test_write_candidate_jsonl_emits_standard_candidate_records(self) -> None:
        module = _load_module()
        with TemporaryDirectory() as directory:
            output = Path(directory) / "candidates.jsonl"
            count = module.write_candidate_jsonl([sample_frame()], output, FakeV20Planner(), top_k=2)
            payloads = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(2, count)
        self.assertEqual("segment-a-100", payloads[0]["frame_name"])
        self.assertEqual("wod_v20_neural_system2_planner", payloads[0]["source"])
        self.assertEqual("v20_mode_0", payloads[0]["candidate_name"])
        self.assertEqual(8.5, payloads[0]["confidence"])
        self.assertEqual(20, len(payloads[0]["trajectory_20wp_4hz"]))


if __name__ == "__main__":
    unittest.main()
