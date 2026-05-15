from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "score_train.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("score_train", SCRIPT)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ScoreTrainTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load_module()

    def test_split_rollouts_keeps_rollout_groups_intact(self) -> None:
        rollout_id = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int64)
        train_mask, val_mask = self.module.split_rollouts(rollout_id, val_frac=0.25, seed=7)
        self.assertEqual(len(rollout_id), int(train_mask.sum() + val_mask.sum()))
        for rid in np.unique(rollout_id):
            group = rollout_id == rid
            self.assertTrue(train_mask[group].all() or val_mask[group].all())

    def test_kinematic_encoder_output_shape(self) -> None:
        encoder = self.module.KinematicTrajectoryEncoder(point_horizon=8)
        traj = torch.zeros(6, 8, 3)
        speed = torch.ones(6)
        out = encoder(traj, speed, 2.0)
        self.assertEqual((6, self.module.TRAJ_HIDDEN), tuple(out.shape))

    def test_trajectory_scorer_accepts_interaction_tensor(self) -> None:
        model = self.module.TrajectoryScorer(point_horizon=8, interaction_dim=7)
        state = torch.zeros(4, 10)
        traj = torch.zeros(4, 9, 8, 3)
        interaction = torch.zeros(4, 9, 7)
        out = model(state, traj, 2.0, interaction)
        self.assertEqual((4, 9), tuple(out.shape))


if __name__ == "__main__":
    unittest.main()
