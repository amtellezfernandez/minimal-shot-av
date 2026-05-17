from __future__ import annotations

import argparse
import importlib.util
import math
import sys
import tempfile
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_alpasim_oracle_actor_proxy.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_alpasim_oracle_actor_proxy", SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BuildAlpaSimOracleActorProxyTests(unittest.TestCase):
    def test_discover_asl_files_recurses_run_directory(self) -> None:
        module = _load_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asl = root / "run" / "rollouts" / "scene" / "rollout.asl"
            asl.parent.mkdir(parents=True)
            asl.write_bytes(b"stub")

            discovered = module._discover_asl_files([], [root])

        self.assertEqual([asl.resolve()], discovered)

    def test_world_actors_for_frame_preserves_world_coordinates(self) -> None:
        module = _load_module()
        poses = {
            "EGO": module.Pose2D(x=10.0, y=5.0, yaw=math.pi / 2.0),
            "actor-1": module.Pose2D(x=10.0, y=15.0, yaw=math.pi / 2.0),
        }
        velocities = {
            "EGO": (0.0, 4.0),
            "actor-1": (0.0, 2.0),
        }
        actor_defs = {
            "actor-1": module.ActorDef(
                actor_id="actor-1",
                label="automobile",
                length=4.5,
                width=2.0,
                height=1.5,
            )
        }
        args = argparse.Namespace(
            forward_min_m=-12.0,
            forward_max_m=90.0,
            lateral_max_m=30.0,
            max_actors_per_frame=24,
        )

        actors, source_ego_pose = module._world_actors_for_frame(poses, velocities, actor_defs, args)

        self.assertEqual(1, len(actors))
        self.assertAlmostEqual(10.0, actors[0]["world_x"], places=5)
        self.assertAlmostEqual(15.0, actors[0]["world_y"], places=5)
        self.assertAlmostEqual(0.0, actors[0]["world_vx"], places=5)
        self.assertAlmostEqual(2.0, actors[0]["world_vy"], places=5)
        self.assertAlmostEqual(10.0, actors[0]["source_rel_x"], places=5)
        self.assertAlmostEqual(0.0, actors[0]["source_rel_y"], places=5)
        self.assertEqual("actor-1", actors[0]["label"])
        self.assertEqual("alpasim_oracle_actor_proxy", actors[0]["source"])
        self.assertAlmostEqual(10.0, source_ego_pose["world_x"], places=5)
        self.assertAlmostEqual(5.0, source_ego_pose["world_y"], places=5)


if __name__ == "__main__":
    unittest.main()
