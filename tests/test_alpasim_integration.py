from __future__ import annotations

from pathlib import Path
import tomllib
import unittest


class AlpaSimIntegrationTests(unittest.TestCase):
    def test_pyproject_registers_alpasim_plugin_entrypoints(self) -> None:
        pyproject = tomllib.loads(Path("pyproject.toml").read_text())
        self.assertEqual(
            pyproject["project"]["entry-points"]["alpasim.models"]["spotlight_reflex"],
            "minimal_shot_av.alpasim_spotlight:SpotlightReflexAlpaSimModel",
        )
        self.assertEqual(
            pyproject["project"]["entry-points"]["alpasim.configs"]["spotlight_reflex"],
            "minimal_shot_av.alpasim_configs",
        )

    def test_alpasim_driver_config_exists(self) -> None:
        config_path = Path("src/minimal_shot_av/alpasim_configs/driver/spotlight_reflex.yaml")
        config = config_path.read_text()
        self.assertIn("model_type: spotlight_reflex", config)
        self.assertIn("output_frequency_hz: 4", config)


if __name__ == "__main__":
    unittest.main()
