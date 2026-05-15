from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_alpasim_local_external import _resolve_alpasim_root as resolve_run_root
from scripts.run_alpasim_local_external import _driver_command, _wizard_command
from scripts.setup_alpasim_local_plugin import (
    _apply_local_alpasim_overrides,
    _bootstrap_alpasim_venv,
    ALPASIM_CORE_DEPENDENCIES,
    ALPASIM_EDITABLE_PACKAGES,
    _resolve_alpasim_root as resolve_setup_root,
)


class AlpaSimSetupScriptTests(unittest.TestCase):
    def test_run_launcher_prefers_cli_root_over_env(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cli_root = Path(tmp) / "cli"
            env_root = Path(tmp) / "env"
            cli_root.mkdir()
            env_root.mkdir()
            with patch.dict(os.environ, {"ALPASIM_ROOT": str(env_root)}):
                self.assertEqual(cli_root.resolve(), resolve_run_root(cli_root))

    def test_run_launcher_uses_env_root_when_cli_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_root = Path(tmp) / "env"
            env_root.mkdir()
            with patch.dict(os.environ, {"ALPASIM_ROOT": str(env_root)}, clear=False):
                self.assertEqual(env_root.resolve(), resolve_run_root(None))

    def test_setup_script_uses_env_root_when_cli_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_root = Path(tmp) / "env"
            env_root.mkdir()
            with patch.dict(os.environ, {"ALPASIM_ROOT": str(env_root)}, clear=False):
                self.assertEqual(env_root.resolve(), resolve_setup_root(None))

    def test_setup_script_applies_repo_tracked_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alpasim_root = Path(tmp) / "alpasim"
            source_root = Path(tmp) / "overrides"
            target_file = alpasim_root / "src" / "wizard" / "alpasim_wizard" / "deployment" / "docker_compose.py"
            source_file = source_root / "src" / "wizard" / "alpasim_wizard" / "deployment" / "docker_compose.py"
            source_file.parent.mkdir(parents=True)
            source_file.write_text("override-file\n", encoding="utf-8")

            with patch("scripts.setup_alpasim_local_plugin.ALPASIM_OVERRIDE_ROOT", source_root):
                _apply_local_alpasim_overrides(alpasim_root)

            self.assertEqual("override-file\n", target_file.read_text(encoding="utf-8"))

    def test_setup_script_also_copies_driver_model_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alpasim_root = Path(tmp) / "alpasim"
            source_root = Path(tmp) / "overrides"
            target_file = (
                alpasim_root
                / "src"
                / "driver"
                / "src"
                / "alpasim_driver"
                / "models"
                / "__init__.py"
            )
            source_file = (
                source_root
                / "src"
                / "driver"
                / "src"
                / "alpasim_driver"
                / "models"
                / "__init__.py"
            )
            source_file.parent.mkdir(parents=True)
            source_file.write_text("driver-model-override\n", encoding="utf-8")

            with patch("scripts.setup_alpasim_local_plugin.ALPASIM_OVERRIDE_ROOT", source_root):
                _apply_local_alpasim_overrides(alpasim_root)

            self.assertEqual("driver-model-override\n", target_file.read_text(encoding="utf-8"))

    def test_bootstrap_alpasim_venv_uses_minimal_editable_install_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            alpasim_root = Path(tmp) / "alpasim"
            for relative in ALPASIM_EDITABLE_PACKAGES:
                (alpasim_root / relative).mkdir(parents=True, exist_ok=True)
            venv_python = alpasim_root / ".venv" / "bin" / "python"
            calls: list[list[str]] = []

            def fake_run(cmd: list[str], *, cwd: Path, capture_output: bool = False):
                calls.append(cmd)
                if cmd[:2] == ["uv", "venv"]:
                    venv_python.parent.mkdir(parents=True, exist_ok=True)
                    venv_python.write_text("", encoding="utf-8")
                return type("Result", (), {"stdout": "", "stderr": "", "returncode": 0})()

            with patch("scripts.setup_alpasim_local_plugin._run", side_effect=fake_run):
                _bootstrap_alpasim_venv(alpasim_root, uv_bin="uv")

            self.assertGreaterEqual(len(calls), 2)
            self.assertEqual(["uv", "venv", str(alpasim_root / ".venv")], calls[0])
            self.assertIn("pip", calls[1])
            self.assertTrue(set(ALPASIM_CORE_DEPENDENCIES).issubset(set(calls[1])))
            editable_targets = [cmd[-1] for cmd in calls[2:]]
            self.assertEqual(
                [str(alpasim_root / relative) for relative in ALPASIM_EDITABLE_PACKAGES],
                editable_targets,
            )

    def test_driver_command_uses_alpasim_venv_python(self) -> None:
        cmd = _driver_command(
            alpasim_python=Path("/tmp/alpasim/.venv/bin/python"),
            driver_config_path=Path("/tmp/run/external-driver-config.yaml"),
        )
        self.assertEqual("/tmp/alpasim/.venv/bin/python", cmd[0])
        self.assertEqual(["-m", "alpasim_driver.main"], cmd[1:3])

    def test_wizard_command_uses_alpasim_venv_binary(self) -> None:
        cmd = _wizard_command(
            alpasim_wizard=Path("/tmp/alpasim/.venv/bin/alpasim_wizard"),
            wizard_driver="spotlight_reflex",
            run_dir=Path("/tmp/run"),
            scene_ids=["scene-1"],
            baseport=6000,
            port=6789,
            timeout=600,
            topology="1gpu",
            dry_run=False,
        )
        self.assertEqual("/tmp/alpasim/.venv/bin/alpasim_wizard", cmd[0])
        self.assertIn("deploy=local_external_driver", cmd)
