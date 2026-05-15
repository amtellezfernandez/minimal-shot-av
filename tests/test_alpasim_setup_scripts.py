from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_alpasim_local_external import _resolve_alpasim_root as resolve_run_root
from scripts.setup_alpasim_local_plugin import (
    _apply_local_alpasim_overrides,
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
