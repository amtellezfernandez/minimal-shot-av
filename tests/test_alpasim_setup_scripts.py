from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_alpasim_local_external import _resolve_alpasim_root as resolve_run_root
from scripts.setup_alpasim_local_plugin import _resolve_alpasim_root as resolve_setup_root


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
