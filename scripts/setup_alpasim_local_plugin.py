from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPASIM_ROOT = ROOT / "alpasim"
REQUIRED_MODELS = ("spotlight_reflex", "token_dagger_bc")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install this repo into the local AlpaSim driver env and verify plugin discovery."
    )
    parser.add_argument("--alpasim-root", type=Path, default=DEFAULT_ALPASIM_ROOT)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip installation and only verify the current AlpaSim driver registry.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alpasim_root = args.alpasim_root.resolve()
    driver_project = alpasim_root / "src" / "driver"
    venv_python = alpasim_root / ".venv" / "bin" / "python"

    if not driver_project.is_dir():
        raise SystemExit(f"AlpaSim driver project not found: {driver_project}")
    if not venv_python.is_file():
        raise SystemExit(f"AlpaSim virtualenv python not found: {venv_python}")

    if not args.check_only:
        _run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(venv_python),
                "-e",
                str(ROOT),
            ],
            cwd=ROOT,
        )

    plugin_names = _plugin_names(driver_project)
    missing = [name for name in REQUIRED_MODELS if name not in plugin_names]
    if missing:
        raise SystemExit(
            "AlpaSim plugin registration is incomplete. "
            f"Missing {missing}; discovered {plugin_names}."
        )

    print("AlpaSim driver registry OK")
    print(f"Models: {', '.join(plugin_names)}")
    print()
    print("Next:")
    print(
        "  ./.venv/bin/python scripts/run_alpasim_local_external.py "
        "--mode print --model token_dagger_iter2 --scene-preset fresh_3scene"
    )


def _plugin_names(driver_project: Path) -> list[str]:
    script = (
        "from importlib.metadata import entry_points; "
        "print('\\n'.join(sorted(ep.name for ep in entry_points(group='alpasim.models'))))"
    )
    result = _run(
        ["uv", "run", "--project", str(driver_project), "python", "-c", script],
        cwd=ROOT,
        capture_output=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=capture_output,
        check=False,
    )
    if result.returncode != 0:
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    if not capture_output:
        if result.stdout:
            sys.stdout.write(result.stdout)
        if result.stderr:
            sys.stderr.write(result.stderr)
    return result


if __name__ == "__main__":
    main()
