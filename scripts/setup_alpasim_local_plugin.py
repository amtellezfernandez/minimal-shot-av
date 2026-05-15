from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from shutil import which
import shutil


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPASIM_ROOT = ROOT / "alpasim"
ALPASIM_OVERRIDE_ROOT = ROOT / "third_party" / "alpasim_overrides"
REQUIRED_MODELS = ("spotlight_reflex", "token_dagger_bc")
ALPASIM_CORE_DEPENDENCIES = (
    "PyYAML>=6",
    "GitPython",
    "boto3",
    "click",
    "dataclasses-json>=0.6.7",
    "filelock",
    "grpcio",
    "grpcio-tools",
    "huggingface_hub",
    "hydra-core",
    "matplotlib",
    "numpy",
    "omegaconf",
    "opencv-python-headless",
    "pandas",
    "pandas-stubs",
    "pillow",
    "polars>=1.0.0",
    "protobuf>=4.0.0,<5.0.0",
    "pyarrow",
    "pygame>=2.5.0",
    "pytest",
    "pytest-asyncio",
    "rich",
    "setuptools<82",
    "torch",
    "tqdm",
    "types-PyYAML",
    "typing-extensions",
)
ALPASIM_EDITABLE_PACKAGES = (
    "src/plugins",
    "src/grpc",
    "src/utils",
    "src/driver",
    "src/wizard",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install this repo into the local AlpaSim driver env and verify plugin discovery."
    )
    parser.add_argument(
        "--alpasim-root",
        type=Path,
        default=None,
        help="AlpaSim checkout root. Defaults to $ALPASIM_ROOT or ./alpasim.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip installation and only verify the current AlpaSim driver registry.",
    )
    parser.add_argument(
        "--skip-overrides",
        action="store_true",
        help="Do not copy repo-tracked AlpaSim override files into ALPASIM_ROOT before checking.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uv_bin = _require_uv()
    alpasim_root = _resolve_alpasim_root(args.alpasim_root)
    driver_project = alpasim_root / "src" / "driver"
    venv_python = alpasim_root / ".venv" / "bin" / "python"

    if not driver_project.is_dir():
        raise SystemExit(f"AlpaSim driver project not found: {driver_project}")
    if not args.skip_overrides:
        _apply_local_alpasim_overrides(alpasim_root)
    _bootstrap_alpasim_venv(alpasim_root, uv_bin=uv_bin)
    if not venv_python.is_file():
        raise SystemExit(f"AlpaSim virtualenv python not found after bootstrap: {venv_python}")

    if not args.check_only:
        _run(
            [
                uv_bin,
                "pip",
                "install",
                "--cache-dir",
                str(ROOT / ".uv-cache"),
                "--python",
                str(venv_python),
                "-e",
                str(ROOT),
                "PyYAML>=6",
            ],
            cwd=ROOT,
        )

    plugin_names = _plugin_names(venv_python)
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
        "  ALPASIM_ROOT="
        + shlex_quote(str(alpasim_root))
        + " ./.venv/bin/python scripts/run_alpasim_local_external.py "
        "--mode print --model token_dagger_iter2_hybrid_clamped --scene-preset fresh_3scene"
    )


def _resolve_alpasim_root(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.getenv("ALPASIM_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_ALPASIM_ROOT.resolve()


def _require_uv() -> str:
    uv_bin = which("uv")
    if uv_bin:
        return uv_bin
    raise SystemExit(
        "uv is required for AlpaSim setup. Install it first, e.g. "
        "`python3 -m pip install --user uv`, then rerun this script."
    )


def _apply_local_alpasim_overrides(alpasim_root: Path) -> None:
    if not ALPASIM_OVERRIDE_ROOT.is_dir():
        return
    copied: list[str] = []
    for source in ALPASIM_OVERRIDE_ROOT.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(ALPASIM_OVERRIDE_ROOT)
        target = alpasim_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(relative))
    if copied:
        print("Applied repo-tracked AlpaSim overrides:")
        for relative in copied:
            print(f"  {relative}")


def _bootstrap_alpasim_venv(alpasim_root: Path, *, uv_bin: str) -> None:
    venv_root = alpasim_root / ".venv"
    venv_python = venv_root / "bin" / "python"
    if not venv_python.is_file():
        _run([uv_bin, "venv", str(venv_root)], cwd=alpasim_root)

    _run(
        [
            uv_bin,
            "pip",
            "install",
            "--cache-dir",
            str(ROOT / ".uv-cache"),
            "--python",
            str(venv_python),
            *ALPASIM_CORE_DEPENDENCIES,
        ],
        cwd=alpasim_root,
    )

    for relative in ALPASIM_EDITABLE_PACKAGES:
        package_path = alpasim_root / relative
        if not package_path.is_dir():
            raise SystemExit(f"Expected AlpaSim package path missing: {package_path}")
        _run(
            [
                uv_bin,
                "pip",
                "install",
                "--cache-dir",
                str(ROOT / ".uv-cache"),
                "--python",
                str(venv_python),
                "--no-deps",
                "-e",
                str(package_path),
            ],
            cwd=alpasim_root,
        )


def _plugin_names(venv_python: Path) -> list[str]:
    script = "\n".join(
        [
            "from importlib.metadata import entry_points",
            "import sys",
            "names = []",
            "failures = []",
            "for ep in sorted(entry_points(group='alpasim.models'), key=lambda item: item.name):",
            "    try:",
            "        ep.load()",
            "        names.append(ep.name)",
            "    except Exception as exc:",
            "        failures.append(f'{ep.name}: {exc}')",
            "if failures:",
            "    sys.stderr.write('Skipped unloadable model entry points:\\n' + '\\n'.join(failures) + '\\n')",
            "print('\\n'.join(names))",
        ]
    )
    result = _run([str(venv_python), "-c", script], cwd=ROOT, capture_output=True)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def shlex_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


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
