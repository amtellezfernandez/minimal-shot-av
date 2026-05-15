from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ALPASIM_ROOT = ROOT / "alpasim"
DEFAULT_RUNS_ROOT = ROOT / "runs"
SCENE_PRESET_ROOT = ROOT / "src" / "minimal_shot_av" / "simulator" / "alpasim_scene_presets"

MODEL_PRESETS = {
    "spotlight_reflex": {
        "config_file": ROOT / "src" / "minimal_shot_av" / "simulator" / "alpasim_configs" / "driver" / "spotlight_reflex.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": None,
        "driver_env": {},
    },
    "token_dagger_iter2": {
        "config_file": ROOT / "src" / "minimal_shot_av" / "simulator" / "alpasim_configs" / "driver" / "token_dagger_bc.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter2" / "token_dagger_bc.pt",
        "driver_env": {},
    },
    "token_dagger_srcdecay": {
        "config_file": ROOT / "src" / "minimal_shot_av" / "simulator" / "alpasim_configs" / "driver" / "token_dagger_srcdecay.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter3_srcdecay" / "token_dagger_bc.pt",
        "driver_env": {},
    },
    "token_dagger_iter2_clamped": {
        "config_file": ROOT
        / "src"
        / "minimal_shot_av"
        / "simulator"
        / "alpasim_configs"
        / "driver"
        / "token_dagger_bc_clamped.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter2" / "token_dagger_bc.pt",
        "driver_env": {
            "MSA_TOKENBC_TRAJECTORY_MODE": "clamped_lateral",
            "MSA_TOKENBC_MAX_LATERAL_OFFSET_M": "2.0",
        },
    },
    "token_dagger_iter2_hybrid": {
        "config_file": ROOT / "src" / "minimal_shot_av" / "simulator" / "alpasim_configs" / "driver" / "token_dagger_bc.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter2" / "token_dagger_bc.pt",
        "driver_env": {
            "MSA_TOKENBC_SELECTION_MODE": "hybrid_veto",
            "MSA_TOKENBC_HYBRID_TOP_K": "3",
            "MSA_TOKENBC_HYBRID_GEOMETRIC_WEIGHT": "0.75",
            "MSA_TOKENBC_HYBRID_POLICY_TEMPERATURE": "1.0",
            "MSA_TOKENBC_HYBRID_VETO_MARGIN": "8.0",
            "MSA_TOKENBC_HYBRID_MAX_GEOMETRIC_RANK": "2",
            "MSA_TOKENBC_SELECTION_LOG_PATH": "{run_dir}/driver/selection-log.jsonl",
        },
    },
    "token_dagger_iter2_hybrid_clamped": {
        "config_file": ROOT
        / "src"
        / "minimal_shot_av"
        / "simulator"
        / "alpasim_configs"
        / "driver"
        / "token_dagger_bc_clamped.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter2" / "token_dagger_bc.pt",
        "driver_env": {
            "MSA_TOKENBC_SELECTION_MODE": "hybrid_veto",
            "MSA_TOKENBC_HYBRID_TOP_K": "3",
            "MSA_TOKENBC_HYBRID_GEOMETRIC_WEIGHT": "0.75",
            "MSA_TOKENBC_HYBRID_POLICY_TEMPERATURE": "1.0",
            "MSA_TOKENBC_HYBRID_VETO_MARGIN": "8.0",
            "MSA_TOKENBC_HYBRID_MAX_GEOMETRIC_RANK": "2",
            "MSA_TOKENBC_TRAJECTORY_MODE": "clamped_lateral",
            "MSA_TOKENBC_MAX_LATERAL_OFFSET_M": "2.0",
            "MSA_TOKENBC_SELECTION_LOG_PATH": "{run_dir}/driver/selection-log.jsonl",
        },
    },
    "token_dagger_srcdecay_clamped": {
        "config_file": ROOT
        / "src"
        / "minimal_shot_av"
        / "simulator"
        / "alpasim_configs"
        / "driver"
        / "token_dagger_srcdecay_clamped.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter3_srcdecay" / "token_dagger_bc.pt",
        "driver_env": {
            "MSA_TOKENBC_TRAJECTORY_MODE": "clamped_lateral",
            "MSA_TOKENBC_MAX_LATERAL_OFFSET_M": "2.0",
        },
    },
    "token_dagger_srcdecay_hybrid": {
        "config_file": ROOT / "src" / "minimal_shot_av" / "simulator" / "alpasim_configs" / "driver" / "token_dagger_srcdecay.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter3_srcdecay" / "token_dagger_bc.pt",
        "driver_env": {
            "MSA_TOKENBC_SELECTION_MODE": "hybrid_veto",
            "MSA_TOKENBC_HYBRID_TOP_K": "3",
            "MSA_TOKENBC_HYBRID_GEOMETRIC_WEIGHT": "0.75",
            "MSA_TOKENBC_HYBRID_POLICY_TEMPERATURE": "1.0",
            "MSA_TOKENBC_HYBRID_VETO_MARGIN": "8.0",
            "MSA_TOKENBC_HYBRID_MAX_GEOMETRIC_RANK": "2",
            "MSA_TOKENBC_SELECTION_LOG_PATH": "{run_dir}/driver/selection-log.jsonl",
        },
    },
    "token_dagger_srcdecay_hybrid_clamped": {
        "config_file": ROOT
        / "src"
        / "minimal_shot_av"
        / "simulator"
        / "alpasim_configs"
        / "driver"
        / "token_dagger_srcdecay_clamped.yaml",
        "wizard_driver": "spotlight_reflex",
        "checkpoint": ROOT / "artifacts" / "bc_models_iter3_srcdecay" / "token_dagger_bc.pt",
        "driver_env": {
            "MSA_TOKENBC_SELECTION_MODE": "hybrid_veto",
            "MSA_TOKENBC_HYBRID_TOP_K": "3",
            "MSA_TOKENBC_HYBRID_GEOMETRIC_WEIGHT": "0.75",
            "MSA_TOKENBC_HYBRID_POLICY_TEMPERATURE": "1.0",
            "MSA_TOKENBC_HYBRID_VETO_MARGIN": "8.0",
            "MSA_TOKENBC_HYBRID_MAX_GEOMETRIC_RANK": "2",
            "MSA_TOKENBC_TRAJECTORY_MODE": "clamped_lateral",
            "MSA_TOKENBC_MAX_LATERAL_OFFSET_M": "2.0",
            "MSA_TOKENBC_SELECTION_LOG_PATH": "{run_dir}/driver/selection-log.jsonl",
        },
    },
}

SCENE_PRESETS = {
    "fresh_3scene": SCENE_PRESET_ROOT / "fresh_3scene.yaml",
    "front_camera_10scene_smoke": SCENE_PRESET_ROOT / "front_camera_10scene_smoke.yaml",
    "front_camera_30scene_merged": SCENE_PRESET_ROOT / "front_camera_30scene_merged.yaml",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or launch matched AlpaSim local external-driver runs."
    )
    parser.add_argument(
        "--mode",
        choices=("print", "driver", "wizard", "both"),
        default="print",
        help="What to launch. 'print' only writes commands and metadata.",
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PRESETS),
        default="token_dagger_iter2",
        help="Model/checkpoint preset to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint override for learned models.",
    )
    parser.add_argument(
        "--scene-preset",
        choices=tuple(SCENE_PRESETS),
        default="fresh_3scene",
        help="Scene list preset extracted from an existing wizard config.",
    )
    parser.add_argument(
        "--scene-id",
        action="append",
        default=[],
        help="Extra scene id override. If provided, replaces the preset scene list.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run output directory. Defaults to runs/alpasim_<model>_<preset>_<timestamp>.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Parent directory for generated run directories.",
    )
    parser.add_argument(
        "--alpasim-root",
        type=Path,
        default=None,
        help="Path to local AlpaSim checkout with .venv and src/{driver,wizard}. Defaults to $ALPASIM_ROOT or ./alpasim.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6789,
        help="External driver port.",
    )
    parser.add_argument(
        "--baseport",
        type=int,
        default=6000,
        help="Wizard baseport override.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Wizard timeout override in seconds.",
    )
    parser.add_argument(
        "--topology",
        default="1gpu",
        help="AlpaSim wizard topology override, e.g. 1gpu or 8gpu_12rollouts.",
    )
    parser.add_argument(
        "--wizard-dry-run",
        action="store_true",
        help="Pass wizard.dry_run=true for config validation without executing rollouts.",
    )
    parser.add_argument(
        "--driver-warmup-seconds",
        type=float,
        default=10.0,
        help="Delay between starting the external driver and launching wizard in mode=both.",
    )
    parser.add_argument(
        "--allow-existing-run-dir",
        action="store_true",
        help="Reuse an existing run dir instead of failing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    alpasim_root = _resolve_alpasim_root(args.alpasim_root)
    driver_project = alpasim_root / "src" / "driver"
    wizard_project = alpasim_root / "src" / "wizard"
    alpasim_python = alpasim_root / ".venv" / "bin" / "python"
    alpasim_wizard = alpasim_root / ".venv" / "bin" / "alpasim_wizard"

    if not driver_project.is_dir():
        raise SystemExit(f"AlpaSim driver project not found: {driver_project}")
    if not wizard_project.is_dir():
        raise SystemExit(f"AlpaSim wizard project not found: {wizard_project}")
    if not alpasim_python.is_file():
        raise SystemExit(f"AlpaSim virtualenv python not found: {alpasim_python}")
    if not alpasim_wizard.is_file():
        raise SystemExit(f"AlpaSim wizard binary not found: {alpasim_wizard}")

    scene_ids = _scene_ids(args.scene_preset, args.scene_id)
    run_dir = _resolve_run_dir(args)
    _prepare_run_dir(run_dir, allow_existing=args.allow_existing_run_dir)

    model_preset = MODEL_PRESETS[args.model]
    checkpoint = args.checkpoint.resolve() if args.checkpoint else model_preset["checkpoint"]
    if checkpoint is not None:
        checkpoint = Path(checkpoint).resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"Checkpoint not found: {checkpoint}")

    # Keep the external-driver config separate from wizard-generated files.
    # The wizard writes its own driver-config.yaml into the run directory, which
    # can otherwise overwrite the learned-model config after launch.
    driver_config_path = run_dir / "external-driver-config.yaml"
    _write_driver_config(
        template_path=Path(model_preset["config_file"]),
        output_path=driver_config_path,
        checkpoint=checkpoint,
        port=args.port,
        output_dir=run_dir / "driver",
        force_cuda=args.model != "spotlight_reflex",
    )

    driver_cmd = _driver_command(
        alpasim_python=alpasim_python,
        driver_config_path=driver_config_path,
    )
    driver_env = _driver_env(model_preset.get("driver_env", {}), run_dir=run_dir)
    wizard_cmd = _wizard_command(
        alpasim_wizard=alpasim_wizard,
        wizard_driver=model_preset["wizard_driver"],
        run_dir=run_dir,
        scene_ids=scene_ids,
        baseport=args.baseport,
        port=args.port,
        timeout=args.timeout,
        topology=args.topology,
        dry_run=args.wizard_dry_run,
    )

    metadata = {
        "model": args.model,
        "scene_preset": args.scene_preset,
        "scene_ids": scene_ids,
        "port": args.port,
        "baseport": args.baseport,
        "timeout": args.timeout,
        "topology": args.topology,
        "wizard_dry_run": args.wizard_dry_run,
        "driver_config_template": str(model_preset["config_file"]),
        "driver_config_path": str(driver_config_path),
        "wizard_driver": model_preset["wizard_driver"],
        "checkpoint": str(checkpoint) if checkpoint else None,
        "driver_env": driver_env,
        "driver_command": driver_cmd,
        "wizard_command": wizard_cmd,
    }
    (run_dir / "launch-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (run_dir / "driver-command.sh").write_text(_shell_script(driver_cmd, env=driver_env))
    (run_dir / "wizard-command.sh").write_text(_shell_script(wizard_cmd))

    print(f"Run dir: {run_dir}")
    print(f"Scenes ({len(scene_ids)}): {', '.join(scene_ids)}")
    print()
    print("Driver:")
    print("  " + _format_cmd(driver_cmd, env=driver_env))
    print()
    print("Wizard:")
    print("  " + _format_cmd(wizard_cmd))

    if args.mode == "print":
        return

    if args.mode == "driver":
        raise SystemExit(_run(driver_cmd, cwd=ROOT, env=driver_env))
    if args.mode == "wizard":
        raise SystemExit(_run(wizard_cmd, cwd=ROOT))

    driver_stdout = (run_dir / "driver.stdout.log").open("w")
    driver_stderr = (run_dir / "driver.stderr.log").open("w")
    process = subprocess.Popen(
        driver_cmd,
        cwd=ROOT,
        env=_merged_env(driver_env),
        stdout=driver_stdout,
        stderr=driver_stderr,
        text=True,
        start_new_session=True,
    )
    try:
        time.sleep(args.driver_warmup_seconds)
        wizard_code = _run(wizard_cmd, cwd=ROOT)
    finally:
        _terminate_process_group(process)
        driver_stdout.close()
        driver_stderr.close()
    raise SystemExit(wizard_code)


def _scene_ids(scene_preset: str, explicit_scene_ids: list[str]) -> list[str]:
    if explicit_scene_ids:
        return explicit_scene_ids
    preset_path = SCENE_PRESETS[scene_preset]
    if not preset_path.is_file():
        raise SystemExit(f"Scene preset file not found: {preset_path}")
    payload = yaml.safe_load(preset_path.read_text())
    scene_ids = payload.get("scenes", {}).get("scene_ids", [])
    if not scene_ids:
        raise SystemExit(f"No scene_ids found in {preset_path}")
    return [str(scene_id) for scene_id in scene_ids]


def _resolve_alpasim_root(cli_value: Path | None) -> Path:
    if cli_value is not None:
        return cli_value.resolve()
    env_value = os.getenv("ALPASIM_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser().resolve()
    return DEFAULT_ALPASIM_ROOT.resolve()


def _resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (args.runs_root.resolve() / f"alpasim_{args.model}_{args.scene_preset}_{stamp}")


def _prepare_run_dir(run_dir: Path, *, allow_existing: bool) -> None:
    if run_dir.exists():
        if not allow_existing:
            raise SystemExit(f"Run dir already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True)


def _driver_command(
    *,
    alpasim_python: Path,
    driver_config_path: Path,
) -> list[str]:
    return [
        str(alpasim_python),
        "-m",
        "alpasim_driver.main",
        f"--config-path={driver_config_path.parent}",
        f"--config-name={driver_config_path.name}",
    ]


def _write_driver_config(
    *,
    template_path: Path,
    output_path: Path,
    checkpoint: Path | None,
    port: int,
    output_dir: Path,
    force_cuda: bool,
) -> None:
    if not template_path.is_file():
        raise SystemExit(f"Driver config template not found: {template_path}")
    payload = yaml.safe_load(template_path.read_text())
    if isinstance(payload.get("log_level"), str) and "${wizard." in payload["log_level"]:
        payload["log_level"] = "INFO"
    payload["port"] = int(port)
    payload["output_dir"] = str(output_dir)
    model_cfg = payload.setdefault("model", {})
    if checkpoint is not None:
        model_cfg["checkpoint_path"] = str(checkpoint)
    if force_cuda:
        model_cfg["device"] = "cuda"
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _wizard_command(
    *,
    alpasim_wizard: Path,
    wizard_driver: str,
    run_dir: Path,
    scene_ids: list[str],
    baseport: int,
    port: int,
    timeout: int,
    topology: str,
    dry_run: bool,
) -> list[str]:
    cmd = [
        str(alpasim_wizard),
        "deploy=local_external_driver",
        f"topology={topology}",
        f"driver={wizard_driver}",
        f"wizard.log_dir={run_dir}",
        f"wizard.baseport={baseport}",
        f"wizard.timeout={timeout}",
        f"wizard.external_services.driver=[localhost:{port}]",
        f"wizard.dry_run={'true' if dry_run else 'false'}",
        f"scenes.scene_ids={json.dumps(scene_ids)}",
    ]
    return cmd


def _shell_script(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    return "#!/usr/bin/env bash\nset -euo pipefail\n" + _format_cmd(cmd, env=env) + "\n"


def _format_cmd(cmd: list[str], *, env: dict[str, str] | None = None) -> str:
    prefix = ""
    if env:
        prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env.items()) + " "
    return prefix + " ".join(shlex.quote(part) for part in cmd)


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> int:
    result = subprocess.run(cmd, cwd=cwd, env=_merged_env(env), check=False)
    return int(result.returncode)


def _driver_env(values: dict[str, str], *, run_dir: Path) -> dict[str, str]:
    return {str(key): str(value).format(run_dir=run_dir) for key, value in values.items()}


def _merged_env(extra: dict[str, str] | None) -> dict[str, str]:
    env = os.environ.copy()
    if extra:
        env.update(extra)
    return env


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


if __name__ == "__main__":
    main()
